/* Copyright 2023-2024 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aso/kernel/graph.h"
#include "aso/kernel/customized.h"
#include "aso/utils/hash_utils.h"
#include "aso/threadblock/serializer/concat_serializer.h"
#include "aso/threadblock/serializer/element_binary_serializer.h"
#include "aso/threadblock/serializer/element_unary_serializer.h"
#include "aso/threadblock/serializer/input_loader_serializer.h"
#include "aso/threadblock/serializer/matmul_serializer.h"
#include "aso/threadblock/serializer/output_saver_serializer.h"
#include "aso/threadblock/serializer/reduction_serializer.h"

#include <iostream>
#include <map>

namespace aso {
namespace kernel {

std::string dtensor_name(int i) {
  return "dtensor"+std::to_string(i);
}

std::string stensor_name(int i) {
  return "stensor"+std::to_string(i);
}

std::string tensor_dims(DTensor const &tensor) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < tensor.num_dims; i++) {
    ss << tensor.dim[i];
    if (i != tensor.num_dims-1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

std::string block_offset_calculation(int off_x, int off_y, int off_z) {
  if (off_x == 0 && off_y == 0 && off_z == 0) {
    return "0";
  }
  std::stringstream ret;
  if (off_x > 0) {
    ret << "bidx * " << off_x;
  }
  if (off_y > 0) {
    if (off_x > 0) {
      ret << " + ";
    }
    ret << "bidy * " << off_y;
  }
  if (off_z > 0) {
    if (off_x > 0 || off_y > 0) {
      ret << " + ";
    }
    ret << "bidz * " << off_z;
  }
  return ret.str();
}

std::string generate_kernel_code(aso::threadblock::NewKernelParams params,
                                 int forloop_range,
                                 std::string func_name,
                                 std::vector<std::string> input_names,
                                 std::vector<std::string> output_names) {
  using namespace aso::threadblock;
  using namespace std;
  stringstream header;
  stringstream main;
  {
    header << "@triton.jit\n";
    header << "def " << func_name << "(";
    for (size_t i = 0; i < input_names.size(); i++) {
      if (i > 0) {
        header << ", ";
      }
      header << input_names[i];
    }
    for (size_t i = 0; i < output_names.size(); i++) {
      header << ", ";
      header << output_names[i];
    }
    header << "):\n";
  }
  main << "\tfor i in range(" << forloop_range << "):\n";
  map<int, string> stensor_guid_to_name;
  int param_idx = 0;
  for (int op = 0; op < params.num_operators; op++) {
    aso::type::TBOperatorType op_type = params.operator_types[op];
    if (op_type == aso::type::TB_INPUT_OP) {
      int3 input_matrix_row_offset_block_stride;
      int3 input_matrix_column_offset_block_stride;
      int input_matrix_row_offset_forloop_stride;
      int input_matrix_column_offset_forloop_stride;
      int3 global_offset_block_stride;
      int global_offset_forloop_stride;
      int2 dtensor_matrix_shape, stensor_matrix_shape;
      int input_smem_offset;
      aso::layout::DmemLayout dtensor_layout;
      aso::layout::SmemLayout stensor_layout;
      aso::threadblock::deserialize_input_loader_parameters(
          params.parameters,
          param_idx,
          input_matrix_row_offset_block_stride,
          input_matrix_column_offset_block_stride,
          input_matrix_row_offset_forloop_stride,
          input_matrix_column_offset_forloop_stride,
          global_offset_block_stride,
          global_offset_forloop_stride,
          dtensor_matrix_shape,
          stensor_matrix_shape,
          dtensor_layout,
          stensor_layout,
          input_smem_offset);
      header << "\t" << stensor_name(input_smem_offset)
           << " = tl.make_block_ptr(\n"
           << "\t\tbase = " << input_names[op] << " + "
           << block_offset_calculation(global_offset_block_stride.x, global_offset_block_stride.y, global_offset_block_stride.z)
           << ",\n";
      header << "\t\tshape = ("
           << dtensor_matrix_shape.x << ", "
           << dtensor_matrix_shape.y << "),\n";
      header << "\t\tblock_shape = ("
           << stensor_matrix_shape.x << ", "
           << stensor_matrix_shape.y << "),\n";
      string tb_offset_row = block_offset_calculation(input_matrix_row_offset_block_stride.x,
                                                      input_matrix_row_offset_block_stride.y,
                                                      input_matrix_row_offset_block_stride.z);
      string tb_offset_col = block_offset_calculation(input_matrix_column_offset_block_stride.x,
                                                      input_matrix_column_offset_block_stride.y,
                                                      input_matrix_column_offset_block_stride.z);
      header << "\t\toffsets = ("
           << tb_offset_row << ", "
           << tb_offset_col << "),\n";
      // Assume row major layout for now
      header << "\t\tstrides = ("
           << stensor_matrix_shape.y << ", 1)\n";
      header << "\t\torders = (1, 0))\n";
    } else if (op_type == aso::type::TB_OUTPUT_OP) {
      int3 output_matrix_row_offset_block_stride;
      int3 output_matrix_column_offset_block_stride;
      int3 global_offset_block_stride;
      int2 dtensor_matrix_shape, stensor_matrix_shape;
      int input_smem_offset, accum_smem_offset;
      aso::layout::DmemLayout dtensor_layout;
      aso::layout::SmemLayout stensor_layout;
      aso::threadblock::deserialize_output_saver_parameters(
          params.parameters,
          param_idx,
          output_matrix_row_offset_block_stride,
          output_matrix_column_offset_block_stride,
          global_offset_block_stride,
          dtensor_matrix_shape,
          stensor_matrix_shape,
          dtensor_layout,
          stensor_layout,
          input_smem_offset,
          accum_smem_offset);     
    } else if (op_type == aso::type::TB_MATMUL_OP) {
      int m, n, k;
      int A_smem_offset, B_smem_offset, C_smem_offset;
      aso::threadblock::deserialize_matmul_op_parameters(
          params.parameters,
          param_idx,
          m,
          n,
          k,
          A_smem_offset,
          B_smem_offset,
          C_smem_offset);
      main << "\t\t" << stensor_name(C_smem_offset) << " = tl.dot("
           << stensor_name(A_smem_offset) << ", "
           << stensor_name(B_smem_offset) << ", out_dtype=tl.float16)\n";
    } else if (op_type == aso::type::TB_EXP_OP) {
      int smem_offset, num_elements;
      aso::threadblock::deserialize_elementunary_op_parameters(
          params.parameters, param_idx, smem_offset, num_elements);
      main << "\t\t" << stensor_name(smem_offset) << " = tl.math.exp("
           << stensor_name(smem_offset) << ".to(tl.float32)).to(tl.float16)\n";
    } else if (op_type == aso::type::TB_DIV_OP) {
      int3 input1_shape, input2_shape;
      int input1_smem_offset, input2_smem_offset, output_smem_offset;
      aso::threadblock::deserialize_elementbinary_op_parameters(
          params.parameters,
          param_idx,
          input1_shape,
          input2_shape,
          input1_smem_offset,
          input2_smem_offset,
          output_smem_offset);
      main << "\t\t" << stensor_name(output_smem_offset) << " = tl.div("
           << stensor_name(input1_smem_offset) << ", "
           << stensor_name(input2_smem_offset) << ")\n";
    } else if ((op_type >= aso::type::TB_REDUCTION_FIRST_OP_ID) &&
               (op_type <= aso::type::TB_REDUCTION_LAST_OP_ID)) {
      int output_num_elements, reduction_degree, inner_range;
      int input_smem_offset, output_smem_offset;
      aso::threadblock::deserialize_reduction_op_parameters(
          params.parameters,
          param_idx,
          output_num_elements,
          reduction_degree,
          inner_range,
          input_smem_offset,
          output_smem_offset);
      int reduction_dim = -1;
      if (op_type >= aso::type::TB_REDUCTION_0_TO_DIMX_OP &&
          op_type <= aso::type::TB_REDUCTION_2_TO_DIMX_OP) {
        reduction_dim = op_type - aso::type::TB_REDUCTION_0_TO_DIMX_OP;
      } else if (op_type >= aso::type::TB_REDUCTION_0_OP &&
                 op_type <= aso::type::TB_REDUCTION_2_OP) {
        reduction_dim = op_type - aso::type::TB_REDUCTION_0_OP;
        main << "\t\t" << stensor_name(output_smem_offset) << " = tl.sum("
             << stensor_name(input_smem_offset) << ", axis="
             << reduction_dim << ", keep_dims=True)\n";
      } else {
        assert(false);
      }
    }
  }
  assert(params.num_parameters == param_idx);
  return header.str() + main.str();
}

std::string Graph::generate_triton_program() {
  using namespace std;
  stringstream header;
  vector<std::string> kernels;
  map<int, string> dtensor_guid_to_name;
  stringstream launcher;
  stringstream main_program;
  main_program << "def main():\n";
  header << "import triton\nimport torch\nimport triton.language as tl\n";
  for (KNOperator *const op : this->operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        assert(op->output_tensors.size() == 1);
        DTensor output = op->output_tensors[0];
        assert(dtensor_guid_to_name.find(output.guid) == dtensor_guid_to_name.end());
        dtensor_guid_to_name[output.guid] = dtensor_name(output.guid);
        main_program << "\t"
            << dtensor_guid_to_name[output.guid]
            << " = torch.randn("
            << tensor_dims(op->output_tensors[0])
            << ", dtype=torch.float16, device=\"cuda\", requires_grad=False)\n";
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        const KNCustomizedOp *customized = static_cast<const KNCustomizedOp*>(op);
        vector<string> input_names;
        vector<string> output_names;
        for (const auto& t : op->input_tensors) {
          input_names.push_back(dtensor_name(t.guid));
        }
        for (const auto& t : op->output_tensors) {
          output_names.push_back(dtensor_name(t.guid));
        }
        aso::threadblock::NewKernelParams params = customized->bgraph.get_new_kernel_params(false/*fingerprint*/);
        string kernel_code = generate_kernel_code(params, customized->bgraph.forloop_range, "graphdef_kernel", input_names, output_names);
        kernels.push_back(kernel_code);
        break;
      }
      default: {
        //assert(false && "Cannot tritonize this operator");
      }
    }
  }
  stringstream output;
  output << header.str();
  for (const auto & k : kernels) {
    output << k;
  }
  output << launcher.str() << main_program.str();
  return output.str();
}

} // namespace kernel
} // namespace aso
