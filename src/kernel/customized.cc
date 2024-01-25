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

#include "aso/kernel/customized.h"
#include "aso/kernel/device_memory_manager.h"
#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"
#include "aso/threadblock/reduction.h"
#include "aso/threadblock/smem_tensor.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

using aso::threadblock::ExecutionPlan;
using aso::threadblock::STensor;

std::vector<DTensor> Graph::customized(std::vector<DTensor> const &inputs,
                                       ExecutionPlan const &plan) {
  KNOperator *op = create_customized_op(inputs, plan);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        ExecutionPlan const &plan) {
  KNCustomizedOp *op = new KNCustomizedOp(inputs, plan);
  return op;
}

KNCustomizedOp::KNCustomizedOp(std::vector<DTensor> const &_inputs,
                               ExecutionPlan const &_plan)
    : KNOperator(aso::type::KN_CUSTOMIZED_OP, _inputs), plan(_plan) {
  assert(_inputs.size() == plan.input_map.size());
  assert(plan.forloop_dim.size() == plan.input_map.size());
  // Step 1: computing input shapes
  // Step 1: creating a stensor for each input
  std::vector<DTensor> inputs;
  for (size_t i = 0; i < _inputs.size(); i++) {
    DTensor shape = _inputs[i];
    for (int d = 0; d < 3; d++) {
      int dim_idx = -1;
      int dim_div = 1;
      if (d == 0 && plan.grid_dim.x > 1) {
        assert(plan.input_map[i].x >= 0);
        dim_idx = plan.input_map[i].x;
        dim_div = plan.grid_dim.x;
      }
      if (d == 1 && plan.grid_dim.y > 1) {
        assert(plan.input_map[i].y > 0);
        dim_idx = plan.input_map[i].y;
        dim_div = plan.grid_dim.y;
      }
      if (d == 2 && plan.grid_dim.z > 1) {
        assert(plan.input_map[i].z > 0);
        dim_idx = plan.input_map[i].z;
        dim_div = plan.grid_dim.z;
      }
      assert(shape.dim[dim_idx] > 0);
      assert(shape.dim[dim_idx] % dim_div == 0);
      shape.dim[dim_idx] /= dim_div;
    }
    inputs.push_back(shape);
    if (plan.forloop_dim[i] >= 0) {
      int dim_idx = plan.forloop_dim[i];
      assert(shape.dim[dim_idx] > 0);
      assert(shape.dim[dim_idx] % plan.forloop_range == 0);
      shape.dim[dim_idx] /= plan.forloop_range;
    }
    std::vector<int> dims;
    for (int i = 0; i < shape.num_dims; i++) {
      dims.push_back(shape.dim[i]);
    }
    bgraph.new_input(dims, shape.data_type);
  }

  auto const &ops = plan.ops;
  for (auto const &op : ops) {
    std::vector<STensor> my_inputs;
    for (auto const &idx : op.second) {
      // assert(bgraph.tensors.find(idx) != bgraph.tensors.end());
      // my_inputs.push_back(bgraph.tensors[idx]);
      assert((int)bgraph.operators.size() > idx.first);
      assert((int)bgraph.operators[idx.first]->output_tensors.size() >
             idx.second);
      my_inputs.push_back(
          bgraph.operators[idx.first]->output_tensors[idx.second]);
    }
    switch (op.first) {
      case aso::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
        break;
      }
      case aso::type::TB_EXP_OP: {
        assert(my_inputs.size() == 1);
        bgraph.exp(my_inputs[0]);
        break;
      }
      case aso::type::TB_REDUCTION_0_OP:
      case aso::type::TB_REDUCTION_1_OP:
      case aso::type::TB_REDUCTION_2_OP:
      {
        assert(my_inputs.size() == 1);
        int reduce_dim = op.first - aso::type::TB_REDUCTION_0_OP;
        bgraph.reduction(my_inputs[0], reduce_dim);
        break;
      }
      default: {
        assert(false && "Unsupported kernel operator");
      }
    }
  }

  assert(output_tensors.size() == 0);
  // Identify outputs: a tensor is an output if it is not used by
  // any other operators
  for (auto const &op : bgraph.operators) {
    for (size_t i = 0; i < op->output_tensors.size(); i++) {
      bool found = false;
      for (auto const &op2 : bgraph.operators) {
        for (size_t j = 0; j < op2->input_tensors.size(); j++) {
          if (op2->input_tensors[j] == op->output_tensors[i]) {
            found = true;
          }
        }
      }
      if (!found) {
        // TODO: change output tensor_shape
        STensor stensor = op->output_tensors[i];
        DTensor dtensor;
        dtensor.num_dims = stensor.num_dims;
        dtensor.data_type = stensor.data_type;
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        for (int d = 0; d < dtensor.num_dims; d++) {
          dtensor.dim[d] = stensor.dim[d];
        }
        for (int d = 0; d < 3; d++) {
          int dim_idx = -1;
          int dim_div = 1;
          if (d == 0 && plan.grid_dim.x > 1) {
            assert(plan.output_map.x >= 0);
            dim_idx = plan.output_map.x;
            dim_div = plan.grid_dim.x;
          }
          if (d == 1 && plan.grid_dim.y > 1) {
            assert(plan.output_map.y > 0);
            dim_idx = plan.output_map.y;
            dim_div = plan.grid_dim.y;
          }
          if (d == 2 && plan.grid_dim.z > 1) {
            assert(plan.output_map.z > 0);
            dim_idx = plan.output_map.z;
            dim_div = plan.grid_dim.z;
          }
          assert(dtensor.dim[dim_idx] > 0);
          assert(dtensor.dim[dim_idx] % dim_div == 0);
          dtensor.dim[dim_idx] /= dim_div;
        }
        for (int d = dtensor.num_dims - 1; d >= 0; d--) {
          dtensor.stride[d] = (d == dtensor.num_dims - 1)
                                  ? 1
                                  : dtensor.stride[d + 1] * dtensor.dim[d + 1];
        }
        DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        dtensor.data_ptr = dmm->allocate(dtensor.size());
        output_tensors.push_back(dtensor);
      }
    }
  }
}

KNCustomizedOp::~KNCustomizedOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i].data_ptr);
  }
}

} // namespace kernel
} // namespace aso
