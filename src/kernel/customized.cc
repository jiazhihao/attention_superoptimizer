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
#include "aso/threadblock/operator.h"
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

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        threadblock::Graph const &_graph) {
  size_t output_size = 0;
  for (threadblock::TBOperator *op : _graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      output_size +=
          static_cast<threadblock::TBOutputOp *>(op)->dtensor.data_size();
    }
  }

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + output_size > dmm->total_size) {
    return nullptr;
  }

  KNCustomizedOp *op = new KNCustomizedOp(inputs, _graph);
  return op;
}

KNCustomizedOp::KNCustomizedOp(std::vector<DTensor> const &_inputs,
                               ExecutionPlan const &_plan)
    : KNOperator(aso::type::KN_CUSTOMIZED_OP, _inputs), plan(_plan),
      bgraph(_plan.grid_dim, _plan.block_dim, _plan.forloop_range) {
  assert(_inputs.size() == plan.input_map.size());
  assert(plan.forloop_dim.size() == plan.input_map.size());
  assert(plan.input_smem_layouts.size() == plan.input_map.size());
  // Step 1: computing input shapes
  // Step 1: creating a stensor for each input
  for (size_t i = 0; i < input_tensors.size(); i++) {
    bgraph.new_input(input_tensors[i],
                     plan.input_map[i],
                     plan.forloop_dim[i],
                     plan.input_smem_layouts[i]);
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
      case aso::type::TB_REDUCTION_2_OP: {
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
      if (!found && op->op_type != aso::type::TB_INPUT_OP) {
        // TODO: change output tensor_shape
        STensor stensor = op->output_tensors[i];
        DTensor dtensor = bgraph.new_output(stensor, plan.output_map);
        printf("stensor.offset(%d)\n", stensor.smem_offset);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        dmm->allocate(dtensor);
        // Update dtensor saved by the output operator
        {
          assert(bgraph.operators.back()->op_type == aso::type::TB_OUTPUT_OP);
          aso::threadblock::TBOutputOp *output = static_cast<aso::threadblock::TBOutputOp*>(bgraph.operators.back());
          output->dtensor = dtensor;
        }
        output_tensors.push_back(dtensor);
      }
    }
  }
}

KNCustomizedOp::KNCustomizedOp(std::vector<DTensor> const &_inputs,
                               aso::threadblock::Graph const &_graph)
    : KNOperator(aso::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim, _graph.block_dim, _graph.forloop_range) {
  ExecutionPlan plan;
  plan.grid_dim = _graph.grid_dim;
  plan.block_dim = _graph.block_dim;
  plan.forloop_range = _graph.forloop_range;

  for (auto const &op : _graph.operators) {
    std::vector<STensor> my_inputs;
    std::vector<std::pair<int, int>> indices;
    for (size_t i = 0; i < op->input_tensors.size(); i++) {
      int op_idx = -1, ts_idx = op->input_tensors[i].owner_ts_idx;
      for (size_t l = 0; l < _graph.operators.size(); l++) {
        if (_graph.operators[l] == op->input_tensors[i].owner_op) {
          assert(op_idx == -1);
          op_idx = static_cast<int>(l);
        }
      }
      assert(op_idx != -1);
      my_inputs.push_back(bgraph.operators[op_idx]->output_tensors[ts_idx]);
      indices.push_back({op_idx, ts_idx});
    }
    if (op->op_type != aso::type::TB_INPUT_OP &&
        op->op_type != aso::type::TB_OUTPUT_OP) {
      plan.ops.push_back({op->op_type, indices});
    }
    switch (op->op_type) {
      case aso::type::TB_INPUT_OP: {
        assert(my_inputs.size() == 0);
        aso::threadblock::TBInputOp *input_op =
            static_cast<aso::threadblock::TBInputOp *>(op);
        bgraph.new_input(input_op->dtensor,
                         input_op->input_map,
                         input_op->forloop_dim,
                         input_op->output_tensors[0].layout);
        plan.input_map.push_back(input_op->input_map);
        plan.forloop_dim.push_back(input_op->forloop_dim);
        break;
      }
      case aso::type::TB_OUTPUT_OP: {
        assert(my_inputs.size() == 1);
        aso::threadblock::TBOutputOp *output_op =
            static_cast<aso::threadblock::TBOutputOp *>(op);
        DTensor dtensor =
            bgraph.new_output(my_inputs[0], output_op->output_map);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        dmm->allocate(dtensor);
        output_tensors.push_back(dtensor);
        plan.output_map = output_op->output_map;
        break;
      }
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
      case aso::type::TB_REDUCTION_2_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - aso::type::TB_REDUCTION_0_OP;
        bgraph.reduction(my_inputs[0], reduce_dim);
        break;
      }
      default: {
        assert(false && "Unsupported kernel operator");
      }
    }
  }
}

KNCustomizedOp::~KNCustomizedOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i]);
  }
}

KNCustomizedOp::operator json() const {
  return bgraph;
}

} // namespace kernel
} // namespace aso
