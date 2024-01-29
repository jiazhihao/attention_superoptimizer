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

#include "aso/threadblock/graph.h"
#include "aso/utils/hash_utils.h"

namespace aso {
namespace threadblock {

Graph::Graph(dim3 _grid_dim, dim3 _block_dim, int _forloop_range)
    : grid_dim(_grid_dim), block_dim(_block_dim), forloop_range(_forloop_range),
      smem_offset(0) {}

const size_t MAX_SMEM_SIZE = 96 * 1024; // 96 KB

size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
  size_t h1 = std::hash<int>{}(p.first);
  size_t h2 = std::hash<int>{}(p.second);
  hash_combine(h1, h2);
  return h1;
}

off_t Graph::allocate(STensor const &tensor) {
  off_t ret = smem_offset;
  smem_offset += tensor.size();
  assert(smem_offset <= (off_t)MAX_SMEM_SIZE);
  allocated_tensors.push_back(std::make_pair(ret, tensor.size()));
  return ret;
}

void Graph::free(STensor const &tensor) {
  assert(allocated_tensors.size() > 0);
  assert(allocated_tensors.back().first == tensor.smem_offset);
  assert(allocated_tensors.back().second == tensor.size());
  smem_offset -= allocated_tensors.back().second;
  allocated_tensors.pop_back();
}

void Graph::free(std::vector<STensor> const &tensors) {
  for (int i = tensors.size() - 1; i >= 0; i--) {
    free(tensors[i]);
  }
}

KernelParams Graph::get_kernel_params() {
  KernelParams params;
  params.forloop_range = this->forloop_range;
  params.num_operators = operators.size();
  params.num_input_dtensors = 0;
  params.num_output_dtensors = 0;
  assert(params.num_operators <= KernelParams::MAX_NUM_OPERATORS);
  for (size_t i = 0; i < operators.size(); i++) {
    params.operator_types[i] = operators[i]->op_type;
    assert(operators[i]->input_tensors.size() <= KernelParams::MAX_NUM_INPUTS);
    for (size_t j = 0; j < operators[i]->input_tensors.size(); j++) {
      params.input_tensors[i][j] = operators[i]->input_tensors[j];
    }
    assert(operators[i]->output_tensors.size() <=
           KernelParams::MAX_NUM_OUTPUTS);
    for (size_t j = 0; j < operators[i]->output_tensors.size(); j++) {
      params.output_tensors[i][j] = operators[i]->output_tensors[j];
    }
    if (operators[i]->op_type == aso::type::TB_INPUT_OP) {
      TBInputOp *input_op = static_cast<TBInputOp *>(operators[i]);
      params.input_device_tensors[params.num_input_dtensors++] =
          input_op->dtensor;
    }
    if (operators[i]->op_type == aso::type::TB_OUTPUT_OP) {
      TBOutputOp *output_op = static_cast<TBOutputOp *>(operators[i]);
      params.output_device_tensors[params.num_output_dtensors++] =
          output_op->dtensor;
    }
  }
  return params;
}

} // namespace threadblock
} // namespace aso
