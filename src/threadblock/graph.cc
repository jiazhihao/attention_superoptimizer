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
  params.num_smem_inputs = 0;
  params.num_smem_outputs = 0;
  params.num_dmem_inputs = 0;
  params.num_dmem_outputs = 0;

  assert(params.num_operators <= KernelParams::MAX_NUM_OPERATORS);
  for (size_t i = 0; i < operators.size(); i++) {
    params.operator_types[i] = operators[i]->op_type;
    params.operator_num_inputs[i] = operators[i]->input_tensors.size();
    params.operator_num_outputs[i] = operators[i]->output_tensors.size();
    for (int j = 0; j < params.operator_num_inputs[i]; j++) {
      params.smem_inputs[params.num_smem_inputs++] =
          operators[i]->input_tensors[j];
      assert(params.num_smem_inputs <= KernelParams::MAX_TOTAL_SMEM_INPUTS);
    }
    for (int j = 0; j < params.operator_num_outputs[i]; j++) {
      params.smem_outputs[params.num_smem_outputs++] =
          operators[i]->output_tensors[j];
      assert(params.num_smem_outputs <= KernelParams::MAX_TOTAL_SMEM_OUTPUTS);
    }
    if (operators[i]->op_type == aso::type::TB_INPUT_OP) {
      TBInputOp *input_op = static_cast<TBInputOp *>(operators[i]);
      params.input_map[params.num_dmem_inputs] = input_op->input_map;
      params.forloop_dim[params.num_dmem_inputs] = input_op->forloop_dim;
      params.dmem_inputs[params.num_dmem_inputs++] = input_op->dtensor;
      assert(params.num_dmem_inputs <= KernelParams::MAX_NUM_DMEM_INPUTS);
    }
    if (operators[i]->op_type == aso::type::TB_OUTPUT_OP) {
      TBOutputOp *output_op = static_cast<TBOutputOp *>(operators[i]);
      params.output_map = output_op->output_map;
      params.dmem_outputs[params.num_dmem_outputs++] = output_op->dtensor;
      assert(params.num_dmem_outputs <= KernelParams::MAX_NUM_DMEM_OUTPUTS);
    }
  }
  return params;
}

Graph::operator json() const {
  json j = {{"grid_dim", grid_dim},
            {"block_dim", block_dim},
            {"forloop_range", forloop_range},
            {"operators", {}},
            {"smem_offset", smem_offset}};
  for (TBOperator *const op : operators) {
    j["operators"].push_back(*op);
  }
  return j;
}

} // namespace threadblock
} // namespace aso
