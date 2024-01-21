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
#include "aso/kernel/graph.h"
#include "aso/kernel/operator_factory.h"
#include "aso/threadblock/graph.h"
#include "aso/threadblock/smem_tensor.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

DTensor Graph::customized(std::vector<DTensor> const &inputs,
                          CustomizedOp::ExecutionPlan const &plan) {
  assert(false);
}

Operator *OperatorFactory::get_or_create_customized(
    std::vector<DTensor> const &inputs,
    CustomizedOp::ExecutionPlan const &plan) {
  CustomizedKey key(inputs, plan);
  CustomizedOp *op = nullptr;
  if (customized.find(key) != customized.end()) {
    op = customized[key];
  } else {
    op = new CustomizedOp(inputs, plan);
    customized[key] = op;
  }
  return op;
}

CustomizedOp::CustomizedOp(std::vector<DTensor> const &_inputs,
                           ExecutionPlan const &_plan)
    : aso::kernel::Operator(_inputs), plan(_plan) {
  assert(_inputs.size() == plan.input_map.size());
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
  }
  auto const &ops = plan.ops;
  aso::threadblock::Graph bgraph(inputs);
  for (auto const &op : ops) {
    std::vector<aso::threadblock::STensor> my_inputs;

    for (auto const &idx : op.second) {
      assert(bgraph.tensors.find(idx) != bgraph.tensors.end());
      my_inputs.push_back(bgraph.tensors[idx]);
    }
    switch (op.first->operator_type()) {
      case aso::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
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
  for (auto const &t : bgraph.tensors) {
    bool found = false;
    for (auto const &e : bgraph.edges) {
      if (e.second == t.first) {
        found = true;
        break;
      }
    }
    if (!found) {
      // TODO: change output tensor_shape
      DTensor tensor_shape; // = t.second.get_shape();
      output_tensors.push_back(tensor_shape);
    }
  }
}

CustomizedOp::~CustomizedOp() {}

CustomizedKey::CustomizedKey(std::vector<DTensor> const &_inputs,
                             CustomizedOp::ExecutionPlan const &_plan)
    : inputs(_inputs), plan(_plan) {}

bool CustomizedKey::operator==(CustomizedKey const &b) const {
  if (inputs.size() != b.inputs.size()) {
    return false;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    if (inputs[i] != b.inputs[i]) {
      return false;
    }
  }
  // TODO: check execution plan equivalence
  assert(false);
  return true;
}

} // namespace kernel
} // namespace aso

namespace std {
size_t hash<aso::kernel::CustomizedKey>::operator()(
    aso::kernel::CustomizedKey const &key) const {
  assert(false);
  size_t ret = 0;
  return ret;
}
}; // namespace std
