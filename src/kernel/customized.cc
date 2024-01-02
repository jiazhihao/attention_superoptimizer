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

#include "aso/kernel/matmul.h"
#include "aso/kernel/graph.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

using aso::kernel::customized::ExecutionPlan;

TensorShape Graph::customized(std::vector<TensorShape> const &inputs) {
}

Operator *OperatorFactory::get_or_create_customized(
    std::vector<TensorShape> const &inputs,
    ExecutionPlan const& plan) {
  customized::Key key(inputs, plan);
  customized::Operator *op = nullptr;
  if (customized.find(key) != customized.end()) {
    op = customized[key];
  } else {
    op = new customized::Operator(inputs, plan);
    customized[key] = op;
  }
  return op;
}

namespace customized {

Operator::Operator(std::vector<TensorShape> const &_inputs,
                   ExecutionPlan const &_plan) 
    : aso::kernel::Operator(_inputs), plan(_plan) {
  assert(_inputs.size() == plan.grad_map.size());
  std::vector<TensorShape> inputs;
  for (int i = 0; i < _inputs.size(); i++) {
    TensorShape shape = _inputs[i];
    for (int d = 0; d < 3; d++) {
      int dim_idx = -1;
      int dim_div = 1;
      if (d == 0 && plan.grid_dim.x > 1) {
        assert(plan.grid_map[i].x >= 0);
        dim_idx = plan.grid_map[i].x;
        dim_div = plan.grid_dim.x;
      }
      if (d == 1 && plan.grid_dim.y > 1) {
        assert(plan.grid_map[i].y > 0);
        dim_idx = plan.grid_map[i].y;
        dim_div = plan.grid_dim.y;
      }
      if (d == 2 && plan.grid_dim.z > 1) {
        assert(plan.grid_map[i].z > 0);
        dim_idx = plan.grid_map[i].z;
        dim_div = plan.grid_dim.z;
      }
      assert(shape.dim[dim_idx] > 0);
      assert(shape.dim[dim_idx] % dim_div == 0);
      shape.dim[dim_idx] /= dim_div;
    }
  }
}

Operator::~Operator() {}

Key::Key(std::vector<TensorShape> const &_inputs,
         ExecutionPlan const &_plan) : inputs(_inputs), plan(_plan) {}

bool Key::operator==(Key const &b) const {
  if (inputs.size() != b.inputs.size())
    return false;
  for (int i = 0; i < inputs.size(); i++) {
    if (inputs[i] != b.inputs[i])
      return false;
  }
  // TODO: check execution plan equivalence
  assert(false);
  return true;
}

} // namespace matmul
} // namespace kernel
} // namespace aso

namespace std {
size_t hash<aso::kernel::customized::Key>::operator()(
    aso::kernel::customized::Key const &key) const {
  assert(false);
  size_t ret = 0;
  return ret;
}
}; // namespace std
