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

#pragma once

#include "aso/kernel/operator.h"
#include "aso/threadblock/operator.h"
#include "aso/tensor.h"
#include <vector_types.h>
#include <tuple>

namespace aso {
namespace kernel {
namespace customized {

// using ExecutionPlan = std::vector<std::pair<aso::threadblock::Operator*,std::vector<std::pair<int, int> > > >;

class ExecutionPlan {
public:
  std::vector<std::pair<aso::threadblock::Operator*,std::vector<std::pair<int, int> > > > ops;
  std::vector<dim3> input_map;
  dim3 output_map; // assume that all output must use the same map
  std::vector<int> forloop_dim;
  dim3 grid_dim, block_dim, warp_dim;
};

class Operator : public aso::kernel::Operator {
public:
  Operator(std::vector<TensorShape> const &inputs,
           ExecutionPlan const &plan);
  ~Operator();
  aso::type::OperatorType operator_type() const;
public:
  ExecutionPlan plan;
  //aso::threadblock::Graph *operator_graph;
};

class Key {
public:
  Key(std::vector<TensorShape> const &inputs,
      ExecutionPlan const &plan);
  bool operator==(Key const &b) const;
  std::vector<TensorShape> inputs;
  ExecutionPlan plan;
};

} // namespace customized
} // namespace kernel
} // namespace aso

namespace std {
template <>
struct hash<aso::kernel::customized::Key> {
  size_t operator()(aso::kernel::customized::Key const &) const;
};
} // namespace std
