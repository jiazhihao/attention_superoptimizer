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

#include "aso/kernel/customized.h"
#include "aso/kernel/operator.h"
// #include "aso/kernel/operator_factory.h"
#include "aso/kernel/device_tensor.h"
#include <vector>

namespace aso {
namespace kernel {

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(void);
  DTensor new_input(std::vector<int> const &dims,
                    aso::type::DataType data_type);
  DTensor matmul(DTensor const &A, DTensor const &B);
  DTensor customized(std::vector<DTensor> const &inputs,
                     CustomizedOp::ExecutionPlan const &plan);

  std::vector<aso::kernel::Operator *> operators;
  // std::unordered_map<std::pair<int, int>, DTensor, pair_hash> tensors;
  // std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>
  // edges; std::vector<std::vector<SrcEdge>> edges;
  // aso::kernel::OperatorFactory *operator_factory;
};

} // namespace kernel
} // namespace aso
