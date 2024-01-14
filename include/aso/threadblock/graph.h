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

#include "aso/threadblock/operator.h"
#include "aso/tensor.h"
#include <vector>

namespace aso {
namespace threadblock {

class Graph {
private:
  struct pair_hash {
    size_t operator () (const std::pair<int,int> &p) const;
  };

public:
  Graph(std::vector<aso::TensorShape> const &_inputs);
  Tensor matmul(Tensor const &A, Tensor const &B);

  std::vector<aso::threadblock::Operator *> operators;
  std::unordered_map<std::pair<int, int>, Tensor, pair_hash> tensors;
  std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>edges;
  // std::vector<std::vector<SrcEdge>> edges;
  // aso::kernel::OperatorFactory *operator_factory;
};

} // namespace threadblock
} // namespace aso