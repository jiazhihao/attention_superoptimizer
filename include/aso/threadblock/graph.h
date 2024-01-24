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

#include "aso/kernel/device_tensor.h"
#include "aso/threadblock/operator.h"
#include "aso/threadblock/smem_tensor.h"
#include <vector>
#include <vector_types.h>

namespace aso {
namespace threadblock {

class ExecutionPlan {
public:
  std::vector<std::pair<TBOperator *, std::vector<std::pair<int, int>>>> ops;
  std::vector<dim3> input_map;
  dim3 output_map; // assume that all output must use the same map
  std::vector<int> forloop_dim;
  int forloop_range;
  dim3 grid_dim, block_dim, warp_dim;
};

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(std::vector<aso::kernel::DTensor> const &_inputs);
  Graph();
  STensor new_input(std::vector<int> const &dims, aso::type::DataType);
  STensor matmul(STensor const &A, STensor const &B);
  STensor exp(STensor const &A);
  STensor reduction(STensor const &A, int dim);

  std::vector<aso::threadblock::TBOperator *> operators;
  // std::unordered_map<std::pair<int, int>, STensor, pair_hash> tensors;
  // std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>
  // edges;
  //  std::vector<std::vector<SrcEdge>> edges;
  //  aso::kernel::OperatorFactory *operator_factory;
};

} // namespace threadblock
} // namespace aso
