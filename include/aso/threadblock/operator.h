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
#include "aso/threadblock/smem_tensor.h"
#include "aso/type.h"
#include <vector>

namespace aso {
namespace threadblock {

class Graph;

class TBOperator {
public:
  TBOperator(Graph* graph,
             aso::type::TBOperatorType,
             STensor const &input1);
  TBOperator(Graph* graph,
             aso::type::TBOperatorType,
             STensor const &input1,
             STensor const &input2);
  TBOperator(Graph* graph,
             aso::type::TBOperatorType,
             std::vector<STensor> const &inputs);
  ~TBOperator();
public:
  Graph *bgraph;
  aso::type::TBOperatorType op_type;
  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

} // namespace threadblock
} // namespace aso
