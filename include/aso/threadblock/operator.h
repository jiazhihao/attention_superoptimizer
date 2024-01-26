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
#include "aso/threadblock/smem_tensor.h"
#include "aso/type.h"
#include <vector>
#include <vector_types.h>

namespace aso {
namespace threadblock {

class Graph;

class TBOperator {
public:
  TBOperator(Graph *graph, aso::type::TBOperatorType);
  TBOperator(Graph *graph, aso::type::TBOperatorType, STensor const &input1);
  TBOperator(Graph *graph,
             aso::type::TBOperatorType,
             STensor const &input1,
             STensor const &input2);
  TBOperator(Graph *graph,
             aso::type::TBOperatorType,
             std::vector<STensor> const &inputs);
  ~TBOperator();

public:
  Graph *bgraph;
  aso::type::TBOperatorType op_type;
  std::vector<STensor> input_tensors;
  std::vector<STensor> output_tensors;
};

class TBInputOp : public TBOperator {
public:
  TBInputOp(Graph *_graph,
            aso::kernel::DTensor const &dtensor,
            int3 input_map,
            int forloop_dim);
  ~TBInputOp();

public:
  aso::kernel::DTensor dtensor;
};

class TBOutputOp : public TBOperator {
public:
  TBOutputOp(Graph *_graph, STensor const &stensor, int3 output_map);
  ~TBOutputOp();

public:
  aso::kernel::DTensor dtensor;
};

} // namespace threadblock
} // namespace aso
