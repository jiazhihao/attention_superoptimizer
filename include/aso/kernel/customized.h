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
#include "aso/kernel/operator.h"
#include "aso/threadblock/graph.h"
#include "aso/threadblock/operator.h"
#include <tuple>
#include <vector_types.h>

namespace aso {
namespace kernel {

class KNCustomizedOp : public aso::kernel::KNOperator {
public:
  struct Params {
    const static int MAX_NUM_OPERATORS = 8;
    const static int MAX_NUM_INPUTS = 3;
    int forloop_range;
    int num_operators;
    aso::type::TBOperatorType operator_types[MAX_NUM_OPERATORS];
    aso::threadblock::STensor input_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
    aso::threadblock::STensor output_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
    // input dtensors in device memory
    int num_inputs;
    off_t input_tensor_offset_in_smem[MAX_NUM_INPUTS];
  };

  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 aso::threadblock::ExecutionPlan const &plan);
  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 aso::threadblock::Graph const &_graph);
  ~KNCustomizedOp();
  void run();
  bool profile(ProfileResult &profile);

public:
  aso::threadblock::ExecutionPlan plan;
  aso::threadblock::Graph bgraph;
};

} // namespace kernel
} // namespace aso
