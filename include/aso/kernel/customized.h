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
#include "aso/threadblock/operator.h"
#include <tuple>
#include <vector_types.h>

namespace aso {
namespace kernel {

// using ExecutionPlan =
// std::vector<std::pair<aso::threadblock::Operator*,std::vector<std::pair<int,
// int> > > >;

class CustomizedOp : public aso::kernel::Operator {
public:
  struct Params {
    const static int MAX_NUM_OPERATORS = 8;
    const static int MAX_NUM_INPUTS = 3;
    int forloop_range;
    int num_operators;
    aso::type::OperatorType operator_types[MAX_NUM_OPERATORS];
    aso::threadblock::STensor input_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
    aso::threadblock::STensor output_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
    // input dtensors in device memory
    int num_inputs;
    off_t input_tensor_offset_in_smem[MAX_NUM_INPUTS];
  };
  class ExecutionPlan {
  public:
    std::vector<std::pair<aso::threadblock::Operator *,
                          std::vector<std::pair<int, int>>>>
        ops;
    std::vector<dim3> input_map;
    dim3 output_map; // assume that all output must use the same map
    std::vector<int> forloop_dim;
    int forloop_range;
    dim3 grid_dim, block_dim, warp_dim;
  };

  CustomizedOp(std::vector<DTensor> const &inputs, ExecutionPlan const &plan);
  ~CustomizedOp();
  aso::type::OperatorType operator_type() const;
  void run();
  bool profile(ProfileResult &profile);

public:
  ExecutionPlan plan;
  // aso::threadblock::Graph *operator_graph;
};

class CustomizedKey {
public:
  CustomizedKey(std::vector<DTensor> const &inputs,
                CustomizedOp::ExecutionPlan const &plan);
  bool operator==(CustomizedKey const &b) const;
  std::vector<DTensor> inputs;
  CustomizedOp::ExecutionPlan plan;
};

} // namespace kernel
} // namespace aso

namespace std {
template <>
struct hash<aso::kernel::CustomizedKey> {
  size_t operator()(aso::kernel::CustomizedKey const &) const;
};
} // namespace std
