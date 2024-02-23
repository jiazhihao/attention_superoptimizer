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

struct KernelParams {
  const static int MAX_NUM_OPERATORS = 10;
  const static int MAX_TOTAL_SMEM_INPUTS = 16;
  const static int MAX_TOTAL_SMEM_OUTPUTS = 10;
  const static int MAX_NUM_DMEM_INPUTS = 3;
  const static int MAX_NUM_DMEM_OUTPUTS = 3;
  int forloop_range;
  int num_operators, num_smem_inputs, num_smem_outputs;
  int operator_num_inputs[MAX_NUM_OPERATORS],
      operator_num_outputs[MAX_NUM_OPERATORS];
  aso::type::TBOperatorType operator_types[MAX_NUM_OPERATORS];
  aso::threadblock::STensor smem_inputs[MAX_TOTAL_SMEM_INPUTS];
  aso::threadblock::STensor smem_outputs[MAX_TOTAL_SMEM_OUTPUTS];
  // input dtensors in device memory
  int num_dmem_inputs, num_dmem_outputs;
  aso::kernel::DTensor dmem_inputs[MAX_NUM_DMEM_INPUTS];
  aso::kernel::DTensor dmem_outputs[MAX_NUM_DMEM_INPUTS];
};

class ExecutionPlan {
public:
  std::vector<
      std::pair<aso::type::TBOperatorType, std::vector<std::pair<int, int>>>>
      ops;
  // input-related fields
  std::vector<int3> input_map;
  std::vector<int> forloop_dim;
  std::vector<aso::layout::SmemLayout> input_smem_layouts;
  // output-related fields
  int3 output_map; // assume that all output must use the same map
  int forloop_range;
  dim3 grid_dim, block_dim;
};

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(dim3 grid_dim, dim3 block_dim, int forloop_range);
  // input operator
  STensor new_input(aso::kernel::DTensor const &dtensor,
                    int3 input_map,
                    int forloop_dim,
                    aso::layout::SmemLayout layout);
  TBOperator *create_input_op(aso::kernel::DTensor const &dtensor,
                              int3 input_map,
                              int forloop_dim,
                              aso::layout::SmemLayout layout);
  // output operator
  aso::kernel::DTensor new_output(STensor const &stensor, int3 output_map);
  TBOperator *create_output_op(STensor const &stensor, int3 output_map);
  // matmul operator
  STensor matmul(STensor const &A, STensor const &B);
  TBOperator *create_matmul_op(STensor const &A, STensor const &B);
  // element unary operator
  STensor exp(STensor const &A);
  TBOperator *create_elementunary_op(STensor const &A,
                                     aso::type::TBOperatorType _type);
  // reduction operator
  STensor reduction(STensor const &A, int dim);
  TBOperator *create_reduction_op(STensor const &A, int dim);

  off_t allocate(STensor const &tensor);
  void free(STensor const &tensor);
  void free(std::vector<STensor> const &tensors);

  KernelParams get_kernel_params();

  operator json() const;

public:
  dim3 grid_dim, block_dim;
  int forloop_range;
  std::vector<aso::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;

  const size_t MAX_SMEM_SIZE = 96 * 1024; // 96 KB
};

} // namespace threadblock
} // namespace aso
