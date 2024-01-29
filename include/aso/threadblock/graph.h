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
  const static int MAX_NUM_INPUTS = 3;
  const static int MAX_NUM_OUTPUTS = 3;
  int forloop_range;
  int num_operators;
  aso::type::TBOperatorType operator_types[MAX_NUM_OPERATORS];
  aso::threadblock::STensor input_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
  aso::threadblock::STensor output_tensors[MAX_NUM_OPERATORS][MAX_NUM_INPUTS];
  // input dtensors in device memory
  int num_input_dtensors, num_output_dtensors;
  aso::kernel::DTensor input_device_tensors[MAX_NUM_INPUTS];
  aso::kernel::DTensor output_device_tensors[MAX_NUM_INPUTS];
};

class ExecutionPlan {
public:
  std::vector<
      std::pair<aso::type::TBOperatorType, std::vector<std::pair<int, int>>>>
      ops;
  std::vector<int3> input_map;
  int3 output_map; // assume that all output must use the same map
  std::vector<int> forloop_dim;
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
                    int forloop_dim);
  TBOperator *create_input_op(aso::kernel::DTensor const &dtensor,
                              int3 input_map,
                              int forloop_dim);
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

public:
  dim3 grid_dim, block_dim;
  int forloop_range;
  std::vector<aso::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;
};

} // namespace threadblock
} // namespace aso
