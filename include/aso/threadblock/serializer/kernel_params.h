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

namespace aso {
namespace threadblock {

struct alignas(16) NewKernelParams {
  const static int MAX_NUM_OPERATORS = 10;
  const static int MAX_NUM_PARAMETERS = 120;
  const static int MAX_NUM_DMEM_INPUTS = 3;
  const static int MAX_NUM_DMEM_OUTPUTS = 3;
  int num_operators, num_parameters, num_dmem_inputs, num_dmem_outputs;
  int parameters[MAX_NUM_PARAMETERS];
  aso::type::TBOperatorType operator_types[MAX_NUM_OPERATORS];
  void *dmem_input_ptrs[MAX_NUM_DMEM_INPUTS];
  void *dmem_output_ptrs[MAX_NUM_DMEM_OUTPUTS];
};

struct alignas(16) KernelParams {
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
  // mappings between input dtensors and stensors
  int3 input_map[MAX_NUM_DMEM_INPUTS];
  int forloop_dim[MAX_NUM_DMEM_INPUTS];
  // mappings between output dtensors and their stensors
  int3 output_map;
};

} // namespace threadblock
} // namespace aso