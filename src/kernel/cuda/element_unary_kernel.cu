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

#include "aso/kernel/device_memory_manager.h"
#include "aso/kernel/graph.h"
#include "aso/kernel/element_unary.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

using namespace aso::type;

bool KNElementUnaryOp::profile(ProfileResult &result) {
  // TODO: to be implemented
  return false;
}

__global__ void compute_elementunary_fingerprint(
    aso::type::KNOperatorType type,
    FPType *exp_lookup_table,
    aso::type::FPType *input_ptr,
    aso::type::FPType *output_ptr,
    int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (type == aso::type::KN_EXP_OP) {
    if (i < num_elements) {
      aso::type::FPType val = input_ptr[i];
      aso::type::FPType q_residual = val % FP_Q;
      output_ptr[i] = exp_lookup_table[q_residual] * FP_Q;
    }
  } else {
    assert(false && "Unimplemented");
  }
}


bool KNElementUnaryOp::fingerprint(void) {
  assert(input_tensors[0].num_elements() == output_tensors[0].num_elements());
  int num_elements = input_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks = (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();
  compute_elementunary_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      op_type,
      dmm->exp_lookup_table,
      input_tensors[0].fp_ptr,
      output_tensors[0].fp_ptr,
      num_elements);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // kernel
} // aso
