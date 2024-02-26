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
#include "aso/kernel/reduction.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

using namespace aso::type;

bool KNReductionOp::profile(ProfileResult &result) {
  // TODO: to be implemented
  return false;
}

__global__ void compute_reduction_fingerprint(FPType *input_ptr,
                                              FPType *output_ptr,
                                              int num_elements,
                                              int reduction_factor,
                                              int input_stride,
                                              int output_stride) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_elements) {
    uint32_t result = 0;
    int n = i / output_stride;
    int m = i % output_stride;
    for (int k = 0; k < reduction_factor; k++) {
      result = (result + input_ptr[n * input_stride + m + k * output_stride]) %
               FP_PQ;
    }
    output_ptr[i] = result;
  }
}

bool KNReductionOp::fingerprint(void) {
  int num_elements = output_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  int output_stride = 1;
  int input_stride = 1;
  for (int i = reduction_dim; i < output_tensors[0].num_dims; i++) {
    output_stride *= output_tensors[0].dim[i];
    input_stride *= input_tensors[0].dim[i];
  }
  assert(output_stride * reduction_factor == input_stride);
  compute_reduction_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      input_tensors[0].fp_ptr,
      output_tensors[0].fp_ptr,
      num_elements,
      reduction_factor,
      input_stride,
      output_stride);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace aso
