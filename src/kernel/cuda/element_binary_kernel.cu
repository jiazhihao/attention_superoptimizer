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
#include "aso/kernel/element_binary.h"
#include "aso/kernel/graph.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

using namespace aso::type;

bool KNElementBinaryOp::profile(ProfileResult &result) {
  // TODO: to be implemented
  return false;
}

__global__ void
    compute_elementbinary_fingerprint(aso::type::KNOperatorType type,
                                      FPType *div_p_lookup_table,
                                      FPType *div_q_lookup_table,
                                      aso::kernel::DTensor input1,
                                      aso::kernel::DTensor input2,
                                      aso::kernel::DTensor output,
                                      int num_elements) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (type == aso::type::KN_DIV_OP) {
    if (i < num_elements) {
      int input1_stride = 1, input1_idx = 0;
      int input2_stride = 1, input2_idx = 0;
      for (int d = output.num_dims - 1; d >= 0; d--) {
        input1_idx += (i % input1.dim[d]) * input1_stride;
        input2_idx += (i % input2.dim[d]) * input2_stride;
        input1_stride *= input1.dim[d];
        input2_stride *= input2.dim[d];
        i /= output.dim[d];
      }
      uint32_t x = input1.fp_ptr[input1_idx];
      uint32_t y = input2.fp_ptr[input2_idx];
      uint32_t z =
          (x % FP_P) * div_p_lookup_table[y % FP_P] * FP_Q_MUL_P_MOD_1 +
          (x % FP_Q) * div_q_lookup_table[y % FP_Q] * FP_P_MUL_Q_MOD_1;
      output.fp_ptr[threadIdx.x + blockIdx.x * blockDim.x] = z % FP_PQ;
      // printf("div: output[%d] = %d input1[%d] = %d input2[%d] = %d\n",
      //     threadIdx.x + blockIdx.x * blockDim.x, z % FP_PQ,
      //     input1_idx, x, input2_idx, y);
    }
  } else {
    assert(false && "Unimplemented");
  }
}

bool KNElementBinaryOp::fingerprint(void) {
  assert(input_tensors[0].num_dims == output_tensors[0].num_dims);
  for (int i = 0; i < output_tensors[0].num_dims; i++) {
    if (input_tensors[0].dim[i] != output_tensors[0].dim[i]) {
      assert(input_tensors[0].dim[i] == 1);
    }
  }
  assert(input_tensors[1].num_dims == output_tensors[0].num_dims);
  for (int i = 0; i < output_tensors[0].num_dims; i++) {
    if (input_tensors[1].dim[i] != output_tensors[0].dim[i]) {
      assert(input_tensors[1].dim[i] == 1);
    }
  }
  int num_elements = output_tensors[0].num_elements();
  int const num_threads_per_blk = 1024;
  int num_blocks =
      (num_elements + num_threads_per_blk - 1) / num_threads_per_blk;
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();
  compute_elementbinary_fingerprint<<<num_blocks, num_threads_per_blk>>>(
      op_type,
      dmm->div_p_lookup_table,
      dmm->div_q_lookup_table,
      input_tensors[0],
      input_tensors[1],
      output_tensors[0],
      num_elements);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace aso