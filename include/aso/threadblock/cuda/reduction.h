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

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

namespace aso {
namespace threadblock {

using namespace cutlass;
using namespace aso::type;

template <typename ElementType>
class RedunctionExecutor {
public:
  // reference implementation: ReduceSameRow function from
  // cutlass/examples/41_fused_multi_head_attention/gemm/mma_accum_lambda_iterator.h
  CUTLASS_DEVICE
  RedunctionExecutor() {}

  void CUTLASS_DEVICE execute_kernel(void) {
    assert(false && "To Be Implemented");
  }
};

class TBReductionFingerprinter {
public:
  CUTLASS_DEVICE
  TBReductionFingerprinter(aso::type::TBOperatorType type,
                           char *smem_buffer,
                           STensor const &input,
                           STensor const &output,
                           int thread_id,
                           int num_threads) {
    FPType *input_ptr = (FPType *)(smem_buffer + input.smem_offset);
    FPType *output_ptr = (FPType *)(smem_buffer + output.smem_offset);
    int num_elements = output.size();
    if (type == TB_REDUCTION_0_OP) {
      int kK = input.dim[0] / output.dim[0];
      int kIterations = (num_elements + num_threads - 1) / num_threads;
      for (int i = 0; i < kIterations; i++) {
        uint32_t result = 0;
        int no = (i * num_threads + thread_id) / output.dim[1];
        int m = (i * num_threads + thread_id) % output.dim[1];
        if (no >= output.dim[0]) {
          continue;
        }
        for (int k = 0; k < kK; k++) {
          int ni = no + k * output.dim[0];
          result =
              (result + input_ptr[ni * input.stride[0] + m * input.stride[1]]) %
              FP_PQ;
        }
        output_ptr[no * output.stride[0] + m * output.stride[1]] = result;
      }
    } else if (type == TB_REDUCTION_1_OP) {
      int kK = input.dim[1] / output.dim[1];
      for (int i = 0; i < (num_elements + num_threads - 1) / num_threads; i++) {
        uint32_t result = 0;
        int n = (i * num_threads + thread_id) / output.dim[1];
        int mo = (i * num_threads + thread_id) % output.dim[1];
        if (n >= output.dim[0]) {
          continue;
        }
        for (int k = 0; k < kK; k++) {
          int mi = mo + k * output.dim[1];
          result =
              (result + input_ptr[n * input.stride[0] + mi * input.stride[1]]) %
              FP_PQ;
        }
        output_ptr[n * output.stride[0] + mo * output.stride[1]] = result;
      }
    } else {
      assert(false && "Unimplemented");
    }
  };
};

} // namespace threadblock
} // namespace aso
