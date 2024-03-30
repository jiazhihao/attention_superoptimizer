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
class ElementBinaryExecutor {
public:
  CUTLASS_DEVICE
  ElementBinaryExecutor(aso::type::TBOperatorType op_type,
                        char *smem_buffer,
                        STensor const &input1,
                        STensor const &input2,
                        STensor const &output,
                        int thread_id,
                        int num_threads) {
    //FIXME: currently we assume broadcast the inner-most dim
    ElementType *input1_ptr = (ElementType *)(smem_buffer + input1.smem_offset);
    ElementType *input2_ptr = (ElementType *)(smem_buffer + input2.smem_offset);
    ElementType *output_ptr = (ElementType *)(smem_buffer + output.smem_offset);
    int num_elements = output.num_elements();
    int factor1 = num_elements / input1.num_elements();
    int factor2 = num_elements / input2.num_elements();
    if (op_type == aso::type::TB_DIV_OP) {
      for (int i = 0; i < num_elements; i += num_threads) {
        output_ptr[i] = input1_ptr[i / factor1] / input2_ptr[i / factor2];
      }
    } else {
      assert(false && "Unsupported operator");
    }
  };
};

class TBElementBinaryFingerPrinter {
public:
  CUTLASS_DEVICE
  TBElementBinaryFingerPrinter(aso::type::TBOperatorType type,
                               FPType *div_p_lookup_table,
                               FPType *div_q_lookup_table,
                               char *smem_buffer,
                               STensor const &input1,
                               STensor const &input2,
                               STensor const &output,
                               int thread_id,
                               int num_threads) {
    FPType *output_ptr = (FPType *)(smem_buffer + output.smem_offset);
    FPType *input1_ptr = (FPType *)(smem_buffer + input1.smem_offset);
    FPType *input2_ptr = (FPType *)(smem_buffer + input2.smem_offset);
    int num_elements = output.num_elements();
    if (type == aso::type::TB_DIV_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        int idx = i;
        int input1_stride = 1, input1_idx = 0;
        int input2_stride = 1, input2_idx = 0;
        for (int d = output.num_dims - 1; d >= 0; d--) {
          input1_idx += (idx % input1.dim[d]) * input1_stride;
          input2_idx += (idx % input2.dim[d]) * input2_stride;
          input1_stride *= input1.dim[d];
          input2_stride *= input2.dim[d];
          idx /= output.dim[d];
        }
        uint32_t x = input1_ptr[input1_idx];
        uint32_t y = input2_ptr[input2_idx];
        uint32_t z =
            (x % FP_P) * div_p_lookup_table[y % FP_P] * FP_Q_MUL_P_MOD_1 +
            (x % FP_Q) * div_q_lookup_table[y % FP_Q] * FP_P_MUL_Q_MOD_1;
        output_ptr[i] = z % FP_PQ;
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace aso
