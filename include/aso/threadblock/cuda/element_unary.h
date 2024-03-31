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
class ElementUnaryExecutor {
public:
  CUTLASS_DEVICE
  ElementUnaryExecutor(aso::type::TBOperatorType op_type,
                       char *smem_buffer,
                       STensor const &input,
                       STensor const &output,
                       int thread_id,
                       int num_threads) {
    assert(input.smem_offset == output.smem_offset);
    int num_elements = output.num_elements();
    ElementType *base_ptr = (ElementType *)(smem_buffer + input.smem_offset);
    if (op_type == aso::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        base_ptr[thread_id] = cutlass::fast_exp(base_ptr[thread_id]);
      }
    }
  }
};

class TBElementUnaryFingerPrinter {
public:
  CUTLASS_DEVICE
  TBElementUnaryFingerPrinter(aso::type::TBOperatorType type,
                              FPType *exp_lookup_table,
                              char *smem_buffer,
                              STensor const &input,
                              STensor const &output,
                              int thread_id,
                              int num_threads) {
    // Assert inplace
    assert(input.smem_offset == output.smem_offset);
    FPType *ptr = (FPType *)(smem_buffer + input.smem_offset);
    int num_elements = output.num_elements();
    if (type == aso::type::TB_EXP_OP) {
      for (int i = thread_id; i < num_elements; i += num_threads) {
        FPType input = ptr[i];
        // FPType p_residual = input % FP_P;
        FPType q_residual = input % FP_Q;
        uint32_t result = exp_lookup_table[q_residual];
        result = (result * FP_Q_MUL_P_MOD_1) % FP_PQ;
        ptr[i] = result;
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

} // namespace threadblock
} // namespace aso
