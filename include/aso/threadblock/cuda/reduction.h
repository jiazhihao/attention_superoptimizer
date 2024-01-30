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

} // namespace threadblock
} // namespace aso