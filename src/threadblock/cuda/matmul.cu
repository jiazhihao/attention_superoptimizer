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

#include "aso/threadblock/graph.h"
#include "aso/threadblock/matmul.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>
#include "cutlass/cutlass.h"

namespace aso {
namespace threadblock {
namespace matmul {

template<
    typename scalar_type,
    typename arch_tag>


static void CUTLASS_DEVICE Operator::compute_kernel(void) {
  extern __shared__ char smem_buffer[];

}

} // namespace matmul
} // namespace threadblock
} // namespace aso