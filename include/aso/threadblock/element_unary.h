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

namespace aso {
namespace threadblock {

using namespace cutlass;

template <typename ElementType>
class ElementUnaryOp {
public:
  ElementType* base_ptr
public:
  CUTLASS_DEVICE
  ElementUnaryOp(ElementType* _base_ptr)
  : base_ptr(_base_ptr) {}

  void CUTLASS_DEVICE execute_kernel(void) {
    // extern __shared__ char smem_buffer[];
  }
};

} // namespace threadblock
} // namespace aso
