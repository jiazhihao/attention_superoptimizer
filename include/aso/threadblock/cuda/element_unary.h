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
class ElementUnaryExecutor {
public:
  ElementType *base_ptr;
  aso::type::TBOperatorType op_type;
  int kElements;

public:
  CUTLASS_DEVICE
  ElementUnaryExecutor(ElementType *_base_ptr,
                       aso::type::TBOperatorType _type,
                       int _kElements)
      : base_ptr(_base_ptr), op_type(_type), kElements(_kElements) {}

  void CUTLASS_DEVICE execute_kernel(void) {
    // extern __shared__ char smem_buffer[];
    if (op_type == aso::type::TB_EXP_OP) {
      for (int i = 0; i < kElements; i += blockDim.x) {
        base_ptr[i] = cutlass::fast_exp(base_ptr[i]);
      }
    } else {
      assert(false && "Unsupported operator");
    }
  }
};

class ElementUnaryFingerprinter {
public:
  aso::type::FPType *base_ptr;
  aso::type::TBOperatorType op_type;
  int kElements;

public:
  CUTLASS_DEVICE
  ElementUnaryFingerprinter(aso::type::FPType *_base_ptr,
                            aso::type::TBOperatorType _type,
                            int _kElements)
      : base_ptr(_base_ptr), op_type(_type), kElements(_kElements) {}

  void CUTLASS_DEVICE compute_fingerprint(void) {
    // extern __shared__ char smem_buffer[];
    if (op_type == aso::type::TB_EXP_OP) {
      for (int i = 0; i < kElements; i += blockDim.x) {
        assert(false && "To Be Implemented");
        // base_ptr[i] = cutlass::fast_exp(base_ptr[i]);
      }
    } else {
      assert(false && "Unsupported operator");
    }
  }
};

} // namespace threadblock
} // namespace aso