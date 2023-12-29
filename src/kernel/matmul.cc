/* Copyright 2023 CMU
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

#include "aso/kernel/matmul.h"
#include <cassert>

namespace aso {
namespace kernel {
namespace matmul {

Operator::Operator(Tensor const &A, Tensor const &B)
    : aso::kernel::Operator(A, B) {
  Tensor C;
  assert(A.num_dims == 2);
  assert(B.num_dims == 2);
  // Currently only support row-major output
  // to be consistent with cutlass
  C.num_dims = 2;
  C.dims[0] = A.dims[0];
  C.dims[1] = B.dims[1];
  C.stride[0] = C.dims[1];
  C.stride[1] = 1;
  C.data_type = A.data_type;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

Key::Key(Tensor const &A, Tensor const &B) : operand_a(A), operand_b(B) {}

} // namespace matmul
} // namespace kernel
} // namespace aso
