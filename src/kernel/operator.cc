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

#include "aso/kernel/operator.h"

namespace aso {
namespace kernel {

Operator::Operator(Tensor const &A, Tensor const &B) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

Operator::~Operator() {}

// aso::base::Operator::Type Operator::get_operator_type(void) {
//   return aso::base::Operator::KERNEL_OPERATOR;
// }

} // namespace kernel
} // namespace aso
