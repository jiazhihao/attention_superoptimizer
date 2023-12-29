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
#include "aso/kernel/graph.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

Tensor Graph::matmul(Tensor const &A,
                     Tensor const &B) {
  Operator* op = operator_factory->get_or_create_matmul(A, B);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

Operator* OperatorFactory::get_or_create_matmul(Tensor const &A,
                                                Tensor const &B) {
  if (A.num_dims != 2 || B.num_dims != 2)
    return nullptr;
  if (A.dim[1] != B.dim[0])
    return nullptr;
  matmul::Key key(A, B);
  matmul::Operator* op = nullptr;
  if (matmul.find(key) != matmul.end()) {
    op = matmul[key];
  } else {
    op = new matmul::Operator(A, B);
    matmul[key] = op;
  }
  return op;
}

namespace matmul {

Operator::Operator(Tensor const &A, Tensor const &B)
    : aso::kernel::Operator(A, B) {
  Tensor C;
  assert(A.num_dims == 2);
  assert(B.num_dims == 2);
  // Currently only support row-major output
  // to be consistent with cutlass
  C.num_dims = 2;
  C.dim[0] = A.dim[0];
  C.dim[1] = B.dim[1];
  C.stride[0] = C.dim[1];
  C.stride[1] = 1;
  C.data_type = A.data_type;
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

Operator::~Operator() {}

Key::Key(Tensor const &A, Tensor const &B) : operand_a(A), operand_b(B) {}

bool Key::operator==(const Key& b) const {
  if (b.operand_a != operand_a)
    return false;
  if (b.operand_b != operand_b)
    return false;
  return true;
}

} // namespace matmul
} // namespace kernel
} // namespace aso

namespace std {
size_t hash<aso::kernel::matmul::Key>::operator()(
    aso::kernel::matmul::Key const &key) const {
  size_t ret = 0;
  hash_combine(ret, key.operand_a);
  hash_combine(ret, key.operand_b);
  return ret;
}
}; // namespace std
