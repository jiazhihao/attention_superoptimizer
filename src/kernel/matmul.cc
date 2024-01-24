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

#include "aso/kernel/matmul.h"
#include "aso/kernel/device_memory_manager.h"
#include "aso/kernel/graph.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

DTensor Graph::matmul(DTensor const &A, DTensor const &B) {
  KNOperator *op = create_matmul_op(A, B);
  assert(op != nullptr);
  operators.push_back(op);
  DTensor output = op->output_tensors[0];
  return output;
}

KNOperator *Graph::create_matmul_op(DTensor const &A, DTensor const &B) {
  if (A.num_dims != 2 || B.num_dims != 2) {
    return nullptr;
  }
  if (A.dim[1] != B.dim[0]) {
    return nullptr;
  }
  KNMatmulOp *op = new KNMatmulOp(A, B);
  return op;
}

KNMatmulOp::KNMatmulOp(DTensor const &A, DTensor const &B)
    : aso::kernel::KNOperator(aso::type::KN_MATMUL_OP, A, B) {
  DTensor C;
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
  C.owner_op = this;
  C.owner_ts_idx = 0;
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  C.data_ptr = dmm->allocate(C.size());
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

KNMatmulOp::~KNMatmulOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i].data_ptr);
  }
}

MatmulKey::MatmulKey(DTensor const &A, DTensor const &B)
    : operand_a(A), operand_b(B) {}

bool MatmulKey::operator==(MatmulKey const &b) const {
  if (b.operand_a != operand_a) {
    return false;
  }
  if (b.operand_b != operand_b) {
    return false;
  }
  return true;
}

} // namespace kernel
} // namespace aso

namespace std {
size_t hash<aso::kernel::MatmulKey>::operator()(
    aso::kernel::MatmulKey const &key) const {
  size_t ret = 0;
  hash_combine(ret, key.operand_a);
  hash_combine(ret, key.operand_b);
  return ret;
}
}; // namespace std
