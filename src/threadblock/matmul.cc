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

#include "aso/threadblock/matmul.h"
#include "aso/threadblock/graph.h"
#include "aso/threadblock/operator.h"

namespace aso {
namespace threadblock {

STensor Graph::matmul(STensor const &A, STensor const &B) {
  TBOperator *op = create_matmul_op(A, B);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_matmul_op(STensor const &A, STensor const &B) {
  if (A.num_dims != 2 || B.num_dims != 2) {
    return nullptr;
  }
  if (A.dim[1] != B.dim[0]) {
    return nullptr;
  }
  TBMatmulOp *op = new TBMatmulOp(this, A, B);
  return op;
}

TBMatmulOp::TBMatmulOp(Graph *_graph, STensor const &A, STensor const &B)
    : TBOperator(_graph, aso::type::TB_MATMUL_OP, A, B) {
  STensor C;
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
  C.smem_offset = bgraph->allocate(C);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(C);
}

TBMatmulOp::~TBMatmulOp() {
  bgraph->free(output_tensors);
}

} // namespace threadblock
} // namespace aso