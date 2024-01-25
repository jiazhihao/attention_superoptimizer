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

#include <cassert>
#include "aso/threadblock/graph.h"
#include "aso/threadblock/reduction.h"

namespace aso {
namespace threadblock {

STensor Graph::reduction(STensor const &input, int dim) {
  TBOperator *op = create_reduction_op(input, dim);
  return op->output_tensors[0];
}

TBOperator* Graph::create_reduction_op(STensor const &input, int dim) {
  TBOperator* op = new TBReductionOp(this, input, dim);
  return op;
}

TBReductionOp::TBReductionOp(Graph *bgraph,
                             STensor const &input,
                             int dim)
  : TBOperator(bgraph, aso::type::TB_REDUCTION_0_OP, input), reduce_dim(dim) {
  aso::type::TBOperatorType type = static_cast<aso::type::TBOperatorType>(aso::type::TB_REDUCTION_0_OP + dim);
  this->op_type = type;
  STensor output = input;
  output.smem_offset = bgraph->allocate(output_tensors[0]);
}

TBReductionOp::~TBReductionOp() {
  bgraph->free(output_tensors[0]);
}

} // namespace threadblock
} // namespace aso
