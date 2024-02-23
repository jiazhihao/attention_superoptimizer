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

#include "aso/threadblock/reduction.h"
#include "aso/threadblock/graph.h"
#include <cassert>

namespace aso {
namespace threadblock {

STensor Graph::reduction(STensor const &input, int dim) {
  TBOperator *op = create_reduction_op(input, dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_reduction_op(STensor const &input, int dim) {
  TBOperator *op = new TBReductionOp(this, input, dim);

  STensor output = input;
  assert(output.num_dims > dim);
  assert(output.layout == aso::layout::SmemRowMajor);
  output.dim[dim] = 1;
  for (int i = output.num_dims - 1; i >= 0; i--) {
    output.stride[i] = (i == output.num_dims - 1)
                           ? 1
                           : output.stride[i + 1] * output.dim[i + 1];
  }

  if (smem_offset + (off_t)output.size() > (off_t)MAX_SMEM_SIZE) {
    return nullptr;
  }

  return op;
}

TBReductionOp::TBReductionOp(Graph *bgraph, STensor const &input, int dim)
    : TBOperator(bgraph, aso::type::TB_REDUCTION_0_OP, input), reduce_dim(dim) {
  aso::type::TBOperatorType type = static_cast<aso::type::TBOperatorType>(
      aso::type::TB_REDUCTION_0_OP + dim);
  this->op_type = type;
  STensor output = input;
  assert(output.num_dims > reduce_dim);
  assert(output.layout == aso::layout::SmemRowMajor);
  output.dim[reduce_dim] = 1;
  for (int i = output.num_dims - 1; i >= 0; i--) {
    output.stride[i] = (i == output.num_dims - 1)
                           ? 1
                           : output.stride[i + 1] * output.dim[i + 1];
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.smem_offset = bgraph->allocate(output);
  output_tensors.push_back(output);
}

TBReductionOp::~TBReductionOp() {
  bgraph->free(output_tensors[0]);
}

TBReductionOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"reduce_dim", reduce_dim}};
}
} // namespace threadblock
} // namespace aso
