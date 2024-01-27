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
#include "aso/threadblock/operator.h"

namespace aso {
namespace threadblock {

aso::kernel::DTensor Graph::new_output(STensor const &stensor,
                                       int3 output_map) {
  TBOperator *op = create_output_op(stensor, output_map);
  assert(op != nullptr);
  operators.push_back(op);
  return static_cast<TBOutputOp *>(op)->dtensor;
}

TBOperator *Graph::create_output_op(STensor const &stensor, int3 output_map) {
  TBOutputOp *op = new TBOutputOp(this, stensor, output_map);
  return op;
}

TBOutputOp::TBOutputOp(Graph *_graph, STensor const &stensor, int3 _output_map)
    : TBOperator(_graph, aso::type::TB_OUTPUT_OP), output_map(_output_map) {
  dtensor.num_dims = stensor.num_dims;
  dtensor.data_type = stensor.data_type;
  for (int i = 0; i < dtensor.num_dims; i++) {
    dtensor.dim[i] = stensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && bgraph->grid_dim.x > 1) {
      dim_idx = output_map.x;
      dim_div = bgraph->grid_dim.x;
    }
    if (d == 1 && bgraph->grid_dim.y > 1) {
      dim_idx = output_map.y;
      dim_div = bgraph->grid_dim.y;
    }
    if (d == 2 && bgraph->grid_dim.z > 1) {
      dim_idx = output_map.z;
      dim_div = bgraph->grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(dtensor.dim[dim_idx] > 0);
      dtensor.dim[dim_idx] *= dim_div;
    }
  }

  for (int i = dtensor.num_dims - 1; i >= 0; i--) {
    dtensor.stride[i] = (i == dtensor.num_dims - 1)
                            ? 1
                            : dtensor.stride[i + 1] * dtensor.dim[i + 1];
  }
}

TBOutputOp::~TBOutputOp() {}

} // namespace threadblock
} // namespace aso
