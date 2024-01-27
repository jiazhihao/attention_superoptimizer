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

STensor Graph::new_input(aso::kernel::DTensor const &dtensor,
                         int3 input_map,
                         int forloop_dim) {
  TBOperator *op = create_input_op(dtensor, input_map, forloop_dim);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_input_op(aso::kernel::DTensor const &dtensor,
                                   int3 input_map,
                                   int forloop_dim) {
  TBInputOp *op = new TBInputOp(this, dtensor, input_map, forloop_dim);
  return op;
}

TBInputOp::TBInputOp(Graph *_graph,
                     aso::kernel::DTensor const &_dtensor,
                     int3 _input_map,
                     int _forloop_dim)
    : TBOperator(_graph, aso::type::TB_INPUT_OP), dtensor(_dtensor),
      input_map(_input_map), forloop_dim(_forloop_dim) {
  STensor tensor;
  tensor.num_dims = dtensor.num_dims;
  tensor.data_type = dtensor.data_type;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = dtensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && bgraph->grid_dim.x > 1) {
      dim_idx = input_map.x;
      dim_div = bgraph->grid_dim.x;
    }
    if (d == 1 && bgraph->grid_dim.y > 1) {
      dim_idx = input_map.y;
      dim_div = bgraph->grid_dim.y;
    }
    if (d == 2 && bgraph->grid_dim.z > 1) {
      dim_idx = input_map.z;
      dim_div = bgraph->grid_dim.z;
    }
    if (dim_idx >= 0) {
      assert(tensor.dim[dim_idx] > 0);
      assert(tensor.dim[dim_idx] % dim_div == 0);
      tensor.dim[dim_idx] /= dim_div;
    }
  }

  if (forloop_dim >= 0) {
    assert(tensor.dim[forloop_dim] > 0);
    assert(tensor.dim[forloop_dim] % bgraph->forloop_range == 0);
    tensor.dim[forloop_dim] /= bgraph->forloop_range;
  }

  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.stride[i] = (i == tensor.num_dims - 1)
                           ? 1
                           : tensor.stride[i + 1] * tensor.dim[i + 1];
  }
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.smem_offset = bgraph->allocate(tensor);
  output_tensors.push_back(tensor);
}

TBInputOp::~TBInputOp() {
  bgraph->free(output_tensors[0]);
}

} // namespace threadblock
} // namespace aso
