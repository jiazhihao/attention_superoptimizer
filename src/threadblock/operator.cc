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

#include "aso/threadblock/operator.h"
#include "aso/threadblock/graph.h"

namespace aso {
namespace threadblock {

TBOperator::TBOperator(Graph* _graph,
                       aso::type::TBOperatorType _type,
                       STensor const &input1)
  : bgraph(_graph), op_type(_type) {
  input_tensors.push_back(input1);  
}

TBOperator::TBOperator(Graph* _graph,
                       aso::type::TBOperatorType _type,
                       STensor const &input1,
                       STensor const &input2)
  : bgraph(_graph), op_type(_type) {
  input_tensors.push_back(input1);
  input_tensors.push_back(input2);
}

TBOperator::~TBOperator() {}

STensor Graph::new_input(aso::kernel::DTensor const &dtensor,
                         dim3 input_map,
                         int forloop_dim,
                         int forloop_range) {
  TBOperator *op = create_input_op(dtensor, input_map, forloop_dim, forloop_range);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

TBOperator *Graph::create_input_op(aso::kernel::DTensor const &dtensor,
                                   dim3 input_map,
                                   int forloop_dim,
                                   int forloop_range) {
  KNInputOp *op = new KNInputOp(dtensor, input_map, forloop_dim, forloop_range);
  return op;
}

KNInputOp::KNInputOp(Graph* _graph,
                     aso::kernel::DTensor const &dtensor,
                     dim3 input_map,
                     int forloop_dim)
    : TBOperator(_graph, aso::type::KN_INPUT_OP) {
  STensor tensor;
  tensor.num_dims = dtensor.num_dims;
  tensor.data_type = dtensor.data_type;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = dtensor.dim[i];
  }

  for (int d = 0; d < 3; d++) {
    int dim_idx = -1;
    int dim_div = 1;
    if (d == 0 && graph->grid_dim.x > 1) {
      assert(input_map.x >= 0);
      dim_idx = input_map.x;
      dim_div = graph->grid_dim.x;
    }
    if (d == 1 && graph->grid_dim.y > 1) {
      assert(input_map.y > 0);
      dim_idx = input_map.y;
      dim_div = graph->grid_dim.y;
    }
    if (d == 2 && graph->grid_dim.z > 1) {
      assert(input_map.z > 0);
      dim_idx = input_map.z;
      dim_div = graph->grid_dim.z;
    }
    assert(tensor.dim[dim_idx] > 0);
    assert(tensor.dim[dim_idx] % dim_div == 0);
    tensor.dim[dim_idx] /= dim_div;
  }

  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
    tensor.stride[i] = (i == tensor.num_dims - 1)
                           ? 1
                           : tensor.stride[i + 1] * tensor.dim[i + 1];
  }
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  tensor.data_ptr = graph->allocate(tensor);
  output_tensors.push_back(tensor);
}

KNInputOp::~KNInputOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->free(output_tensors[0].data_ptr);
}

} // namespace threadblock
} // namespace aso
