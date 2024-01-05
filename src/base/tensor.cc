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

#include "aso/tensor.h"
#include "aso/utils/hash_utils.h"
#include <functional>

namespace aso {
TensorShape::TensorShape() {
  data_type = aso::type::DT_UNKNOWN;
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    stride[i] = 0;
  }
}

Tensor::Tensor(TensorShape const &_shape,
               int _owner_op_idx,
               int _owner_ts_idx) {
  data_type = _shape.data_type;
  num_dims = _shape.num_dims;
  for (int i = 0; i < num_dims; i++) {
    dim[i] = _shape.dim[i];
    stride[i] = _shape.stride[i];
  }
  owner_op_idx = _owner_op_idx;
  owner_ts_idx = _owner_ts_idx;
}

Tensor::Tensor() {
  data_type = aso::type::DT_UNKNOWN;
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    stride[i] = 0;
  }
  owner_op_idx = -1000;
  owner_ts_idx = -1000;
}

TensorShape Tensor::get_shape() const {
  TensorShape shape;
  shape.data_type = data_type;
  shape.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    shape.dim[i] = dim[i];
    shape.stride[i] = stride[i];
  }
  return shape;
}

} // namespace aso

namespace std {

size_t
    hash<aso::TensorShape>::operator()(aso::TensorShape const &tensor) const {
  size_t ret = hash<int>()((tensor.data_type));
  hash_combine(ret, tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    hash_combine(ret, tensor.stride[i]);
  }
  return ret;
}

} // namespace std
