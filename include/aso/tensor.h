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

#pragma once

#include "aso/data_type.h"
#include <cstddef>
#include <functional>

namespace aso {

#define MAX_TENSOR_DIMS 4

class Operator;

struct TensorShape {
  TensorShape();
  inline bool operator==(TensorShape const &b) const {
    if (data_type != b.data_type) {
      return false;
    }
    if (num_dims != b.num_dims) {
      return false;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return false;
      }
      if (stride[i] != b.stride[i]) {
        return false;
      }
    }
    return true;
  }
  inline bool operator!=(TensorShape const &b) const {
    if (data_type != b.data_type) {
      return true;
    }
    if (num_dims != b.num_dims) {
      return true;
    }
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i]) {
        return true;
      }
      if (stride[i] != b.stride[i]) {
        return true;
      }
    }
    return false;
  }

  inline size_t size() const {
    size_t num_elements = 1;
    using namespace aso::datatype;
    size_t data_type_size = get_datatype_size(data_type);
    for (int i = 0; i < num_dims; i++) {
      num_elements *= dim[i];
    }
    return num_elements * data_type_size;
  }

  inline bool is_row_major() const {
    if (num_dims != 2) {
      return false;
    }
    if (stride[0] > 1) {
      return false;
    }
    assert(stride[0] == 1 && stride[1] == dim[0]);
    return true;
  }

  inline bool is_column_major() const {
    if (num_dims != 2) {
      return false;
    }
    if (stride[1] > 1) {
      return false;
    }
    assert(stride[1] == 1 && stride[0] == dim[1]);
    return true;
  }

  aso::datatype::Type data_type;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
};

struct Tensor {
  // TensorShape fields
  Tensor(TensorShape const &shape, int owner_op_idx, int owner_ts_idx);
  Tensor();
  TensorShape get_shape() const;
  aso::datatype::Type data_type;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
  // Tensor fields
  int owner_op_idx;
  int owner_ts_idx;
};

} // namespace aso

namespace std {
template <>
struct hash<aso::TensorShape> {
  size_t operator()(aso::TensorShape const &) const;
};
}; // namespace std
