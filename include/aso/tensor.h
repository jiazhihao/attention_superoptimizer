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

struct Tensor {
  Tensor();
  // Note that Tensor equivalence does not check owner_operator
  inline bool operator==(const Tensor& b) const {
    if (data_type != b.data_type) return false;
    if (num_dims != b.num_dims) return false;
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i])
        return false;
      if (stride[i] != b.stride[i])
        return false;
    }
    return true;
  }
  inline bool operator!=(const Tensor& b) const {
    if (data_type != b.data_type) return true;
    if (num_dims != b.num_dims) return true;
    for (int i = 0; i < num_dims; i++) {
      if (dim[i] != b.dim[i])
        return true;
      if (stride[i] != b.stride[i])
        return true;
    }
    return false;
  }

  aso::datatype::Type data_type;
  int num_dims;
  int dim[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
  Operator *owner_operator;
  int owner_output_idx;
};

} // namespace aso

namespace std {
template <>
struct hash<aso::Tensor> {
  size_t operator()(aso::Tensor const &) const;
};
}; // namespace std
