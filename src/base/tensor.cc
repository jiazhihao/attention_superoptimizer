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
Tensor::Tensor() {
  data_type = aso::datatype::UNKNOWN;
  num_dims = 0;
  for (int i = 0; i < MAX_TENSOR_DIMS; i++) {
    dim[i] = 0;
    stride[i] = 0;
  }
  owner_operator = nullptr;
  owner_output_idx = 0;
}
} // namespace aso

namespace std {

size_t hash<aso::Tensor>::operator()(
    aso::Tensor const &tensor) const {
  size_t ret = hash<int>()((tensor.data_type));
  hash_combine(ret, tensor.num_dims);
  for (int i = 0; i < tensor.num_dims; i++) {
    hash_combine(ret, tensor.dim[i]);
    hash_combine(ret, tensor.stride[i]);
  }
  return ret;
}

} // namespace std
