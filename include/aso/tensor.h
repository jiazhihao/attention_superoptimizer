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

namespace aso {

#define MAX_TENSOR_DIMS 4

class Operator;

struct Tensor {
  aso::datatype::Type data_type;
  int num_dims;
  int dims[MAX_TENSOR_DIMS];
  int stride[MAX_TENSOR_DIMS];
  Operator *owner_operator;
  int owner_output_idx;
};

} // namespace aso
