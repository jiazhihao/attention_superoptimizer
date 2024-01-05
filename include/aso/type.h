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

#include <cassert>
#include <cstddef>

namespace aso {
namespace type {

enum DataType {
  DT_INT8,
  DT_BFLOAT16,
  DT_FLOAT16,
  DT_FLOAT32,
  DT_DOUBLE,
  DT_UNKNOWN,
};

size_t get_datatype_size(DataType type);

enum OperatorType {
  KN_UNKOWN = 1000,
  KN_MATMUL = 1001,
  KN_CUSTOMIZED = 1999,
  TB_UNKOWN = 2000,
  TB_MATMUL = 2001,
  TB_CUSTOMIZED = 2999
};

} // namespace datatype
} // namespace aso
