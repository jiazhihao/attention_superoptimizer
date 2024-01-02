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

#include "aso/data_type.h"

namespace aso {
namespace datatype {

size_t get_datatype_size(Type type) {
  switch (type) {
    case INT8:
      return 1;
    case BFLOAT16:
    case FLOAT16:
      return 2;
    case FLOAT32:
      return 4;
    case DOUBLE:
      return 8;
    case UNKNOWN:
    default:
      assert(false);
  }
}

} // namespace datatype
} // namespace aso
