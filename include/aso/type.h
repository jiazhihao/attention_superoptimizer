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
#include <cstdint>

namespace aso {
namespace type {

typedef uint16_t FPType;
const uint16_t FP_P = 227;
const uint16_t FP_Q = 113;
const uint16_t FP_PQ = 25651;

enum DataType {
  DT_INT8,
  DT_UINT16,
  DT_BFLOAT16,
  DT_FLOAT16,
  DT_FLOAT32,
  DT_DOUBLE,
  DT_UNKNOWN,
};

size_t get_datatype_size(DataType type);

enum KNOperatorType {
  KN_UNKOWN = 1000,
  KN_INPUT_OP = 1001,
  KN_OUTPUT_OP = 1002,
  KN_MATMUL_OP = 1003,
  KN_REDUCTION_0_OP = 1004,
  KN_REDUCTION_1_OP = 1005,
  KN_REDUCTION_2_OP = 1006,
  KN_EXP_OP = 1007,
  KN_DIV_OP = 1008,
  KN_CUSTOMIZED_OP = 1999,
};

enum TBOperatorType {
  TB_UNKOWN = 2000,
  TB_INPUT_OP = 2001,
  TB_OUTPUT_OP = 2002,
  TB_MATMUL_OP = 2003,
  TB_REDUCTION_0_OP = 2004,
  TB_REDUCTION_1_OP = 2005,
  TB_REDUCTION_2_OP = 2006,
  TB_EXP_OP = 2007,
  TB_DIV_OP = 2008,
  TB_CUSTOMIZED_OP = 2999
};

} // namespace type
} // namespace aso
