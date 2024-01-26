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

#pragma once

#include "aso/kernel/device_tensor.h"
#include "aso/profile_result.h"
#include <vector>

namespace aso {
namespace kernel {

class KNOperator {
public:
  KNOperator(aso::type::KNOperatorType _type);
  KNOperator(aso::type::KNOperatorType _type, DTensor const &input1);
  KNOperator(aso::type::KNOperatorType _type,
             DTensor const &input1,
             DTensor const &input2);
  KNOperator(aso::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs);
  ~KNOperator();
  virtual bool profile(ProfileResult &result) = 0;
  aso::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(std::vector<int> const &dims, aso::type::DataType data_type);
  ~KNInputOp();
  bool profile(ProfileResult &profile);
};

} // namespace kernel
} // namespace aso
