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

#include "aso/profile_result.h"
#include "aso/tensor.h"

namespace aso {
namespace kernel {

class Operator {
public:
  Operator(TensorShape const &input1, TensorShape const &input2);
  ~Operator();
  // aso::base::Operator::Type get_operator_type(void)
  bool profile(ProfileResult &result);
  std::vector<aso::TensorShape> input_tensors;
  std::vector<aso::TensorShape> output_tensors;
};

} // namespace kernel
} // namespace aso
