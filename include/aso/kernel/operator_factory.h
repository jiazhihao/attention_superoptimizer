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

#include "aso/kernel/customized.h"
#include "aso/kernel/matmul.h"
#include "aso/kernel/operator.h"
#include <unordered_map>
#include <vector>

namespace aso {
namespace kernel {

class OperatorFactory {
public:
  using Op = Operator *;
  static OperatorFactory *singleton;
  OperatorFactory(void);
  Op get_or_create_matmul(TensorShape const &A, TensorShape const &B);
  Op get_or_create_customized(
      std::vector<TensorShape> const &inputs,
      aso::kernel::customized::ExecutionPlan const &plan);

public:
  static OperatorFactory *get_instance();

public:
  std::unordered_map<aso::kernel::matmul::Key, aso::kernel::matmul::Operator *>
      matmul;
  std::unordered_map<aso::kernel::customized::Key,
                     aso::kernel::customized::Operator *>
      customized;
};

} // namespace kernel
} // namespace aso
