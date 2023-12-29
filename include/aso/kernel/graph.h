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

#include "aso/kernel/operator.h"
#include "aso/kernel/operator_factory.h"
#include "aso/tensor.h"
#include <vector>

namespace aso {
namespace kernel {

class Graph {
public:
  Graph(void);
  Tensor matmul(Tensor const &A, Tensor const &B);
  std::vector<aso::kernel::Operator *> operators;
  aso::kernel::OperatorFactory *operator_factory;
};

} // namespace kernel
} // namespace aso
