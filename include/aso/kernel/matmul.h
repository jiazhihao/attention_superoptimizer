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

#include "aso/kernel/operator.h"

namespace aso {
namespace kernel {
namespace matmul {

class Operator : public aso::kernel::Operator {
public:
  Operator(TensorShape const &A, TensorShape const &B);
  ~Operator();
  aso::type::OperatorType operator_type() const;
  bool profile(ProfileResult &profile);
};

class Key {
public:
  Key(TensorShape const &A, TensorShape const &B);
  bool operator==(Key const &b) const;
  TensorShape operand_a;
  TensorShape operand_b;
};

} // namespace matmul
} // namespace kernel
} // namespace aso

namespace std {
template <>
struct hash<aso::kernel::matmul::Key> {
  size_t operator()(aso::kernel::matmul::Key const &) const;
};
} // namespace std
