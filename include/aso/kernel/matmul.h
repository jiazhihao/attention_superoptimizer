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

namespace aso {
namespace kernel {
namespace matmul {

class Operator : public aso::kernel::Operator {
public:
  Operator(Tensor const &A, Tensor const &B);
  ~Operator();
  bool profile(ProfileResult &profile);
};

class Key {
public:
  Key(Tensor const &A, Tensor const &B);
  bool operator==(Key const &b) const;
  Tensor operand_a;
  Tensor operand_b;
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
