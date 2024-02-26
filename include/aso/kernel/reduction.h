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

class KNReductionOp : public aso::kernel::KNOperator {
public:
  KNReductionOp(DTensor const &input, int reduction_dim, int reduction_factor);
  ~KNReductionOp();
  bool profile(ProfileResult &profile) override;
  bool fingerprint(void) override;

  operator json() const override;
public:
  int reduction_dim, reduction_factor;
};

} // namespace kernel
} // namespace aso

