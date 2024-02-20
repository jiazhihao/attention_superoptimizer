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
#include "aso/kernel/operator.h"
#include "aso/threadblock/graph.h"
#include "aso/threadblock/operator.h"
#include <tuple>
#include <vector_types.h>

namespace aso {
namespace kernel {

class KNCustomizedOp : public aso::kernel::KNOperator {
public:
  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 aso::threadblock::ExecutionPlan const &plan);
  KNCustomizedOp(std::vector<DTensor> const &inputs,
                 aso::threadblock::Graph const &_graph);
  ~KNCustomizedOp();
  bool profile(ProfileResult &profile);
  void run(void);
  bool fingerprint(void);

  operator json() const override;

public:
  aso::threadblock::ExecutionPlan plan;
  aso::threadblock::Graph bgraph;
};

} // namespace kernel
} // namespace aso
