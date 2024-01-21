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
  ~OperatorFactory(void);
  Op get_or_create_matmul(DTensor const &A, DTensor const &B);
  Op create_input(std::vector<int> const &dims, aso::type::DataType data_type);
  Op get_or_create_customized(
      std::vector<DTensor> const &inputs,
      aso::kernel::CustomizedOp::ExecutionPlan const &plan);
  void *allocate(size_t size_in_bytes);
  void free(void *ptr);

public:
  static OperatorFactory *get_instance();

public:
  std::unordered_map<aso::kernel::MatmulKey, aso::kernel::MatmulKNOp *> matmul;
  std::unordered_map<aso::kernel::CustomizedKey, aso::kernel::CustomizedOp *>
      customized;
  // fields for managing the preallocated cuda buffer
  char *base_ptr;
  off_t offset;
  size_t total_size;
  std::vector<std::pair<void *, size_t>> allocated_tensors;
};

} // namespace kernel
} // namespace aso
