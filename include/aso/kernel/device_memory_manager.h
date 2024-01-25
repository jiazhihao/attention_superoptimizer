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

#include <cublas_v2.h>
namespace aso {
namespace kernel {

class DeviceMemoryManager {
public:
  static DeviceMemoryManager *singleton;
  DeviceMemoryManager(void);
  ~DeviceMemoryManager(void);
  void *allocate(size_t size_in_bytes);
  void free(void *ptr);
public:
  static DeviceMemoryManager *get_instance();
public:
  // fields for managing the preallocated cuda buffer
  char *base_ptr;
  off_t offset;
  size_t total_size;
  std::vector<std::pair<void *, size_t>> allocated_tensors;
public:
  cublasHandle_t blas;
  // cudnnHandle_t cudnn;
};

} // namespace kernel
} // namespace aso
