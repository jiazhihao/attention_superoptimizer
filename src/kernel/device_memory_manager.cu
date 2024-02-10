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

#include "aso/kernel/device_memory_manager.h"
#include "aso/utils/cuda_helper.h"

namespace aso {
namespace kernel {

DeviceMemoryManager *DeviceMemoryManager::singleton = nullptr;

DeviceMemoryManager::DeviceMemoryManager() {
  // preallocate 10 GB of device memory
  total_size = (size_t)10 * 1024 * 1024 * 1024;
  offset = 0;
  checkCUDA(cudaMalloc(&base_ptr, total_size));
  checkCUDA(cublasCreate(&blas));
  checkCUDA(cublasSetMathMode(blas, CUBLAS_TENSOR_OP_MATH));
}

DeviceMemoryManager::~DeviceMemoryManager() {
  checkCUDA(cudaFree(base_ptr));
  checkCUDA(cublasDestroy(blas));
}

bool DeviceMemoryManager::allocate(DTensor &tensor, bool allocate_fingerprint) {
  void *ret_ptr = base_ptr + offset;
  offset += tensor.data_size();
  tensor.data_ptr = ret_ptr;
  allocated_tensors.push_back(std::make_pair(ret_ptr, tensor.data_size()));

  if (allocate_fingerprint) {
    ret_ptr = base_ptr + offset;
    offset += tensor.fingerprint_size();
    tensor.fp_ptr = (aso::type::FPType *)ret_ptr;
    allocated_tensors.push_back(
        std::make_pair(ret_ptr, tensor.fingerprint_size()));
  }
  // Assert that we haven't used more than what we pre-allocated
  assert(offset <= total_size);

  return true;
}

bool DeviceMemoryManager::free(DTensor &tensor) {
  // Currently assume that tensors are freed in the reverse order
  // so ptr must be the last tensor we have created
  if (tensor.fp_ptr != nullptr) {
    assert(allocated_tensors.size() > 0);
    assert(allocated_tensors.back().first == tensor.fp_ptr);
    offset -= allocated_tensors.back().second;
    allocated_tensors.pop_back();
  }
  assert(allocated_tensors.size() > 0);
  assert(allocated_tensors.back().first == tensor.data_ptr);
  offset -= allocated_tensors.back().second;
  allocated_tensors.pop_back();
  return true;
}

DeviceMemoryManager *DeviceMemoryManager::get_instance() {
  if (singleton == nullptr) {
    singleton = new DeviceMemoryManager();
  }
  return singleton;
}

} // namespace kernel
} // namespace aso
