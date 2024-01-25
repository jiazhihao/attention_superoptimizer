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

#include "aso/kernel/graph.h"
#include "aso/kernel/matmul.h"
#include "aso/kernel/device_memory_manager.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

bool KNMatmulOp::profile(ProfileResult &result) {
  float alpha = 1.0f, beta = 0.0f;
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();
  void *A = input_tensors[0].data_ptr;
  void *B = input_tensors[1].data_ptr;
  void *C = output_tensors[0].data_ptr;
  // checkCUDA(cudaMalloc(&A, input_tensors[0].size()));
  // checkCUDA(cudaMalloc(&B, input_tensors[1].size()));
  // checkCUDA(cudaMalloc(&C, output_tensors[0].size()));
  int row_A = input_tensors[0].dim[0];
  int column_A = input_tensors[0].dim[1];
  int row_B = input_tensors[1].dim[0];
  int column_B = input_tensors[1].dim[1];
  int row_C = output_tensors[0].dim[0];
  int column_C = output_tensors[0].dim[1];
  assert(column_A == row_B);
  assert(row_C == row_A);
  assert(column_C == column_B);
  cudaDataType_t type_A =
      aso::utils::to_cuda_datatype(input_tensors[0].data_type);
  cudaDataType_t type_B =
      aso::utils::to_cuda_datatype(input_tensors[1].data_type);
  cudaDataType_t type_C =
      aso::utils::to_cuda_datatype(output_tensors[0].data_type);
  // TODO: currently set the default to CUBLAS_COMPUTE_16F for best performance
  cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
  cublasOperation_t trans_A = CUBLAS_OP_N;
  cublasOperation_t trans_B = CUBLAS_OP_N;
  if (input_tensors[0].is_column_major()) {
    trans_A = CUBLAS_OP_T;
  } else {
    assert(input_tensors[0].is_row_major());
  }
  if (input_tensors[1].is_column_major()) {
    trans_B = CUBLAS_OP_T;
  } else {
    assert(input_tensors[1].is_row_major());
  }
  // Currently assume C must be in row major;
  assert(output_tensors[0].is_row_major());
  int lda = input_tensors[0].is_row_major() ? row_A : column_A;
  int ldb = input_tensors[1].is_row_major() ? row_B : column_B;
  int ldc = row_C;

  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 16; i++) {
    checkCUDA(cublasGemmEx(dmm->blas,
                           trans_A,
                           trans_B,
                           row_C,
                           column_C,
                           column_A,
                           &alpha,
                           A,
                           type_A,
                           lda,
                           B,
                           type_B,
                           ldb,
                           &beta,
                           C,
                           type_C,
                           ldc,
                           compute_type,
                           CUBLAS_GEMM_DEFAULT));
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / 16;
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

} // namespace kernel
} // namespace aso
