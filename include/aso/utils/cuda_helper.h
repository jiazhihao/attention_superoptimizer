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

#include "aso/type.h"
#include <cublas_v2.h>
#include <cudnn.h>
#include <cutlass/cutlass.h>
#include <iostream>
#include <sstream>
#include <string>

namespace aso {

#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    assert(false);                                                             \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCURAND(status)                                                    \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CURAND_STATUS_SUCCESS) {                                     \
      _error << "CURAND failure: " << status;                                  \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCUDA(status)                                                      \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

template <typename T>
CUTLASS_DEVICE T warp_uniform(T value) {
  struct {
    union {
      T value;
      uint32_t asInt;
    };
  } p;
  p.value = value;
  p.asInt = __shfl_sync(0xffffffff, (unsigned)p.asInt, 0);
  return p.value;
}

template <typename T>
CUTLASS_DEVICE T *warp_uniform(T *ptr) {
  struct {
    union {
      T *ptr;
      uint32_t asInt[2];
    };
  } p;
  p.ptr = ptr;
  p.asInt[0] = warp_uniform(p.asInt[0]);
  p.asInt[1] = warp_uniform(p.asInt[1]);
  return p.ptr;
}

template <int WARPS_PER_BLOCK = 4, int WARP_SIZE = 32>
CUTLASS_DEVICE float block_sum_fp32(float sum) {

  __shared__ float red_smem[WARPS_PER_BLOCK];
  // Decompose the thread index into warp / lane.
  int warp = threadIdx.x / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;

// Compute the sum per warp.
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Warp leaders store the data to shared memory.
  if (lane == 0) {
    red_smem[warp] = sum;
  }

  // Make sure the data is in shared memory.
  __syncthreads();

  // The warps compute the final sums.
  if (lane < WARPS_PER_BLOCK) {
    sum = red_smem[lane];
  }

// Parallel reduction inside the warp.
#pragma unroll
  for (int mask = WARPS_PER_BLOCK / 2; mask >= 1; mask /= 2) {
    sum += __shfl_xor_sync(uint32_t(-1), sum, mask);
  }

  // Broadcast to other threads.
  return __shfl_sync(uint32_t(-1), sum, 0);
}

namespace utils {

cudaDataType_t to_cuda_datatype(aso::type::DataType type);

size_t get_max_shared_mem();

using namespace aso::type;

CUTLASS_HOST_DEVICE
int get_reduction_dim(TBOperatorType type) {
  if (type >= TB_REDUCTION_0_TO_DIMX_OP && type <= TB_REDUCTION_2_TO_DIMX_OP) {
    return type - TB_REDUCTION_0_TO_DIMX_OP;
  } else if (type >= TB_REDUCTION_0_OP && type <= TB_REDUCTION_2_OP) {
    return type - TB_REDUCTION_0_OP;
  } else {
    assert(false);
    return -1;
  }
}

} // namespace utils
} // namespace aso
