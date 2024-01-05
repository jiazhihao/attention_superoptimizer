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

namespace utils {

cudaDataType_t to_cuda_datatype(aso::type::DataType type);

} // namespace utils
} // namespace aso
