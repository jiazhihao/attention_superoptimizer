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

#include "aso/kernel/customized.h"
#include "aso/threadblock/cuda/element_unary.h"
#include "aso/threadblock/cuda/input.h"
#include "aso/threadblock/cuda/matmul.h"
#include "aso/threadblock/cuda/reduction.h"
#include "aso/threadblock/graph.h"
#include "aso/utils/cuda_helper.h"

namespace aso {
namespace kernel {

__global__ void
    customized_kernel_function(aso::threadblock::KernelParams const &params) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    int device_input_idx = 0;
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          break;
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          cutlass::MatrixCoord threadblock_offset = {0, 0};
          aso::threadblock::GenericInputLoader loader(
              smem_buffer,
              params.input_device_tensors[device_input_idx++],
              params.output_tensors[op][0],
              threadIdx.x,
              blockDim.x,
              threadblock_offset);
          break;
        }
        case aso::type::TB_MATMUL_OP: {
          break;
          using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
          using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
          using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
          int thread_idx = threadIdx.x;
          // Broadcast the warp_id computed by lane 0 to ensure dependent code
          // is compiled as warp-uniform.
          int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
          int lane_idx = threadIdx.x % 32;
          aso::threadblock::MatmulExecutor<ThreadblockShape,
                                           WarpShape,
                                           InstructionShape,
                                           cutlass::half_t,
                                           cutlass::layout::RowMajor,
                                           cutlass::layout::ColumnMajor>
              executor(thread_idx, warp_idx, lane_idx);
          executor.execute_kernel();
          break;
        }
        case aso::type::TB_EXP_OP: {
          break;
          aso::threadblock::STensor tensor = params.input_tensors[op][0];
          cutlass::half_t *input_ptr =
              (cutlass::half_t *)(smem_buffer + tensor.smem_offset);
          aso::threadblock::ElementUnaryExecutor<cutlass::half_t> executor(
              input_ptr, params.operator_types[op], tensor.size());
          executor.execute_kernel();
          break;
        }
        default: {
          assert(false && "Unsupported threadblock operator");
        }
      }
    }
  }
}

void KNCustomizedOp::run() {
  int smem_size = 48 * 1024; // 48 KB
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  assert(bgraph.smem_offset <= smem_size);
  customized_kernel_function<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      params);
}

bool KNCustomizedOp::profile(ProfileResult &result) {
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  for (int i = 0; i < 16; i++) {
    run();
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
