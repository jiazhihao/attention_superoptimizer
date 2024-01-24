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
#include "aso/threadblock/cuda/matmul.h"
#include "aso/threadblock/cuda/reduction.h"

namespace aso {
namespace kernel {

__global__ void
    customized_kernel_function(KNCustomizedOp::Params const &params) {
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          break;
        }
        case aso::type::TB_MATMUL_OP: {
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
  Params params;
  customized_kernel_function<<<plan.grid_dim, plan.block_dim, smem_size>>>(
      params);
}

bool KNCustomizedOp::profile(ProfileResult &result) {}

} // namespace kernel
} // namespace aso
