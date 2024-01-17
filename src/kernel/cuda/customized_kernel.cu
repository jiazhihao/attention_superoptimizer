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

namespace aso {
namespace kernel {
namespace customized {

CUTLASS_DEVICE
void kernel_function(Params const &params) {
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < forloop_range; i++) {
    // TODO: prologue for loading data into shared memory
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      if (params.operators[op].type == aso::type::TB_MATMUL) {
        using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 32>;
        using WarpShape = cutlass::gemm::GemmShape<32, 32, 32>;
        using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
        int thread_idx = threadId.x;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;

        aso::threadblock::matmul::MatmulExecutor<ThreadblockShape,
            WarpShape, InstructionShape, cutlass::half_t,
            cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>
                executor(thread_idx, warp_idx, lane_idx);
        executor.compute_kernel();
      }
    }
  }
}

void Customized::run() {
  int smem_size = 48 * 1024 * 1024;
  kernel_function<<<grid_dim, block_dim, smem_size>>>();
}

bool Customized::profile(ProfileResult &result) {
}

} // namespace customized
} // namespace kernel
} // namespace aso
