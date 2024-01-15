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

#include "aso/threadblock/graph.h"
#include "aso/threadblock/matmul.h"
#include "aso/utils/cuda_helper.h"
#include "aso/utils/hash_utils.h"
#include <cassert>
#include "cutlass/cutlass.h"

namespace aso {
namespace threadblock {
namespace matmul {

template<
    typename scalar_type,
    typename arch_tag>

void CUTLASS_DEVICE Operator::compute_kernel(void) {
  extern __shared__ char smem_buffer[];
  WarpLoadedFragmentA warp_loaded_frag_A[2];
  WarpLoadedFragmentB warp_loaded_frag_B[2];
  FragmentC accum;
  WarpTransformedFragmentA warp_transformed_frag_A[2];
  WarpTransformedFragmentB warp_transformed_frag_B[2];
  Epilogue epilogue;

  WarpOperator warp_mma;

  warp_tile_iterator_A.set_kgroup_index(0);
  warp_tile_iterator_B.set_kgroup_index(0);
  warp_tile_iterator_A.load(warp_loaded_frag_A[0]);
  warp_tile_iterator_B.load(warp_loaded_frag_B[0]);
  ++warp_tile_iterator_A;
  ++warp_tile_iterator_B;

  int gemm_k_iterations = ;
  CUTLASS_GEMM_LOOP
  for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; warp_mma_k++) {
    // Load warp-level tiles from shared memory, wrapping to k offset if
    // this is the last group as the case may be.
    warp_tile_iterator_A.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
    warp_tile_iterator_B.set_kgroup_index((warp_mma_k + 1) % kWarpGemmIterations);
    warp_tile_iterator_A.load(warp_loaded_frag_A[(warp_mma_k + 1) % 2]);
    warp_tile_iterator_B.load(warp_loaded_frag_B[(warp_mma_k + 1) % 2]);
    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;
    iterator_A.clear_mask(gemm_k_iterations_0 == 0);
    warp_mma.transform(warp_transformed_frag_A[warp_mma_k % 2],
                       warp_transformed_frag_B[warp_mma_k % 2],
                       warp_loaded_frag_A[warp_mma_k % 2],
                       warp_loaded_frag_B[warp_mma_k % 2]);
    warp_mma(
      accum,
      warp_transformed_frag_A[warp_mma_k % 2],
      warp_transformed_frag_B[warp_mma_k % 2],
      accum
    );
  }
  epilogue(output_op, smem_iterator_D, accum);
  __syncthreads();
}

} // namespace matmul
} // namespace threadblock
} // namespace aso
