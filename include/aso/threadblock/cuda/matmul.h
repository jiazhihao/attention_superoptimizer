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

#include "cutlass/aligned_buffer.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"

namespace aso {
namespace threadblock {

using namespace cutlass;

template <typename ThreadblockShape,
          typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename LayoutA,
          typename LayoutB>
class MatmulExecutor {
public:
  static int const kStages = 2; // 2 stages for pipelined executions
#ifdef DEADCODE
  template <typename Shape_, typename Policy_>
  class SharedStorage {
  public:
    using Shape = Shape_;
    using Policy = Policy_;
    using Operator = typename Policy::Operator;
    using TensorRefA =
        TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;
    using TensorRefB =
        TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;
    /// Shape of the A matrix operand in shared memory
    using ShapeA =
        MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                    Shape::kK * kStages + Policy::SmemPaddingA::kColumn>;
    /// Shape of the B matrix operand in shared memory
    using ShapeB = MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                               Shape::kN + Policy::SmemPaddingB::kColumn>;

  public:
    /// Buffer for A operand
    AlignedBuffer<typename Operator::ElementA, ShapeA::kCount> operand_A;

    /// Buffer for B operand
    AlignedBuffer<typename Operator::ElementB, ShapeB::kCount> operand_B;

  public:
    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({ShapeA::kRow, ShapeA::kColumn});
    }
    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({ShapeB::kRow, ShapeB::kColumn});
    }
    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{operand_A.data(), LayoutA()};
    }
    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{operand_B.data(), LayoutB()};
    }
  };
#endif
public:
  // cutlass typename
  using MmaCore =
      typename gemm::threadblock::DefaultMmaCore<ThreadblockShape,
                                                 WarpShape,
                                                 InstructionShape,
                                                 ElementType,
                                                 LayoutA,
                                                 ElementType,
                                                 LayoutB,
                                                 ElementType,
                                                 layout::RowMajor,
                                                 arch::OpClassTensorOp,
                                                 2 /*kStages*/,
                                                 arch::OpMultiplyAdd>;
  using Operator = typename MmaCore::MmaPolicy::Operator;
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;
  using WarpFragmentC = typename Operator::FragmentC;
  using MmaTensorOp = typename MmaCore::MmaTensorOp;
  using FragmentIteratorAccumulator =
      epilogue::warp::FragmentIteratorTensorOp<WarpShape,
                                               InstructionShape,
                                               ElementType,
                                               WarpFragmentC,
                                               layout::RowMajor>;
  // Create an epilogue
  // Iterator to store to shared-memory
  using SmemAccumulatorLayout = layout::RowMajor;
  using SmemIteratorD =
      typename epilogue::warp::TileIteratorTensorOp<WarpShape,
                                                    InstructionShape,
                                                    ElementType,
                                                    SmemAccumulatorLayout>;
  // We need to provide an operation for the epilogue. Let's create an
  // operation that does nothing (ScaleType::Nothing)
  using OutputOpNoOp = cutlass::epilogue::thread::LinearCombination<
      typename SmemIteratorD::Element, // ElementOutput
      FragmentIteratorAccumulator::Fragment::kElements,
      ElementType,                     // ElementAccumulator
      typename SmemIteratorD::Element, // ElementCompute
      cutlass::epilogue::thread::ScaleType::Nothing>;
  using Epilogue = cutlass::epilogue::threadblock::EpilogueSmemAccumulator<
      SmemIteratorD,
      FragmentIteratorAccumulator,
      SmemIteratorD,
      OutputOpNoOp>;
  // Number of warp-level GEMM oeprations
  using WarpGemm = typename Operator::Shape;
  /// Shape describing the number of warps filling the CTA
  using WarpCount = gemm::GemmShape<MmaCore::Shape::kM / WarpGemm::kM,
                                    MmaCore::Shape::kN / WarpGemm::kN,
                                    MmaCore::Shape::kK / WarpGemm::kK>;
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  // cutlass fields
  typename Operator::IteratorA warp_tile_iterator_A;
  typename Operator::IteratorB warp_tile_iterator_B;
  SmemIteratorD smem_iterator_D;
  OutputOpNoOp output_op;

  CUTLASS_DEVICE
  MatmulExecutor( // TODO: add SharedStorage &shared_storage,
                  ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx)
      // warp_tile_iterator_A(shared_storage.operand_A_ref(), lane_idx),
      // warp_tile_iterator_B(shared_storage.operand_B_ref(), lane_idx),
      // smem_iterator_D(shared_storage.accumulator_shared_storage0.accum_ref(),
      // lane_idx),
      : output_op({}) {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (WarpCount::kM * WarpCount::kN);
    int warp_idx_k = warp_idx / (WarpCount::kM * WarpCount::kN);

    int warp_idx_m = warp_idx_mn % WarpCount::kM;
    int warp_idx_n = warp_idx_mn / WarpCount::kM;

    int tile_offset_k = kWarpGemmIterations * warp_idx_k;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A.add_tile_offset({warp_idx_m, tile_offset_k});
    warp_tile_iterator_B.add_tile_offset({tile_offset_k, warp_idx_n});

    // Add smem accumulator iterator warp offset
    smem_iterator_D.add_tile_offset(
        {warp_idx_m * SmemIteratorD::TileIterations::kRow,
         warp_idx_n * SmemIteratorD::TileIterations::kColumn});
  }

  void CUTLASS_DEVICE execute_kernel(void) {
    // extern __shared__ char smem_buffer[];

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    WarpFragmentC accum;
    Epilogue epilogue;

    MmaTensorOp warp_mma;

    warp_tile_iterator_A.set_kgroup_index(0);
    warp_tile_iterator_B.set_kgroup_index(0);
    warp_tile_iterator_A.load(warp_frag_A[0]);
    warp_tile_iterator_B.load(warp_frag_B[0]);
    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;

    CUTLASS_GEMM_LOOP
    for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; warp_mma_k++) {
      // Load warp-level tiles from shared memory, wrapping to k offset if
      // this is the last group as the case may be.
      if (warp_mma_k == kWarpGemmIterations - 1) {
        // Write fragments to shared memory
        // reference:
        // cutlass/examples/13/threadblock/b2b_mma_pipelined_smem_accumulators.h:376
        assert(false);
      }
      warp_tile_iterator_A.set_kgroup_index((warp_mma_k + 1) %
                                            kWarpGemmIterations);
      warp_tile_iterator_B.set_kgroup_index((warp_mma_k + 1) %
                                            kWarpGemmIterations);
      warp_tile_iterator_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
      warp_tile_iterator_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
      ++warp_tile_iterator_A;
      ++warp_tile_iterator_B;
      if (warp_mma_k == 0) {
        // Write fragments to shared memory
        // reference:
        // cutlass/examples/13/threadblock/b2b_mma_pipelined_smem_accumulators.h:376
        assert(false);
      }
      warp_mma(accum,
               warp_frag_A[warp_mma_k % 2],
               warp_frag_B[warp_mma_k % 2],
               accum);
    }
    // TODO: enable epilogue
    // epilogue(OutputOpNoOp({}), smem_iterator_D, accum);
    __syncthreads();
  }
};

} // namespace threadblock
} // namespace aso
