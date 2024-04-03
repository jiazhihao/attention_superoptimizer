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

#include "aso/threadblock/smem_tensor.h"
#include "aso/utils/static_switch.h"

#include "cutlass/aligned_buffer.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/threadblock/default_gemv_core.h"
#include "cutlass/gemm_coord.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/tensor_ref.h"

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"
#include "cutlass/transform/threadblock/vector_iterator.h"
#include "cutlass/transform/warp/vector_fragment_iterator.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/threadblock/epilogue_smem_accumulator.h"
#include "cutlass/epilogue/warp/fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"

#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/warp/mma_tensor_op_tile_iterator.h"

namespace aso {
namespace threadblock {

using namespace cutlass;
using namespace aso::type;

template <typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename SmemLayoutC>
class MatmulExecutorV0 {
public:
  using TensorRefA = TensorRef<ElementType, SmemLayoutA>;
  using TensorRefB = TensorRef<ElementType, SmemLayoutB>;
  using TensorRefC = TensorRef<ElementType, SmemLayoutC>;
  TensorRefA ref_A;
  TensorRefB ref_B;
  TensorRefC ref_C;
  int const m, n, k;
  size_t pos;
  size_t const block_size;
  CUTLASS_DEVICE
  MatmulExecutorV0(TensorRefA const &ref_A,
                   TensorRefB const &ref_B,
                   TensorRefC const &ref_C,
                   int m,
                   int n,
                   int k,
                   int thread_idx,
                   int warp_idx,
                   int lane_idx)
      : ref_A(ref_A), ref_B(ref_B), ref_C(ref_C), m(m), n(n), k(k),
        pos(thread_idx), block_size(blockDim.x * blockDim.y * blockDim.z) {}
  CUTLASS_DEVICE
  void execute_kernel(void) {
    size_t const mn = m * n;
    while (pos < mn) {
      int const i = pos / n;
      int const j = pos % n;
      ElementType sum = static_cast<ElementType>(0);
      for (int l = 0; l < k; l++) {
        sum += ref_A.at({i, l}) * ref_B.at({l, j});
      }
      ref_C.at({i, j}) += sum;
      pos += block_size;
    }
  }
};

template <typename ThreadblockShape,
          typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename LayoutA,
          typename LayoutB>
class MatmulExecutorV1 {
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
                                                 cutlass::layout::RowMajor,
                                                 arch::OpClassTensorOp,
                                                 2 /*kStages*/,
                                                 arch::OpMultiplyAdd>;
  using Operator = typename MmaCore::MmaPolicy::Operator;
  using WarpFragmentA = typename Operator::FragmentA;
  using WarpFragmentB = typename Operator::FragmentB;
  using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
  using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
  using WarpFragmentC = typename Operator::FragmentC;
  using MmaTensorOp = typename MmaCore::MmaTensorOp;
  using FragmentIteratorAccumulator =
      epilogue::warp::FragmentIteratorTensorOp<WarpShape,
                                               InstructionShape,
                                               ElementType,
                                               WarpFragmentC,
                                               cutlass::layout::RowMajor>;
  // Create an epilogue
  // Iterator to store to shared-memory
  using SmemAccumulatorLayout = cutlass::layout::RowMajor;
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
  MatmulExecutorV1( // TODO: add SharedStorage &shared_storage,
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

template <typename ThreadBlockShape,
          typename ThreadShape,
          typename ElementType,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename SmemLayoutC>
class GemvExecutorV0 {
public:
  using Core =
      typename cutlass::gemm::threadblock::DefaultGemvCore<ThreadBlockShape,
                                                           ThreadShape,
                                                           ElementType,
                                                           SmemLayoutA,
                                                           ElementType,
                                                           SmemLayoutB,
                                                           ElementType,
                                                           SmemLayoutC>;
  using Operator = typename Core::Operator;

  /// Iterates over A in global memory
  using IteratorA = typename Core::IteratorA;

  /// Iterates over B in global memory
  using IteratorB = typename Core::IteratorB;

  /// Fragment of operand C loaded from global memory
  using IteratorC = typename Core::IteratorC;

  using LayoutCD = SmemLayoutC;

  /// Policy for the iterator that reads/writes C/D
  using IteratorPolicyCD = typename cutlass::platform::conditional<
      cutlass::platform::is_same<LayoutCD, cutlass::layout::RowMajor>::value,
      cutlass::transform::PitchLinearTilePolicyStripminedThreadContiguous<
          cutlass::layout::PitchLinearShape<ThreadBlockShape::kN,
                                            ThreadBlockShape::kM>,
          Core::kThreadsPerN,
          ThreadShape::kN>,
      cutlass::transform::PitchLinearTilePolicyStripminedThreadStrided<
          cutlass::layout::PitchLinearShape<ThreadBlockShape::kM,
                                            ThreadBlockShape::kN>,
          Core::kThreadsPerN,
          ThreadShape::kM>>::type;

  /// Iterator that reads/writes C/D
  using IteratorCD = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<ThreadBlockShape::kM, ThreadBlockShape::kN>,
      ElementType,
      LayoutCD,
      0,
      IteratorPolicyCD>;

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of operand accumulator loaded/stored to global memory
  using FragmentC = typename Operator::FragmentC;

  using Params_A = typename IteratorA::Params;
  using Params_B = typename IteratorB::Params;

  using TensorRefA = typename IteratorA::TensorRef;
  using TensorRefB = typename IteratorB::TensorRef;
  using TensorRefC = typename IteratorCD::TensorRef;

  FragmentC accum;
  IteratorA iterator_A;
  IteratorB iterator_B;
  int gemm_k;

  CUTLASS_DEVICE
  GemvExecutorV0(TensorRefA const &ref_A,
                 TensorRefB const &ref_B,
                 TensorRefC const &ref_C,
                 int m,
                 int n,
                 int k,
                 int thread_idx,
                 int warp_idx,
                 int lane_idx)
      : iterator_A(Params_A(ref_A.layout()), ref_A.data(), {1, k}, 0, {0, 0}),
        iterator_B(
            Params_B(ref_B.layout()), ref_B.data(), {k, n}, thread_idx, {0, 0}),
        gemm_k(k) {}
  CUTLASS_DEVICE
  void execute_kernel(void) {
    FragmentA frag_A;
    FragmentB frag_B;
    frag_A.clear();
    frag_B.clear();

    iterator_A.load(frag_A);
    iterator_B.load(frag_B);
    ++iterator_A;
    ++iterator_B;

    //
    // Mainloop
    //
    Operator thread_mma;

    if (gemm_k < ThreadBlockShape::kK) {
      iterator_A.clear_mask();
      iterator_B.clear_mask();
    }

    // iterate over K to accumulate result
    CUTLASS_GEMM_LOOP
    for (; gemm_k > 0; gemm_k -= ThreadBlockShape::kK) {
      thread_mma(accum, frag_A, frag_B, accum);

      iterator_A.load(frag_A);
      iterator_B.load(frag_B);
      ++iterator_A;
      ++iterator_B;

      if (gemm_k < ThreadBlockShape::kK) {
        iterator_A.clear_mask();
        iterator_B.clear_mask();
      }
    }
  }
};

template <typename WarpShape,
          typename InstructionShape,
          typename ElementType,
          typename SmemLayoutA,
          typename SmemLayoutB,
          typename SmemLayoutC>
class MatmulExecutorV2 {
public:
  using SmemLayoutA_T = cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementType>::value,
      WarpShape::kK>;

  // Shared memory layout
  using SmemLayoutB_T =
      cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<
          sizeof_bits<ElementType>::value,
          WarpShape::kK>;
  using MmaTensorOp = typename gemm::warp::DefaultMmaTensorOp<
      WarpShape,
      InstructionShape,
      ElementType,   // Data type of A elements
      SmemLayoutA_T, // Layout of A matrix
      ElementType,   // Data type of B elements
      SmemLayoutB_T, // Layout of B matrix
      ElementType,   // Data type of C elements
      SmemLayoutC,   // Layout of C matrix
      arch::OpMultiplyAdd,
      1>::Type;

  using WarpFragmentA = typename MmaTensorOp::FragmentA;
  using WarpFragmentB = typename MmaTensorOp::FragmentB;
  using WarpFragmentC = typename MmaTensorOp::FragmentC;
  using WarpTransformedFragmentA = typename MmaTensorOp::TransformedFragmentA;
  using WarpTransformedFragmentB = typename MmaTensorOp::TransformedFragmentB;

  // Define a 'FragmentIterator' to iterate over slices of accumulators
  // See:
  // cutlass/examples/13_two_tensor_op_fusion/threadblock/default_b2b_mma_smem_accumulator.h:134
  using FragmentIteratorAccumulator = epilogue::warp::FragmentIteratorTensorOp<
      WarpShape,
      InstructionShape,
      ElementType,
      typename MmaTensorOp::Policy::Operator::FragmentC,
      SmemLayoutC>;

  // Create an epilogue
  // Iterator to store to shared-memory
  using SmemIteratorD = epilogue::warp::TileIteratorTensorOp<WarpShape,
                                                             InstructionShape,
                                                             ElementType,
                                                             SmemLayoutC>;

  // We need to provide an operation for the epilogue. Let's create an
  // operation that simply accumulates (ScaleType::NoBetaScaling)
  using OutputOpAccumulate = epilogue::thread::LinearCombination<
      typename SmemIteratorD::Element, // ElementOutput
      FragmentIteratorAccumulator::Fragment::kElements,
      ElementType,                     // ElementAccumulator
      typename SmemIteratorD::Element, // ElementCompute
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using Epilogue = epilogue::threadblock::EpilogueSmemAccumulator<
      SmemIteratorD,
      FragmentIteratorAccumulator,
      SmemIteratorD,
      OutputOpAccumulate>;

  using TensorRefA = typename MmaTensorOp::IteratorA::TensorRef;
  using TensorRefB = typename MmaTensorOp::IteratorB::TensorRef;
  using TensorRefC = typename SmemIteratorD::TensorRef;

  // Number of gemm iterations along the k dimension
  static constexpr int kWarpGemmIterations =
      (WarpShape::kK + InstructionShape::kK - 1) / InstructionShape::kK;

  /// cutlass fields
  typename MmaTensorOp::IteratorA warp_tile_iterator_A;
  typename MmaTensorOp::IteratorB warp_tile_iterator_B;
  int _warp_idx;
  int _lane_idx;
  SmemIteratorD smem_iterator_D;
  OutputOpAccumulate output_op;
  int gemm_k_iterations_0;

  CUTLASS_DEVICE
  MatmulExecutorV2(TensorRefA const &ref_A,
                   TensorRefB const &ref_B,
                   TensorRefC const &ref_C,
                   int m,
                   int n,
                   int k,
                   int thread_idx,
                   int warp_idx,
                   int lane_idx)
      : output_op({}),
        // See:
        // cutlass/examples/13_two_tensor_op_fusion/threadblock/b2b_mma_base.h:223
        warp_tile_iterator_A(ref_A, lane_idx),
        warp_tile_iterator_B(ref_B, lane_idx),
        smem_iterator_D(ref_C, lane_idx) {
    int warp_count_m = (m + WarpShape::kM - 1) / WarpShape::kM;
    int warp_count_n = (n + WarpShape::kN - 1) / WarpShape::kN;
    assert(warp_count_m > 0);
    assert(warp_count_n > 0);
    // We have exactly 4 warps in a thread block
    assert(warp_count_m * warp_count_n == 4);
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   warp_idx_m: the warp's position within the threadblock along the M
    //               dimension
    //   warp_idx_n: the warp's position within the threadblock along the N
    //               dimension
    //   warp_idx_k: the warp's position within the threadblock along the K
    //               dimension

    int warp_idx_mn = warp_idx % (warp_count_m * warp_count_n);
    int warp_idx_k = warp_idx / (warp_count_m * warp_count_n);
    if (false && lane_idx == 0) {
      printf("warp_idx(%d) warp_count_m(%d) warp_count_n(%d) m(%d) n(%d)\n",
             warp_idx,
             warp_count_m,
             warp_count_n,
             m,
             n);
    }
    // Currently assume that we don't parittion over k within a threadblock
    // we do it across threadblocks
    if (warp_idx_k > 0) {
      printf("warp_idx(%d) warp_count_m(%d) warp_count_n(%d) m(%d) n(%d)\n",
             warp_idx,
             warp_count_m,
             warp_count_n,
             m,
             n);
      assert(warp_idx_k == 0);
    }
    int warp_idx_m = warp_idx_mn % warp_count_m;
    int warp_idx_n = warp_idx_mn / warp_count_m;

    int tile_offset_k = kWarpGemmIterations * warp_idx_k;

    // Add per-warp offsets in units of warp-level tiles
    warp_tile_iterator_A.add_tile_offset({warp_idx_m, tile_offset_k});
    warp_tile_iterator_B.add_tile_offset({tile_offset_k, warp_idx_n});
    gemm_k_iterations_0 = (k + WarpShape::kK - 1) / WarpShape::kK;

    // Add smem accumulator iterator warp offset
    smem_iterator_D.add_tile_offset(
        {warp_idx_m * SmemIteratorD::TileIterations::kRow,
         warp_idx_n * SmemIteratorD::TileIterations::kColumn});
    _warp_idx = warp_idx;
    _lane_idx = lane_idx;
  }

  CUTLASS_DEVICE
  void execute_kernel(void) {
    // extern __shared__ char smem_buffer[];

    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
    WarpFragmentC accum;
    accum.clear();

    warp_tile_iterator_A.set_kgroup_index(0);
    warp_tile_iterator_B.set_kgroup_index(0);
    warp_tile_iterator_A.load(warp_frag_A[0]);
    warp_tile_iterator_B.load(warp_frag_B[0]);
    ++warp_tile_iterator_A;
    ++warp_tile_iterator_B;

    MmaTensorOp warp_mma;

    // We should unroll the k groups, and the main loop is hard-coded in the
    // threadblock graph.
    warp_mma.transform(warp_transformed_frag_A_[0],
                       warp_transformed_frag_B_[0],
                       warp_frag_A[0],
                       warp_frag_B[0]);

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations_0 > 0; --gemm_k_iterations_0) {

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < kWarpGemmIterations; warp_mma_k++) {

        warp_tile_iterator_A.set_kgroup_index((warp_mma_k + 1) %
                                              kWarpGemmIterations);
        warp_tile_iterator_B.set_kgroup_index((warp_mma_k + 1) %
                                              kWarpGemmIterations);

        // skip warp tile loading for the last kgroup
        // if (warp_mma_k < kWarpGemmIterations - 1) {
        warp_tile_iterator_A.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        warp_tile_iterator_B.load(warp_frag_B[(warp_mma_k + 1) % 2]);
        // }
        ++warp_tile_iterator_A;
        ++warp_tile_iterator_B;
        if (warp_mma_k > 0) {
          warp_mma.transform(warp_transformed_frag_A_[warp_mma_k % 2],
                             warp_transformed_frag_B_[warp_mma_k % 2],
                             warp_frag_A[warp_mma_k % 2],
                             warp_frag_B[warp_mma_k % 2]);
        }

        if (false && _lane_idx == 0) {
          printf(
              "warp_idx(%d) warp_mma_k(%d) lane_idx(%d) got A = %f, B = %f\n",
              _warp_idx,
              warp_mma_k,
              _lane_idx,
              static_cast<float>(warp_frag_A[warp_mma_k % 2].front()),
              static_cast<float>(warp_frag_B[warp_mma_k % 2].front()));
        }

        warp_mma(accum,
                 warp_transformed_frag_A_[warp_mma_k % 2],
                 warp_transformed_frag_B_[warp_mma_k % 2],
                 accum);
        if (warp_mma_k + 1 == kWarpGemmIterations) {
          warp_mma.transform(warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
                             warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
                             warp_frag_A[(warp_mma_k + 1) % 2],
                             warp_frag_B[(warp_mma_k + 1) % 2]);
        }
      }
    }

    Epilogue epilogue;
    // accum.fill(static_cast<ElementType>(_warp_idx)); // for debugging
    epilogue(output_op, smem_iterator_D, accum);
    __syncthreads();
  }
};

CUTLASS_DEVICE
void calculate_warp_shape(
    int m, int n, int inst_m, int inst_n, int &warp_m, int &warp_n) {
  int m_factor = (m + inst_m - 1) / inst_m;
  int n_factor = (n + inst_n - 1) / inst_n;
  // We should have at least 4 warps in a thread block
  assert(m_factor * n_factor >= 4);
  if (m_factor >= 2 && n_factor >= 2) {
    assert(m_factor % 2 == 0);
    assert(n_factor % 2 == 0);
    m_factor = m_factor / 2;
    n_factor = n_factor / 2;
  } else if (m_factor == 1) {
    assert(n_factor % 4 == 0);
    n_factor = n_factor / 4;
  } else if (n_factor == 1) {
    assert(m_factor % 4 == 0);
    m_factor = m_factor / 4;
  }
  warp_m = m_factor * inst_m;
  warp_n = n_factor * inst_n;
}

class GenericMatmulExecutor {
public:
  CUTLASS_DEVICE
  GenericMatmulExecutor(half_t* A_ptr,
                        half_t* B_ptr,
                        half_t* C_ptr,
                        int m,
                        int n,
                        int k,
                        int thread_id,
                        int warp_id,
                        int lane_id) {
    //assert(A.num_dims == B.num_dims);
    //int m = A.dim[A.num_dims - 2];
    //int n = B.dim[B.num_dims - 1];
    //int k = A.dim[A.num_dims - 1];
    using InstructionShape = gemm::GemmShape<16, 8, 16>;
    int warp_m = 0;
    int warp_n = 0;
    calculate_warp_shape(
        m, n, InstructionShape::kM, InstructionShape::kN, warp_m, warp_n);
    // if (thread_id == 0 && blockIdx.x == 0) {
    //   printf("warp_m(%d) warp_n(%d)\n", warp_m, warp_n);
    // }
    WARP_SHAPE_M_SWITCH(warp_m, WARP_M, [&] {
      WARP_SHAPE_N_SWITCH(warp_n, WARP_N, [&] {
        using WarpShape = gemm::GemmShape<WARP_M, WARP_N, InstructionShape::kK>;
        // TODO: consider cutlass' RowMajorTensorOpMultiplicandCrosswise
        // layout
        // using SmemLayoutA =
        // cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16, 32>; using
        // SmemLayoutB =
        // cutlass::layout::ColumnMajorTensorOpMultiplicandCrosswise<16, 32>;
        using Executor = MatmulExecutorV2<WarpShape,
                                          InstructionShape,
                                          half_t,
                                          cutlass::layout::RowMajor,
                                          cutlass::layout::ColumnMajor,
                                          cutlass::layout::RowMajor>;
        // assert(A.layout == STensor::ROW_MAJOR &&
        //        B.layout == STensor::COLUMN_MAJOR &&
        //        "Layouts: mismatch between inputs and Executor.");
        //half_t *A_ptr = (half_t *)(smem_buffer + A.smem_offset);
        //half_t *B_ptr = (half_t *)(smem_buffer + B.smem_offset);
        //half_t *C_ptr = (half_t *)(smem_buffer + C.smem_offset);

        Executor executor({A_ptr, Executor::TensorRefA::Layout::packed({m, k})},
                          {B_ptr, Executor::TensorRefB::Layout::packed({k, n})},
                          {C_ptr, Executor::TensorRefC::Layout::packed({m, n})},
                          m,
                          n,
                          k,
                          thread_id,
                          warp_id,
                          lane_id);
        executor.execute_kernel();
      });
    });
  }
};

class GenericGemvExecutor {
public:
  CUTLASS_DEVICE
  GenericGemvExecutor(char *smem_buffer,
                      STensor const &A,
                      STensor const &B,
                      STensor const &C,
                      int thread_id,
                      int warp_id,
                      int lane_id) {
    int m = A.dim[0];
    int n = B.dim[1];
    int k = A.dim[1];
    using ThreadBlockShape = cutlass::gemm::GemmShape<1, 64, 4>;
    using ThreadShape = cutlass::gemm::GemmShape<1, 4, 4>;
    using Executor = GemvExecutorV0<ThreadBlockShape,
                                    ThreadShape,
                                    half_t,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor,
                                    cutlass::layout::RowMajor>;
    half_t *A_ptr = (half_t *)(smem_buffer + A.smem_offset);
    half_t *B_ptr = (half_t *)(smem_buffer + B.smem_offset);
    half_t *C_ptr = (half_t *)(smem_buffer + C.smem_offset);

    Executor executor({A_ptr, Executor::TensorRefA::Layout::packed({m, k})},
                      {B_ptr, Executor::TensorRefB::Layout::packed({k, n})},
                      {C_ptr, Executor::TensorRefC::Layout::packed({m, n})},
                      m,
                      n,
                      k,
                      thread_id,
                      warp_id,
                      lane_id);
    executor.execute_kernel();
  }
};

class TBMatmulFingerprinter {
public:
  CUTLASS_DEVICE
  TBMatmulFingerprinter(FPType* A_ptr,
                        FPType* B_ptr,
                        FPType* C_ptr,
                        int a_m_size,
                        int c_n_size,
                        int a_k_size,
                        int thread_id,
                        int num_threads) {
    // Note that we assume all tensors are in row-major layouts
    // when computing fingerprints
    //FPType *A_ptr = (FPType *)(smem_buffer + A.smem_offset);
    //FPType *B_ptr = (FPType *)(smem_buffer + B.smem_offset);
    //FPType *C_ptr = (FPType *)(smem_buffer + C.smem_offset);
    //int num_batches = 1;
    //for (int i = 0; i < C.num_dims - 2; i++) {
    //  num_batches *= C.dim[i];
    //}
    // Do not support batch matmul in TB
    //assert(num_batches == 1);
    int num_elements = a_m_size * c_n_size;
    //int c_n_size = C.dim[C.num_dims - 1];
    //int a_k_size = A.dim[A.num_dims - 1];
    int b_n_size = c_n_size;
    for (int i = thread_id; i < num_elements; i += num_threads) {
      uint32_t result = 0;
      int m = i / c_n_size;
      int n = i % c_n_size;
      for (int k = 0; k < a_k_size; k++) {
        uint32_t a_value = A_ptr[m * a_k_size + k];
        uint32_t b_value = B_ptr[k * b_n_size + n];
        result = (result + a_value * b_value) % FP_PQ;
        if (false && thread_id == 0) {
          printf("i(%d) block(%d %d %d) result(%d) a_value(%d) b_value(%d)"
                 "n(%d) m(%d) k(%d) a_k_size(%d) b_n_size(%d) c_n_size(%d)\n",
                 i,
                 blockIdx.x,
                 blockIdx.y,
                 blockIdx.z,
                 (int)result,
                 (int)a_value,
                 (int)b_value,
                 n,
                 m,
                 k,
                 a_k_size,
                 b_n_size,
                 c_n_size);
        }
      }
      C_ptr[i] = result;
    } // for i
  }
};

} // namespace threadblock
} // namespace aso
