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

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/matrix_coord.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"
#include "cutlass/transform/threadblock/predicated_vector_access_iterator.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_iterator_tensor_op.h"

namespace aso {
namespace threadblock {

using namespace cutlass;

template <typename ElementType,
          int kRow,
          int kColumn,
          int kThreads,
          typename DmemLayout,
          typename SmemLayout>
class RowMajorInputLoader {
public:
  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Number of threads per warp
  static int const kWarpSize = 32;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      kColumn / (kAccessSizeInBits / sizeof_bits<ElementType>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  /// Shared memory layout
  // using DmemLayout = layout::RowMajor;

  /// Shared memory layout
  // using SmemLayout = layout::RowMajorTensorOpMultiplicandCrosswise<
  //     sizeof_bits<ElementType>::value,
  //     kColumn>;

  /// ThreadMap of iterator A
  using IteratorThreadMap = transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kColumn, kRow>,
      kThreads,
      cutlass::layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                        kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementType>::value>;

  // Define iterators over tiles from the A operand
  using DmemIterator =
      transform::threadblock::PredicatedTileIterator<MatrixShape<kRow, kColumn>,
                                                     ElementType,
                                                     DmemLayout,
                                                     1,
                                                     IteratorThreadMap>;

  /// Shared memory iterator to A operand
  using SmemIterator =
      transform::threadblock::RegularTileIterator<MatrixShape<kRow, kColumn>,
                                                  ElementType,
                                                  SmemLayout,
                                                  0,
                                                  IteratorThreadMap>;

  /// Fragment of operand loaded from global memory
  using Fragment = typename DmemIterator::Fragment;

public:
  CUTLASS_DEVICE
  RowMajorInputLoader(ElementType *dmem_ptr,
                      ElementType *smem_ptr,
                      MatrixCoord extent,
                      int thread_id,
                      MatrixCoord matrix_offset)
      : dmem_iterator(DmemLayout::packed(extent),
                      dmem_ptr,
                      extent,
                      thread_id,
                      matrix_offset),
        smem_iterator({smem_ptr, SmemLayout::packed({kRow, kColumn})},
                      thread_id) {
    Fragment tb_fragment;
    // The last kblock is loaded in the prolog
    dmem_iterator.load(tb_fragment);
    smem_iterator.store(tb_fragment);
  }

public:
  DmemIterator dmem_iterator;
  SmemIterator smem_iterator;
};

template <typename ElementType,
          int kRow,
          int kColumn,
          int kThreads,
          typename DmemLayout,
          typename SmemLayout>
class ColumnMajorInputLoader {
public:
  /// Size of a threadblock-scoped access
  static int const kAccessSizeInBits = 128;

  /// Number of threads per warp
  static int const kWarpSize = 32;

  // Warp thread arrangement
  static int const kWarpThreadArrangementContiguousA =
      kRow / (kAccessSizeInBits / sizeof_bits<ElementType>::value);

  static int const kWarpThreadArrangementStridedA =
      kWarpSize / kWarpThreadArrangementContiguousA;

  /// ThreadMap of iterator A
  using IteratorThreadMap = transform::PitchLinearWarpRakedThreadMap<
      cutlass::layout::PitchLinearShape<kRow, kColumn>,
      kThreads,
      cutlass::layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
                                        kWarpThreadArrangementStridedA>,
      kAccessSizeInBits / sizeof_bits<ElementType>::value>;

  // Define iterators over tiles from the A operand
  using DmemIterator =
      transform::threadblock::PredicatedTileIterator<MatrixShape<kRow, kColumn>,
                                                     ElementType,
                                                     DmemLayout,
                                                     0,
                                                     IteratorThreadMap>;

  /// Shared memory iterator to A operand
  using SmemIterator =
      transform::threadblock::RegularTileIterator<MatrixShape<kRow, kColumn>,
                                                  ElementType,
                                                  SmemLayout,
                                                  1,
                                                  IteratorThreadMap>;

  /// Fragment of operand loaded from global memory
  using Fragment = typename DmemIterator::Fragment;

public:
  CUTLASS_DEVICE
  ColumnMajorInputLoader(ElementType *dmem_ptr,
                         ElementType *smem_ptr,
                         MatrixCoord extent,
                         int thread_id,
                         MatrixCoord matrix_offset)
      : dmem_iterator(DmemLayout::packed(extent),
                      dmem_ptr,
                      extent,
                      thread_id,
                      matrix_offset),
        smem_iterator({smem_ptr, SmemLayout::packed({kRow, kColumn})},
                      thread_id) {
    Fragment tb_fragment;
    // The last kblock is loaded in the prolog
    dmem_iterator.load(tb_fragment);
    smem_iterator.store(tb_fragment);
  }

public:
  DmemIterator dmem_iterator;
  SmemIterator smem_iterator;
};

template <int kRow, int kColumn>
class ShapedInputLoader {
public:
  CUTLASS_DEVICE
  ShapedInputLoader(void *dtensor_ptr,
                    void *stensor_ptr,
                    int2 dtensor_matrix_shape,
                    aso::layout::DmemLayout dlayout,
                    aso::layout::SmemLayout slayout,
                    int thread_id,
                    int num_threads,
                    MatrixCoord matrix_offset,
                    int global_offset) {
    // assert(stensor.dim[stensor.num_dims - 2] == kRow);
    // assert(stensor.dim[stensor.num_dims - 1] == kColumn);
    //  Currently only support half precision
    int const kThreads = 128;
    // assert(num_threads == kThreads);
    // assert(stensor.data_type == aso::type::DT_FLOAT16);
    // assert(dtensor.data_type == aso::type::DT_FLOAT16);
    MatrixCoord extent(
        {dtensor_matrix_shape.x, dtensor_matrix_shape.y});
    //aso::layout::DmemLayout dlayout = dtensor.layout;
    //aso::layout::SmemLayout slayout = stensor.layout;
    if (dlayout == aso::layout::DmemRowMajor) {
      using DmemLayout = cutlass::layout::RowMajor;
      if (slayout == aso::layout::SmemRowMajor) {
        using SmemLayout = cutlass::layout::RowMajor;
        using InputLoader = RowMajorInputLoader<cutlass::half_t,
                                                kRow,
                                                kColumn,
                                                kThreads,
                                                DmemLayout,
                                                SmemLayout>;
        InputLoader loader(
            ((cutlass::half_t *)dtensor_ptr) + global_offset,
            (cutlass::half_t *)stensor_ptr,
            extent,
            thread_id,
            matrix_offset);
      } else if (
          slayout ==
              aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise16 ||
          slayout ==
              aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise32 ||
          slayout ==
              aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise64) {
        using SmemLayout =
            cutlass::layout::RowMajorTensorOpMultiplicandCrosswise<16 /*bits*/,
                                                                   kColumn>;
        using InputLoader = RowMajorInputLoader<cutlass::half_t,
                                                kRow,
                                                kColumn,
                                                kThreads,
                                                DmemLayout,
                                                SmemLayout>;
        InputLoader loader(
            ((cutlass::half_t *)dtensor_ptr) + global_offset,
            (cutlass::half_t *)stensor_ptr,
            extent,
            thread_id,
            matrix_offset);
      }
    } else {
      // assert(dlayout == aso::layout::DmemColumnMajor);
      using DmemLayout = cutlass::layout::ColumnMajor;
      if (slayout == aso::layout::SmemColumnMajor) {
        using SmemLayout = cutlass::layout::ColumnMajor;
        using InputLoader = ColumnMajorInputLoader<cutlass::half_t,
                                                   kRow,
                                                   kColumn,
                                                   kThreads,
                                                   DmemLayout,
                                                   SmemLayout>;
        InputLoader loader(
            ((cutlass::half_t *)dtensor_ptr) + global_offset,
            (cutlass::half_t *)stensor_ptr,
            extent,
            thread_id,
            matrix_offset);
      } else if (
          slayout ==
              aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16 ||
          slayout ==
              aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32 ||
          slayout ==
              aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64) {
        using SmemLayout = cutlass::layout::
            ColumnMajorTensorOpMultiplicandCrosswise<16 /*bits*/, kRow>;
        using InputLoader = ColumnMajorInputLoader<cutlass::half_t,
                                                   kRow,
                                                   kColumn,
                                                   kThreads,
                                                   DmemLayout,
                                                   SmemLayout>;
        InputLoader loader(
            ((cutlass::half_t *)dtensor_ptr) + global_offset,
            (cutlass::half_t *)stensor_ptr,
            extent,
            thread_id,
            matrix_offset);
      }
    }
  }
};

class GenericInputLoader {
public:
  CUTLASS_DEVICE
  GenericInputLoader(void* dtensor_ptr,
                     void* stensor_ptr,
                     int2 dtensor_matrix_shape,
                     int2 stensor_matrix_shape,
                     aso::layout::DmemLayout dlayout,
                     aso::layout::SmemLayout slayout,
                     int thread_id,
                     int num_threads,
                     MatrixCoord matrix_offset,
                     int global_offset) {
    int kRow = stensor_matrix_shape.x;
    int kColumn = stensor_matrix_shape.y;

    if (kRow == 64 && kColumn == 64) {
      ShapedInputLoader<64, 64>(dtensor_ptr,
                                stensor_ptr,
                                dtensor_matrix_shape,
                                dlayout,
                                slayout,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 32 && kColumn == 64) {
      ShapedInputLoader<32, 64>(dtensor_ptr,
                                stensor_ptr,
                                dtensor_matrix_shape,
                                dlayout,
                                slayout,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 64 && kColumn == 32) {
      ShapedInputLoader<64, 32>(dtensor_ptr,
                                stensor_ptr,
                                dtensor_matrix_shape,
                                dlayout,
                                slayout,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 16 && kColumn == 64) {
      ShapedInputLoader<16, 64>(dtensor_ptr,
                                stensor_ptr,
                                dtensor_matrix_shape,
                                dlayout,
                                slayout,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 1 && kColumn == 64) {
      ShapedInputLoader<64, 64>(smem_buffer,
                                dtensor,
                                stensor,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else {
      // if (threadIdx.x == 0 && blockIdx.x == 0) {
      //   printf("kRow = %d kColumn = %d\n", kRow, kColumn);
      // }
      assert(false && "Unimplemented");
    }
  }
};

class TBInputLoaderFingerprinter {
public:
  CUTLASS_DEVICE
  TBInputLoaderFingerprinter(aso::type::FPType* dtensor_ptr,
                             aso::type::FPType* stensor_ptr,
                             int2 dtensor_matrix_shape,
                             int2 stensor_matrix_shape,
                             aso::layout::DmemLayout dlayout,
                             aso::layout::SmemLayout slayout,
                             int thread_id,
                             int num_threads,
                             MatrixCoord matrix_offset,
                             int global_offset) {
    aso::type::FPType *smem_ptr = stensor_ptr;
    aso::type::FPType *dmem_ptr = dtensor_ptr + global_offset;
    int num_elements = stensor_matrix_shape.x * stensor_matrix_shape.y;
    int smem_num_column = stensor_matrix_shape.y;
    int dmem_num_column = dtensor_matrix_shape.y;
    for (int idx = thread_id; idx < num_elements; idx += num_threads) {
      int dmem_row_idx = matrix_offset.row() + idx / smem_num_column;
      int dmem_column_idx = matrix_offset.column() + idx % smem_num_column;
      assert(dmem_column_idx < dmem_num_column);
      smem_ptr[idx] =
          dmem_ptr[dmem_row_idx * dmem_num_column + dmem_column_idx];
      // if (thread_id == 0) {
      //   printf("fp_ptr(%p) smem_offset(%d) idx(%d) blc(%d %d %d) val(%d)
      //   dmem_row_idx(%d) dmem_column_idx(%d) smem_num_column(%d)
      //   dmem_num_column(%d)\n",
      //       dtensor.fp_ptr, (int)stensor.smem_offset, idx, blockIdx.x,
      //       blockIdx.y, blockIdx.z, (int)smem_ptr[idx], dmem_row_idx,
      //       dmem_column_idx, smem_num_column, dmem_num_column);
      // }
    }
  }
};

} // namespace threadblock
} // namespace aso
