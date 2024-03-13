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
#include "cutlass/transform/threadblock/regular_tile_access_iterator_pitch_linear.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op.h"
#include "cutlass/transform/threadblock/regular_tile_access_iterator_tensor_op_sm80.h"

namespace aso {
namespace threadblock {

using namespace cutlass;
using namespace aso::type;

template <typename ElementType,
          int kRow,
          int kColumn,
          int kThreads,
          typename DmemLayout,
          typename SmemLayout>
class RowMajorOutputSaver {
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
  RowMajorOutputSaver(ElementType *dmem_ptr,
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
                      thread_id) {}

  void CUTLASS_DEVICE execute_kernel(void) {
    Fragment tb_fragment;
    // The last kblock is loaded in the prolog
    smem_iterator.load(tb_fragment);
    dmem_iterator.store(tb_fragment);
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
class ColumnMajorOutputSaver {
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
  ColumnMajorOutputSaver(ElementType *dmem_ptr,
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
                      thread_id) {}

  void CUTLASS_DEVICE execute_kernel(void) {
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
class ShapedOutputSaver {
public:
  CUTLASS_DEVICE
  ShapedOutputSaver(char *smem_buffer,
                    aso::kernel::DTensor const &dtensor,
                    aso::threadblock::STensor const &stensor,
                    int thread_id,
                    int num_threads,
                    MatrixCoord matrix_offset,
                    int global_offset) {
    assert(stensor.dim[stensor.num_dims - 2] == kRow);
    assert(stensor.dim[stensor.num_dims - 1] == kColumn);
    // Currently only support half precision
    int const kThreads = 128;
    assert(num_threads == kThreads);
    assert(stensor.data_type == aso::type::DT_FLOAT16);
    assert(dtensor.data_type == aso::type::DT_FLOAT16);
    MatrixCoord extent({dtensor.dim[stensor.num_dims - 2], dtensor.dim[stensor.num_dims - 1]});
    if (dtensor.layout == aso::layout::DmemRowMajor) {
      using DmemLayout = cutlass::layout::RowMajor;
      switch (stensor.layout) {
        case aso::layout::SmemRowMajor: {
          using SmemLayout = cutlass::layout::RowMajor;
          using OutputSaver = RowMajorOutputSaver<cutlass::half_t,
                                                  kRow,
                                                  kColumn,
                                                  kThreads,
                                                  DmemLayout,
                                                  SmemLayout>;
          OutputSaver loader(
              ((cutlass::half_t *)dtensor.data_ptr) + global_offset,
              (cutlass::half_t *)(stensor.smem_offset + smem_buffer),
              extent,
              thread_id,
              matrix_offset);
          loader.execute_kernel();
          break;
        }
        case aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise16:
        case aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise32:
        case aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise64: {
          using SmemLayout = cutlass::layout::
              RowMajorTensorOpMultiplicandCrosswise<16 /*bits*/, kColumn>;
          using OutputSaver = RowMajorOutputSaver<cutlass::half_t,
                                                  kRow,
                                                  kColumn,
                                                  kThreads,
                                                  DmemLayout,
                                                  SmemLayout>;
          OutputSaver loader(
              ((cutlass::half_t *)dtensor.data_ptr) + global_offset,
              (cutlass::half_t *)(stensor.smem_offset + smem_buffer),
              extent,
              thread_id,
              matrix_offset);
          loader.execute_kernel();
          break;
        }
        default: {
          printf("smem layout = %d\n", stensor.layout);
          assert(false && "Unsupported smem layout");
        }
      }
    } else {
      assert(dtensor.layout == aso::layout::DmemColumnMajor);
      using DmemLayout = cutlass::layout::ColumnMajor;
      switch (stensor.layout) {
        case aso::layout::SmemColumnMajor: {
          using SmemLayout = cutlass::layout::ColumnMajor;
          using OutputSaver = ColumnMajorOutputSaver<cutlass::half_t,
                                                     kRow,
                                                     kColumn,
                                                     kThreads,
                                                     DmemLayout,
                                                     SmemLayout>;
          OutputSaver loader(
              ((cutlass::half_t *)dtensor.data_ptr) + global_offset,
              (cutlass::half_t *)(stensor.smem_offset + smem_buffer),
              extent,
              thread_id,
              matrix_offset);
          loader.execute_kernel();
          break;
        }
        case aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16:
        case aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32:
        case aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64: {
          using SmemLayout = cutlass::layout::
              ColumnMajorTensorOpMultiplicandCrosswise<16 /*bits*/, kRow>;
          using OutputSaver = ColumnMajorOutputSaver<cutlass::half_t,
                                                     kRow,
                                                     kColumn,
                                                     kThreads,
                                                     DmemLayout,
                                                     SmemLayout>;
          OutputSaver saver(
              ((cutlass::half_t *)dtensor.data_ptr) + global_offset,
              (cutlass::half_t *)(stensor.smem_offset + smem_buffer),
              extent,
              thread_id,
              matrix_offset);
          saver.execute_kernel();
          break;
        }
        default: {
          printf("smem layout = %d\n", stensor.layout);
          assert(false && "Unsupported smem layout");
        }
      }
    }
  }
};

class GenericOutputSaver {
public:
  CUTLASS_DEVICE
  GenericOutputSaver(char *smem_buffer,
                     aso::kernel::DTensor const &dtensor,
                     aso::threadblock::STensor const &stensor,
                     int thread_id,
                     int num_threads,
                     MatrixCoord matrix_offset,
                     int global_offset) {
    int kRow = stensor.dim[stensor.num_dims - 2];
    int kColumn = stensor.dim[stensor.num_dims - 1];
    if (kRow == 64 && kColumn == 64) {
      ShapedOutputSaver<64, 64>(smem_buffer,
                                dtensor,
                                stensor,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 32 && kColumn == 64) {
      ShapedOutputSaver<32, 64>(smem_buffer,
                                dtensor,
                                stensor,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    } else if (kRow == 64 && kColumn == 32) {
      ShapedOutputSaver<64, 32>(smem_buffer,
                                dtensor,
                                stensor,
                                thread_id,
                                num_threads,
                                matrix_offset,
                                global_offset);
    }
  }
};

class TBOutputAccumFingerprinter {
public:
  CUTLASS_DEVICE
  TBOutputAccumFingerprinter(char *smem_buffer,
                             aso::threadblock::STensor const &input,
                             aso::threadblock::STensor const &output,
                             bool is_first_loop,
                             int thread_id,
                             int num_threads) {
    aso::type::FPType *input_ptr =
        (aso::type::FPType *)(input.smem_offset + smem_buffer);
    aso::type::FPType *output_ptr =
        (aso::type::FPType *)(output.smem_offset + smem_buffer);
    int num_elements = (int)input.num_elements();
    if (is_first_loop) {
      for (int idx = thread_id; idx < num_elements; idx += num_threads) {
        output_ptr[idx] = input_ptr[idx];
      }
      if (thread_id == 0) {
        // printf("Accumu(0): block(%d %d %d) output(%d) input(%d)\n",
        //        blockIdx.x,
        //        blockIdx.y,
        //        blockIdx.z,
        //        output_ptr[thread_id],
        //        input_ptr[thread_id]);
      }
    } else {
      for (int idx = thread_id; idx < num_elements; idx += num_threads) {
        uint32_t value = input_ptr[idx];
        if (thread_id == 0) {
          // printf("Accumu(1): block(%d %d %d) output_old(%d) input(%d) "
          //        "output_new(%d)\n",
          //        blockIdx.x,
          //        blockIdx.y,
          //        blockIdx.z,
          //        output_ptr[thread_id],
          //        input_ptr[thread_id],
          //        (value + output_ptr[idx]) % FP_PQ);
        }
        output_ptr[idx] = (value + output_ptr[idx]) % FP_PQ;
      }
    }
  }
};

class TBOutputSaverFingerprinter {
public:
  CUTLASS_DEVICE
  TBOutputSaverFingerprinter(char *smem_buffer,
                             aso::kernel::DTensor const &dtensor,
                             aso::threadblock::STensor const &stensor,
                             int thread_id,
                             int num_threads,
                             MatrixCoord matrix_offset,
                             int global_offset) {
    aso::type::FPType *smem_ptr =
        (aso::type::FPType *)(stensor.smem_offset + smem_buffer);
    aso::type::FPType *dmem_ptr = dtensor.fp_ptr + global_offset;
    int num_elements = (int)stensor.num_elements();
    int smem_num_column = stensor.dim[stensor.num_dims - 1];
    int dmem_num_column = dtensor.dim[dtensor.num_dims - 1];
    for (int idx = thread_id; idx < num_elements; idx += num_threads) {
      int dmem_row_idx = matrix_offset.row() + idx / smem_num_column;
      int dmem_column_idx = matrix_offset.column() + idx % smem_num_column;
      assert(dmem_column_idx < dmem_num_column);
      if (thread_id == 0) {
        // printf("output:fp_ptr(%p) global_offset(%d) idx(%d) blc(%d %d %d) "
        //        "val(%d) smem_offset(%d)"
        //        "dmem_row_idx(%d) dmem_column_idx(%d) smem_num_column(%d) "
        //        "dmem_num_column(%d)\n",
        //        dtensor.fp_ptr,
        //        global_offset,
        //        idx,
        //        blockIdx.x,
        //        blockIdx.y,
        //        blockIdx.z,
        //        (int)smem_ptr[idx],
        //        stensor.smem_offset,
        //        dmem_row_idx,
        //        dmem_column_idx,
        //        smem_num_column,
        //        dmem_num_column);
      }
      dmem_ptr[dmem_row_idx * dmem_num_column + dmem_column_idx] =
          smem_ptr[idx];
    }
  }
};

} // namespace threadblock
} // namespace aso
