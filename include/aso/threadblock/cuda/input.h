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

// TODO: currently we fix the data layout of the tensor in device and shared
// memory these data layout should be provided as a typename instead
template <typename ElementType, int kRow, int kColumn, int kThreads>
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
  using DmemLayout = layout::RowMajor;

  /// Shared memory layout
  using SmemLayout = layout::RowMajorTensorOpMultiplicandCrosswise<
      sizeof_bits<ElementType>::value,
      kColumn>;

  /// ThreadMap of iterator A
  using IteratorThreadMap = transform::PitchLinearWarpRakedThreadMap<
      layout::PitchLinearShape<kColumn, kRow>,
      kThreads,
      layout::PitchLinearShape<kWarpThreadArrangementContiguousA,
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
  using SmemIterator = transform::threadblock::RegularTileAccessIterator<
      MatrixShape<kRow, kColumn>,
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
                      MatrixCoord threadblock_offset)
      : dmem_iterator(DmemLayout::packed(extent),
                      dmem_ptr,
                      extent,
                      thread_id,
                      threadblock_offset),
        smem_iterator({smem_ptr, SmemLayout::packed({kRow, kColumn})},
                      thread_id) {}

  void CUTLASS_DEVICE execute_kernel(void) {
    Fragment tb_fragment;
    // The last kblock is loaded in the prolog
    dmem_iterator.load(tb_fragment);
    // smem_iterator.store(tb_fragment);
  }

public:
  DmemIterator dmem_iterator;
  SmemIterator smem_iterator;
};

class GenericInputLoader {
public:
  CUTLASS_DEVICE
  GenericInputLoader(char *smem_buffer,
                     aso::kernel::DTensor const &dtensor,
                     aso::threadblock::STensor const &stensor,
                     int thread_id,
                     int num_threads,
                     MatrixCoord threadblock_offset) {
    // Currently only support half precision
    assert(stensor.data_type == aso::type::DT_FLOAT16);
    assert(dtensor.data_type == aso::type::DT_FLOAT16);
    // Currently assume the dtensor must in row major
    assert(dtensor.stride[1] == 1);
    assert(dtensor.stride[0] == dtensor.dim[1]);
    MatrixCoord extent({dtensor.dim[0], dtensor.dim[1]});
    if (stensor.is_row_major()) {
      if (stensor.dim[0] == 64 && stensor.dim[1] == 64) {
        int const kRow = 64;
        int const kColumn = 64;
        int const kThreads = 128;
        assert(num_threads == kThreads);
        using InputLoader =
            RowMajorInputLoader<cutlass::half_t, kRow, kColumn, kThreads>;
        InputLoader loader(
            (cutlass::half_t *)dtensor.data_ptr,
            (cutlass::half_t *)(stensor.smem_offset + smem_buffer),
            extent,
            thread_id,
            threadblock_offset);
        loader.execute_kernel();
      }
    } else if (stensor.is_column_major()) {
      assert(false && "To be implemented");
    }
  }
};

} // namespace threadblock
} // namespace aso
