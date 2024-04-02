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
#include "aso/kernel/device_memory_manager.h"
#include "aso/threadblock/cuda/element_binary.h"
#include "aso/threadblock/cuda/element_unary.h"
#include "aso/threadblock/cuda/input_loader.h"
#include "aso/threadblock/cuda/matmul.h"
#include "aso/threadblock/cuda/output_saver.h"
#include "aso/threadblock/cuda/reduction.h"
#include "aso/threadblock/serializer/input_loader_serializer.h"
#include "aso/threadblock/serializer/output_saver_serializer.h"
#include "aso/threadblock/graph.h"
#include "aso/utils/cuda_helper.h"
#include "aso/warp/cuda/matmul.h"

namespace aso {
namespace kernel {

__global__ void
    customized_kernel_function(aso::threadblock::KernelParams const params,
                               aso::threadblock::NewKernelParams const new_params,
                               int forloop_range) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  if (blockDim.y > 1 || blockDim.z > 1) {
    assert(false && "blockDim.y and blockDim.z must be 1");
  }

  extern __shared__ char smem_buffer[];

  int param_idx = 0;
  for (int i = 0; i < forloop_range; i++) {
    int smem_input_idx = 0, smem_output_idx = 0;
    // start executing operators
    param_idx = 0;
    for (int op = 0; op < new_params.num_operators; op++) {
      aso::type::TBOperatorType op_type = new_params.operator_types[op];
      if (op_type == aso::type::TB_INPUT_OP) {
        //aso::kernel::DTensor dtensor = params.dmem_inputs[op];
        // Assume that InputLoaders are the first operators
        void* dtensor_ptr = new_params.dmem_input_ptrs[op];
        //aso::threadblock::STensor stensor =
        //    params.smem_outputs[smem_output_idx];
        int3 input_matrix_row_offset_block_stride;
        int3 input_matrix_column_offset_block_stride;
        int input_matrix_row_offset_forloop_stride;
        int input_matrix_column_offset_forloop_stride;
        int3 global_offset_block_stride;
        int global_offset_forloop_stride;
        int2 dtensor_matrix_shape, stensor_matrix_shape;
        int input_smem_offset;
        aso::layout::DmemLayout dtensor_layout;
        aso::layout::SmemLayout stensor_layout;
        aso::threadblock::deserialize_input_loader_parameters(
            new_params.parameters,
            param_idx,
            input_matrix_row_offset_block_stride,
            input_matrix_column_offset_block_stride,
            input_matrix_row_offset_forloop_stride,
            input_matrix_column_offset_forloop_stride,
            global_offset_block_stride,
            global_offset_forloop_stride,
            dtensor_matrix_shape,
            stensor_matrix_shape,
            dtensor_layout,
            stensor_layout,
            input_smem_offset);

        int tb_offset_row = blockIdx.x * input_matrix_row_offset_block_stride.x
                          + blockIdx.y * input_matrix_row_offset_block_stride.y
                          + blockIdx.z * input_matrix_row_offset_block_stride.z
                          + i * input_matrix_row_offset_forloop_stride;
        int tb_offset_column = blockIdx.x * input_matrix_column_offset_block_stride.x
                             + blockIdx.y * input_matrix_column_offset_block_stride.y
                             + blockIdx.z * input_matrix_column_offset_block_stride.z
                             + i * input_matrix_column_offset_forloop_stride;
        int global_offset = blockIdx.x * global_offset_block_stride.x
                          + blockIdx.y * global_offset_block_stride.y
                          + blockIdx.z * global_offset_block_stride.z
                          + i * global_offset_forloop_stride;
        cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
        cutlass::half_t *stensor_ptr = (cutlass::half_t *)(smem_buffer + input_smem_offset);
        aso::threadblock::GenericInputLoader loader(dtensor_ptr,
                                                    stensor_ptr,
                                                    dtensor_matrix_shape,
                                                    stensor_matrix_shape,
                                                    dtensor_layout,
                                                    stensor_layout,
                                                    threadIdx.x,
                                                    blockDim.x,
                                                    matrix_offset,
                                                    global_offset);
        __syncthreads();
      } else if (op_type == aso::type::TB_OUTPUT_OP) {
        // Only save outputs after forloop
        // So we do nothing for output saver
      } else if (op_type == aso::type::TB_MATMUL_OP) {
        int thread_idx = threadIdx.x;
        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
        int lane_idx = threadIdx.x % 32;
        aso::threadblock::GenericMatmulExecutor executor(
            smem_buffer,
            params.smem_inputs[smem_input_idx],
            params.smem_inputs[smem_input_idx + 1],
            params.smem_outputs[smem_output_idx],
            thread_idx,
            warp_idx,
            lane_idx);
        __syncthreads();
      } else if (op_type == aso::type::TB_EXP_OP) {
        aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
        //aso::threadblock::STensor output = params.smem_outputs[smem_output_idx];
        // Assert inline
        //assert(input.smem_offset == output.smem_offset);
        cutlass::half_t *base_ptr = (cutlass::half_t *)(smem_buffer + input.smem_offset);
        aso::threadblock::ElementUnaryExecutor<cutlass::half_t> executor(
            op_type,
            base_ptr,
            input.num_elements(),
            threadIdx.x,
            blockDim.x);
        __syncthreads();
      } else if (op_type == aso::type::TB_DIV_OP) {
        aso::threadblock::STensor input1 = params.smem_inputs[smem_input_idx];
        aso::threadblock::STensor input2 =
            params.smem_inputs[smem_input_idx + 1];
        aso::threadblock::STensor output = params.smem_outputs[smem_output_idx];
        aso::threadblock::ElementBinaryExecutor<cutlass::half_t> executor(
            op_type,
            smem_buffer,
            input1,
            input2,
            output,
            threadIdx.x,
            blockDim.x);
        __syncthreads();
      } else if ((op_type >= aso::type::TB_REDUCTION_FIRST_OP_ID) &&
                 (op_type <= aso::type::TB_REDUCTION_LAST_OP_ID)) {
        aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
        aso::threadblock::STensor output = params.smem_outputs[smem_output_idx];
        aso::threadblock::SimpleRedunctionExecutor<cutlass::half_t> executor(
            params.operator_types[op],
            smem_buffer,
            input,
            output,
            threadIdx.x,
            blockDim.x);
        __syncthreads();
      } else {
        // Assert the uncaptured operator must be output saver
        // Only save outputs after forloop
        // So we do nothing for output saver
        // if (op_type != aso::type::TB_OUTPUT_OP) {
        //  assert(false && "Unsupported threadblock operator");
        //}
      }
      // increment indices
      smem_input_idx += params.operator_num_inputs[op];
      smem_output_idx += params.operator_num_outputs[op];
    }
    // assert smem/dmem inputs/outputs aligned
    // assert(params.num_smem_inputs == smem_input_idx);
    // assert(params.num_smem_outputs == smem_output_idx);
  }
  // Save output
  int output_saver_start_idx = new_params.num_operators - new_params.num_dmem_outputs;
  for (int op = output_saver_start_idx; op < new_params.num_operators; op++) {
    assert(new_params.operator_types[op] == aso::type::TB_OUTPUT_OP);
    void* dtensor_ptr = new_params.dmem_output_ptrs[op - output_saver_start_idx];
    //aso::threadblock::STensor stensor =
    //    params.smem_outputs[smem_output_idx];
    int3 output_matrix_row_offset_block_stride;
    int3 output_matrix_column_offset_block_stride;
    int3 global_offset_block_stride;
    int2 dtensor_matrix_shape, stensor_matrix_shape;
    int input_smem_offset, accum_smem_offset;
    aso::layout::DmemLayout dtensor_layout;
    aso::layout::SmemLayout stensor_layout;
    aso::threadblock::deserialize_output_saver_parameters(
        new_params.parameters,
        param_idx,
        output_matrix_row_offset_block_stride,
        output_matrix_column_offset_block_stride,
        global_offset_block_stride,
        dtensor_matrix_shape,
        stensor_matrix_shape,
        dtensor_layout,
        stensor_layout,
        input_smem_offset,
        accum_smem_offset);
    int tb_offset_row = blockIdx.x * output_matrix_row_offset_block_stride.x
                      + blockIdx.y * output_matrix_row_offset_block_stride.y
                      + blockIdx.z * output_matrix_row_offset_block_stride.z;
    int tb_offset_column = blockIdx.x * output_matrix_column_offset_block_stride.x
                         + blockIdx.y * output_matrix_column_offset_block_stride.y
                         + blockIdx.z * output_matrix_column_offset_block_stride.z;
    // calculate global offset beyond the last two dimensions
    // global_offset captures offsets caused by partitioning other dimensions
    // such as batch matmul
    // global_offset is directly added to dtensor_ptr by the output saver
    int global_offset = blockIdx.x * global_offset_block_stride.x
                      + blockIdx.y * global_offset_block_stride.y
                      + blockIdx.z * global_offset_block_stride.z;

    // FIXME: use cutlass prologue for loading data into shared memory
    // examples/13_two_tensor_op_fusion/threadblock/
    // b2b_mma_pipelined_smem_accumulator.h prologue iterators
    cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
    cutlass::half_t *stensor_ptr = (cutlass::half_t *)(smem_buffer + accum_smem_offset);
    aso::threadblock::GenericOutputSaver saver(dtensor_ptr,
                                               stensor_ptr,
                                               dtensor_matrix_shape,
                                               stensor_matrix_shape,
                                               dtensor_layout,
                                               stensor_layout,
                                               threadIdx.x,
                                               blockDim.x,
                                               matrix_offset,
                                               global_offset);
    // No need to synchronize for output saver
    //__syncthreads();
  }
  assert(new_params.num_parameters == param_idx);
}

__global__ void
    compute_customizedop_fingerprint(aso::threadblock::KernelParams params,
                                     aso::threadblock::NewKernelParams new_params,
                                     int forloop_range,
                                     aso::type::FPType *exp_lookup_table,
                                     aso::type::FPType *div_p_lookup_table,
                                     aso::type::FPType *div_q_lookup_table) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  extern __shared__ char smem_buffer[];
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);

  int param_idx = 0;
  int output_saver_start_idx = new_params.num_operators - new_params.num_dmem_outputs;
  for (int i = 0; i < forloop_range; i++) {
    int smem_input_idx = 0, smem_output_idx = 0;
    param_idx = 0;
    // start executing operators
    for (int op = 0; op < new_params.num_operators; op++) {
      if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        printf("i(%d) op(%d) op_type(%d)\n", i, op, new_params.operator_types[op]);
      switch (new_params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          //aso::kernel::DTensor dtensor = params.dmem_inputs[op];
          // Assume that InputLoaders are the first operators
          aso::type::FPType* dtensor_ptr = (aso::type::FPType*)new_params.dmem_input_ptrs[op];
          //aso::threadblock::STensor stensor =
          //    params.smem_outputs[smem_output_idx];
          int3 input_matrix_row_offset_block_stride;
          int3 input_matrix_column_offset_block_stride;
          int input_matrix_row_offset_forloop_stride;
          int input_matrix_column_offset_forloop_stride;
          int3 global_offset_block_stride;
          int global_offset_forloop_stride;
          int2 dtensor_matrix_shape, stensor_matrix_shape;
          int input_smem_offset;
          aso::layout::DmemLayout dtensor_layout;
          aso::layout::SmemLayout stensor_layout;
          aso::threadblock::deserialize_input_loader_parameters(
              new_params.parameters,
              param_idx,
              input_matrix_row_offset_block_stride,
              input_matrix_column_offset_block_stride,
              input_matrix_row_offset_forloop_stride,
              input_matrix_column_offset_forloop_stride,
              global_offset_block_stride,
              global_offset_forloop_stride,
              dtensor_matrix_shape,
              stensor_matrix_shape,
              dtensor_layout,
              stensor_layout,
              input_smem_offset);
  
          // Note that input_matrix_offset_forloop_stride's x and y indicates
          // row and column
          int tb_offset_row = blockIdx.x * input_matrix_row_offset_block_stride.x
                            + blockIdx.y * input_matrix_row_offset_block_stride.y
                            + blockIdx.z * input_matrix_row_offset_block_stride.z
                            + i * input_matrix_row_offset_forloop_stride;
          int tb_offset_column = blockIdx.x * input_matrix_column_offset_block_stride.x
                               + blockIdx.y * input_matrix_column_offset_block_stride.y
                               + blockIdx.z * input_matrix_column_offset_block_stride.z
                               + i * input_matrix_column_offset_forloop_stride;
          int global_offset = blockIdx.x * global_offset_block_stride.x
                            + blockIdx.y * global_offset_block_stride.y
                            + blockIdx.z * global_offset_block_stride.z
                            + i * global_offset_forloop_stride;
          cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
          aso::type::FPType *stensor_ptr = (aso::type::FPType*)(smem_buffer + input_smem_offset);
          aso::threadblock::TBInputLoaderFingerprinter fp(dtensor_ptr,
                                                          stensor_ptr,
                                                          dtensor_matrix_shape,
                                                          stensor_matrix_shape,
                                                          dtensor_layout,
                                                          stensor_layout,
                                                          threadIdx.x,
                                                          blockDim.x,
                                                          matrix_offset,
                                                          global_offset);
          __syncthreads();
          break;
        }
        case aso::type::TB_OUTPUT_OP: {
          int3 output_matrix_row_offset_block_stride;
          int3 output_matrix_column_offset_block_stride;
          int3 global_offset_block_stride;
          int2 dtensor_matrix_shape, stensor_matrix_shape;
          int input_smem_offset, accum_smem_offset;
          aso::layout::DmemLayout dtensor_layout;
          aso::layout::SmemLayout stensor_layout;
          aso::threadblock::deserialize_output_saver_parameters(
              new_params.parameters,
              param_idx,
              output_matrix_row_offset_block_stride,
              output_matrix_column_offset_block_stride,
              global_offset_block_stride,
              dtensor_matrix_shape,
              stensor_matrix_shape,
              dtensor_layout,
              stensor_layout,
              input_smem_offset,
              accum_smem_offset);
          aso::type::FPType *input_stensor_ptr = (aso::type::FPType*)(smem_buffer + input_smem_offset);
          aso::type::FPType *accum_stensor_ptr = (aso::type::FPType*)(smem_buffer + accum_smem_offset);
          aso::threadblock::TBOutputAccumFingerprinter fp(
              input_stensor_ptr, accum_stensor_ptr, stensor_matrix_shape, (i == 0), threadIdx.x, blockDim.x);
          __syncthreads();
          // Step 2: Save final output to dmem if this is the last forloop
          if (i == forloop_range - 1) {
            assert(op >= output_saver_start_idx);
            aso::type::FPType* dtensor_ptr = (aso::type::FPType*)new_params.dmem_output_ptrs[op - output_saver_start_idx];
            int tb_offset_row = blockIdx.x * output_matrix_row_offset_block_stride.x
                              + blockIdx.y * output_matrix_row_offset_block_stride.y
                              + blockIdx.z * output_matrix_row_offset_block_stride.z;
            int tb_offset_column = blockIdx.x * output_matrix_column_offset_block_stride.x
                                 + blockIdx.y * output_matrix_column_offset_block_stride.y
                                 + blockIdx.z * output_matrix_column_offset_block_stride.z;
            // calculate global offset beyond the last two dimensions
            // global_offset captures offsets caused by partitioning other dimensions
            // such as batch matmul
            // global_offset is directly added to dtensor_ptr by the output saver
            int global_offset = blockIdx.x * global_offset_block_stride.x
                              + blockIdx.y * global_offset_block_stride.y
                              + blockIdx.z * global_offset_block_stride.z;
            cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
            aso::threadblock::TBOutputSaverFingerprinter fp(dtensor_ptr,
                                                            accum_stensor_ptr,
                                                            dtensor_matrix_shape,
                                                            stensor_matrix_shape,
                                                            dtensor_layout,
                                                            stensor_layout,
                                                            threadIdx.x,
                                                            blockDim.x,
                                                            matrix_offset,
                                                            global_offset);
            // No need to syncthread when saving output to dmem
            //__syncthreads();
          }
          break;
        }
        case aso::type::TB_MATMUL_OP: {
          aso::threadblock::TBMatmulFingerprinter fp(
              smem_buffer,
              params.smem_inputs[smem_input_idx],
              params.smem_inputs[smem_input_idx + 1],
              params.smem_outputs[smem_output_idx],
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case aso::type::TB_EXP_OP: {
          aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
          aso::threadblock::STensor output =
              params.smem_outputs[smem_output_idx];
          // Assert inline
          assert(input.smem_offset == output.smem_offset);
          // cutlass::half_t *input_ptr =
          //     (cutlass::half_t *)(smem_buffer + input.smem_offset);
          aso::threadblock::TBElementUnaryFingerPrinter fp(
              params.operator_types[op],
              exp_lookup_table /*lookup_table*/,
              smem_buffer,
              input,
              output,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case aso::type::TB_DIV_OP: {
          aso::threadblock::STensor input1 = params.smem_inputs[smem_input_idx];
          aso::threadblock::STensor input2 =
              params.smem_inputs[smem_input_idx + 1];
          aso::threadblock::STensor output =
              params.smem_outputs[smem_output_idx];
          aso::threadblock::TBElementBinaryFingerPrinter fp(
              params.operator_types[op],
              div_p_lookup_table /*div_p_lookup*/,
              div_q_lookup_table /*div_q_lookup*/,
              smem_buffer,
              input1,
              input2,
              output,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        case aso::type::TB_REDUCTION_0_OP:
        case aso::type::TB_REDUCTION_1_OP:
        case aso::type::TB_REDUCTION_2_OP:
        case aso::type::TB_REDUCTION_0_TO_DIMX_OP:
        case aso::type::TB_REDUCTION_1_TO_DIMX_OP:
        case aso::type::TB_REDUCTION_2_TO_DIMX_OP: {
          aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
          aso::threadblock::STensor output =
              params.smem_outputs[smem_output_idx];
          aso::threadblock::TBReductionFingerprinter fp(
              params.operator_types[op],
              smem_buffer,
              input,
              output,
              threadIdx.x,
              blockDim.x);
          __syncthreads();
          break;
        }
        default: {
          assert(false && "Unsupported threadblock operator");
        }
      }
      // increment indices
      smem_input_idx += params.operator_num_inputs[op];
      smem_output_idx += params.operator_num_outputs[op];
    }
    // assert smem/dmem inputs/outputs aligned
    assert(params.num_smem_inputs == smem_input_idx);
    assert(params.num_smem_outputs == smem_output_idx);
    assert(new_params.num_parameters == param_idx);
  }

#ifdef DEADCODE
  // Save output
  int dmem_output_idx = 0, smem_output_idx = 0;
  int output_saver_start_idx = new_params.num_operators - new_params.num_dmem_outputs;
  for (int op = 0; op < new_params.num_operators; op++) {
    if (new_params.operator_types[op] == aso::type::TB_OUTPUT_OP) {
      assert(op >= output_saver_start_idx);
      aso::type::FPType* dtensor_ptr = (aso::type::FPType*)new_params.dmem_output_ptrs[op - output_saver_start_idx];
      int3 output_matrix_row_offset_block_stride;
      int3 output_matrix_column_offset_block_stride;
      int3 global_offset_block_stride;
      int2 dtensor_matrix_shape, stensor_matrix_shape;
      int input_smem_offset, accum_smem_offset;
      aso::layout::DmemLayout dtensor_layout;
      aso::layout::SmemLayout stensor_layout;
      aso::threadblock::deserialize_output_saver_parameters(
          new_params.parameters,
          param_idx,
          output_matrix_row_offset_block_stride,
          output_matrix_column_offset_block_stride,
          global_offset_block_stride,
          dtensor_matrix_shape,
          stensor_matrix_shape,
          dtensor_layout,
          stensor_layout,
          input_smem_offset,
          accum_smem_offset);
      int tb_offset_row = blockIdx.x * output_matrix_row_offset_block_stride.x
                        + blockIdx.y * output_matrix_row_offset_block_stride.y
                        + blockIdx.z * output_matrix_row_offset_block_stride.z;
      int tb_offset_column = blockIdx.x * output_matrix_column_offset_block_stride.x
                           + blockIdx.y * output_matrix_column_offset_block_stride.y
                           + blockIdx.z * output_matrix_column_offset_block_stride.z;
      // calculate global offset beyond the last two dimensions
      // global_offset captures offsets caused by partitioning other dimensions
      // such as batch matmul
      // global_offset is directly added to dtensor_ptr by the output saver
      int global_offset = blockIdx.x * global_offset_block_stride.x
                        + blockIdx.y * global_offset_block_stride.y
                        + blockIdx.z * global_offset_block_stride.z;
      cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
      aso::type::FPType *stensor_ptr = (aso::type::FPType*)(smem_buffer + accum_smem_offset);
      // Perform alignment
      {
        int3 output_map = params.output_map;
        aso::kernel::DTensor dtensor2 = params.dmem_outputs[dmem_output_idx];
        aso::threadblock::STensor stensor2 = params.smem_outputs[smem_output_idx];
        // assert(dtensor.num_dims == 2);
        // assert(stensor.num_dims == 2);
        int num_dims = stensor2.num_dims;
        int3 row_stride = {
            output_map.x == num_dims - 2 ? stensor2.dim[num_dims - 2] : 0,
            output_map.y == num_dims - 2 ? stensor2.dim[num_dims - 2] : 0,
            output_map.z == num_dims - 2 ? stensor2.dim[num_dims - 2] : 0};
        int3 column_stride = {
            output_map.x == num_dims - 1 ? stensor2.dim[num_dims - 1] : 0,
            output_map.y == num_dims - 1 ? stensor2.dim[num_dims - 1] : 0,
            output_map.z == num_dims - 1 ? stensor2.dim[num_dims - 1] : 0};
        int tb_offset_row2 = blockIdx.x * row_stride.x +
                            blockIdx.y * row_stride.y + blockIdx.z * row_stride.z;
        int tb_offset_column2 = blockIdx.x * column_stride.x +
                               blockIdx.y * column_stride.y +
                               blockIdx.z * column_stride.z;
        // FIXME: use cutlass prologue for loading data into shared memory
        // examples/13_two_tensor_op_fusion/threadblock/
        // b2b_mma_pipelined_smem_accumulator.h prologue iterators
        cutlass::MatrixCoord matrix_offset2 = {tb_offset_row, tb_offset_column};
        // calculate global offset beyond the last two dimensions
        // global_offset captures offsets caused by partitioning other dimensions
        // such as batch matmul
        // global_offset is directly added to dtensor.data_ptr by the input loader
        int global_offset2 = 0;
        if (num_dims > 2) {
          int strides[MAX_TENSOR_DIMS];
          strides[num_dims - 3] =
              dtensor2.dim[num_dims - 2] * dtensor2.dim[num_dims - 1];
          for (int j = num_dims - 4; j >= 0; j--) {
            strides[j] = strides[j + 1] * dtensor2.dim[j + 1];
          }
          if (output_map.x < num_dims - 2 && output_map.x >= 0) {
            global_offset2 += blockIdx.x * strides[output_map.x];
          }
          if (output_map.y < num_dims - 2 && output_map.y >= 0) {
            global_offset2 += blockIdx.y * strides[output_map.y];
          }
          if (output_map.z < num_dims - 2 && output_map.z >= 0) {
            global_offset2 += blockIdx.z * strides[output_map.z];
          }
        }
        if (tb_offset_row2 != tb_offset_row && threadIdx.x == 0) {
          printf("num_params(%d) param_idx(%d) blk(%d %d %d) output_row_offset(%d %d %d) row_stride(%d %d %d)\n",
              new_params.num_parameters, param_idx,
              blockIdx.x, blockIdx.y, blockIdx.z,
              output_matrix_row_offset_block_stride.x,
              output_matrix_row_offset_block_stride.y,
              output_matrix_row_offset_block_stride.z,
              row_stride.x, row_stride.y, row_stride.z);
        }
        assert(tb_offset_row2 == tb_offset_row);
        assert(tb_offset_column2 == tb_offset_column);
        assert(global_offset2 == global_offset);
        assert(dtensor2.fp_ptr == dtensor_ptr);
        assert(dtensor_layout == dtensor2.layout);
        assert(stensor_layout == stensor2.layout);
        assert(dtensor_matrix_shape.x == dtensor2.dim[dtensor2.num_dims-2]);
        assert(dtensor_matrix_shape.y == dtensor2.dim[dtensor2.num_dims-1]);
        assert(stensor_matrix_shape.x == stensor2.dim[stensor2.num_dims-2]);
        assert(stensor_matrix_shape.y == stensor2.dim[stensor2.num_dims-1]);
        assert(input_smem_offset == stensor2.smem_offset);
      }
      aso::threadblock::TBOutputSaverFingerprinter fp(dtensor_ptr,
                                                      stensor_ptr,
                                                      dtensor_matrix_shape,
                                                      stensor_matrix_shape,
                                                      dtensor_layout,
                                                      stensor_layout,
                                                      threadIdx.x,
                                                      blockDim.x,
                                                      matrix_offset,
                                                      global_offset);
      __syncthreads();
      dmem_output_idx++;
    }
    smem_output_idx += params.operator_num_outputs[op];
  }
  assert(params.num_smem_outputs == smem_output_idx);
  assert(params.num_dmem_outputs == dmem_output_idx);
  assert(new_params.num_parameters == param_idx);
#endif
}

void KNCustomizedOp::run() {
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  aso::threadblock::NewKernelParams new_params = bgraph.get_new_kernel_params(false/*fingerprint_kernel*/);
  customized_kernel_function<<<bgraph.grid_dim,
                               bgraph.block_dim,
                               bgraph.smem_offset>>>(params, new_params, bgraph.forloop_range);
}

bool KNCustomizedOp::profile(ProfileResult &result) {
  printf("stensor size = %zu dtensor size %zu\n", sizeof(aso::threadblock::STensor), sizeof(aso::kernel::DTensor));
  int max_smem_size = aso::type::MAX_SMEM_SIZE;
  assert(bgraph.smem_offset <= max_smem_size);
  if (bgraph.smem_offset > 48 * 1024) {
    checkCUDA(cudaFuncSetAttribute(customized_kernel_function,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
  checkCUDA(cudaDeviceSynchronize());
  cudaEvent_t events[2];
  checkCUDA(cudaEventCreate(&events[0]));
  checkCUDA(cudaEventCreate(&events[1]));
  checkCUDA(cudaEventRecord(events[0]));
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  aso::threadblock::NewKernelParams new_params = bgraph.get_new_kernel_params(false/*fingerprint_kernel*/);
  for (int i = 0; i < ProfileResult::NUM_ITERATIONS; i++) {
    customized_kernel_function<<<bgraph.grid_dim,
                                 bgraph.block_dim,
                                 bgraph.smem_offset>>>(params, new_params, bgraph.forloop_range);
  }
  float runtime_ms = 0;
  checkCUDA(cudaEventRecord(events[1]));
  checkCUDA(cudaEventSynchronize(events[1]));
  checkCUDA(cudaEventElapsedTime(&runtime_ms, events[0], events[1]));
  result.run_time = runtime_ms / ProfileResult::NUM_ITERATIONS;
  printf("KNCustomizedOp: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

bool KNCustomizedOp::fingerprint(void) {
  int max_smem_size = aso::type::MAX_SMEM_SIZE;
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  aso::threadblock::NewKernelParams new_params = bgraph.get_new_kernel_params(true/*fingerprint_kernel*/);
  for (int i = 0; i < params.num_smem_outputs; i++) {
    printf("params.smem_outputs[%d].smem_offset = %d\n",
           i,
           params.smem_outputs[i].smem_offset);
  }
  assert(bgraph.smem_offset <= max_smem_size);
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();
  if (bgraph.smem_offset > 48 * 1024) {
    checkCUDA(cudaFuncSetAttribute(compute_customizedop_fingerprint,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
  compute_customizedop_fingerprint<<<bgraph.grid_dim,
                                     bgraph.block_dim,
                                     bgraph.smem_offset>>>(
      params,
      new_params,
      bgraph.forloop_range,
      dmm->exp_lookup_table,
      dmm->div_p_lookup_table,
      dmm->div_q_lookup_table);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace aso
