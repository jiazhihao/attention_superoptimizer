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
#include "aso/threadblock/graph.h"
#include "aso/utils/cuda_helper.h"
#include "aso/warp/cuda/matmul.h"

namespace aso {
namespace kernel {

__global__ void
    customized_kernel_function(aso::threadblock::KernelParams params) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  if (false && threadIdx.x == 0) {
    printf("threadIdx(%d) blockIdx(%d %d %d)\n",
           threadIdx.x,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z);
  }
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    int dmem_input_idx = 0;
    int smem_input_idx = 0, smem_output_idx = 0;
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          int3 input_map = params.input_map[dmem_input_idx];
          int forloop_dim = params.forloop_dim[dmem_input_idx];
          aso::kernel::DTensor dtensor = params.dmem_inputs[dmem_input_idx];
          aso::threadblock::STensor stensor =
              params.smem_outputs[smem_output_idx];
          // assert(dtensor.num_dims == 2);
          // assert(stensor.num_dims == 2);
          int num_dims = stensor.num_dims;
          int3 row_stride = {
              (input_map.x == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1),
              (input_map.y == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1),
              (input_map.z == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1)};
          int3 column_stride = {
              (input_map.x == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1),
              (input_map.y == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1),
              (input_map.z == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1)};
          int tb_offset_row = blockIdx.x * row_stride.x +
                              blockIdx.y * row_stride.y +
                              blockIdx.z * row_stride.z;
          int tb_offset_column = blockIdx.x * column_stride.x +
                                 blockIdx.y * column_stride.y +
                                 blockIdx.z * column_stride.z;
          if (forloop_dim == num_dims - 2) {
            tb_offset_row += i * stensor.dim[num_dims - 2];
          }
          if (forloop_dim == num_dims - 1) {
            tb_offset_column += i * stensor.dim[num_dims - 1];
          }
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          cutlass::MatrixCoord matrix_offset = {tb_offset_row,
                                                tb_offset_column};
          // calculate global offset beyond the last two dimensions
          // global_offset captures offsets caused by partitioning other
          // dimensions such as batch matmul global_offset is directly added to
          // dtensor.data_ptr by the input loader
          int global_offset = 0;
          if (num_dims > 2) {
            int strides[MAX_TENSOR_DIMS];
            strides[num_dims - 3] =
                dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
            for (int j = num_dims - 4; j >= 0; j--) {
              strides[j] = strides[j + 1] * dtensor.dim[j + 1];
            }
            if (input_map.x < num_dims - 2 && input_map.x >= 0) {
              global_offset += blockIdx.x * strides[input_map.x];
            }
            if (input_map.y < num_dims - 2 && input_map.y >= 0) {
              global_offset += blockIdx.y * strides[input_map.y];
            }
            if (input_map.z < num_dims - 2 && input_map.z >= 0) {
              global_offset += blockIdx.z * strides[input_map.z];
            }
            if (forloop_dim < num_dims - 2 && forloop_dim >= 0) {
              global_offset +=
                  i * stensor.dim[forloop_dim] * strides[forloop_dim];
            }
          }
          aso::threadblock::GenericInputLoader loader(smem_buffer,
                                                      dtensor,
                                                      stensor,
                                                      threadIdx.x,
                                                      blockDim.x,
                                                      matrix_offset,
                                                      global_offset);
          __syncthreads();
          dmem_input_idx++;
          break;
        }
        case aso::type::TB_OUTPUT_OP: {
          // Only save outputs after forloop
          // So we do nothing here
          break;
        }
        case aso::type::TB_MATMUL_OP: {
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
          break;
        }
        case aso::type::TB_EXP_OP: {
          aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
          aso::threadblock::STensor output =
              params.smem_outputs[smem_output_idx];
          // Assert inline
          assert(input.smem_offset == output.smem_offset);
          cutlass::half_t *input_ptr =
              (cutlass::half_t *)(smem_buffer + input.smem_offset);
          aso::threadblock::ElementUnaryExecutor<cutlass::half_t> executor(
              input_ptr, params.operator_types[op], input.size());
          executor.execute_kernel();
          break;
        }
        case aso::type::TB_REDUCTION_0_OP:
        case aso::type::TB_REDUCTION_1_OP:
        case aso::type::TB_REDUCTION_2_OP: {
          // aso::threadblock::STensor input =
          // params.smem_inputs[smem_input_idx]; aso::threadblock::STensor
          // output = params.smem_outputs[smem_output_idx];
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
    assert(params.num_dmem_inputs == dmem_input_idx);
  }
  // Save output
  int dmem_output_idx = 0, smem_input_idx = 0;
  for (int op = 0; op < params.num_operators; op++) {
    if (params.operator_types[op] == aso::type::TB_OUTPUT_OP) {
      int3 output_map = params.output_map;
      aso::kernel::DTensor dtensor = params.dmem_outputs[dmem_output_idx];
      aso::threadblock::STensor stensor = params.smem_inputs[smem_input_idx];
      // assert(dtensor.num_dims == 2);
      // assert(stensor.num_dims == 2);
      int num_dims = stensor.num_dims;
      int3 row_stride = {
          output_map.x == num_dims - 2 ? stensor.dim[num_dims - 2] : 0,
          output_map.y == num_dims - 2 ? stensor.dim[num_dims - 2] : 0,
          output_map.z == num_dims - 2 ? stensor.dim[num_dims - 2] : 0};
      int3 column_stride = {
          output_map.x == num_dims - 1 ? stensor.dim[num_dims - 1] : 0,
          output_map.y == num_dims - 1 ? stensor.dim[num_dims - 1] : 0,
          output_map.z == num_dims - 1 ? stensor.dim[num_dims - 1] : 0};
      int tb_offset_row = blockIdx.x * row_stride.x +
                          blockIdx.y * row_stride.y + blockIdx.z * row_stride.z;
      int tb_offset_column = blockIdx.x * column_stride.x +
                             blockIdx.y * column_stride.y +
                             blockIdx.z * column_stride.z;
      // FIXME: use cutlass prologue for loading data into shared memory
      // examples/13_two_tensor_op_fusion/threadblock/
      // b2b_mma_pipelined_smem_accumulator.h prologue iterators
      cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
      // calculate global offset beyond the last two dimensions
      // global_offset captures offsets caused by partitioning other dimensions
      // such as batch matmul
      // global_offset is directly added to dtensor.data_ptr by the input loader
      int global_offset = 0;
      if (num_dims > 2) {
        int strides[MAX_TENSOR_DIMS];
        strides[num_dims - 3] =
            dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
        for (int j = num_dims - 4; j >= 0; j--) {
          strides[j] = strides[j + 1] * dtensor.dim[j + 1];
        }
        if (output_map.x < num_dims - 2 && output_map.x >= 0) {
          global_offset += blockIdx.x * strides[output_map.x];
        }
        if (output_map.y < num_dims - 2 && output_map.y >= 0) {
          global_offset += blockIdx.y * strides[output_map.y];
        }
        if (output_map.z < num_dims - 2 && output_map.z >= 0) {
          global_offset += blockIdx.z * strides[output_map.z];
        }
      }
      aso::threadblock::GenericOutputSaver saver(smem_buffer,
                                                 dtensor,
                                                 stensor,
                                                 threadIdx.x,
                                                 blockDim.x,
                                                 matrix_offset,
                                                 global_offset);
      __syncthreads();
      dmem_output_idx++;
    }
    smem_input_idx += params.operator_num_inputs[op];
  }
  assert(params.num_smem_inputs == smem_input_idx);
  assert(params.num_dmem_outputs == dmem_output_idx);
}

__global__ void
    compute_customizedop_fingerprint(aso::threadblock::KernelParams params,
                                     aso::type::FPType *exp_lookup_table,
                                     aso::type::FPType *div_p_lookup_table,
                                     aso::type::FPType *div_q_lookup_table) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  if (threadIdx.x == 0) {
    printf("threadIdx(%d) blockIdx(%d %d %d) num_operators(%d)\n",
           threadIdx.x,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z,
           params.num_operators);
  }
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    int dmem_input_idx = 0;
    int smem_input_idx = 0, smem_output_idx = 0;
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          int3 input_map = params.input_map[dmem_input_idx];
          int forloop_dim = params.forloop_dim[dmem_input_idx];
          aso::kernel::DTensor dtensor = params.dmem_inputs[dmem_input_idx];
          aso::threadblock::STensor stensor =
              params.smem_outputs[smem_output_idx];
          if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            printf("op(%d) num_ops(%d) smem_output_idx(%d) "
                   "stensor.smem_offset(%d)\n",
                   op,
                   params.num_operators,
                   smem_output_idx,
                   (int)stensor.smem_offset);
          }
          // assert(dtensor.num_dims == 2);
          // assert(stensor.num_dims == 2);
          int num_dims = stensor.num_dims;
          int3 row_stride = {
              (input_map.x == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1),
              (input_map.y == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1),
              (input_map.z == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                  (forloop_dim == num_dims - 2 ? params.forloop_range : 1)};
          int3 column_stride = {
              (input_map.x == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1),
              (input_map.y == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1),
              (input_map.z == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                  (forloop_dim == num_dims - 1 ? params.forloop_range : 1)};
          int tb_offset_row = blockIdx.x * row_stride.x +
                              blockIdx.y * row_stride.y +
                              blockIdx.z * row_stride.z;
          int tb_offset_column = blockIdx.x * column_stride.x +
                                 blockIdx.y * column_stride.y +
                                 blockIdx.z * column_stride.z;
          if (forloop_dim == num_dims - 2) {
            tb_offset_row += i * stensor.dim[num_dims - 2];
          }
          if (forloop_dim == num_dims - 1) {
            tb_offset_column += i * stensor.dim[num_dims - 1];
          }
          // calculate global offset beyond the last two dimensions
          // global_offset captures offsets caused by partitioning other
          // dimensions such as batch matmul global_offset is directly added to
          // dtensor.data_ptr by the input loader
          int global_offset = 0;
          if (num_dims > 2) {
            int strides[MAX_TENSOR_DIMS];
            strides[num_dims - 3] =
                dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
            for (int j = num_dims - 4; j >= 0; j--) {
              strides[j] = strides[j + 1] * dtensor.dim[j + 1];
            }
            if (input_map.x < num_dims - 2 && input_map.x >= 0) {
              global_offset += blockIdx.x * strides[input_map.x];
            }
            if (input_map.y < num_dims - 2 && input_map.y >= 0) {
              global_offset += blockIdx.y * strides[input_map.y];
            }
            if (input_map.z < num_dims - 2 && input_map.z >= 0) {
              global_offset += blockIdx.z * strides[input_map.z];
            }
            if (forloop_dim < num_dims - 2 && forloop_dim >= 0) {
              global_offset +=
                  i * stensor.dim[forloop_dim] * strides[forloop_dim];
            }
          }
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          cutlass::MatrixCoord matrix_offset = {tb_offset_row,
                                                tb_offset_column};
          aso::threadblock::TBInputLoaderFingerprinter fp(smem_buffer,
                                                          dtensor,
                                                          stensor,
                                                          threadIdx.x,
                                                          blockDim.x,
                                                          matrix_offset,
                                                          global_offset);
          __syncthreads();
          dmem_input_idx++;
          break;
        }
        case aso::type::TB_OUTPUT_OP: {
          aso::threadblock::STensor input = params.smem_inputs[smem_input_idx];
          aso::threadblock::STensor output =
              params.smem_outputs[smem_output_idx];
          aso::threadblock::TBOutputAccumFingerprinter fp(
              smem_buffer, input, output, (i == 0), threadIdx.x, blockDim.x);
          __syncthreads();
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
    assert(params.num_dmem_inputs == dmem_input_idx);
  }

  // Save output
  int dmem_output_idx = 0, smem_output_idx = 0;
  for (int op = 0; op < params.num_operators; op++) {
    if (params.operator_types[op] == aso::type::TB_OUTPUT_OP) {
      int3 output_map = params.output_map;
      aso::kernel::DTensor dtensor = params.dmem_outputs[dmem_output_idx];
      aso::threadblock::STensor stensor = params.smem_outputs[smem_output_idx];
      // assert(dtensor.num_dims == 2);
      // assert(stensor.num_dims == 2);
      int num_dims = stensor.num_dims;
      int3 row_stride = {
          output_map.x == num_dims - 2 ? stensor.dim[num_dims - 2] : 0,
          output_map.y == num_dims - 2 ? stensor.dim[num_dims - 2] : 0,
          output_map.z == num_dims - 2 ? stensor.dim[num_dims - 2] : 0};
      int3 column_stride = {
          output_map.x == num_dims - 1 ? stensor.dim[num_dims - 1] : 0,
          output_map.y == num_dims - 1 ? stensor.dim[num_dims - 1] : 0,
          output_map.z == num_dims - 1 ? stensor.dim[num_dims - 1] : 0};
      int tb_offset_row = blockIdx.x * row_stride.x +
                          blockIdx.y * row_stride.y + blockIdx.z * row_stride.z;
      int tb_offset_column = blockIdx.x * column_stride.x +
                             blockIdx.y * column_stride.y +
                             blockIdx.z * column_stride.z;
      // FIXME: use cutlass prologue for loading data into shared memory
      // examples/13_two_tensor_op_fusion/threadblock/
      // b2b_mma_pipelined_smem_accumulator.h prologue iterators
      cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};
      // calculate global offset beyond the last two dimensions
      // global_offset captures offsets caused by partitioning other dimensions
      // such as batch matmul
      // global_offset is directly added to dtensor.data_ptr by the input loader
      int global_offset = 0;
      if (num_dims > 2) {
        int strides[MAX_TENSOR_DIMS];
        strides[num_dims - 3] =
            dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
        for (int j = num_dims - 4; j >= 0; j--) {
          strides[j] = strides[j + 1] * dtensor.dim[j + 1];
        }
        if (output_map.x < num_dims - 2 && output_map.x >= 0) {
          global_offset += blockIdx.x * strides[output_map.x];
        }
        if (output_map.y < num_dims - 2 && output_map.y >= 0) {
          global_offset += blockIdx.y * strides[output_map.y];
        }
        if (output_map.z < num_dims - 2 && output_map.z >= 0) {
          global_offset += blockIdx.z * strides[output_map.z];
        }
      }
      aso::threadblock::TBOutputSaverFingerprinter fp(smem_buffer,
                                                      dtensor,
                                                      stensor,
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
}

void KNCustomizedOp::run() {
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  customized_kernel_function<<<bgraph.grid_dim,
                               bgraph.block_dim,
                               bgraph.smem_offset>>>(params);
}

bool KNCustomizedOp::profile(ProfileResult &result) {
  int max_smem_size = aso::type::MAX_SMEM_SIZE;
  assert(bgraph.smem_offset <= max_smem_size);
  if (bgraph.smem_offset > 64 * 1024) {
    checkCUDA(cudaFuncSetAttribute(customized_kernel_function,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
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
  printf("KNCustomizedOp: runtime(%.8lfms)\n", result.run_time);
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

bool KNCustomizedOp::fingerprint(void) {
  int max_smem_size = aso::type::MAX_SMEM_SIZE;
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  for (int i = 0; i < params.num_smem_outputs; i++) {
    printf("params.smem_outputs[%d].smem_offset = %d\n",
           i,
           params.smem_outputs[i].smem_offset);
  }
  assert(bgraph.smem_offset <= max_smem_size);
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();
  if (bgraph.smem_offset > 64 * 1024) {
    checkCUDA(cudaFuncSetAttribute(compute_customizedop_fingerprint,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   bgraph.smem_offset));
  }
  compute_customizedop_fingerprint<<<bgraph.grid_dim,
                                     bgraph.block_dim,
                                     bgraph.smem_offset>>>(
      params,
      dmm->exp_lookup_table,
      dmm->div_p_lookup_table,
      dmm->div_q_lookup_table);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace aso
