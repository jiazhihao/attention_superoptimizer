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
#include "aso/threadblock/cuda/element_unary.h"
#include "aso/threadblock/cuda/input.h"
#include "aso/threadblock/cuda/matmul.h"
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
  if (threadIdx.x == 0) {
    printf("threadIdx(%d) blockIdx(%d %d %d)\n",
           threadIdx.x,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z);
  }
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    int dmem_input_idx = 0, dmem_output_idx = 0;
    int smem_input_idx = 0, smem_output_idx = 0;
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          cutlass::MatrixCoord threadblock_offset = {0, 0};
          aso::threadblock::GenericInputLoader loader(
              smem_buffer,
              params.dmem_inputs[dmem_input_idx++],
              params.smem_outputs[smem_output_idx],
              threadIdx.x,
              blockDim.x,
              threadblock_offset);
          break;
        }
        case aso::type::TB_OUTPUT_OP: {
          dmem_output_idx++;
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
    assert(params.num_dmem_outputs == dmem_output_idx);
  }
}

__global__ void
    compute_customizedop_fingerprint(aso::threadblock::KernelParams params,
                                     aso::type::FPType *exp_lookup_table) {
  // since we are using cutlass, we group all threads within a threadblock
  // as a 1-D list of threads, therefore blockDim.y and blockDim.z must be
  // 1
  assert(blockDim.y == 1);
  assert(blockDim.z == 1);
  if (threadIdx.x == 0) {
    printf("threadIdx(%d) blockIdx(%d %d %d)\n",
           threadIdx.x,
           blockIdx.x,
           blockIdx.y,
           blockIdx.z);
  }
  extern __shared__ char smem_buffer[];
  for (int i = 0; i < params.forloop_range; i++) {
    int dmem_input_idx = 0, dmem_output_idx = 0;
    int smem_input_idx = 0, smem_output_idx = 0;
    // start executing operators
    for (int op = 0; op < params.num_operators; op++) {
      switch (params.operator_types[op]) {
        case aso::type::TB_INPUT_OP: {
          // FIXME: use cutlass prologue for loading data into shared memory
          // examples/13_two_tensor_op_fusion/threadblock/
          // b2b_mma_pipelined_smem_accumulator.h prologue iterators
          cutlass::MatrixCoord threadblock_offset = {0, 0};
          aso::threadblock::GenericInputLoader loader(
              smem_buffer,
              params.dmem_inputs[dmem_input_idx++],
              params.smem_outputs[smem_output_idx],
              threadIdx.x,
              blockDim.x,
              threadblock_offset);
          break;
        }
        case aso::type::TB_OUTPUT_OP: {
          dmem_output_idx++;
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
    assert(params.num_dmem_outputs == dmem_output_idx);
  }
}

void KNCustomizedOp::run() {
  int smem_size = 48 * 1024; // 48 KB
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  assert(bgraph.smem_offset <= smem_size);
  customized_kernel_function<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      params);
}

bool KNCustomizedOp::profile(ProfileResult &result) {
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
  checkCUDA(cudaEventDestroy(events[0]));
  checkCUDA(cudaEventDestroy(events[1]));
  return true;
}

bool KNCustomizedOp::fingerprint(void) {
  int smem_size = 48 * 1024; // 48 KB
  aso::threadblock::KernelParams params = bgraph.get_kernel_params();
  assert(bgraph.smem_offset <= smem_size);
  aso::kernel::DeviceMemoryManager *dmm =
      aso::kernel::DeviceMemoryManager::get_instance();

  compute_customizedop_fingerprint<<<bgraph.grid_dim,
                                     bgraph.block_dim,
                                     smem_size>>>(params,
                                                  dmm->exp_lookup_table);
  checkCUDA(cudaDeviceSynchronize());
  return true;
}

} // namespace kernel
} // namespace aso
