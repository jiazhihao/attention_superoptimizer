#include "aso/kernel/graph.h"
#include "aso/threadblock/cuda/input_loader.h"
#include "aso/threadblock/cuda/matmul.h"
#include "aso/threadblock/cuda/output_saver.h"
#include "aso/threadblock/graph.h"

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include "common.h"

using namespace aso::threadblock;
using namespace aso::kernel;

__global__ void launch_matmul_kernel(
    DTensor D_A, STensor A, DTensor D_B, STensor B, DTensor D_C, STensor C) {
  extern __shared__ char smem_buffer[];

  // copy_global_to_shared(smem_buffer, A, A_data);
  // copy_global_to_shared(smem_buffer, B, B_data);
  // copy_global_to_shared(smem_buffer, C, C_data);

  //   printf("A smem offset %d\n", A.smem_offset);
  //   printf("B smem offset %d\n", B.smem_offset);
  //   printf("C smem offset %d\n", C.smem_offset);

  int tb_offset_row = 0;
  int tb_offset_column = 0;
  int global_offset = blockIdx.x * (64 * 64);

  cutlass::MatrixCoord matrix_offset = {tb_offset_row, tb_offset_column};

  // load A & B
  aso::threadblock::GenericInputLoader loader_A(smem_buffer,
                                                D_A,
                                                A,
                                                threadIdx.x,
                                                blockDim.x,
                                                matrix_offset,
                                                global_offset);
  aso::threadblock::GenericInputLoader loader_B(smem_buffer,
                                                D_B,
                                                B,
                                                threadIdx.x,
                                                blockDim.x,
                                                matrix_offset,
                                                global_offset);
  __syncthreads();

  GenericMatmulExecutor executor(smem_buffer,
                                 A,
                                 B,
                                 C,
                                 threadIdx.x,
                                 cutlass::canonical_warp_idx_sync(),
                                 cutlass::canonical_lane_idx());

  // output C
  aso::threadblock::GenericOutputSaver saver(smem_buffer,
                                             D_C,
                                             C,
                                             threadIdx.x,
                                             blockDim.x,
                                             matrix_offset,
                                             global_offset);
  __syncthreads();
}

TEST(threadblock_tests, matmul) {

  aso::kernel::Graph kgraph;
  aso::threadblock::Graph bgraph = create_single_threadblock_graph(128);

  constexpr int m = 64, n = 64, k = 64;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A(
      cutlass::MatrixCoord(m, k)),
      C_ours(cutlass::MatrixCoord(m, n)), C_ref(cutlass::MatrixCoord(m, n));
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(
      cutlass::MatrixCoord(k, n));
  random_fill_tensor(A, 'A');
  random_fill_tensor(B, 'B');
  zero_fill_tensor(C_ours);

  // Copy C_ours into C_ref because we do accumulation.
  cutlass::device_memory::copy_device_to_device(
      C_ref.device_data(), C_ours.device_data(), C_ours.capacity());
  C_ref.sync_host();

  aso::kernel::DTensor D_A = kgraph.new_input(
      {m, k}, aso::type::DT_FLOAT16, aso::layout::DmemLayout::DmemRowMajor);
  aso::kernel::DTensor D_B = kgraph.new_input(
      {k, n}, aso::type::DT_FLOAT16, aso::layout::DmemLayout::DmemColumnMajor);
  aso::kernel::DTensor D_C_ours = kgraph.new_input(
      {m, n}, aso::type::DT_FLOAT16, aso::layout::DmemLayout::DmemRowMajor);

  // copy inputs
  cutlass::device_memory::copy_device_to_device(
      static_cast<cutlass::half_t *>(D_A.data_ptr),
      A.device_data(),
      A.capacity());
  cutlass::device_memory::copy_device_to_device(
      static_cast<cutlass::half_t *>(D_B.data_ptr),
      B.device_data(),
      B.capacity());

  aso::threadblock::STensor A_s = bgraph.new_input(
      D_A,
      {0, -1, -1},
      -1,
      aso::layout::SmemRowMajorTensorOpMultiplicand_Crosswise64);
  aso::threadblock::STensor B_s = bgraph.new_input(
      D_B,
      {0, -1, -1},
      -1,
      aso::layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64);
  aso::threadblock::STensor C_s =
      bgraph.new_input(D_C_ours, {0, -1, -1}, -1, aso::layout::SmemRowMajor);

  int smem_size = 48 * 1024; // 48 KB
  launch_matmul_kernel<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      D_A, A_s, D_B, B_s, D_C_ours, C_s);

  cutlass::device_memory::copy_device_to_device(
      C_ours.device_data(),
      static_cast<cutlass::half_t *>(D_C_ours.data_ptr),
      C_ours.capacity());

  A.sync_host();
  B.sync_host();
  C_ours.sync_host();

  cutlass::reference::host::Gemm<cutlass::half_t,
                                 cutlass::layout::RowMajor,
                                 cutlass::half_t,
                                 cutlass::layout::ColumnMajor,
                                 cutlass::half_t,
                                 cutlass::layout::RowMajor,
                                 cutlass::half_t,
                                 float>
      gemm_ref;
  gemm_ref(
      {m, n, k}, 1.0_hf, A.host_ref(), B.host_ref(), 1.0_hf, C_ref.host_ref());

  if (!cutlass::reference::host::TensorRelativelyEquals(
          C_ref.host_view(), C_ours.host_view(), 0.2_hf, 0.1_hf)) {
    char const *filename = "errors_matmul.csv";
    std::cerr << "Error - Our kernel differs from reference. Wrote computed "
                 "and reference results to '"
              << filename << "'" << std::endl;
    std::ofstream file(filename);
    // file << "\n\nA =\n" << A.host_view() << std::endl;
    // file << "\n\nB =\n" << B.host_view() << std::endl;
    file << "\n\nOurs =\n" << C_ours.host_view() << std::endl;
    file << "\n\nReference =\n" << C_ref.host_view() << std::endl;
    file.close();
    ASSERT_TRUE(false && "Our kernel differs from reference");
  }
}

int wtf(int argc, char **argv) {
  using namespace aso;
  kernel::Graph graph;
  kernel::DTensor A = graph.new_input({4096, 1024},
                                      aso::type::DT_FLOAT16,
                                      aso::layout::DmemLayout::DmemRowMajor);
  kernel::DTensor B = graph.new_input({1024, 4096},
                                      aso::type::DT_FLOAT16,
                                      aso::layout::DmemLayout::DmemColumnMajor);

  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    plan.input_map.push_back({0, -1, -1});
    plan.input_map.push_back({-1, 1, -1});
    plan.output_map = {0, 1, -1};
    plan.forloop_dim = {1, 0};
    plan.grid_dim = {64, 64, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 16;
    graph.customized({A, B}, plan);
  }
  ProfileResult result;
  graph.operators.back()->profile(result);
  std::cout << "Execution time: " << result.run_time << "ms" << std::endl;
  return 0;
}
