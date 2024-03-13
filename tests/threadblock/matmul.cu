#include "aso/kernel/graph.h"
#include "aso/threadblock/cuda/matmul.h"
#include "aso/threadblock/graph.h"

#include <fstream>
#include <iostream>

#include <gtest/gtest.h>

#include "common.h"

using namespace aso::threadblock;

__global__ void launch_matmul_kernel(STensor A,
                                     cutlass::half_t *A_data,
                                     STensor B,
                                     cutlass::half_t *B_data,
                                     STensor C,
                                     cutlass::half_t *C_data) {
  extern __shared__ char smem_buffer[];

  copy_global_to_shared(smem_buffer, A, A_data);
  copy_global_to_shared(smem_buffer, B, B_data);
  copy_global_to_shared(smem_buffer, C, C_data);

  GenericMatmulExecutor executor(smem_buffer,
                                 A,
                                 B,
                                 C,
                                 threadIdx.x,
                                 cutlass::canonical_warp_idx_sync(),
                                 cutlass::canonical_lane_idx());

  copy_shared_to_global(smem_buffer, C, C_data);
}

TEST(threadblock_tests, matmul) {
  Graph bgraph = create_single_threadblock_graph(128);

  constexpr int m = 64, n = 64, k = 64;

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::RowMajor> A(
      cutlass::MatrixCoord(m, k)),
      C_ours(cutlass::MatrixCoord(m, n)), C_ref(cutlass::MatrixCoord(m, n));
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor> B(
      cutlass::MatrixCoord(k, n));

  random_fill_tensor(A, 'A');
  random_fill_tensor(B, 'B');
  // random_fill_tensor(C_ours, 'C');
  zero_fill_tensor(C_ours);

  // Copy C_ours into C_ref because we do accumulation.
  cutlass::device_memory::copy_device_to_device(
      C_ref.device_data(), C_ours.device_data(), C_ours.capacity());
  C_ref.sync_host();

  STensor A_s = allocate_stensor(bgraph, A), B_s = allocate_stensor(bgraph, B),
          C_s = allocate_stensor(bgraph, C_ours);

  int smem_size = 48 * 1024; // 48 KB
  launch_matmul_kernel<<<bgraph.grid_dim, bgraph.block_dim, smem_size>>>(
      A_s, A.device_data(), B_s, B.device_data(), C_s, C_ours.device_data());

  cudaDeviceSynchronize();

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
                                 cutlass::half_t>
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
