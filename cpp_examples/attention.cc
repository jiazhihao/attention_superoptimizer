#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"

using namespace aso;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input(
        {16, 64, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input(
        {16, 64, 512}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input(
        {16, 512, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    ref_graph.matmul(D, V);
  }

  size_t device_mem_size = (size_t)10 * 1024 * 1024 * 1024; // 10 GB
  size_t shared_mem_size = (size_t)64 * 1024; // 64 KB

  search::KernelGraphGenerator gen(ref_graph, device_mem_size, shared_mem_size);

  gen.generate_kernel_graphs();

  return 0;
}
