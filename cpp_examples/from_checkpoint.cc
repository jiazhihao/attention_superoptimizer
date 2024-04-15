#include "aso/kernel/graph.h"
#include "aso/search/search.h"
#include "aso/threadblock/graph.h"

#include <iostream>

using namespace aso;
using namespace aso::search;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Missing checkpoint file" << std::endl;
    return 1;
  }

  clock_t st = clock();
  // kernel::Graph ref_graph;
  // {
  //   kernel::DTensor Q = ref_graph.new_input(
  //       {16, 64, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  //   kernel::DTensor K = ref_graph.new_input(
  //       {16, 64, 512}, type::DT_FLOAT16, layout::DmemColumnMajor);
  //   kernel::DTensor V = ref_graph.new_input(
  //       {16, 512, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  //   kernel::DTensor A = ref_graph.matmul(Q, K);
  //   kernel::DTensor E = ref_graph.exp(A);
  //   kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
  //   kernel::DTensor D = ref_graph.div(E, S);
  //   ref_graph.matmul(D, V);
  // }

  // search::KernelGraphGenerator gen(ref_graph, GeneratorConfig::get_default_config(), "checkpoint.json");
  search::KernelGraphGenerator gen(argv[1]);

  gen.generate_kernel_graphs();

  clock_t et = clock();

  std::cout << "running time: " << (double)(et - st) / CLOCKS_PER_SEC << " sec" << std::endl;

  return 0;
}
