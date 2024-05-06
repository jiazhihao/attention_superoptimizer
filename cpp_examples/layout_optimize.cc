#include "mirage/kernel/graph.h"
#include "mirage/search/search.h"
#include "mirage/threadblock/graph.h"

#include <iostream>

using namespace mirage;
using namespace mirage::search;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Missing checkpoint file" << std::endl;
    return 1;
  }

  clock_t st = clock();

  search::KernelGraphGenerator gen(argv[1]);
  gen.config = GeneratorConfig::get_default_config();

  for (json const &j : gen.generated_graphs) {
    kernel::Graph g;
    from_json(j, g);
    gen.optimize_layout(g);
    gen.save_checkpoint();
    while (!g.operators.empty()) {
        delete g.operators.back();
        g.operators.pop_back();
    }
  }

  clock_t et = clock();

  std::cout << "running time: " << (double)(et - st) / CLOCKS_PER_SEC << " sec" << std::endl;

  return 0;
}
