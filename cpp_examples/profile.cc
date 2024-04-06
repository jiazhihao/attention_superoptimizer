#include "aso/kernel/graph.h"
#include "aso/search/search.h"
#include "aso/threadblock/graph.h"

#include <iostream>
#include <fstream>

using namespace aso;
using namespace aso::search;

int main(int argc, char **argv) {  
  if (argc < 2) {
    std::cerr << "Miss graph file name" << std::endl;
    return 1;
  }

  kernel::Graph g;
  std::ifstream ifs(argv[1]);
  json j;
  ifs >> j;
  from_json(j, g);

  std::cout << json(g) << std::endl;

  float run_time = 0;
  for (auto op : g.operators) {
    ProfileResult op_result;
    op->profile(op_result);
    run_time += op_result.run_time;
  }

  std::cout << "Profiled running time: " << run_time << std::endl;

  return 0;
}
