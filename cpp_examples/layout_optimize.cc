#include "aso/kernel/graph.h"
#include "aso/search/search.h"
#include "aso/threadblock/graph.h"

#include <iostream>

using namespace aso;
using namespace aso::search;

bool has_no_small_tensor(kernel::Graph const &g) {
  for (auto const &op : g.operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      for (auto const &bop : static_cast<kernel::KNCustomizedOp *>(op)->bgraph.operators) {
        if (bop->op_type == type::TB_OUTPUT_OP) {
          for (auto const &output : bop->output_tensors) {
            for (int d = output.num_dims - 2; d < output.num_dims; ++d) {
              if (output.dim[d] < 8) {
                std::cerr << output.dim[0] << ", " << output.dim[1] << ", " << output.dim[2] << std::endl;
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
}

bool filter(kernel::Graph const &g) {
  for (auto const &op : g.operators) {
    if (op->op_type == type::KN_CUSTOMIZED_OP) {
      return false;
    }
  }
  return true;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Missing checkpoint file" << std::endl;
    return 1;
  }

  clock_t st = clock();

  std::unordered_set<int> index_to_skip;
  for (int i = 2; i < argc; ++i) {
    index_to_skip.insert(std::atoi(argv[i]));
  }

  search::KernelGraphGenerator gen(argv[1]);

  int index = 0;
  for (json const &j : gen.generated_graphs) {
    std::cout << "optimizing " << j << std::endl;
    if (index_to_skip.find(index) == index_to_skip.end()) {
      kernel::Graph g;
      from_json(j, g);
      gen.optimize_layout(g);
      gen.save_checkpoint();
      while (!g.operators.empty()) {
          delete g.operators.back();
          g.operators.pop_back();
      }
    }
    std::cout << "finished graph" << (index++) << std::endl;
  }

  clock_t et = clock();

  std::cout << "running time: " << (double)(et - st) / CLOCKS_PER_SEC << " sec" << std::endl;

  return 0;
}
