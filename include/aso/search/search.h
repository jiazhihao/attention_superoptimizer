#pragma once

#include <unordered_map>
#include <unordered_set>

#include "aso/search/algebraic_pattern.h"
#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"

namespace aso {
namespace search {

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph);
  void generate_kernel_graphs();

  kernel::Graph computation_graph;
  std::vector<kernel::Graph> kernel_graph_candidates;

private:
  struct SearchContext {
    std::unordered_map<int, std::shared_ptr<AlgebraicPattern>> opid2pattern;
    std::unordered_map<int, size_t> opid2rdeg; // Should extend to per-dimension rdeg later
    std::unordered_set<size_t> existing_op_hash;
    std::unordered_map<int, int> opid2odeg; // Output degree

    std::vector<threadblock::Graph> tb_graph_candidates;
  };

  std::shared_ptr<AlgebraicPattern> final_pattern;
  size_t max_rdeg;

  void generate_threadblock_graphs(SearchContext &c, threadblock::Graph g);
  void generate_next_kernel(SearchContext &c, kernel::Graph g);
};

std::vector<std::shared_ptr<AlgebraicPattern>> pattern_eval(
  kernel::Graph const &g,
  std::unordered_map<int, std::shared_ptr<AlgebraicPattern>> const &input_pattern);

std::vector<std::shared_ptr<AlgebraicPattern>> pattern_eval(
  threadblock::Graph const &g,
  std::unordered_map<int, std::shared_ptr<AlgebraicPattern>> const &input_pattern);

bool verify(kernel::Graph const &g);

} // namespace search
} // namespace aso