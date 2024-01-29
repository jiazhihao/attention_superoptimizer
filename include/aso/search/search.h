#pragma once

#include <unordered_map>
#include <unordered_set>

#include "aso/kernel/graph.h"
#include "aso/search/algebraic_pattern.h"
#include "aso/threadblock/graph.h"

namespace aso {
namespace search {

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph);
  void generate_kernel_graphs();

  kernel::Graph computation_graph;
  std::vector<kernel::Graph> kernel_graph_candidates;

private:
  template <typename OpType, typename TensorType>
  struct SearchContext {
    std::unordered_map<TensorType, std::shared_ptr<AlgebraicPattern>>
        algebraic_pattern;
    std::unordered_map<TensorType, size_t>
        rdeg; // Should extend to per-dimension rdeg later
    std::unordered_set<size_t> existing_op_hash;
    std::unordered_map<OpType *, int> output_degree; // Output degree
  };

  std::shared_ptr<AlgebraicPattern> final_pattern;
  size_t max_rdeg;

  void generate_threadblock_graphs(
      SearchContext<TBOperator, STensor> &c,
      threadblock::Graph g,
      std::vector<std::shared_ptr<AlgebraicPattern>> output_patterns,
      std::vector<int> output_rdegs,
      std::vector<threadblock::Graph> &result_graphs,
      std::vector<std::vector<std::shared_ptr<AlgebraicPattern>>>
          &result_output_patterns,
      std::vector<std::vector<int>> &result_output_rdegs);
  void generate_next_kernel(SearchContext<KNOperator, DTensor> &c,
                            kernel::Graph g);

  bool is_finished_graph(SearchContext<TBOperator, STensor> &c,
                         threadblock::Graph const &g);
};

std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> pattern_eval(
    kernel::Graph const &g,
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> const
        &input_pattern);

bool verify(kernel::Graph const &g);

} // namespace search
} // namespace aso