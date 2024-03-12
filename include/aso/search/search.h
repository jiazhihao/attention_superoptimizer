#pragma once

#include <unordered_map>
#include <unordered_set>

#include "aso/kernel/graph.h"
#include "aso/search/algebraic_pattern.h"
#include "aso/threadblock/graph.h"

namespace aso {
namespace search {

int const MAX_NUM_THREADBLOCK_GRAPH_OP = 7; // Outputs not counted
int const MAX_NUM_KERNEL_GRAPH_OP = 5;
int const MAX_NUM_THREADBLOCK = 2;
int const MAX_NUM_THREADBLOCK_INPUT = 3;
int const MAX_NUM_THREADBLOCK_OUTPUT = 2;

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

struct Order {
  std::vector<int> v;
  int type;
  Order(std::vector<int> const &v, int type);
  bool operator<(Order const &) const;
  bool operator<=(Order const &) const;
};

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph);
  void generate_kernel_graphs();

  kernel::Graph computation_graph;

public:
  template <typename OpType, typename TensorType>
  struct SearchContext {
    std::vector<TensorType> all_tensors;
    std::vector<std::shared_ptr<AlgebraicPattern>> algebraic_pattern;
    std::vector<int> num_consumers;

    // std::unordered_set<size_t> existing_op_hash;
    std::vector<Order> op_order;

    std::vector<std::shared_ptr<AlgebraicPattern>> output_pattern;
  };

  std::vector<std::shared_ptr<AlgebraicPattern>> final_patterns;
  std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>>
      computation_graph_patterns;

  std::vector<DTensor> output_tensors;

  int num_kernels;

  void generate_next_tb_operator(SearchContext<TBOperator, STensor> &c,
                                 threadblock::Graph &g,
                                 std::function<void()> const &callback);
  void generate_next_kn_operator(SearchContext<KNOperator, DTensor> &c,
                                 kernel::Graph &g);

  bool finish_tb_graph(SearchContext<TBOperator, STensor> &c,
                       threadblock::Graph &g,
                       int3 output_map);

  void process_outputs();

  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);

  void pattern_eval();
  void fingerprint_eval();
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;

  bool verify(SearchContext<KNOperator, DTensor> &c, kernel::Graph const &g);
};

} // namespace search
} // namespace aso