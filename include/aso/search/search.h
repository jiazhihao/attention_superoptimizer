#pragma once

#include <unordered_map>
#include <unordered_set>

#include "aso/kernel/graph.h"
#include "aso/search/algebraic_pattern.h"
#include "aso/threadblock/graph.h"

namespace aso {
namespace search {

int const MAX_NUM_THREADBLOCK_GRAPH_OP = 9;
int const MAX_NUM_KERNEL_GRAPH_OP = 6;
int const MAX_NUM_THREADBLOCK = 2;
int const MAX_NUM_THREADBLOCK_OUTPUT = 3;

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

struct Order {
  std::vector<int> v;
  bool operator<(Order const &) const;
};

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       size_t device_mem_size,
                       size_t shared_mem_size);
  void generate_kernel_graphs();

  kernel::Graph computation_graph;
  size_t device_mem_size, shared_mem_size;

private:
  template <typename OpType, typename TensorType>
  struct SearchContext {
    std::vector<TensorType> all_tensors;
    std::vector<std::shared_ptr<AlgebraicPattern>> algebraic_pattern;
    std::vector<int> num_consumers;

    std::unordered_set<size_t> existing_op_hash;
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
                       threadblock::Graph &g);

  void process_outputs();

  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);

  void pattern_eval();
  void fingerprint_eval();
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;

  bool verify(kernel::Graph const &g, SearchContext<KNOperator, DTensor> &c);
};

} // namespace search
} // namespace aso