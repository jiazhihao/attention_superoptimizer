#pragma once

#include <unordered_map>
#include <unordered_set>

#include "aso/search/config.h"
#include "aso/search/dim_strategy.h"
#include "aso/search/order.h"

namespace aso {
namespace search {

template <typename OpType, typename TensorType>
struct SearchContext {
  std::vector<TensorType> all_tensors;
  std::vector<std::shared_ptr<AlgebraicPattern>> algebraic_pattern;
  std::vector<int> num_consumers;
  std::vector<Order> op_order;
  std::vector<std::shared_ptr<AlgebraicPattern>> output_pattern;
};

class KernelGraphGenerator {
public:
  KernelGraphGenerator(kernel::Graph const &computation_graph,
                       GeneratorConfig const &config);
  void generate_kernel_graphs();

  kernel::Graph computation_graph;
  kernel::Graph best_graph;
  ProfileResult best_profile_result;

  GeneratorConfig config;
  DimStrategy dim_strategy;

private:
  std::vector<std::shared_ptr<AlgebraicPattern>> final_patterns;
  std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>>
      computation_graph_patterns;

  std::vector<DTensor> output_tensors;

  void generate_next_tb_operator(
      SearchContext<TBOperator, STensor> &c,
      threadblock::Graph &g,
      std::function<void()> const &create_customized_then_next_kn);
  void generate_next_kn_operator(SearchContext<KNOperator, DTensor> &c,
                                 kernel::Graph &g);

  void update_best_graph(kernel::Graph &g);

  bool create_tb_outputs(SearchContext<TBOperator, STensor> &c,
                         threadblock::Graph &g,
                         int3 output_map);

  void process_outputs();

  bool check_pattern(std::shared_ptr<AlgebraicPattern> pattern);

  void pattern_eval();
  void fingerprint_eval();
  bool have_same_fingerprint(std::vector<DTensor> const &outputs,
                             std::vector<int> const &match) const;

  bool verify(SearchContext<KNOperator, DTensor> &c, kernel::Graph const &g);

  int random_test_counter;
  int verify_counter;
  int tbgraph_counter;
};

} // namespace search
} // namespace aso