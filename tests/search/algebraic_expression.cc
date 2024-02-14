#include <gtest/gtest.h>

#include "aso/kernel/graph.h"
#include "aso/search/search.h"
#include "aso/threadblock/graph.h"

using namespace aso;
using namespace search;

TEST(algebraic_expression, basic) {
  kernel::Graph graph;
  kernel::DTensor Q = graph.new_input({64, 4096}, aso::type::DT_FLOAT16);
  kernel::DTensor K = graph.new_input({4096, 16384}, aso::type::DT_FLOAT16);

  std::shared_ptr<AlgebraicPattern> Q_pattern = std::make_shared<Var>("q");
  std::shared_ptr<AlgebraicPattern> K_pattern = std::make_shared<Var>("k");

  kernel::DTensor matmul = graph.matmul(Q, K);

  {
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>>
        input_expression_map{{Q, Q_pattern}, {K, K_pattern}};
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> results =
        pattern_eval(graph, input_expression_map);

    std::shared_ptr<AlgebraicPattern> output_pattern = results.at(matmul);
    std::shared_ptr<AlgebraicPattern> target_pattern =
        std::make_shared<Mul>(K_pattern, Q_pattern);

    EXPECT_TRUE(output_pattern->subpattern_to(*target_pattern));
    EXPECT_TRUE(target_pattern->subpattern_to(*output_pattern));
  }
}