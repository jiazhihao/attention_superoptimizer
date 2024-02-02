#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"

using namespace aso;

int main(int argc, char **argv) {
  kernel::Graph graph;
  kernel::DTensor Q = graph.new_input({64, 4096}, aso::type::DT_FLOAT16);
  kernel::DTensor K = graph.new_input({16384, 4096}, aso::type::DT_FLOAT16);
  kernel::DTensor V = graph.new_input({16384, 4096}, aso::type::DT_FLOAT16);

  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    plan.ops.push_back({aso::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({aso::type::TB_REDUCTION_1_OP, {{4, 0}}});
    plan.input_map.push_back({1, -1, -1});
    plan.input_map.push_back({1, 0, -1});
    plan.input_map.push_back({1, 0, -1});
    plan.output_map = {1, 0, -1};
    plan.forloop_dim = {-1, 0, 0};
    plan.grid_dim = {64, 16, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 16;
    graph.customized({Q, K, V}, plan);
  }
  ProfileResult result;
  graph.operators.back()->profile(result);
  return 0;
}
