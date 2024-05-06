#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"

#include <iostream>

using namespace mirage;

int main(int argc, char **argv) {
  kernel::Graph graph;
  kernel::DTensor Q =
      graph.new_input({16, 64, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor K =
      graph.new_input({16, 64, 512}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor V =
      graph.new_input({16, 512, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  bool construct_from_plan = true;
  if (construct_from_plan) {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    plan.ops.push_back({mirage::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({mirage::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({mirage::type::TB_REDUCTION_2_TO_DIMX_OP, {{5, 0}}});
    plan.input_map.push_back({0, -1, -1});
    plan.input_map.push_back({0, 2, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor, layout::SmemRowMajor, layout::SmemRowMajor};
    plan.output_map = {0, 2, -1};
    plan.forloop_dim = {1, 2, 1};
    plan.grid_dim = {16, 8, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 4;
    std::vector<kernel::DTensor> outputs = graph.customized({Q, K, V}, plan);
    assert(outputs.size() == 1);
  } else {
    threadblock::Graph tb_graph({16, 8, 1}, {128, 1, 1}, 4);
    threadblock::STensor q =
        tb_graph.new_input(Q, {0, -1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor k =
        tb_graph.new_input(K, {0, 2, -1}, 2, layout::SmemRowMajor);
    threadblock::STensor v =
        tb_graph.new_input(V, {0, 1, -1}, 1, layout::SmemRowMajor);
    threadblock::STensor matmul = tb_graph.matmul(q, k);
    threadblock::STensor exp = tb_graph.exp(matmul);
    threadblock::STensor matmul2 = tb_graph.matmul(exp, v);
    threadblock::STensor red = tb_graph.reduction_to_dimx(matmul2, 2);
    kernel::DTensor output = tb_graph.new_output(red, {0, 2, -1});

    auto op = graph.create_customized_op({Q, K, V}, tb_graph);
    graph.operators.push_back(op);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  return 0;
}
