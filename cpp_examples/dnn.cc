#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"

using namespace aso;

int main(int argc, char **argv) {
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input(
        {16, 64, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input(
        {16, 64, 512}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input(
        {16, 512, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    ref_graph.matmul(D, V);
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
  }
  kernel::Graph graph;
  kernel::DTensor Q =
      graph.new_input({16, 64, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor K =
      graph.new_input({16, 64, 512}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor V =
      graph.new_input({16, 512, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    // plan.ops.push_back({aso::type::TB_MATMUL_OP, {{3, 0}, {2, 0}}});
    plan.ops.push_back({aso::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({aso::type::TB_REDUCTION_2_OP, {{4, 0}}});
    plan.input_map.push_back({0, -1, -1});
    plan.input_map.push_back({0, 2, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        // layout::SmemRowMajor,
        // layout::SmemColumnMajor,
        // layout::SmemColumnMajor,
        layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
        layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
        layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
    };
    plan.output_map = {0, 2, -1};
    plan.forloop_dim = {-1, 2, 1};
    plan.grid_dim = {16, 2, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 4;
    std::vector<kernel::DTensor> outputs = graph.customized({Q, K, V}, plan);
    assert(outputs.size() == 2);
    kernel::DTensor o1 = graph.reduction(outputs[0], 2 /*dim*/, 2 /*factor*/);
    kernel::DTensor o2 = graph.reduction(outputs[1], 2 /*dim*/, 2 /*factor*/);
    graph.div(o1, o2);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
      graph.operators.back()->output_tensors[0]));
  // ProfileResult result;
  // for (auto const &op : graph.operators) {
  //   op->profile(result);
  // }
  return 0;
}
