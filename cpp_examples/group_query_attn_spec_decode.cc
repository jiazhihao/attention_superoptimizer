#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"
#include "common.h"

using namespace aso;

int main(int argc, char **argv) {
  // Currently only optimize for these two batch sizes
  int batch_size = asotest::BATCH_SIZE;
  assert(batch_size == 1 || batch_size == 8);
  kernel::Graph ref_graph;
  {
    kernel::DTensor Q = ref_graph.new_input(
        {2 * batch_size, 128, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
    kernel::DTensor K = ref_graph.new_input(
        {2 * batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor V = ref_graph.new_input(
        {2 * batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
    kernel::DTensor A = ref_graph.matmul(Q, K);
    kernel::DTensor E = ref_graph.exp(A);
    kernel::DTensor S = ref_graph.reduction(E, 2 /*dim*/);
    kernel::DTensor D = ref_graph.div(E, S);
    ref_graph.matmul(D, V);
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
    ProfileResult result;
    float total_runtime = 0.0f;
    for (auto const &op : ref_graph.operators) {
      op->profile(result);
      total_runtime = total_runtime + result.run_time;
    }
    printf("[cudnn kernel graph] Total runtime = %.4lfms\n", total_runtime);
  }
  kernel::Graph graph;
  kernel::DTensor Q =
      graph.new_input({2 * batch_size, 128, 64}, type::DT_FLOAT16, layout::DmemRowMajor);
  kernel::DTensor K =
      graph.new_input({2 * batch_size, 64, 4096}, type::DT_FLOAT16, layout::DmemColumnMajor);
  kernel::DTensor V =
      graph.new_input({2 * batch_size, 4096, 64}, type::DT_FLOAT16, layout::DmemColumnMajor);
  std::vector<kernel::DTensor> outputs;
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{0, 0}, {1, 0}}});
    // plan.ops.push_back({aso::type::TB_MATMUL_OP, {{3, 0}, {2, 0}}});
    plan.ops.push_back({aso::type::TB_EXP_OP, {{3, 0}}});
    plan.ops.push_back({aso::type::TB_MATMUL_OP, {{4, 0}, {2, 0}}});
    plan.ops.push_back({aso::type::TB_REDUCTION_2_OP, {{4, 0}}});
    plan.input_map.push_back({0, -1, 1});
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
    plan.output_map = {0, 2, 1};
    plan.forloop_dim = {-1, 2, 1};
    if (batch_size == 1) {
      plan.grid_dim = {2, 16, 4};
    } else {
      plan.grid_dim = {16, 8, 2};
    }
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 4;
    plan.reduction_dimx = 64;
    outputs = graph.customized({Q, K, V}, plan);
    assert(outputs.size() == 2);
    // kernel::DTensor o1 = graph.reduction(outputs[0], 2 /*dim*/, 64 /*size*/);
    // kernel::DTensor o2 = graph.reduction(outputs[1], 2 /*dim*/);
    // graph.div(o1, o2);
  }
  {
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_REDUCTION_2_TO_DIMX_OP, {{0, 0}}});
    plan.ops.push_back({aso::type::TB_REDUCTION_2_OP, {{1, 0}}});
    plan.ops.push_back({aso::type::TB_DIV_OP, {{2, 0}, {3, 0}}});
    plan.input_map.push_back({0, 1, -1});
    plan.input_map.push_back({0, 1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor,
        layout::SmemRowMajor,
    };
    plan.output_map = {0, 1, -1};
    plan.forloop_dim = {-1, -1};
    plan.grid_dim = {2, 16, 1};
    if (batch_size == 8) {
      plan.grid_dim = {16, 8, 1};
    }
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 1;
    plan.reduction_dimx = 64;
    outputs = graph.customized({outputs[0], outputs[1]}, plan);
    assert(outputs.size() == 1);
  }
  for (auto const &op : graph.operators) {
    op->fingerprint();
  }
  assert(ref_graph.operators.back()->output_tensors[0].has_same_fingerprint(
      graph.operators.back()->output_tensors[0]));
  ProfileResult result;
  float total_ms = 0.0f;
  for (auto const &op : graph.operators) {
    op->profile(result);
    total_ms = total_ms + result.run_time;
  }
  printf("[2 Block Graphs] Total runtime = %.4lfms\n", total_ms);
  return 0;
}
