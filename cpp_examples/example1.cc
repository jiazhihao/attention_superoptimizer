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
    threadblock::ExecutionPlan plan;
    plan.ops.push_back({aso::type::TB_REDUCTION_1_OP, {{0, 0}}});
    plan.input_map.push_back({0, -1, -1});
    plan.input_smem_layouts = {
        layout::SmemRowMajor
    };
    plan.output_map = {0, 2, -1};
    plan.forloop_dim = {1};
    plan.grid_dim = {16, 2, 1};
    plan.block_dim = {128, 1, 1};
    plan.forloop_range = 4;
    std::vector<kernel::DTensor> outputs = ref_graph.customized({K}, plan);
    assert(outputs.size() == 1);
    for (auto const &op : ref_graph.operators) {
      op->fingerprint();
    }
  }
  return 0;
}
