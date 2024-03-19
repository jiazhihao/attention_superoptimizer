#include "aso/search/search.h"
#include "aso/kernel/customized.h"
#include "aso/search/dim_strategy.h"
#include "aso/search/op_utils.h"
#include "aso/search/tensor_utils.h"
#include "aso/utils/containers.h"

#include <iostream>

namespace aso {
namespace search {

bool Order::operator<(Order const &other) const {
  for (size_t i = 0; i < v.size(); ++i) {
    if (i < other.v.size() && v[i] < other.v[i]) {
      return true;
    }
    if (i < other.v.size() && v[i] > other.v[i]) {
      return false;
    }
    if (i >= other.v.size()) {
      return false;
    }
  }
  if (v.size() < other.v.size()) {
    return true;
  }
  return type < other.type;
}

bool Order::operator<=(Order const &other) const {
  return !(other < *this);
}

Order::Order(std::vector<int> const &v, int type) : v(v), type(type) {}

std::vector<std::vector<int>> get_matches(int num_outputs) {
  std::vector<std::vector<int>> results;
  std::vector<int> perm;
  for (int i = 0; i < num_outputs; ++i) {
    perm.push_back(i);
  }
  do {
    results.push_back(perm);
  } while (std::next_permutation(perm.begin(), perm.end()));
  return results;
}

bool KernelGraphGenerator::finish_tb_graph(
    SearchContext<TBOperator, STensor> &c,
    threadblock::Graph &g,
    int3 output_map) {
  assert(c.output_pattern.empty());

  std::vector<TBOperator *> output_ops;
  std::vector<std::shared_ptr<AlgebraicPattern>> patterns;

  auto free_operators = [&]() {
    while (!output_ops.empty()) {
      delete output_ops.back();
      output_ops.pop_back();
    }
  };

  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      if (c.all_tensors[i].owner_op->op_type ==
          type::TBOperatorType::TB_INPUT_OP) {
        free_operators();
        return false;
      }
      STensor input = c.all_tensors[i];
      std::shared_ptr<AlgebraicPattern> pattern =
          g.forloop_range > 1
              ? std::make_shared<Red>(g.forloop_range, c.algebraic_pattern[i])
              : c.algebraic_pattern[i];
      if (!check_pattern(pattern)) {
        free_operators();
        return false;
      }
      TBOperator *new_op = g.create_output_op(input, output_map);
      if (!new_op) {
        free_operators();
        return false;
      }
      output_ops.push_back(new_op);
      patterns.push_back(pattern);
    }
  }

  if (output_ops.size() > MAX_NUM_THREADBLOCK_OUTPUT) {
    free_operators();
    return false;
  }

  for (size_t i = 0; i < output_ops.size(); ++i) {
    g.operators.push_back(output_ops[i]);
    c.output_pattern.push_back(patterns[i]);
  }
  return true;
}

void KernelGraphGenerator::generate_next_tb_operator(
    SearchContext<TBOperator, STensor> &c,
    threadblock::Graph &g,
    std::function<void()> const &create_customized_then_next_kn) {

  // Finish threadblock graph search and continue to search the next kernel
  // operator
  for (int3 output_map : {int3{0, 2, -1}, int3{0, 1, -1}, int3{0, -1, -1}}) {
    if ((g.grid_dim.x == 1 && output_map.x != -1) || (g.grid_dim.x > 1 && output_map.x == -1)) {
      continue;
    }
    if ((g.grid_dim.y == 1 && output_map.y != -1) || (g.grid_dim.y > 1 && output_map.y == -1)) {
      continue;
    }
    if ((g.grid_dim.z == 1 && output_map.z != -1) || (g.grid_dim.z > 1 && output_map.z == -1)) {
      continue;
    }
    if (finish_tb_graph(c, g, output_map)) {
      create_customized_then_next_kn();
      while (!c.output_pattern.empty()) {
        c.output_pattern.pop_back();
        delete g.operators.back();
        g.operators.pop_back();
      }
    }
  }

  if (g.operators.size() >= MAX_NUM_THREADBLOCK_GRAPH_OP) {
    return;
  }

  tbgraph_counter++;
  if (tbgraph_counter % 10000 == 0) {
    std::cerr << "tbgraph counter: " << tbgraph_counter << std::endl;
    std::cerr << c.algebraic_pattern.size() << " " << c.all_tensors.size() << " " << c.num_consumers.size() << " " << c.op_order.size() << " " << c.output_pattern.size() << std::endl;
  }

  std::vector<type::TBOperatorType> op_to_explore{
      type::TBOperatorType::TB_MATMUL_OP,
      // type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP,
      type::TBOperatorType::TB_EXP_OP,
      // type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP,
      type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP,
      type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP,
      type::TBOperatorType::TB_DIV_OP};

  for (type::TBOperatorType op_type : op_to_explore) {
    for (auto const &input_idx : get_input_cand_idx(op_type, c.all_tensors)) {
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          continue;
        }
        std::vector<STensor> input_tensors;
        std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
        for (int i : input_idx) {
          input_tensors.push_back(c.all_tensors[i]);
          input_patterns.push_back(c.algebraic_pattern[i]);
        }
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input_tensors, input_patterns);
        if (!check_pattern(pattern)) {
          continue;
        }

        threadblock::TBOperator *new_op =
            create_op(g, op_type, input_tensors);

        if (!new_op) {
          continue;
        }

        STensor output = new_op->output_tensors[0];

        g.operators.push_back(new_op);
        c.all_tensors.push_back(output);
        c.algebraic_pattern.push_back(pattern);
        c.num_consumers.push_back(0);
        for (int i : input_idx) {
          c.num_consumers[i]++;
        }
        c.op_order.push_back(order);
        generate_next_tb_operator(c, g, create_customized_then_next_kn);
        c.op_order.pop_back();
        for (int i : input_idx) {
          c.num_consumers[i]--;
        }
        c.num_consumers.pop_back();
        c.algebraic_pattern.pop_back();
        c.all_tensors.pop_back();
        assert(g.operators.back() == new_op);
        delete g.operators.back();
        g.operators.pop_back();
    }
  }
}

void KernelGraphGenerator::generate_next_kn_operator(
    SearchContext<KNOperator, DTensor> &c, kernel::Graph &g) {
  if (verify(c, g)) {
    std::cerr << "kernel graph candidate: " << json(g) << std::endl;
    return;
  }

  if (g.operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
    return;
  }

  // TODO(@wmdi): make it user-defined
  std::vector<type::KNOperatorType> op_to_explore{
      type::KNOperatorType::KN_MATMUL_OP,
      // type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_EXP_OP,
      type::KNOperatorType::KN_DIV_OP,
      type::KNOperatorType::KN_CUSTOMIZED_OP};

  for (type::KNOperatorType op_type : op_to_explore) {
    if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      for (auto const &input_idx : get_input_cand_idx(op_type, c.all_tensors)) {
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          continue;
        }
        std::vector<DTensor> input_tensors;
        std::vector<std::shared_ptr<AlgebraicPattern>> input_patterns;
        for (int i : input_idx) {
          input_tensors.push_back(c.all_tensors[i]);
          input_patterns.push_back(c.algebraic_pattern[i]);
        }
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input_tensors, input_patterns);
        if (!check_pattern(pattern)) {
          continue;
        }
        KNOperator *new_op = create_op(g, op_type, input_tensors);
        if (!new_op) {
          continue;
        }
        DTensor output = new_op->output_tensors[0];

        g.operators.push_back(new_op);
        c.all_tensors.push_back(output);
        c.algebraic_pattern.push_back(pattern);
        c.num_consumers.push_back(0);
        for (int i : input_idx) {
          c.num_consumers[i]++;
        }
        c.op_order.push_back(order);
        generate_next_kn_operator(c, g);
        c.op_order.pop_back();
        for (int i : input_idx) {
          c.num_consumers[i]--;
        }
        c.num_consumers.pop_back();
        c.algebraic_pattern.pop_back();
        c.all_tensors.pop_back();
        assert(g.operators.back() == new_op);
        delete g.operators.back();
        g.operators.pop_back();
      }
    } else {
      // Customized op
      if (num_kernels >= MAX_NUM_THREADBLOCK) {
        continue;
      }

      // FIXME: simplify the search space
      for (auto const &input_tensor_idx :
           get_customized_input_cand_idx(c.all_tensors)) {
        if (input_tensor_idx.size() > MAX_NUM_THREADBLOCK_INPUT) {
          continue;
        }
        std::vector<DTensor> input_tensors;
        for (int i : input_tensor_idx) {
          input_tensors.push_back(c.all_tensors[i]);
        }
        Order order(input_tensor_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          continue;
        }
        for (std::vector<int3> const &input_map :
             get_input_map_cand(input_tensors)) {
          for (dim3 grid_dim : get_grid_dim_cand(input_tensors, input_map)) {
            for (dim3 block_dim :
                 get_block_dim_cand(input_tensors, input_map, grid_dim)) {
              for (std::vector<int> const &forloop_dim :
                   get_forloop_dim_cand(input_tensors)) {
                for (int forloop_range : get_forloop_range_cand(input_tensors,
                                                                input_map,
                                                                grid_dim,
                                                                block_dim,
                                                                forloop_dim)) {
                  SearchContext<TBOperator, STensor> tb_context;
                  threadblock::Graph tb_graph(
                      grid_dim, block_dim, forloop_range);

                  bool input_created = true;
                  for (size_t i = 0; i < input_tensors.size(); ++i) {
                    DTensor tensor = input_tensors[i];
                    TBOperator *input_op =
                        tb_graph.create_input_op(tensor,
                                                 input_map[i],
                                                 forloop_dim[i],
                                                 layout::SmemRowMajor);
                    if (input_op == nullptr) {
                      input_created = false;
                      break;
                    }
                    tb_graph.operators.push_back(input_op);
                    STensor output = input_op->output_tensors[0];
                    tb_context.all_tensors.push_back(output);
                    tb_context.algebraic_pattern.push_back(
                        c.algebraic_pattern[input_tensor_idx[i]]);
                    tb_context.num_consumers.push_back(0);
                    tb_context.op_order.push_back(Order(
                        {},
                        static_cast<int>(type::TBOperatorType::TB_INPUT_OP)));
                  }
                  if (!input_created) {
                    while (!tb_graph.operators.empty()) {
                      delete tb_graph.operators.back();
                      tb_graph.operators.pop_back();
                    }
                    continue;
                  }

                  auto create_customized_then_next_kn = [&]() {
                    KNOperator *new_op =
                        g.create_customized_op(input_tensors, tb_graph);
                    if (!new_op) {
                      return;
                    }
                    g.operators.push_back(new_op);
                    assert(new_op->output_tensors.size() ==
                           tb_context.output_pattern.size());
                    for (size_t i = 0; i < new_op->output_tensors.size(); ++i) {
                      c.all_tensors.push_back(new_op->output_tensors[i]);
                      c.algebraic_pattern.push_back(
                          tb_context.output_pattern[i]);
                      c.num_consumers.push_back(0);
                    }
                    for (int input_idx : input_tensor_idx) {
                      c.num_consumers[input_idx]++;
                    }
                    c.op_order.push_back(order);
                    num_kernels++;
                    generate_next_kn_operator(c, g);
                    num_kernels--;
                    c.op_order.pop_back();
                    for (int input_idx : input_tensor_idx) {
                      c.num_consumers[input_idx]--;
                    }
                    for (size_t j = 0; j < new_op->output_tensors.size(); ++j) {
                      c.all_tensors.pop_back();
                      c.algebraic_pattern.pop_back();
                      c.num_consumers.pop_back();
                    }
                    assert(g.operators.back() == new_op);
                    delete g.operators.back();
                    g.operators.pop_back();
                  };

                  generate_next_tb_operator(tb_context, tb_graph, create_customized_then_next_kn);

                  while (!tb_graph.operators.empty()) {
                    delete tb_graph.operators.back();
                    tb_graph.operators.pop_back();
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

void KernelGraphGenerator::generate_kernel_graphs() {
  pattern_eval();
  fingerprint_eval();

  kernel::Graph g;
  SearchContext<KNOperator, DTensor> c;

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      g.operators.push_back(op);
      DTensor output_tensor = op->output_tensors[0];
      assert(contains_key(computation_graph_patterns, output_tensor));
      c.all_tensors.push_back(output_tensor);
      c.algebraic_pattern.push_back(
          computation_graph_patterns.at(output_tensor));
      c.num_consumers.push_back(0);
      c.op_order.push_back(
          Order({}, static_cast<int>(type::KNOperatorType::KN_INPUT_OP)));
    }
  }

  process_outputs();
  generate_next_kn_operator(c, g);

  std::cerr << "random test counter: " << random_test_counter << std::endl;
}

void KernelGraphGenerator::fingerprint_eval() {
  for (auto const &op : computation_graph.operators) {
    op->fingerprint();
  }
}

void KernelGraphGenerator::process_outputs() {
  std::unordered_map<DTensor, int> num_consumers;
  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &input : op->input_tensors) {
      num_consumers[input]++;
    }
  }

  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &output : op->output_tensors) {
      if (num_consumers[output] == 0) {
        output_tensors.push_back(output);
        final_patterns.push_back(computation_graph_patterns.at(output));
      }
    }
  }
  for (auto const &pattern : final_patterns) {
    std::cerr << "final pattern: " << pattern->to_string() << std::endl;
  }
}

bool KernelGraphGenerator::check_pattern(
    std::shared_ptr<AlgebraicPattern> pattern) {
  for (auto const &final_pattern : final_patterns) {
    if (pattern->subpattern_to(*final_pattern)) {
      return true;
    }
  }
  return false;
}

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph)
    : computation_graph(computation_graph), num_kernels(0), random_test_counter(0), verify_counter(0), tbgraph_counter(0) {}

void KernelGraphGenerator::pattern_eval() {
  // Assume operators are in topological order
  int input_id = 0;
  for (KNOperator *op : computation_graph.operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Var>("v_" + std::to_string(input_id))});
        input_id++;
        break;
      case type::KNOperatorType::KN_MATMUL_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1],
                 std::make_shared<Mul>(
                     computation_graph_patterns.at(op->input_tensors[0]),
                     computation_graph_patterns.at(op->input_tensors[1])))});
        break;
      case type::KNOperatorType::KN_REDUCTION_0_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[0],
                 computation_graph_patterns.at(op->input_tensors[0]))});
      case type::KNOperatorType::KN_REDUCTION_1_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[1],
                 computation_graph_patterns.at(op->input_tensors[0]))});
      case type::KNOperatorType::KN_REDUCTION_2_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Red>(
                 op->input_tensors[0].dim[2],
                 computation_graph_patterns.at(op->input_tensors[0]))});
        break;
      case type::KNOperatorType::KN_DIV_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Div>(
                 computation_graph_patterns.at(op->input_tensors[0]),
                 computation_graph_patterns.at(op->input_tensors[1]))});
        break;
      case type::KNOperatorType::KN_EXP_OP:
        computation_graph_patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Exp>(
                 computation_graph_patterns.at(op->input_tensors[0]))});
        break;
      default:
        assert(false && "Unsupported computation graph operator");
    }
  }
}

bool KernelGraphGenerator::verify(SearchContext<KNOperator, DTensor> &c,
                                  kernel::Graph const &g) {
  verify_counter++;
  if (verify_counter % 1000 == 0) {
    std::cerr << "verify counter: " << verify_counter << std::endl;
    std::cerr << c.algebraic_pattern.size() << " " << c.all_tensors.size() << " " << c.num_consumers.size() << " " << c.op_order.size() << " " << c.output_pattern.size() << std::endl;
  }
  size_t num_outputs = 0;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      ++num_outputs;
    }
  }
  if (num_outputs != final_patterns.size()) {
    return false;
  }

  std::vector<DTensor> outputs;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      outputs.push_back(c.all_tensors[i]);
    }
  }

  random_test_counter++;
  if (random_test_counter < 10 || random_test_counter % 10 == 0) {
    std::cerr << "random test counter: " << random_test_counter << std::endl;
  }

  for (auto const &op : g.operators) {
    op->fingerprint();
  }

  for (auto const &match : get_matches(outputs.size())) {
    if (have_same_fingerprint(outputs, match)) {
      return true;
    }
  }

  return false;
}

bool KernelGraphGenerator::have_same_fingerprint(
    std::vector<DTensor> const &outputs, std::vector<int> const &match) const {
  assert(outputs.size() == match.size());
  for (int i = 0; i < static_cast<int>(match.size()); ++i) {
    if (!output_tensors[i].has_same_fingerprint(outputs[match[i]])) {
      return false;
    }
  }
  return true;
}

void KernelGraphGenerator::update_best_graph(kernel::Graph &g) {
  std::cerr << "kernel graph candidate: " << json(g) << std::endl;
  ProfileResult result;
  for (auto op : g.operators) {
    ProfileResult op_result;
    op->profile(op_result);
    result.run_time += op_result.run_time;
  }
  if (result.run_time < best_profile_result.run_time) {
    best_graph = g;
    best_profile_result = result;
  }
  return;
}

} // namespace search
} // namespace aso
