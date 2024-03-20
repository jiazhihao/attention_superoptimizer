#include "aso/search/search.h"
#include "aso/kernel/customized.h"
#include "aso/search/dim_strategy.h"
#include "aso/search/op_utils.h"
#include "aso/search/tensor_utils.h"
#include "aso/utils/containers.h"

#include <iostream>

namespace aso {
namespace search {

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph, GeneratorConfig const &config)
    : computation_graph(computation_graph), config(config),
      dim_strategy(DimStrategy(config)), random_test_counter(0),
      verify_counter(0), tbgraph_counter(0) {}

KernelGraphGenerator::KernelGraphGenerator(Checkpoint const &checkpoint)
    : best_graph(best_graph), best_profile_result(best_profile_result),
      config(config), dim_strategy(DimStrategy(config)), final_patterns(final_patterns),
      output_tensors(output_tensors), random_test_counter(random_test_counter),
      verify_counter(verify_counter), tbgraph_counter(tbgraph_counter),
      callstack(callstack) {}

int count_op(type::KNOperatorType op_type, kernel::Graph const &g) {
  int counter = 0;
  for (auto const &op : g.operators) {
    if (op->op_type == op_type) {
      ++counter;
    }
  }
  return counter;
}

std::vector<int> get_open_tensor_idx(SearchContext<DTensor> &c,
                                     kernel::Graph const &g) {
  std::vector<int> result;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      result.push_back(i);
    }
  }
  return result;
}

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

bool KernelGraphGenerator::create_tb_outputs(SearchContext<STensor> &c,
                                             threadblock::Graph &g,
                                             int3 output_map) {

  assert(c.output_pattern.empty());

  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      if (c.all_tensors[i].owner_op->op_type ==
          type::TBOperatorType::TB_INPUT_OP) {
        return false;
      }
      STensor input = c.all_tensors[i];
      std::shared_ptr<AlgebraicPattern> pattern =
          g.forloop_range > 1
              ? std::make_shared<Red>(g.forloop_range, c.algebraic_pattern[i])
              : c.algebraic_pattern[i];
      if (!check_pattern(pattern)) {
        return false;
      }
      TBOperator *new_op = g.create_output_op(input, output_map);
      if (!new_op) {
        return false;
      }
      g.operators.push_back(new_op);
      c.output_pattern.push_back(pattern);
    }
  }
  return true;
}

void KernelGraphGenerator::generate_next_tb_operator(
    SearchContext<STensor> &c,
    threadblock::Graph &g,
    std::function<void()> const &create_customized_then_next_kn,
    int depth) {

  if (depth >= callstack.size()) {
    callstack.push_back(LayerCheckpoint{});
  }
  LayerCheckpoint &checkpoint = callstack.back();

  // Finish threadblock graph search and continue to search the next kernel
  // operator
  for (int3 output_map : dim_strategy.get_output_map_cand(g.grid_dim)) {
    if (contains(checkpoint.output_map_explored, output_map)) {
      continue;
    }
    if (create_tb_outputs(c, g, output_map)) {
      create_customized_then_next_kn();
    }
    while (!c.output_pattern.empty()) {
      c.output_pattern.pop_back();
      delete g.operators.back();
      g.operators.pop_back();
    }
    checkpoint.output_map_explored.insert(output_map);
  }

  if (g.operators.size() >= MAX_NUM_THREADBLOCK_GRAPH_OP) {
    return;
  }

  tbgraph_counter++;
  // if (tbgraph_counter % 50000 == 0) {
  //   std::cerr << "tbgraph counter: " << tbgraph_counter << std::endl;
  // }

  for (type::TBOperatorType op_type : config.tbop_to_explore) {
    if (contains(checkpoint.tbop_explored, op_type)) {
      continue;
    }
    checkpoint.input_idx_explored.clear();
    for (auto const &input_idx :
         dim_strategy.get_input_cand_idx(op_type, c.all_tensors)) {
      if (contains(checkpoint.input_idx_explored, input_idx)) {
        continue;
      }
      Order order(input_idx, static_cast<int>(op_type));
      if (order <= c.op_order.back()) {
        checkpoint.input_idx_explored.insert(input_idx);
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
        checkpoint.input_idx_explored.insert(input_idx);
        continue;
      }

      threadblock::TBOperator *new_op = create_op(g, op_type, input_tensors);

      if (!new_op) {
        checkpoint.input_idx_explored.insert(input_idx);
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
      generate_next_tb_operator(
          c, g, create_customized_then_next_kn, depth + 1);
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
      checkpoint.input_idx_explored.insert(input_idx);
    }
    checkpoint.tbop_explored.insert(op_type);
  }
}

void KernelGraphGenerator::generate_next_kn_operator(SearchContext<DTensor> &c,
                                                     kernel::Graph &g,
                                                     int depth) {
  if (depth >= callstack.size()) {
    callstack.push_back(LayerCheckpoint{});
  }
  LayerCheckpoint &checkpoint = callstack.back();

  if (verify(c, g)) {
    std::cerr << "kernel graph candidate: " << json(g) << std::endl;
    return;
  }

  if (g.operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
    return;
  }

  for (type::KNOperatorType op_type : config.knop_to_explore) {
    if (contains(checkpoint.knop_explored, op_type)) {
      continue;
    }
    checkpoint.input_idx_explored.clear();
    if (op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      for (auto const &input_idx :
           dim_strategy.get_input_cand_idx(op_type, c.all_tensors)) {
        if (contains(checkpoint.input_idx_explored, input_idx)) {
          continue;
        }
        Order order(input_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          checkpoint.input_map_explored.insert(input_idx);
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
          checkpoint.input_map_explored.insert(input_idx);
          continue;
        }
        KNOperator *new_op = create_op(g, op_type, input_tensors);
        if (!new_op) {
          checkpoint.input_map_explored.insert(input_idx);
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
        checkpoint.input_map_explored.insert(input_idx);
      }
    } else {
      if (count_op(type::KNOperatorType::KN_CUSTOMIZED_OP, g) >=
          MAX_NUM_THREADBLOCK) {
        checkpoint.knop_explored.insert(op_type);
        continue;
      }

      for (auto const &input_tensor_idx :
           dim_strategy.get_customized_input_cand_idx(
               c.all_tensors, get_open_tensor_idx(c, g))) {
        if (contains(checkpoint.input_idx_explored, input_tensor_idx)) {
          continue;
        }
        checkpoint.grid_dim_explored.clear();
        std::vector<DTensor> input_tensors;
        for (int i : input_tensor_idx) {
          input_tensors.push_back(c.all_tensors[i]);
        }
        Order order(input_tensor_idx, static_cast<int>(op_type));
        if (order <= c.op_order.back()) {
          checkpoint.input_idx_explored.insert(input_tensor_idx);
          continue;
        }
        for (dim3 grid_dim : dim_strategy.get_grid_dim_cand(input_tensors)) {
          if (contains(checkpoint.grid_dim_explored, grid_dim)) {
            continue;
          }
          checkpoint.block_dim_explored.clear();
          for (dim3 block_dim :
               dim_strategy.get_block_dim_cand(input_tensors, grid_dim)) {
            if (contains(checkpoint.block_dim_explored, block_dim)) {
              continue;
            }
            checkpoint.input_map_explored.clear();
            for (std::vector<int3> const &input_map :
                 dim_strategy.get_input_map_cand(input_tensors, grid_dim)) {
              if (contains(checkpoint.input_map_explored, input_map)) {
                continue;
              }
              checkpoint.forloop_dim_explored.clear();
              for (std::vector<int> const &forloop_dim :
                   dim_strategy.get_forloop_dim_cand(input_tensors)) {
                if (contains(checkpoint.forloop_dim_explored, forloop_dim)) {
                  continue;
                }
                checkpoint.forloop_range_explored.clear();
                for (int forloop_range :
                     dim_strategy.get_forloop_range_cand(input_tensors,
                                                         input_map,
                                                         grid_dim,
                                                         block_dim,
                                                         forloop_dim)) {
                  if (contains(checkpoint.forloop_range_explored,
                               forloop_range)) {
                    continue;
                  }
                  SearchContext<STensor> tb_context;
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

                  if (input_created) {
                    generate_next_tb_operator(tb_context, tb_graph, [&]() {
                      KNOperator *new_op =
                          g.create_customized_op(input_tensors, tb_graph);
                      if (!new_op) {
                        return;
                      }
                      g.operators.push_back(new_op);
                      assert(new_op->output_tensors.size() ==
                             tb_context.output_pattern.size());
                      for (size_t i = 0; i < new_op->output_tensors.size();
                           ++i) {
                        c.all_tensors.push_back(new_op->output_tensors[i]);
                        c.algebraic_pattern.push_back(
                            tb_context.output_pattern[i]);
                        c.num_consumers.push_back(0);
                      }
                      for (int input_idx : input_tensor_idx) {
                        c.num_consumers[input_idx]++;
                      }
                      c.op_order.push_back(order);
                      generate_next_kn_operator(c, g);
                      c.op_order.pop_back();
                      for (int input_idx : input_tensor_idx) {
                        c.num_consumers[input_idx]--;
                      }
                      for (size_t j = 0; j < new_op->output_tensors.size();
                           ++j) {
                        c.all_tensors.pop_back();
                        c.algebraic_pattern.pop_back();
                        c.num_consumers.pop_back();
                      }
                      assert(g.operators.back() == new_op);
                      delete g.operators.back();
                      g.operators.pop_back();
                    });
                  }

                  while (!tb_graph.operators.empty()) {
                    delete tb_graph.operators.back();
                    tb_graph.operators.pop_back();
                  }
                  checkpoint.forloop_range_explored.insert(forloop_range);
                }
                checkpoint.forloop_dim_explored.insert(forloop_dim);
              }
              checkpoint.input_map_explored.insert(input_map);
            }
            checkpoint.block_dim_explored.insert(block_dim);
          }
          checkpoint.grid_dim_explored.insert(grid_dim);
        }
        checkpoint.input_idx_explored.insert(input_tensor_idx);
      }
    }
    checkpoint.knop_explored.insert(op_type);
  }
}

void KernelGraphGenerator::generate_kernel_graphs() {
  pattern_eval();
  fingerprint_eval();

  kernel::Graph g;
  SearchContext<DTensor> c;

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

bool KernelGraphGenerator::verify(SearchContext<DTensor> &c,
                                  kernel::Graph const &g) {
  verify_counter++;
  if (verify_counter % 1000 == 0) {
    std::cerr << "verify counter: " << verify_counter << std::endl;
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

void KernelGraphGenerator::save_checkpoint() const {
  
}

} // namespace search
} // namespace aso
