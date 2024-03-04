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
  return true;
}

std::vector<int> to_dim_vector(int num_dims, int *dim) {
  std::vector<int> dims;
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(dim[i]);
  }
  return dims;
}

std::vector<std::vector<int>>
    get_input_cand(std::vector<DTensor> const &all_inputs) {
  std::vector<std::vector<int>> results;
  for (int bitmap = 1; bitmap < (1 << all_inputs.size()); ++bitmap) {
    std::vector<int> inputs;
    for (size_t i = 0; i < all_inputs.size(); ++i) {
      if ((bitmap >> i) & 1) {
        inputs.push_back(i);
      }
    }
    results.push_back(inputs);
  }
  return results;
}

bool KernelGraphGenerator::finish_tb_graph(
    SearchContext<TBOperator, STensor> &c, threadblock::Graph &g) {
  assert(c.output_pattern.empty());

  std::vector<TBOperator *> output_ops;
  std::vector<std::shared_ptr<AlgebraicPattern>> patterns;

  auto fail_to_finish = [&]() {
    while (!output_ops.empty()) {
      delete output_ops.back();
      output_ops.pop_back();
    }
  };

  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      STensor input = c.all_tensors[i];
      std::shared_ptr<AlgebraicPattern> pattern =
          g.forloop_range > 1
              ? std::make_shared<Red>(g.forloop_range, c.algebraic_pattern[i])
              : c.algebraic_pattern[i];
      if (!check_pattern(pattern)) {
        fail_to_finish();
        return false;
      }
      for (int3 output_map : {int3{0, 2, -1}}) {
        TBOperator *new_op = g.create_output_op(input, output_map);
        if (!new_op) {
          fail_to_finish();
          return false;
        }
        output_ops.push_back(new_op);
        patterns.push_back(pattern);
      }
    }
  }

  if (output_ops.size() > MAX_NUM_THREADBLOCK_OUTPUT) {
    fail_to_finish();
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
    std::function<void()> const &callback) {

  // Finish threadblock graph search and continue to search the next kernel
  // operator
  if (finish_tb_graph(c, g)) {
    callback();
    while (!c.output_pattern.empty()) {
      c.output_pattern.pop_back();
      delete g.operators.back();
      g.operators.pop_back();
    }
  }

  if (g.operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
    return;
  }

  std::vector<type::TBOperatorType> op_to_explore{
      type::TBOperatorType::TB_MATMUL_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP,
      type::TBOperatorType::TB_EXP_OP,
      type::TBOperatorType::TB_DIV_OP};

  for (type::TBOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (size_t i = 0; i < c.all_tensors.size(); ++i) {
        for (size_t j = 0; j < c.all_tensors.size(); ++j) {
          size_t hash = get_operator_hash(i, j, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          Order order{{i, j}};
          if (order < c.op_order.back()) {
            continue;
          }
          STensor input1 = c.all_tensors[i], input2 = c.all_tensors[j];
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type,
                          input1,
                          input2,
                          c.algebraic_pattern[i],
                          c.algebraic_pattern[j]);
          if (!check_pattern(pattern)) {
            continue;
          }

          threadblock::TBOperator *new_op =
              create_op(g, op_type, input1, input2);

          if (!new_op) {
            continue;
          }

          STensor output = new_op->output_tensors[0];

          g.operators.push_back(new_op);
          c.all_tensors.push_back(output);
          c.algebraic_pattern.push_back(pattern);
          c.num_consumers.push_back(0);
          c.num_consumers[i]++;
          c.num_consumers[j]++;
          c.existing_op_hash.insert(hash);
          c.op_order.push_back(order);
          generate_next_tb_operator(c, g, callback);
          c.op_order.pop_back();
          c.existing_op_hash.erase(hash);
          c.num_consumers[j]--;
          c.num_consumers[i]--;
          c.num_consumers.pop_back();
          c.algebraic_pattern.pop_back();
          c.all_tensors.pop_back();
          delete g.operators.back();
          g.operators.pop_back();
        }
      }
    } else if (is_unary(op_type)) {
      for (size_t i = 0; i < c.all_tensors.size(); ++i) {
        size_t hash = get_operator_hash(i, op_type);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        Order order{{i}};
        if (order < c.op_order.back()) {
          continue;
        }
        STensor input = c.all_tensors[i];
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input, c.algebraic_pattern[i]);
        if (!check_pattern(pattern)) {
          continue;
        }

        threadblock::TBOperator *new_op = create_op(g, op_type, input);
        if (!new_op) {
          continue;
        }
        STensor output = new_op->output_tensors[0];

        g.operators.push_back(new_op);
        c.all_tensors.push_back(output);
        c.algebraic_pattern.push_back(pattern);
        c.num_consumers.push_back(0);
        c.num_consumers[i]++;
        c.existing_op_hash.insert(hash);
        c.op_order.push_back(order);
        generate_next_tb_operator(c, g, callback);
        c.op_order.pop_back();
        c.existing_op_hash.erase(hash);
        c.num_consumers[i]--;
        c.num_consumers.pop_back();
        c.algebraic_pattern.pop_back();
        c.all_tensors.pop_back();
        delete g.operators.back();
        g.operators.pop_back();
      }
    }
  }
}

void KernelGraphGenerator::generate_next_kn_operator(
    SearchContext<KNOperator, DTensor> &c, kernel::Graph &g) {
  if (verify(g, c)) {
    std::cout << "kernel graph candidate: " << json(g) << std::endl;
    return;
  }

  if (g.operators.size() >= MAX_NUM_KERNEL_GRAPH_OP) {
    return;
  }

  // TODO(@wmdi): make it user-defined
  std::vector<type::KNOperatorType> op_to_explore{
      type::KNOperatorType::KN_MATMUL_OP,
      // type::KNOperatorType::KN_REDUCTION_0_OP,
      // type::KNOperatorType::KN_REDUCTION_1_OP,
      // type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_CUSTOMIZED_OP};

  for (type::KNOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (size_t i = 0; i < c.all_tensors.size(); ++i) {
        for (size_t j = 0; j < c.all_tensors.size(); ++j) {
          size_t hash = get_operator_hash(i, j, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          Order order{{i, j}};
          if (order < c.op_order.back()){
            continue;
          }
          DTensor input1 = c.all_tensors[i], input2 = c.all_tensors[j];
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type,
                          input1,
                          input2,
                          c.algebraic_pattern[i],
                          c.algebraic_pattern[j]);
          if (!check_pattern(pattern)) {
            continue;
          }
          KNOperator *new_op = create_op(g, op_type, input1, input2);
          if (!new_op) {
            continue;
          }
          DTensor output = new_op->output_tensors[0];

          g.operators.push_back(new_op);
          c.all_tensors.push_back(output);
          c.algebraic_pattern.push_back(pattern);
          c.num_consumers.push_back(0);
          c.num_consumers[i]++;
          c.num_consumers[j]++;
          c.existing_op_hash.insert(hash);
          c.op_order.push_back(order);
          generate_next_kn_operator(c, g);
          c.op_order.pop_back();
          c.existing_op_hash.erase(hash);
          c.num_consumers[j]--;
          c.num_consumers[i]--;
          c.num_consumers.pop_back();
          c.algebraic_pattern.pop_back();
          c.all_tensors.pop_back();
          delete g.operators.back();
          g.operators.pop_back();
        }
      }
    } else if (is_unary(op_type)) {
      for (size_t i = 0; i < c.all_tensors.size(); ++i) {
        size_t hash = get_operator_hash(i, op_type);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        Order order{{i}};
        if (order < c.op_order.back()) {
          continue;
        }
        DTensor input = c.all_tensors[i];
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input, c.algebraic_pattern[i]);
        if (!check_pattern(pattern)) {
          continue;
        }

        kernel::KNOperator *new_op = create_op(g, op_type, input);
        if (!new_op) {
          continue;
        }

        DTensor output = new_op->output_tensors[0];

        g.operators.push_back(new_op);
        c.all_tensors.push_back(output);
        c.algebraic_pattern.push_back(pattern);
        c.num_consumers.push_back(0);
        c.num_consumers[i]++;
        c.existing_op_hash.insert(hash);
        c.op_order.push_back(order);
        generate_next_kn_operator(c, g);
        c.op_order.pop_back();
        c.existing_op_hash.erase(hash);
        c.num_consumers[i]--;
        c.num_consumers.pop_back();
        c.algebraic_pattern.pop_back();
        c.all_tensors.pop_back();
        delete g.operators.back();
        g.operators.pop_back();
      }
    } else if (op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      if (num_kernels >= 1) {
        continue;
      }

      // FIXME: simplify the search space
      for (auto const &input_tensor_idx : get_input_cand(c.all_tensors)) {
        if (input_tensor_idx.size() >
            threadblock::KernelParams::MAX_NUM_DMEM_INPUTS) {
          continue;
        }
        size_t hash = 0;
        for (auto const &idx : input_tensor_idx) {
          hash_combine(hash, idx);
        }
        hash_combine(hash, type::KNOperatorType::KN_CUSTOMIZED_OP);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        std::vector<DTensor> input_tensors;
        for (int i : input_tensor_idx) {
          input_tensors.push_back(c.all_tensors[i]);
        }
        Order order{input_tensor_idx};
        if (order < c.op_order.back()) {
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
                    tb_context.op_order.push_back({{-1}});
                  }
                  if (!input_created) {
                    while (!tb_graph.operators.empty()) {
                      delete tb_graph.operators.back();
                      tb_graph.operators.pop_back();
                    }
                    continue;
                  }

                  auto callback = [&]() {
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
                    c.existing_op_hash.insert(hash);
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
                    c.existing_op_hash.erase(hash);
                    for (size_t j = 0; j < new_op->output_tensors.size(); ++j) {
                      c.all_tensors.pop_back();
                      c.algebraic_pattern.pop_back();
                      c.num_consumers.pop_back();
                    }
                    delete g.operators.back();
                    g.operators.pop_back();
                  };

                  generate_next_tb_operator(tb_context, tb_graph, callback);

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
      c.op_order.push_back({{-1}});
    }
  }

  find_final_patterns();
  generate_next_kn_operator(c, g);
}

void KernelGraphGenerator::find_final_patterns() {
  std::unordered_map<DTensor, int> num_consumers;
  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &input : op->input_tensors) {
      num_consumers[input]++;
    }
  }

  for (kernel::KNOperator *op : computation_graph.operators) {
    for (DTensor const &output : op->output_tensors) {
      if (num_consumers[output] == 0) {
        final_patterns.push_back(computation_graph_patterns.at(output));
      }
    }
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
    kernel::Graph const &computation_graph,
    size_t device_mem_size,
    size_t shared_mem_size)
    : computation_graph(computation_graph), device_mem_size(device_mem_size),
      shared_mem_size(shared_mem_size), num_kernels(0) {}

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
                 op->input_tensors[0].dim[1],
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
      default:
        assert(false && "Unsupported computation graph operator");
    }
  }
}

bool KernelGraphGenerator::verify(kernel::Graph const &g,
                                  SearchContext<KNOperator, DTensor> &c) {
  size_t num_output = 0;
  for (size_t i = 0; i < c.all_tensors.size(); ++i) {
    if (c.num_consumers[i] == 0) {
      if (num_output >= final_patterns.size() ||
          !(*c.algebraic_pattern[i] == *final_patterns[num_output])) {
        return false;
      }
      ++num_output;
    }
  }
  if (num_output != final_patterns.size()) {
    return false;
  }

  // assert(false && "TBD");
  return true;
}

} // namespace search
} // namespace aso
