#include "aso/search/search.h"
#include "aso/kernel/customized.h"
#include "aso/search/dim_strategy.h"
#include "aso/search/op_utils.h"
#include "aso/search/tensor_utils.h"
#include "aso/utils/containers.h"

#include <iostream>

std::ostream &operator<<(std::ostream &os, dim3 const &d) {
  os << "(" << d.x << "," << d.y << "," << d.z << ")";
  return os;
}

std::ostream &operator<<(std::ostream &os, int3 const &d) {
  os << "(" << d.x << "," << d.y << "," << d.z << ")";
  return os;
}

namespace aso {
namespace search {

std::vector<int> to_dim_vector(int num_dims, int *dim) {
  std::vector<int> dims;
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(dim[i]);
  }
  return dims;
}

bool KernelGraphGenerator::is_finished_graph(
    SearchContext<TBOperator, STensor> &c, threadblock::Graph const &g) {
  for (auto op : g.operators) {
    if (c.output_degree[op] == 0 &&
        op->op_type != type::TBOperatorType::TB_OUTPUT_OP) {
      return false;
    }
  }
  return true;
}

std::vector<std::vector<DTensor>>
    get_input_cand(std::vector<DTensor> const &all_inputs) {
  assert(all_inputs.size() <= threadblock::KernelParams::MAX_NUM_DMEM_INPUTS);
  std::vector<std::vector<DTensor>> results;
  for (int bitmap = 1; bitmap < (1 << all_inputs.size()); ++bitmap) {
    std::vector<DTensor> inputs;
    for (size_t i = 0; i < all_inputs.size(); ++i) {
      if ((bitmap >> i) & 1) {
        inputs.push_back(all_inputs[i]);
      }
    }
    results.push_back(inputs);
  }
  return results;
}

void KernelGraphGenerator::generate_threadblock_graphs(
    SearchContext<TBOperator, STensor> &c,
    threadblock::Graph g,
    std::vector<std::shared_ptr<AlgebraicPattern>> output_patterns,
    std::vector<threadblock::Graph> &result_graphs,
    std::vector<std::vector<std::shared_ptr<AlgebraicPattern>>>
        &result_output_patterns) {

  if (is_finished_graph(c, g)) {
    result_graphs.push_back(g);
    result_output_patterns.push_back(output_patterns);
    return;
  }

  if (g.operators.size() >= 8) {
    return;
  }

  std::vector<type::TBOperatorType> op_to_explore{
      type::TBOperatorType::TB_MATMUL_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      // type::TBOperatorType::TB_REDUCTION_2_OP,
      type::TBOperatorType::TB_EXP_OP,
      type::TBOperatorType::TB_DIV_OP,
      type::TBOperatorType::TB_OUTPUT_OP};

  for (type::TBOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (TBOperator *op1 : g.operators) {
        if (op1->op_type == type::TB_OUTPUT_OP) {
          continue;
        }
        for (TBOperator *op2 : g.operators) {
          if (op2->op_type == type::TB_OUTPUT_OP) {
            continue;
          }
          size_t hash = get_operator_hash(op1, op2, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          STensor input1 = op1->output_tensors[0],
                  input2 = op2->output_tensors[0];
          if (!check_tensor_shape(op_type, input1, input2)) {
            continue;
          }
          assert(contains_key(c.algebraic_pattern, input1));
          assert(contains_key(c.algebraic_pattern, input2));
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type,
                          input1,
                          input2,
                          c.algebraic_pattern.at(input1),
                          c.algebraic_pattern.at(input2));
          if (!check_pattern(pattern)) {
            continue;
          }

          threadblock::Graph ng = g;
          threadblock::TBOperator *new_op = nullptr;
          switch (op_type) {
            case type::TBOperatorType::TB_MATMUL_OP:
              new_op = ng.create_matmul_op(input1, input2);
              break;
            case type::TBOperatorType::TB_DIV_OP:
              assert(false && "TBD");
              break;
            default:
              assert(false && "Unsupported operator");
          }

          if (!new_op) {
            continue;
          }

          ng.operators.push_back(new_op);
          STensor output = new_op->output_tensors[0];

          c.algebraic_pattern.insert({output, pattern});
          c.existing_op_hash.insert(hash);
          c.output_degree[op1]++;
          c.output_degree[op2]++;
          generate_threadblock_graphs(
              c, ng, output_patterns, result_graphs, result_output_patterns);
          c.algebraic_pattern.erase(output);
          c.existing_op_hash.erase(hash);
          c.output_degree[op1]--;
          c.output_degree[op2]--;
        }
      }
    } else if (is_unary(op_type)) {
      for (auto op : g.operators) {
        if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
          continue;
        }
        size_t hash = get_operator_hash(op, op_type);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        STensor input = op->output_tensors[0];
        if (!check_tensor_shape(op_type, input)) {
          continue;
        }
        assert(contains_key(c.algebraic_pattern, input));
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, input, c.algebraic_pattern.at(input));
        if (!check_pattern(pattern)) {
          continue;
        }

        threadblock::Graph ng = g;
        threadblock::TBOperator *new_op = nullptr;
        switch (op_type) {
          case type::TBOperatorType::TB_EXP_OP:
            new_op = ng.create_elementunary_op(input, op_type);
            break;
          case type::TBOperatorType::TB_REDUCTION_0_OP:
            new_op = ng.create_reduction_op(input, 0);
            break;
          case type::TBOperatorType::TB_REDUCTION_1_OP:
            new_op = ng.create_reduction_op(input, 1);
            break;
          case type::TBOperatorType::TB_REDUCTION_2_OP:
            new_op = ng.create_reduction_op(input, 2);
            break;
          default:
            assert(false && "Unsupported operator");
        }

        if (!new_op) {
          continue;
        }
        ng.operators.push_back(new_op);
        STensor output = new_op->output_tensors[0];

        c.algebraic_pattern.insert({output, pattern});
        c.existing_op_hash.insert(hash);
        c.output_degree[op]++;
        generate_threadblock_graphs(
            c, ng, output_patterns, result_graphs, result_output_patterns);
        c.algebraic_pattern.erase(output);
        c.existing_op_hash.erase(hash);
        c.output_degree[op]--;
      }
    } else if (op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      for (TBOperator *op : g.operators) {
        if (op->op_type == type::TBOperatorType::TB_INPUT_OP ||
            op->op_type == type::TBOperatorType::TB_OUTPUT_OP ||
            c.output_degree[op] > 0 ||
            output_patterns.size() >=
                threadblock::KernelParams::MAX_NUM_DMEM_OUTPUTS) {
          continue;
        }

        STensor input = op->output_tensors[0];
        assert(contains_key(c.algebraic_pattern, input));
        std::shared_ptr<AlgebraicPattern> pattern =
            g.forloop_range > 1
                ? std::make_shared<Red>(g.forloop_range,
                                        c.algebraic_pattern.at(input))
                : c.algebraic_pattern.at(input);
        if (!check_pattern(pattern)) {
          continue;
        }

        for (int3 output_map : {int3{0, 1, -1}, int3{1, 0, -1}}) {
          threadblock::Graph ng = g;
          threadblock::TBOperator *new_op =
              ng.create_output_op(input, output_map);
          if (!new_op) {
            continue;
          }
          ng.operators.push_back(new_op);

          output_patterns.push_back(pattern);
          c.output_degree[op]++;
          generate_threadblock_graphs(
              c, ng, output_patterns, result_graphs, result_output_patterns);
          output_patterns.pop_back();
          c.output_degree[op]--;
        }
      }
    }
  }
}

void KernelGraphGenerator::generate_next_kernel(
    SearchContext<KNOperator, DTensor> &c, kernel::Graph &g) {
  if (verify(g, c)) {
    std::cout << "kernel graph candidate: " << json(g) << std::endl;
    return;
  }

  std::vector<type::KNOperatorType> op_to_explore{
      type::KNOperatorType::KN_MATMUL_OP,
      // type::KNOperatorType::KN_REDUCTION_0_OP,
      // type::KNOperatorType::KN_REDUCTION_1_OP,
      // type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_CUSTOMIZED_OP};

  for (type::KNOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      int num_op = g.operators.size();
      for (int op_idx1 = 0; op_idx1 < num_op; ++op_idx1) {
        for (int op_idx2 = 0; op_idx2 < num_op; ++op_idx2) {
          KNOperator *op1 = g.operators[op_idx1], *op2 = g.operators[op_idx2];
          for (DTensor input1 : op1->output_tensors) {
            for (DTensor input2 : op2->output_tensors) {
              if (!check_tensor_shape(op_type, input1, input2)) {
                continue;
              }
              size_t hash = get_operator_hash(input1, input2, op_type);
              if (contains(c.existing_op_hash, hash)) {
                continue;
              }
              assert(contains_key(c.algebraic_pattern, input1));
              assert(contains_key(c.algebraic_pattern, input2));
              std::shared_ptr<AlgebraicPattern> pattern =
                  get_pattern(op_type,
                              input1,
                              input2,
                              c.algebraic_pattern.at(input1),
                              c.algebraic_pattern.at(input2));
              if (!check_pattern(pattern)) {
                continue;
              }
              kernel::KNOperator *new_op = nullptr;
              switch (op_type) {
                case type::KNOperatorType::KN_MATMUL_OP:
                  new_op = g.create_matmul_op(input1, input2);
                  break;
                default:
                  assert(false && "Unsupported operator");
              }

              if (!new_op) {
                continue;
              }
              g.operators.push_back(new_op);
              DTensor output = new_op->output_tensors[0];

              c.algebraic_pattern.insert({output, pattern});
              c.existing_op_hash.insert(hash);
              c.output_degree[op1]++;
              c.output_degree[op2]++;
              generate_next_kernel(c, g);
              c.algebraic_pattern.erase(output);
              c.existing_op_hash.erase(hash);
              c.output_degree[op1]--;
              c.output_degree[op2]--;
              delete g.operators.back();
              g.operators.pop_back();
            }
          }
        }
      }
    } else if (is_unary(op_type)) {
      int num_op = g.operators.size();
      for (int op_idx = 0; op_idx < num_op; ++op_idx) {
        KNOperator *op = g.operators[op_idx];
        for (DTensor input : op->output_tensors) {
          if (!check_tensor_shape(op_type, input)) {
            continue;
          }
          size_t hash = get_operator_hash(input, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          assert(contains_key(c.algebraic_pattern, input));
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type, input, c.algebraic_pattern.at(input));
          if (!check_pattern(pattern)) {
            continue;
          }

          kernel::KNOperator *new_op = nullptr;
          switch (op_type) {
            case type::KNOperatorType::KN_REDUCTION_0_OP:
            case type::KNOperatorType::KN_REDUCTION_1_OP:
            case type::KNOperatorType::KN_REDUCTION_2_OP:
              assert(false && "TBD");
            default:
              assert(false && "Unsupported operator");
          }

          if (!new_op) {
            continue;
          }
          g.operators.push_back(new_op);
          DTensor output = new_op->output_tensors[0];

          c.algebraic_pattern.insert({output, pattern});
          c.existing_op_hash.insert(hash);
          c.output_degree[op]++;
          generate_next_kernel(c, g);
          c.algebraic_pattern.erase(output);
          c.existing_op_hash.erase(hash);
          c.output_degree[op]--;
          delete g.operators.back();
          g.operators.pop_back();
        }
      }
    } else if (op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      if (num_kernels >= 1) {
        continue;
      }

      std::vector<DTensor> all_input_tensors;
      std::vector<KNOperator *> input_operators;

      for (auto op : g.operators) {
        input_operators.push_back(op);
        for (DTensor tensor : op->output_tensors) {
          all_input_tensors.push_back(tensor);
        }
      }

      if (all_input_tensors.size() >
          threadblock::KernelParams::MAX_NUM_DMEM_INPUTS) {
        continue;
      }

      // FIXME: simplify the search space
      for (auto const &input_tensors : get_input_cand(all_input_tensors)) {
        size_t hash = 0;
        for (auto const &input : input_tensors) {
          hash_combine(hash, input);
        }
        hash_combine(hash, type::KNOperatorType::KN_CUSTOMIZED_OP);
        if (contains(c.existing_op_hash, hash)) {
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
                  SearchContext<TBOperator, STensor> nc;
                  threadblock::Graph ng(grid_dim, block_dim, forloop_range);

                  bool input_created = true;
                  for (size_t i = 0; i < input_tensors.size(); ++i) {
                    DTensor tensor = input_tensors[i];
                    TBOperator *input_op =
                        ng.create_input_op(tensor,
                                           input_map[i],
                                           forloop_dim[i],
                                           layout::SmemRowMajor);
                    if (input_op == nullptr) {
                      input_created = false;
                      break;
                    }
                    ng.operators.push_back(input_op);
                    STensor output = input_op->output_tensors[0];
                    assert(contains_key(c.algebraic_pattern, tensor));
                    nc.algebraic_pattern.insert(
                        {output, c.algebraic_pattern.at(tensor)});
                  }
                  if (!input_created) {
                    continue;
                  }

                  std::vector<threadblock::Graph> tbgs;
                  std::vector<std::vector<std::shared_ptr<AlgebraicPattern>>>
                      output_patterns;
                  generate_threadblock_graphs(
                      nc, ng, {}, tbgs, output_patterns);

                  assert(tbgs.size() == output_patterns.size());

                  for (size_t i = 0; i < tbgs.size(); ++i) {
                    threadblock::Graph const &tbg = tbgs[i];
                    kernel::KNOperator *new_op =
                        g.create_customized_op(input_tensors, tbg);
                    if (!new_op) {
                      continue;
                    }
                    g.operators.push_back(new_op);
                    assert(new_op->output_tensors.size() ==
                           output_patterns[i].size());
                    for (size_t j = 0; j < new_op->output_tensors.size(); ++j) {
                      c.algebraic_pattern.insert(
                          {new_op->output_tensors[j], output_patterns[i][j]});
                    }
                    c.existing_op_hash.insert(hash);
                    for (auto op : input_operators) {
                      c.output_degree[op]++;
                    }
                    num_kernels++;
                    generate_next_kernel(c, g);
                    num_kernels--;
                    for (size_t j = 0; j < new_op->output_tensors.size(); ++j) {
                      c.algebraic_pattern.erase(new_op->output_tensors[j]);
                    }
                    c.existing_op_hash.erase(hash);
                    for (auto op : input_operators) {
                      c.output_degree[op]--;
                    }
                    delete g.operators.back();
                    g.operators.pop_back();
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
      DTensor output_tensor = op->output_tensors[0];
      DTensor input =
          g.new_input(to_dim_vector(output_tensor.num_dims, output_tensor.dim),
                      output_tensor.data_type,
                      layout::DmemRowMajor);
      assert(contains_key(computation_graph_patterns, output_tensor));
      c.algebraic_pattern.insert(
          {input, computation_graph_patterns.at(output_tensor)});
    }
  }

  find_final_patterns(c.algebraic_pattern);
  generate_next_kernel(c, g);
}

void KernelGraphGenerator::find_final_patterns(
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> const
        &input_pattern) {
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
  for (KNOperator *op : g.operators) {
    if (c.output_degree[op] == 0) {
      for (DTensor const &tensor : op->output_tensors) {
        if (num_output >= final_patterns.size() ||
            !c.algebraic_pattern.at(tensor)->subpattern_to(
                *final_patterns[num_output]) ||
            !final_patterns[num_output]->subpattern_to(
                *c.algebraic_pattern.at(tensor))) {
          return false;
        }
        ++num_output;
      }
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
