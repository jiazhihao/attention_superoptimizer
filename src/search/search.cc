#include "aso/search/search.h"
#include "aso/kernel/customized.h"
#include "aso/search/dim_strategy.h"
#include "aso/search/op_utils.h"
#include "aso/search/tensor_utils.h"
#include "aso/utils/containers.h"

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
    if (c.output_degree.at(op) == 0 &&
        op->op_type != type::TBOperatorType::TB_OUTPUT_OP) {
      return false;
    }
  }
  return true;
}

void KernelGraphGenerator::generate_threadblock_graphs(
    SearchContext<TBOperator, STensor> &c,
    threadblock::Graph g,
    std::vector<std::shared_ptr<AlgebraicPattern>> output_patterns,
    std::vector<int> output_rdegs,
    std::vector<threadblock::Graph> &result_graphs,
    std::vector<std::vector<std::shared_ptr<AlgebraicPattern>>>
        &result_output_patterns,
    std::vector<std::vector<int>> &result_output_rdegs) {

  if (is_finished_graph(c, g)) {
    result_graphs.push_back(g);
    result_output_patterns.push_back(output_patterns);
    result_output_rdegs.push_back(output_rdegs);
    return;
  }

  std::vector<type::TBOperatorType> op_to_explore{
      type::TBOperatorType::TB_MATMUL_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP,
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
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type,
                          c.algebraic_pattern[input1],
                          c.algebraic_pattern[input2]);
          if (!pattern->subpattern_to(*final_pattern)) {
            continue;
          }
          size_t rdeg = std::max(c.rdeg.at(input1), c.rdeg.at(input2));

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
          c.rdeg.insert({output, rdeg});
          c.existing_op_hash.insert(hash);
          c.output_degree[op1]++;
          c.output_degree[op2]++;
          generate_threadblock_graphs(c,
                                      ng,
                                      output_patterns,
                                      output_rdegs,
                                      result_graphs,
                                      result_output_patterns,
                                      result_output_rdegs);
          c.algebraic_pattern.erase(output);
          c.rdeg.erase(output);
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
        std::shared_ptr<AlgebraicPattern> pattern =
            get_pattern(op_type, c.algebraic_pattern.at(input));
        if (!pattern->subpattern_to(*final_pattern)) {
          continue;
        }
        size_t rdeg = get_rdeg(op_type, c.rdeg.at(input), input);
        if (rdeg > max_rdeg) {
          continue;
        }

        threadblock::Graph ng = g;
        threadblock::TBOperator *new_op = nullptr;
        switch (op_type) {
          case type::TBOperatorType::TB_EXP_OP:
            new_op = ng.create_elementunary_op(input, op_type);
            break;
          case type::TBOperatorType::TB_REDUCTION_0_OP:
          case type::TBOperatorType::TB_REDUCTION_1_OP:
          case type::TBOperatorType::TB_REDUCTION_2_OP:
            new_op = ng.create_elementunary_op(input, op_type);
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
        c.rdeg.insert({output, rdeg});
        c.existing_op_hash.insert(hash);
        c.output_degree[op]++;
        generate_threadblock_graphs(c,
                                    ng,
                                    output_patterns,
                                    output_rdegs,
                                    result_graphs,
                                    result_output_patterns,
                                    result_output_rdegs);
        c.algebraic_pattern.erase(output);
        c.rdeg.erase(output);
        c.existing_op_hash.erase(hash);
        c.output_degree[op]--;
      }
    } else if (op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      for (TBOperator *op : g.operators) {
        if (op->op_type == type::TBOperatorType::TB_INPUT_OP) {
          continue;
        }

        STensor input = op->output_tensors[0];

        threadblock::Graph ng = g;
        for (int3 output_map : {int3{0, 1}, int3{1, 0}}) {
          threadblock::TBOperator *new_op =
              ng.create_output_op(input, output_map);
          if (!new_op) {
            continue;
          }
          ng.operators.push_back(new_op);

          output_patterns.push_back(c.algebraic_pattern.at(input));
          output_rdegs.push_back(c.rdeg.at(input));
          c.output_degree[op]++;
          generate_threadblock_graphs(c,
                                      ng,
                                      output_patterns,
                                      output_rdegs,
                                      result_graphs,
                                      result_output_patterns,
                                      result_output_rdegs);
          output_patterns.pop_back();
          output_rdegs.pop_back();
          c.output_degree[op]--;
        }
      }
    }
  }
}

void KernelGraphGenerator::generate_next_kernel(
    SearchContext<KNOperator, DTensor> &c, kernel::Graph g) {
  if (verify(g)) {
    kernel_graph_candidates.push_back(g);
    return;
  }

  std::vector<type::KNOperatorType> op_to_explore{
      type::KNOperatorType::KN_MATMUL_OP,
      type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_CUSTOMIZED_OP};

  for (type::KNOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (auto op1 : g.operators) {
        for (auto op2 : g.operators) {
          for (DTensor input1 : op1->output_tensors) {
            for (DTensor input2 : op2->output_tensors) {
              if (!check_tensor_shape(op_type, input1, input2)) {
                continue;
              }
              size_t hash = get_operator_hash(input1, input2, op_type);
              if (contains(c.existing_op_hash, hash)) {
                continue;
              }
              std::shared_ptr<AlgebraicPattern> pattern =
                  get_pattern(op_type,
                              c.algebraic_pattern.at(input1),
                              c.algebraic_pattern.at(input2));
              if (!pattern->subpattern_to(*final_pattern)) {
                continue;
              }
              size_t rdeg = std::max(c.rdeg.at(input1), c.rdeg.at(input2));
              kernel::Graph ng = g;
              kernel::KNOperator *new_op = nullptr;
              switch (op_type) {
                case type::KNOperatorType::KN_MATMUL_OP:
                  new_op = ng.create_matmul_op(input1, input2);
                default:
                  assert(false && "Unsupported operator");
              }

              if (!new_op) {
                continue;
              }
              ng.operators.push_back(new_op);
              DTensor output = new_op->output_tensors[0];

              c.algebraic_pattern.insert({output, pattern});
              c.rdeg.insert({output, rdeg});
              c.existing_op_hash.insert(hash);
              c.output_degree[op1]++;
              c.output_degree[op2]++;
              generate_next_kernel(c, ng);
              c.algebraic_pattern.erase(output);
              c.rdeg.erase(output);
              c.existing_op_hash.erase(hash);
              c.output_degree[op1]--;
              c.output_degree[op2]--;
            }
          }
        }
      }
    } else if (is_unary(op_type)) {
      for (auto op : g.operators) {
        for (DTensor input : op->output_tensors) {
          if (!check_tensor_shape(op_type, input)) {
            continue;
          }
          size_t hash = get_operator_hash(input, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          std::shared_ptr<AlgebraicPattern> pattern =
              get_pattern(op_type, c.algebraic_pattern.at(input));
          if (!pattern->subpattern_to(*final_pattern)) {
            continue;
          }
          size_t rdeg = get_rdeg(op_type, c.rdeg.at(input), input);
          if (rdeg > max_rdeg) {
            continue;
          }

          kernel::Graph ng = g;
          kernel::KNOperator *new_op = nullptr;
          switch (op_type) {
            case type::KNOperatorType::KN_REDUCTION_0_OP:
            case type::KNOperatorType::KN_REDUCTION_1_OP:
            case type::KNOperatorType::KN_REDUCTION_2_OP:
            case type::KNOperatorType::KN_OUTPUT_OP:
              assert(false && "TBD");
            default:
              assert(false && "Unsupported operator");
          }

          if (!new_op) {
            continue;
          }
          ng.operators.push_back(new_op);
          DTensor output = new_op->output_tensors[0];

          c.algebraic_pattern.insert({output, pattern});
          c.rdeg.insert({output, rdeg});
          c.existing_op_hash.insert(hash);
          c.output_degree[op]++;
          generate_next_kernel(c, ng);
          c.algebraic_pattern.erase(output);
          c.rdeg.erase(output);
          c.existing_op_hash.erase(hash);
          c.output_degree[op]--;
        }
      }
    } else if (op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::vector<DTensor> input_tensors;
      std::vector<KNOperator *> input_operators;
      size_t hash = 0;

      for (auto op : g.operators) {
        if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP ||
            c.output_degree.at(op) > 0) {
          continue;
        }
        input_operators.push_back(op);
        hash_combine(hash, op);
        for (DTensor tensor : op->output_tensors) {
          input_tensors.push_back(tensor);
        }
      }

      hash_combine(hash, type::KNOperatorType::KN_CUSTOMIZED_OP);

      if (input_tensors.size() >
          threadblock::KernelParams::MAX_NUM_DMEM_INPUTS) {
        continue;
      }

      // FIXME: simplify the search space
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

                for (int i = 0; i < input_tensors.size(); ++i) {
                  DTensor tensor = input_tensors[i];
                  STensor output =
                      ng.new_input(tensor, input_map[i], forloop_dim[i]);
                  nc.algebraic_pattern.insert(
                      {output, c.algebraic_pattern.at(tensor)});
                  nc.rdeg.insert({output, c.rdeg.at(tensor)});
                }

                std::vector<threadblock::Graph> tbgs;
                std::vector<std::vector<std::shared_ptr<AlgebraicPattern>>>
                    output_patterns;
                std::vector<std::vector<int>> output_rdegs;
                generate_threadblock_graphs(
                    nc, ng, {}, {}, tbgs, output_patterns, output_rdegs);

                assert(tbgs.size() == output_patterns.size());
                assert(tbgs.size() == output_rdegs.size());

                for (int i = 0; i < tbgs.size(); ++i) {
                  threadblock::Graph const &tbg = tbgs[i];
                  kernel::Graph ng = g;
                  kernel::KNOperator *new_op =
                      ng.create_customized_op(input_tensors, tbg);
                  if (!new_op) {
                    continue;
                  }
                  ng.operators.push_back(new_op);
                  std::vector<DTensor> outputs = new_op->output_tensors;
                  assert(outputs.size() == output_patterns[i].size());
                  for (int j = 0; j < outputs.size(); ++j) {
                    c.algebraic_pattern.insert(
                        {outputs[i], output_patterns[i][j]});
                    c.rdeg.insert({outputs[i], output_rdegs[i][j]});
                    c.existing_op_hash.insert(hash);
                    for (auto op : input_operators) {
                      c.output_degree[op]++;
                    }
                    generate_next_kernel(c, ng);
                    c.algebraic_pattern.erase(outputs[i]);
                    c.rdeg.erase(outputs[i]);
                    c.existing_op_hash.erase(hash);
                    for (auto op : input_operators) {
                      c.output_degree[op]--;
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
}

void KernelGraphGenerator::generate_kernel_graphs() {
  kernel::Graph g;
  SearchContext<KNOperator, DTensor> c;

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      int opid = g.operators.size();
      DTensor output_tensor = op->output_tensors[0];
      DTensor input =
          g.new_input(to_dim_vector(output_tensor.num_dims, output_tensor.dim),
                      output_tensor.data_type);
      c.algebraic_pattern.insert(
          {input, std::make_shared<Var>("v_" + std::to_string(opid))});
      c.rdeg.insert({input, 1});
    }

    for (DTensor t : op->output_tensors) {
      max_rdeg = std::max(max_rdeg, t.data_size());
    }
  }

  std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> patterns =
      pattern_eval(computation_graph, c.algebraic_pattern);
  for (auto op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      final_pattern = patterns.at(op->output_tensors[0]);
      break;
    }
  }

  generate_next_kernel(c, g);
}

KernelGraphGenerator::KernelGraphGenerator(
    kernel::Graph const &computation_graph,
    size_t device_mem_size,
    size_t shared_mem_size)
    : computation_graph(computation_graph), device_mem_size(device_mem_size),
      shared_mem_size(shared_mem_size), final_pattern(nullptr), max_rdeg(0) {}

std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> pattern_eval(
    kernel::Graph const &g,
    std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> const
        &input_pattern) {
  std::unordered_map<DTensor, std::shared_ptr<AlgebraicPattern>> patterns;
  // Assume operators are in topological order
  for (KNOperator *op : g.operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP:
        patterns.insert(
            {op->output_tensors[0], input_pattern.at(op->output_tensors[0])});
        break;
      case type::KNOperatorType::KN_OUTPUT_OP:
        patterns.insert(
            {op->output_tensors[0], patterns.at(op->input_tensors[0])});
        break;
      case type::KNOperatorType::KN_MATMUL_OP:
        patterns.insert(
            {op->output_tensors[0],
             std::make_shared<Mul>(patterns.at(op->input_tensors[0]),
                                   patterns.at(op->input_tensors[1]))});
        break;
      case type::KNOperatorType::KN_REDUCTION_0_OP:
      case type::KNOperatorType::KN_REDUCTION_1_OP:
      case type::KNOperatorType::KN_REDUCTION_2_OP:
        patterns.insert(
            {op->output_tensors[0], patterns.at(op->input_tensors[0])});
        break;
      default:
        assert(false && "Unsupported computation graph operator");
    }
  }
}

} // namespace search
} // namespace aso
