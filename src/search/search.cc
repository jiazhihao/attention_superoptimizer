#include "aso/search/search.h"
#include "aso/kernel/customized.h"
#include "aso/utils/containers.h"
#include "aso/utils/hash_utils.h"

namespace aso {
namespace search {

std::vector<int> to_dim_vector(int num_dims, int *dim) {
  std::vector<int> dims;
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(dim[i]);
  }
  return dims;
}

template <typename Op, typename OpType>
size_t get_operator_hash(Op i, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, op);
  return h;
}

template <typename Op, typename OpType>
size_t get_operator_hash(Op i, Op j, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, j);
  hash_combine(h, op);
  return h;
}

bool is_binary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_MATMUL_OP, type::TBOperatorType::TB_DIV_OP};
  return contains(true_values, op);
}

bool is_unary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_EXP_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP};
  return contains(true_values, op);
}

bool is_binary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_MATMUL_OP};
  return contains(true_values, op);
}

bool is_unary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
  };
  return contains(true_values, op);
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return opd;
    case type::KNOperatorType::KN_OUTPUT_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      return opd;
    case type::TBOperatorType::TB_OUTPUT_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs) {

  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs) {

  switch (op) {
    case type::TBOperatorType::TB_MATMUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    case type::TBOperatorType::TB_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    default:
      assert(false);
  }
}

size_t get_rdeg(type::KNOperatorType op,
                int input_rdeg,
                DTensor const &input_tensor) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return input_rdeg * input_tensor.dim[0];
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      return input_rdeg * input_tensor.dim[1];
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return input_rdeg * input_tensor.dim[2];
    default:
      return input_rdeg;
  }
}

size_t get_rdeg(type::TBOperatorType op,
                int input_rdeg,
                STensor const &input_tensor) {
  switch (op) {
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      return input_rdeg * input_tensor.dim[0];
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      return input_rdeg * input_tensor.dim[1];
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      return input_rdeg * input_tensor.dim[2];
    default:
      return input_rdeg;
  }
}

bool check_tensor_shape(type::TBOperatorType op, STensor const &input) {
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return true;
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      return input.dim[0] != 1;
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      return input.num_dims >= 2 && input.dim[1] != 1;
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      return input.num_dims >= 3 && input.dim[2] != 1;
    default:
      assert(false && "Unsupported Operator");
  }
}

bool check_tensor_shape(type::TBOperatorType op,
                        STensor const &input1,
                        STensor const &input2) {
  switch (op) {
    case type::TBOperatorType::TB_MATMUL_OP:
      return input1.num_dims == 2 && input2.num_dims == 2 &&
             input1.dim[1] == input2.dim[0];
    case type::TBOperatorType::TB_DIV_OP:
      return input1.num_dims == 2 && input2.num_dims == 2 && input2.dim[1] == 1;
    default:
      assert(false && "Unsupported Operator");
  }
}

bool check_tensor_shape(type::KNOperatorType op, DTensor const &input) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return input.dim[0] != 1;
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      return input.num_dims >= 2 && input.dim[1] != 1;
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return input.num_dims >= 3 && input.dim[2] != 1;
    default:
      assert(false && "Unsupported Operator");
  }
}

bool check_tensor_shape(type::KNOperatorType op,
                        DTensor const &input1,
                        DTensor const &input2) {
  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return input1.num_dims == 2 && input2.num_dims == 2 &&
             input1.dim[1] == input2.dim[0];
    default:
      assert(false && "Unsupported Operator");
  }
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

          STensor output;
          threadblock::Graph ng = g;
          switch (op_type) {
            case type::TBOperatorType::TB_MATMUL_OP:
              output = ng.matmul(input1, input2);
              break;
            case type::TBOperatorType::TB_DIV_OP:
              assert(false && "TBD");
              break;
            default:
              assert(false && "Unsupported operator");
          }

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

        STensor output;
        threadblock::Graph ng = g;
        switch (op_type) {
          case type::TBOperatorType::TB_EXP_OP:
            output = ng.exp(input);
            break;
          case type::TBOperatorType::TB_REDUCTION_0_OP:
            output = ng.reduction(input, 0);
            break;
          case type::TBOperatorType::TB_REDUCTION_1_OP:
            output = ng.reduction(input, 1);
            break;
          case type::TBOperatorType::TB_REDUCTION_2_OP:
            output = ng.reduction(input, 2);
            break;
          default:
            assert(false && "Unsupported operator");
        }

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
        int3 output_map; /* TODO: determine the output_map */
        ng.new_output(input, output_map);

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
              DTensor output;
              switch (op_type) {
                case type::KNOperatorType::KN_MATMUL_OP:
                  output = ng.matmul(input1, input2);
                default:
                  assert(false && "Unsupported operator");
              }

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
          DTensor output;
          switch (op_type) {
            case type::KNOperatorType::KN_REDUCTION_0_OP:
            case type::KNOperatorType::KN_REDUCTION_1_OP:
            case type::KNOperatorType::KN_REDUCTION_2_OP:
            case type::KNOperatorType::KN_OUTPUT_OP:
              assert(false && "TBD");
            default:
              assert(false && "Unsupported operator");
          }

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
          kernel::KNCustomizedOp::Params::MAX_NUM_INPUTS) {
        continue;
      }

      /*
        TODO: decide the values for dim and forloop_range
      */
      for (dim3 block_dim : {dim3(1, 1, 1)}) {
        dim3 grid_dim;
        for (int forloop_range : {1}) {

          // TODO: enumerate input_map and forloop_dim
          int forloop_dim = 0;
          int3 input_map{0, 1, 2};

          SearchContext<TBOperator, STensor> nc;
          threadblock::Graph ng(grid_dim, block_dim, forloop_range);

          for (DTensor tensor : input_tensors) {
            STensor output = ng.new_input(tensor, input_map, forloop_dim);
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
            std::vector<DTensor> outputs = ng.customized(input_tensors, tbg);
            assert(outputs.size() == output_patterns[i].size());
            for (int j = 0; j < outputs.size(); ++j) {
              c.algebraic_pattern.insert({outputs[i], output_patterns[i][j]});
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
      max_rdeg = std::max(max_rdeg, t.size());
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
    kernel::Graph const &computation_graph)
    : computation_graph(computation_graph), final_pattern(nullptr),
      max_rdeg(0) {}

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