#include "aso/search/search.h"
#include "aso/utils/hash_utils.h"
#include "aso/utils/containers.h"
#include "aso/kernel/customized.h"

namespace aso {
namespace search {

std::vector<int> to_dim_vector(int num_dims, int* dim) {
  std::vector<int> dims;
  for (int i = 0; i < num_dims; ++i) {
    dims.push_back(dim[i]);
  }
  return dims;
}

template <typename OpType>
size_t get_operator_hash(int i, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, op);
  return h;
}

template <typename OpType>
size_t get_operator_hash(int i, int j, OpType op) {
  size_t h = 0;
  hash_combine(h, i);
  hash_combine(h, j);
  hash_combine(h, op);
  return h;
}

bool is_binary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values {
    type::TBOperatorType::TB_MATMUL_OP,
    type::TBOperatorType::TB_DIV_OP
  };
  return contains(true_values, op);
}

bool is_unary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values {
    type::TBOperatorType::TB_EXP_OP,
    type::TBOperatorType::TB_REDUCTION_0_OP,
    type::TBOperatorType::TB_REDUCTION_1_OP,
    type::TBOperatorType::TB_REDUCTION_2_OP
  };
  return contains(true_values, op);
}

bool is_binary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values {
    type::KNOperatorType::KN_MATMUL_OP
  };
  return contains(true_values, op);
}

bool is_unary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values {
    type::KNOperatorType::KN_REDUCTION_0_OP,
    type::KNOperatorType::KN_REDUCTION_1_OP,
    type::KNOperatorType::KN_REDUCTION_2_OP,
  };
  return contains(true_values, op);
}

std::shared_ptr<AlgebraicPattern> get_pattern(
  type::KNOperatorType op,
  std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern> get_pattern(
  type::TBOperatorType op,
  std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern> get_pattern(
  type::KNOperatorType op,
  std::shared_ptr<AlgebraicPattern> lhs,
  std::shared_ptr<AlgebraicPattern> rhs) {
  
  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return std::make_shared<Mul>(lhs, rhs);
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern> get_pattern(
  type::TBOperatorType op,
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

std::optional<size_t> get_rdeg(type::KNOperatorType op, int input_rdeg, std::vector<int> input_tensor_shape) {
  switch (op) {
  case type::KNOperatorType::KN_REDUCTION_0_OP:
    if (input_tensor_shape[0] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[0];
  case type::KNOperatorType::KN_REDUCTION_1_OP:
    if (input_tensor_shape[1] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[1];
  case type::KNOperatorType::KN_REDUCTION_2_OP:
    if (input_tensor_shape[2] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[2];
  default:
    return input_rdeg;
  }
}

std::optional<size_t> get_rdeg(type::TBOperatorType op, int input_rdeg, std::vector<int> input_tensor_shape) {
  switch (op) {
  case type::TBOperatorType::TB_REDUCTION_0_OP:
    if (input_tensor_shape[0] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[0];
  case type::TBOperatorType::TB_REDUCTION_1_OP:
    if (input_tensor_shape[1] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[1];
  case type::TBOperatorType::TB_REDUCTION_2_OP:
    if (input_tensor_shape[2] == 1) {
      return std::nullopt;
    }
    return input_rdeg * input_tensor_shape[2];
  default:
    return input_rdeg;
  }
}

std::vector<int> get_output_tensor_shape(threadblock::Graph const &g, int id) {
  threadblock::STensor t = g.operators[id]->output_tensors[0];
  return to_dim_vector(t.num_dims, t.dim);
}

std::vector<int> get_output_tensor_shape(kernel::Graph const &g, int id) {
  kernel::DTensor t = g.operators[id]->output_tensors[0];
  return to_dim_vector(t.num_dims, t.dim);
}

void KernelGraphGenerator::generate_threadblock_graphs(SearchContext &c, threadblock::Graph g) {
  c.tb_graph_candidates.push_back(g);

  auto try_next = [&](std::shared_ptr<AlgebraicPattern> pattern,
                      size_t reduction_degree,
                      size_t hash) {
    int new_op_id = g.operators.size();
    c.opid2pattern.emplace(new_op_id, pattern);
    c.opid2rdeg.emplace(new_op_id, reduction_degree);
    c.existing_op_hash.insert(hash);
    threadblock::Graph ng = g;
    /*
      TODO: add new operator into ng
    */
    generate_threadblock_graphs(c, ng);
    c.opid2pattern.erase(new_op_id);
    c.opid2rdeg.erase(new_op_id);
    c.existing_op_hash.erase(hash);
  };

  std::vector<type::TBOperatorType> op_to_explore {
    type::TBOperatorType::TB_MATMUL_OP,
    type::TBOperatorType::TB_REDUCTION_0_OP,
    type::TBOperatorType::TB_REDUCTION_1_OP,
    type::TBOperatorType::TB_REDUCTION_2_OP,
    type::TBOperatorType::TB_EXP_OP,
    type::TBOperatorType::TB_DIV_OP
  };

  for (type::TBOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (int i = 0; i < g.operators.size(); ++i) {
        for (int j = 0; j < g.operators.size(); ++j) {
          size_t hash = get_operator_hash(i, j, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          std::shared_ptr<AlgebraicPattern> pattern = get_pattern(op_type, c.opid2pattern.at(i), c.opid2pattern.at(j));
          if (!pattern->subpattern_to(*final_pattern)) {
            continue;
          }
          size_t rdeg = std::max(c.opid2rdeg.at(i), c.opid2rdeg.at(j));
          try_next(pattern, rdeg, hash);
        }
      }
    } else if (is_unary(op_type)) {
      for (int i = 0; i < g.operators.size(); ++i) {
        size_t hash = get_operator_hash(i, op_type);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        std::shared_ptr<AlgebraicPattern> pattern = get_pattern(op_type, c.opid2pattern.at(i));
        if (!pattern->subpattern_to(*final_pattern)) {
            continue;
        }
        std::optional<size_t> rdeg = get_rdeg(op_type, c.opid2rdeg.at(i), get_output_tensor_shape(g, i));
        if (!rdeg || rdeg.value() > max_rdeg) {
          continue;
        }
        try_next(pattern, rdeg.value(), hash);
      }
    }
  }
}

void KernelGraphGenerator::generate_next_kernel(SearchContext &c, kernel::Graph g) {
  if (verify(g)) {
    kernel_graph_candidates.push_back(g);
    return;
  }

  auto try_next = [&](std::shared_ptr<AlgebraicPattern> pattern,
                      int reduction_degree,
                      size_t hash) {
    int new_op_id = g.operators.size();
    c.opid2pattern.emplace(new_op_id, pattern);
    c.opid2rdeg.emplace(new_op_id, reduction_degree);
    c.existing_op_hash.insert(hash);
    kernel::Graph ng = g;
    /*
      TODO: add new operator into ng
    */
    generate_next_kernel(c, ng);
    c.opid2pattern.erase(new_op_id);
    c.opid2rdeg.erase(new_op_id);
    c.existing_op_hash.erase(hash);
  };

  std::vector<type::KNOperatorType> op_to_explore {
    type::KNOperatorType::KN_MATMUL_OP,
    type::KNOperatorType::KN_REDUCTION_0_OP,
    type::KNOperatorType::KN_REDUCTION_1_OP,
    type::KNOperatorType::KN_REDUCTION_2_OP,
    type::KNOperatorType::KN_CUSTOMIZED_OP
  };

  for (type::KNOperatorType op_type : op_to_explore) {
    if (is_binary(op_type)) {
      for (int i = 0; i < g.operators.size(); ++i) {
        for (int j = 0; j < g.operators.size(); ++j) {
          size_t hash = get_operator_hash(i, j, op_type);
          if (contains(c.existing_op_hash, hash)) {
            continue;
          }
          std::shared_ptr<AlgebraicPattern> pattern = get_pattern(op_type, c.opid2pattern.at(i), c.opid2pattern.at(j));
          if (!pattern->subpattern_to(*final_pattern)) {
            continue;
          }
          size_t rdeg = std::max(c.opid2rdeg.at(i), c.opid2rdeg.at(j));
          try_next(pattern, rdeg, hash);
        }
      }
    } else if (is_unary(op_type)) {
      for (int i = 0; i < g.operators.size(); ++i) {
        size_t hash = get_operator_hash(i, op_type);
        if (contains(c.existing_op_hash, hash)) {
          continue;
        }
        std::shared_ptr<AlgebraicPattern> pattern = get_pattern(op_type, c.opid2pattern.at(i));
        if (!pattern->subpattern_to(*final_pattern)) {
            continue;
        }
        std::optional<size_t> rdeg = get_rdeg(op_type, c.opid2rdeg.at(i), get_output_tensor_shape(g, i));
        if (!rdeg || rdeg.value() > max_rdeg) {
          continue;
        }
        try_next(pattern, rdeg.value(), hash);
      }
    } else if (op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      /*
        TODO: decide the values for dim and forloop_range
      */
      for (dim3 dim : {dim3(1, 1, 1)}) {
        for (int forloop_range : {1}) {
          SearchContext nc;
          threadblock::Graph tbg(dim, forloop_range);
          
          for (int i = 0; i < g.operators.size(); ++i) {
            if (c.opid2odeg.at(i) == 0) {
              /*
                TODO: generate inputs for the next kernel
              */
            }
          }

          if (tbg.operators.size() > kernel::KNCustomizedOp::Params::MAX_NUM_INPUTS) {
            continue;
          }

          generate_threadblock_graphs(nc, tbg);

          /*
            TODO: append the generated threadblock graphs to the current graph
          */
        }
      }
    }
  }
}

void KernelGraphGenerator::generate_kernel_graphs() {
  kernel::Graph g;
  SearchContext c;

  for (kernel::KNOperator *op : computation_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_INPUT_OP) {
      int opid = g.operators.size();
      kernel::DTensor output_tensor = op->output_tensors[0];
      g.new_input(to_dim_vector(output_tensor.num_dims, output_tensor.dim), output_tensor.data_type);
      c.opid2pattern.emplace(opid, std::make_shared<Var>("v_" + std::to_string(opid)));
      c.opid2rdeg.emplace(opid, 1);
      c.opid2odeg.emplace(opid, 0);
    }

    for (kernel::DTensor t : op->output_tensors) {
      max_rdeg = std::max(max_rdeg, t.size());
    }
  }

  std::vector<std::shared_ptr<AlgebraicPattern>> patterns = pattern_eval(computation_graph, c.opid2pattern);
  int final_op_id /* TODO: determine the id of the global output*/;
  final_pattern = patterns[final_op_id];

  generate_next_kernel(c, g);
}

KernelGraphGenerator::KernelGraphGenerator(kernel::Graph const &computation_graph)
  : computation_graph(computation_graph), final_pattern(nullptr), max_rdeg(0) {}

} // namespace search
} // namespace aso