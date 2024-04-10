#include "aso/search/op_utils.h"
#include "aso/utils/containers.h"

namespace aso {
namespace search {

bool is_binary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_ADD_OP,
      type::TBOperatorType::TB_MATMUL_OP,
      type::TBOperatorType::TB_DIV_OP};
  return contains(true_values, op);
}

bool is_unary(type::TBOperatorType op) {
  std::unordered_set<type::TBOperatorType> true_values{
      type::TBOperatorType::TB_EXP_OP,
      type::TBOperatorType::TB_REDUCTION_0_OP,
      type::TBOperatorType::TB_REDUCTION_1_OP,
      type::TBOperatorType::TB_REDUCTION_2_OP,
      type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP,
      type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP,
      type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP,
  };
  return contains(true_values, op);
}

bool is_binary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_ADD_OP,
      type::KNOperatorType::KN_MATMUL_OP,
      type::KNOperatorType::KN_DIV_OP};
  return contains(true_values, op);
}

bool is_unary(type::KNOperatorType op) {
  std::unordered_set<type::KNOperatorType> true_values{
      type::KNOperatorType::KN_REDUCTION_0_OP,
      type::KNOperatorType::KN_REDUCTION_1_OP,
      type::KNOperatorType::KN_REDUCTION_2_OP,
      type::KNOperatorType::KN_EXP_OP,
  };
  return contains(true_values, op);
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return std::make_shared<Red>(tensor.dim[0], opd);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      if (tensor.num_dims <= 1) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[1], opd);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      if (tensor.num_dims <= 2) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[2], opd);
    case type::KNOperatorType::KN_EXP_OP:
      return std::make_shared<Exp>(opd);
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd) {
  // Retrieve reduction_dimx from threadblock graph
  assert(tensor.owner_op != nullptr);
  assert(tensor.owner_op->bgraph != nullptr);
  int reduction_dimx = tensor.owner_op->bgraph->reduction_dimx;
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      return std::make_shared<Red>(tensor.dim[0], opd);
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      if (tensor.num_dims <= 1) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[1], opd);
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      if (tensor.num_dims <= 2) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[2], opd);
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
      if (tensor.dim[0] <= reduction_dimx) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[0] / reduction_dimx, opd);
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
      if (tensor.num_dims <= 1 || tensor.dim[1] <= reduction_dimx) {
        return nullptr;
      }
      return std::make_shared<Red>(tensor.dim[1] / reduction_dimx, opd);
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP:
      if (tensor.num_dims <= 2 || tensor.dim[2] <= reduction_dimx) {
        return nullptr;
      }
     return std::make_shared<Red>(tensor.dim[2] / reduction_dimx, opd);
    case type::TBOperatorType::TB_OUTPUT_OP:
      return opd;
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &tensor_l,
                DTensor const &tensor_r,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs) {

  switch (op) {
    case type::KNOperatorType::KN_MATMUL_OP:
      if (tensor_l.dim[tensor_l.num_dims - 1] > 1) {
        return std::make_shared<Red>(tensor_l.dim[tensor_l.num_dims - 1],
                                     std::make_shared<Mul>(lhs, rhs));
      } else {
        return std::make_shared<Mul>(lhs, rhs);
      }
    case type::KNOperatorType::KN_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::KNOperatorType::KN_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &tensor_l,
                STensor const &tensor_r,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs) {

  switch (op) {
    case type::TBOperatorType::TB_MATMUL_OP: 
      if (tensor_l.dim[tensor_l.num_dims - 1] > 1) {
        return std::make_shared<Red>(tensor_l.dim[tensor_l.num_dims - 1],
                                     std::make_shared<Mul>(lhs, rhs));
      } else {
        return std::make_shared<Mul>(lhs, rhs);
      }
    case type::TBOperatorType::TB_ADD_OP:
      return std::make_shared<Add>(lhs, rhs);
    case type::TBOperatorType::TB_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    default:
      assert(false);
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input) {
  switch (type) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return g.create_reduction_op(input, 0, 1);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      return g.create_reduction_op(input, 1, 1);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return g.create_reduction_op(input, 2, 1);
    case type::KNOperatorType::KN_EXP_OP:
      return g.create_elementunary_op(input, type);
    default:
      assert(false && "Unsupported operator");
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      DTensor const &input1,
                      DTensor const &input2) {
  switch (type) {
    case type::KNOperatorType::KN_MATMUL_OP:
      return g.create_matmul_op(input1, input2);
    case type::KNOperatorType::KN_DIV_OP:
      return g.create_elementbinary_op(input1, input2, type);
    default:
      assert(false && "Unsupported operator");
  }
}

KNOperator *create_op(kernel::Graph &g,
                      type::KNOperatorType type,
                      std::vector<DTensor> const &inputs) {
  if (inputs.size() == 1) {
    return create_op(g, type, inputs[0]);
  }
  if (inputs.size() == 2) {
    return create_op(g, type, inputs[0], inputs[1]);
  }
  return nullptr;
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input) {
  switch (type) {
    case type::TBOperatorType::TB_EXP_OP:
      return g.create_elementunary_op(input, type);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP: {
      int dim = (int)type - (int)type::TBOperatorType::TB_REDUCTION_0_OP;
      if (input.num_dims <= dim || (input.num_dims > dim && input.dim[dim] == 1)) {
        return nullptr;
      }
      return g.create_reduction_op(input, dim);
    }
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      int dim = (int)type - (int)type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP;
      if (input.num_dims <= dim) {
        return nullptr;
      }
      if ((input.dim[dim] <= g.reduction_dimx) || (input.dim[dim] % g.reduction_dimx != 0)) {
        return nullptr;
      }
      return g.create_reduction_to_dimx_op(input, dim);
    }
    default:
      assert(false && "Unsupported operator");
  }
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      STensor const &input1,
                      STensor const &input2) {
  switch (type) {
    case type::TBOperatorType::TB_MATMUL_OP:
      return g.create_matmul_op(input1, input2);
    case type::TBOperatorType::TB_DIV_OP:
      return g.create_elementbinary_op(input1, input2, type);
    default:
      assert(false && "Unsupported operator");
  }
}

TBOperator *create_op(threadblock::Graph &g,
                      type::TBOperatorType type,
                      std::vector<STensor> const &inputs) {
  if (inputs.size() == 1) {
    return create_op(g, type, inputs[0]);
  }
  if (inputs.size() == 2) {
    return create_op(g, type, inputs[0], inputs[1]);
  }
  return nullptr;
}

} // namespace search
} // namespace aso
