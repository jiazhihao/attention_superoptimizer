#include "aso/search/op_utils.h"
#include "aso/utils/containers.h"

namespace aso {
namespace search {

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
                DTensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::KNOperatorType::KN_REDUCTION_0_OP:
      return std::make_shared<Red>(tensor.dim[0], opd);
    case type::KNOperatorType::KN_REDUCTION_1_OP:
      return std::make_shared<Red>(tensor.dim[1], opd);
    case type::KNOperatorType::KN_REDUCTION_2_OP:
      return std::make_shared<Red>(tensor.dim[2], opd);
    default:
      assert(false);
  }
}

std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd) {
  switch (op) {
    case type::TBOperatorType::TB_EXP_OP:
      return std::make_shared<Exp>(opd);
    case type::TBOperatorType::TB_REDUCTION_0_OP:
      assert(tensor.dim[0] > 1);
      return std::make_shared<Red>(tensor.dim[0], opd);
    case type::TBOperatorType::TB_REDUCTION_1_OP:
      assert(tensor.dim[1] > 1);
      return std::make_shared<Red>(tensor.dim[1], opd);
    case type::TBOperatorType::TB_REDUCTION_2_OP:
      assert(tensor.dim[2] > 1);
      return std::make_shared<Red>(tensor.dim[2], opd);
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
      if (tensor_l.dim[2] > 1) {
        return std::make_shared<Red>(tensor_l.dim[2],
                                     std::make_shared<Mul>(lhs, rhs));
      } else {
        return std::make_shared<Mul>(lhs, rhs);
      }
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
      if (tensor_l.dim[2] > 1) {
        return std::make_shared<Red>(tensor_l.dim[2],
                                     std::make_shared<Mul>(lhs, rhs));
      } else {
        return std::make_shared<Mul>(lhs, rhs);
      }
    case type::TBOperatorType::TB_DIV_OP:
      return std::make_shared<Div>(lhs, rhs);
    default:
      assert(false);
  }
}

} // namespace search
} // namespace aso