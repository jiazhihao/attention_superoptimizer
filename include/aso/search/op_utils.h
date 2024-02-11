#pragma once

#include "aso/search/search.h"
#include "aso/utils/hash_utils.h"

namespace aso {
namespace search {

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

bool is_binary(type::TBOperatorType op);
bool is_unary(type::TBOperatorType op);
bool is_binary(type::KNOperatorType op);
bool is_unary(type::KNOperatorType op);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &tensor,
                std::shared_ptr<AlgebraicPattern> opd);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::KNOperatorType op,
                DTensor const &input1,
                DTensor const &input2,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs);
std::shared_ptr<AlgebraicPattern>
    get_pattern(type::TBOperatorType op,
                STensor const &input1,
                STensor const &input2,
                std::shared_ptr<AlgebraicPattern> lhs,
                std::shared_ptr<AlgebraicPattern> rhs);

} // namespace search
} // namespace aso