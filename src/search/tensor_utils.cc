#include "aso/search/tensor_utils.h"

namespace aso {
namespace search {

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

} // namespace search
} // namespace aso