#pragma once

#include "aso/search/search.h"

namespace aso {
namespace search {

size_t get_rdeg(type::KNOperatorType op,
                int input_rdeg,
                DTensor const &input_tensor);

size_t get_rdeg(type::TBOperatorType op,
                int input_rdeg,
                STensor const &input_tensor);

bool check_tensor_shape(type::TBOperatorType op, STensor const &input);

bool check_tensor_shape(type::TBOperatorType op,
                        STensor const &input1,
                        STensor const &input2);

bool check_tensor_shape(type::KNOperatorType op, DTensor const &input);

bool check_tensor_shape(type::KNOperatorType op,
                        DTensor const &input1,
                        DTensor const &input2);

} // namespace search
} // namespace aso