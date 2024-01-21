/* Copyright 2023 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "aso/kernel/operator.h"
#include "aso/kernel/graph.h"
#include "aso/kernel/operator_factory.h"

namespace aso {
namespace kernel {

Operator::Operator(void) {}

Operator::Operator(DTensor const &A) {
  input_tensors.push_back(A);
}

Operator::Operator(DTensor const &A, DTensor const &B) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

Operator::~Operator() {}

DTensor Graph::new_input(std::vector<int> const &dims,
                         aso::type::DataType data_type) {
  OperatorFactory *operator_factory = OperatorFactory::get_instance();
  Operator *op = operator_factory->create_input(dims, data_type);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

Operator *OperatorFactory::create_input(std::vector<int> const &dims,
                                        aso::type::DataType data_type) {
  InputKNOp *op = new InputKNOp(dims, data_type);
  return op;
}

InputKNOp::InputKNOp(std::vector<int> const &dims,
                     aso::type::DataType data_type) {
  DTensor tensor;
  tensor.num_dims = dims.size();
  for (int i = tensor.num_dims - 1; i >= 0; i--) {
    tensor.dim[i] = dims[i];
    tensor.stride[i] = (i == tensor.num_dims - 1)
                           ? 1
                           : tensor.stride[i + 1] * tensor.dim[i + 1];
  }
  tensor.data_type = data_type;
  tensor.owner_op = this;
  tensor.owner_ts_idx = 0;
  OperatorFactory *operator_factory = OperatorFactory::get_instance();
  tensor.data_ptr = operator_factory->allocate(tensor.size());
  output_tensors.push_back(tensor);
}

aso::type::OperatorType InputKNOp::operator_type(void) const {
  return aso::type::KN_INPUT_OP;
}

} // namespace kernel
} // namespace aso
