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
#include "aso/kernel/device_memory_manager.h"
#include "aso/kernel/graph.h"

namespace aso {
namespace kernel {

KNOperator::KNOperator(aso::type::KNOperatorType _type) : op_type(_type) {}

KNOperator::KNOperator(aso::type::KNOperatorType _type, DTensor const &A)
    : op_type(_type) {
  input_tensors.push_back(A);
}

KNOperator::KNOperator(aso::type::KNOperatorType _type,
                       DTensor const &A,
                       DTensor const &B)
    : op_type(_type) {
  input_tensors.push_back(A);
  input_tensors.push_back(B);
}

KNOperator::KNOperator(aso::type::KNOperatorType _type,
                       std::vector<DTensor> const &inputs)
    : op_type(_type) {
  for (auto const &i : inputs) {
    input_tensors.push_back(i);
  }
}

KNOperator::~KNOperator() {}

DTensor Graph::new_input(std::vector<int> const &dims,
                         aso::type::DataType data_type) {
  KNOperator *op = create_input_op(dims, data_type);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors[0];
}

KNOperator *Graph::create_input_op(std::vector<int> const &dims,
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

  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (dmm->offset + tensor.data_size() > dmm->total_size) {
    return nullptr;
  }
  KNInputOp *op = new KNInputOp(dims, data_type);
  return op;
}

KNInputOp::KNInputOp(std::vector<int> const &dims,
                     aso::type::DataType data_type)
    : KNOperator(aso::type::KN_INPUT_OP) {
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
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->allocate(tensor);
  output_tensors.push_back(tensor);
}

KNInputOp::~KNInputOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->free(output_tensors[0]);
}

bool KNInputOp::profile(ProfileResult &profile) {
  profile.run_time = 0.0f;
  return true;
}

KNInputOp::operator json() const {
  return json{{"op_type", op_type}, {"input_tensors", input_tensors}, {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace aso
