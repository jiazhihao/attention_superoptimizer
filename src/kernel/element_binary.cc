/* Copyright 2023-2024 CMU
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

#include "aso/kernel/element_binary.h"
#include "aso/kernel/device_memory_manager.h"
#include "aso/kernel/graph.h"
#include "aso/layout.h"
#include "aso/utils/hash_utils.h"
#include <cassert>

namespace aso {
namespace kernel {

DTensor Graph::div(DTensor const &input1, DTensor const &input2) {
  KNOperator *op =
      create_elementbinary_op(input1, input2, aso::type::KN_DIV_OP);
  assert(op != nullptr);
  operators.push_back(op);
  assert(op->output_tensors.size() == 1);
  DTensor output = op->output_tensors[0];
  return output;
}

KNOperator *Graph::create_elementbinary_op(DTensor const &input1,
                                           DTensor const &input2,
                                           aso::type::KNOperatorType type) {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  if (input1.num_dims != input2.num_dims) {
    return nullptr;
  }
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i] && input1.dim[i] > 1 &&
        input2.dim[i] > 1) {
      return nullptr;
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }

  if (dmm->offset + output.data_size() > dmm->total_size) {
    return nullptr;
  }

  KNElementBinaryOp *op = new KNElementBinaryOp(input1, input2, type);
  return op;
}

KNElementBinaryOp::KNElementBinaryOp(DTensor const &input1,
                                     DTensor const &input2,
                                     aso::type::KNOperatorType type)
    : aso::kernel::KNOperator(type, input1, input2) {
  assert(input1.num_dims == input2.num_dims);
  for (int i = 0; i < input1.num_dims; i++) {
    if (input1.dim[i] != input2.dim[i]) {
      assert(input1.dim[i] == 1 || input2.dim[i] == 1);
    }
  }
  DTensor output = input1;
  for (int i = 0; i < output.num_dims; i++) {
    output.dim[i] = std::max(input1.dim[i], input2.dim[i]);
  }
  output.owner_op = this;
  output.owner_ts_idx = 0;
  output.guid = DTensor::next_guid++;
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  dmm->allocate(output);
  assert(output_tensors.size() == 0);
  output_tensors.push_back(output);
}

KNElementBinaryOp::~KNElementBinaryOp() {
  DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    dmm->free(output_tensors[i]);
  }
}

KNElementBinaryOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors}};
}

} // namespace kernel
} // namespace aso
