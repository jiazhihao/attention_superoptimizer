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

#include "aso/kernel/graph.h"
#include "aso/kernel/customized.h"
#include "aso/utils/hash_utils.h"

#include <iostream>

namespace aso {
namespace kernel {

std::string input_name(int i) {
  return "input"+std::to_string(i);
}

std::string tensor_dims(DTensor const &tensor) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < tensor.num_dims; i++) {
    ss << tensor.dim[i];
    if (i != tensor.num_dims-1) {
      ss << ", ";
    }
  }
  ss << ")";
  return ss.str();
}

std::string generate_kernel_code(aso::threadblock::Graph const &bgraph) {
  std::stringstream header;
  std::stringstream main;
  header << "@triton.jit\n";


  return main.str();
}

std::string Graph::generate_triton_program() {
  using namespace std;
  stringstream header;
  vector<std::string> kernels;
  stringstream launcher;
  stringstream main_program;
  main_program << "def main():\n";
  header << "import triton\nimport torch\nimport triton.language as tl\n";
  int num_inputs = 0;
  for (KNOperator *const op : this->operators) {
    switch (op->op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        main_program << "\t" << input_name(num_inputs++) << " = torch.randn("
           << tensor_dims(op->output_tensors[0])
           << ", dtype=torch.float16, device=\"cuda\", requires_grad=False)\n";
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        
      }
      default: {
        //assert(false && "Cannot tritonize this operator");
      }
    }
  }
  stringstream output;
  output << header.str();
  for (const auto & k : kernels) {
    output << k;
  }
  output << launcher.str() << main_program.str();
  return output.str();
}

} // namespace kernel
} // namespace aso
