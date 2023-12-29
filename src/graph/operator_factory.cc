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

#include "aso/graph/operator_factory.h"
#include "aso/graph/kernel_graph.h"

namespace aso {
namespace graph {

static OperatorFactory *operator_factory_singleton = nullptr;

OperatorFactory::OperatorFactory() {}

KernelGraph::KernelGraph() {
  if (operator_factory_singleton == nullptr) {
    operator_factory_singleton = new OperatorFactory();
  }
  operator_factory = operator_factory_singleton;
}

} // namespace graph
} // namespace aso
