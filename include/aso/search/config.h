#pragma once

#include <vector>
#include <vector_types.h>

#include "aso/kernel/graph.h"
#include "aso/threadblock/graph.h"
#include "aso/type.h"

namespace aso {
namespace search {

using kernel::DTensor;
using kernel::KNOperator;
using threadblock::STensor;
using threadblock::TBOperator;

int const MAX_NUM_THREADBLOCK_GRAPH_OP = 7; // Outputs not counted
int const MAX_NUM_KERNEL_GRAPH_OP = 5;
int const MAX_NUM_THREADBLOCK = 2;
int const MAX_NUM_THREADBLOCK_INPUT = 3;
int const MAX_NUM_THREADBLOCK_OUTPUT = 2;

struct GeneratorConfig {
  std::vector<type::KNOperatorType> knop_to_explore;
  std::vector<type::TBOperatorType> tbop_to_explore;
  std::vector<int3> imap_to_explore;
  std::vector<int3> omap_to_explore;
  std::vector<dim3> grid_dim_to_explore;
  std::vector<dim3> block_dim_to_explore;
  std::vector<int> fmap_to_explore;
  std::vector<int> frange_to_explore;

  static GeneratorConfig get_default_config();
};

} // namespace search
} // namespace aso