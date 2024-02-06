#pragma once

#include "aso/search/search.h"

namespace aso {
namespace search {

std::vector<std::vector<int3>>
    get_input_map_cand(std::vector<DTensor> const &tensors);

std::vector<dim3> get_grid_dim_cand(std::vector<DTensor> const &tensors,
                                    std::vector<int3> const &input_map);

std::vector<dim3> get_block_dim_cand(std::vector<DTensor> const &tensors,
                                     std::vector<int3> const &input_map,
                                     dim3 grid_dim);

std::vector<std::vector<int>>
    get_forloop_dim_cand(std::vector<DTensor> const &input_tensers);

std::vector<int>
    get_forloop_range_cand(std::vector<DTensor> const &input_tensors,
                           std::vector<int3> const &input_map,
                           dim3 grid_dim,
                           dim3 block_dim,
                           std::vector<int> const &forloop_dim);

} // namespace search
} // namespace aso