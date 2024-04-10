#include "aso/search/dim_strategy.h"

namespace aso {
namespace search {

DimStrategy::DimStrategy(GeneratorConfig const &config) : config(config) {}

std::vector<dim3>
    DimStrategy::get_grid_dim_cand(std::vector<DTensor> const &tensors) {
  return config.grid_dim_to_explore;
}

std::vector<dim3>
    DimStrategy::get_block_dim_cand(std::vector<DTensor> const &tensors,
                                    dim3 grid_dim) {
  return config.block_dim_to_explore;
}

bool is_all_replicate_x(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.x != -1) {
      return false;
    }
  }
  return true;
}

bool is_all_replicate_y(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.y != -1) {
      return false;
    }
  }
  return true;
}

bool is_all_replicate_z(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.z != -1) {
      return false;
    }
  }
  return true;
}

void generate_input_map_cand(std::vector<DTensor> const &tensors,
                             dim3 grid_dim,
                             std::vector<int3> imap_to_explore,
                             std::vector<int3> cur,
                             std::vector<std::vector<int3>> &results) {
  if (cur.size() == tensors.size()) {
    if ((is_all_replicate_x(cur) && grid_dim.x > 1) ||
        (is_all_replicate_y(cur) && grid_dim.y > 1) ||
        (is_all_replicate_z(cur) && grid_dim.z > 1)) {
      return;
    }
    results.push_back(cur);
    return;
  }
  DTensor const &tensor = tensors[cur.size()];
  for (int3 input_map : imap_to_explore) {
    if (tensor.num_dims <= input_map.x || tensor.num_dims <= input_map.y ||
        tensor.num_dims <= input_map.z) {
      continue;
    }
    if ((grid_dim.x == 1 && input_map.x != -1) ||
        (grid_dim.y == 1 && input_map.y != -1) ||
        (grid_dim.z == 1 && input_map.z != -1)) {
      continue;
    }
    if ((input_map.x != -1 && tensor.dim[input_map.x] % grid_dim.x != 0) ||
        (input_map.y != -1 && tensor.dim[input_map.y] % grid_dim.y != 0) ||
        (input_map.z != -1 && tensor.dim[input_map.z] % grid_dim.z != 0)) {
      continue;
    }
    cur.push_back(input_map);
    generate_input_map_cand(tensors, grid_dim, imap_to_explore, cur, results);
    cur.pop_back();
  }
}

std::vector<std::vector<int3>>
    DimStrategy::get_input_map_cand(std::vector<DTensor> const &tensors,
                                    dim3 grid_dim) {
  std::vector<std::vector<int3>> results;
  generate_input_map_cand(
      tensors, grid_dim, config.imap_to_explore, {}, results);
  return results;
}

std::vector<int3> DimStrategy::get_output_map_cand(dim3 grid_dim) {
  std::vector<int3> results;
  for (int3 output_map : config.omap_to_explore) {
    if ((grid_dim.x == 1 && output_map.x != -1) ||
        (grid_dim.x > 1 && output_map.x == -1)) {
      continue;
    }
    if ((grid_dim.y == 1 && output_map.y != -1) ||
        (grid_dim.y > 1 && output_map.y == -1)) {
      continue;
    }
    if ((grid_dim.z == 1 && output_map.z != -1) ||
        (grid_dim.z > 1 && output_map.z == -1)) {
      continue;
    }
    results.push_back(output_map);
  }
  return results;
}

void generate_forloop_dim(std::vector<DTensor> const &input_tensors,
                          std::vector<int> fmap_to_explore,
                          std::vector<int> cur,
                          std::vector<std::vector<int>> &results) {
  if (cur.size() == input_tensors.size()) {
    results.push_back(cur);
    return;
  }

  DTensor const &tensor = input_tensors[cur.size()];
  for (int dim : fmap_to_explore) {
    if ((dim == -1) ||
        (dim != -1 && dim < tensor.num_dims && tensor.dim[dim] > 1)) {
      cur.push_back(dim);
      generate_forloop_dim(input_tensors, fmap_to_explore, cur, results);
      cur.pop_back();
    }
  }
}

std::vector<std::vector<int>> DimStrategy::get_forloop_dim_cand(
    std::vector<DTensor> const &input_tensors) {
  std::vector<std::vector<int>> results;
  generate_forloop_dim(input_tensors, config.fmap_to_explore, {}, results);
  return results;
}

std::vector<int> DimStrategy::get_forloop_range_cand(
    std::vector<DTensor> const &input_tensors,
    std::vector<int3> const &input_map,
    dim3 grid_dim,
    dim3 block_dim,
    std::vector<int> const &forloop_dim) {
  bool no_use = true;
  for (int dim : forloop_dim) {
    if (dim >= 0) {
      no_use = false;
    }
  }
  if (no_use) {
    return {1};
  }

  std::vector<int> results;

  for (int x : config.frange_to_explore) {
    bool feasible = true;
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (forloop_dim[i] == -1) {
        continue;
      }
      int dim = input_tensors[i].dim[forloop_dim[i]];
      if (input_map[i].x == forloop_dim[i]) {
        assert(dim % grid_dim.x == 0);
        dim /= grid_dim.x;
      }
      if (input_map[i].y == forloop_dim[i]) {
        assert(dim % grid_dim.y == 0);
        dim /= grid_dim.y;
      }
      if (input_map[i].z == forloop_dim[i]) {
        assert(dim % grid_dim.z == 0);
        dim /= grid_dim.z;
      }
      if (dim % x != 0) {
        feasible = false;
        break;
      }
    }
    if (feasible) {
      results.push_back(x);
    }
  }
  return results;
}

std::vector<std::vector<int>> DimStrategy::get_unary_input(int num_tensors) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    result.push_back({i});
  }
  return result;
}

std::vector<std::vector<int>> DimStrategy::get_binary_input(int num_tensors) {
  std::vector<std::vector<int>> result;
  for (int i = 0; i < num_tensors; ++i) {
    for (int j = 0; j < num_tensors; ++j) {
      result.push_back({i, j});
    }
  }
  return result;
}

std::vector<std::vector<int>> DimStrategy::get_customized_input_cand_idx(
    std::vector<DTensor> const &all_input,
    std::vector<int> const &open_tensor_idx) {

  int num_inputs = all_input.size();

  int opt_level = 0;
  if (opt_level == 0)
  {
    if (all_input.size() == 3) {
      return {{0, 1, 2}, {0, 1}};
    } else {
      return {{num_inputs - 2, num_inputs - 1}};
    }
  } else if (opt_level == 1) {
    std::vector<std::vector<int>> results {{0, 1}, {0, 1, 2}};
    if (num_inputs >= 4) {
      results.push_back({2, num_inputs - 1});
    }
    if (num_inputs >= 5) {
      results.push_back({num_inputs - 2, num_inputs - 1});
      // results.push_back({2, num_inputs - 2, num_inputs - 1});
    }
    return results;
  }

  std::vector<std::vector<int>> results;

  for (uint bitmap = 1; bitmap < (1u << open_tensor_idx.size()); ++bitmap) {
    std::vector<int> input_idx;
    for (size_t i = 0; i < open_tensor_idx.size(); ++i) {
      if (bitmap >> i & 1) {
        input_idx.push_back(open_tensor_idx[i]);
      }
    }
    if (input_idx.size() > 1 && input_idx.size() <= MAX_NUM_THREADBLOCK_INPUT) {
      results.push_back(input_idx);
    }
  }

  return results;
}

} // namespace search
} // namespace aso
