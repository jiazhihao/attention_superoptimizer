#include "aso/search/dim_strategy.h"

namespace aso {
namespace search {

unsigned int get_num_threadblock(dim3 const &grid_dim) {
  return grid_dim.x * grid_dim.y * grid_dim.z;
}

std::vector<dim3> get_grid_dim_cand(std::vector<DTensor> const &tensors,
                                    std::vector<int3> const &input_map) {
  std::vector<dim3> results;
  for (unsigned int x = 16; x <= 16; x *= 2) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (input_map[i].x != -1 && tensors[i].num_dims > input_map[i].x &&
          tensors[i].dim[input_map[i].x] % x != 0) {
        return results;
      }
    }
    for (unsigned int y = 8; y <= 8; y *= 2) {
      bool feasible = true;
      for (size_t i = 0; i < tensors.size(); ++i) {
        if (input_map[i].y != -1 && tensors[i].num_dims > input_map[i].y &&
            tensors[i].dim[input_map[i].y] % y != 0) {
          feasible = false;
          break;
        }
      }
      if (!feasible) {
        break;
      }
      results.push_back(dim3{x, y, 1});
    }
  }
  return results;
}

std::vector<dim3> get_block_dim_cand(std::vector<DTensor> const &tensors,
                                     std::vector<int3> const &input_map,
                                     dim3 grid_dim) {
  return std::vector<dim3>{{128, 1, 1}};
  // std::vector<dim3> results;
  // for (unsigned int x : {32, 64, 128}) {
  //   bool feasible = true;
  //   for (size_t i = 0; i < tensors.size(); ++i) {
  //     if (input_map[i].x < tensors[i].num_dims &&
  //         (tensors[i].dim[input_map[i].x] / grid_dim.x) % x != 0) {
  //       feasible = false;
  //       break;
  //     }
  //   }
  //   if (feasible) {
  //     std::cerr << "block_dim cand:" << dim3{x, 1, 1} << std::endl;
  //     results.push_back(dim3{x, 1, 1});
  //   }
  // }
  // return results;
}

bool is_all_replicate(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.x != -1 || input_map.y != -1 || input_map.z != -1) {
      return false;
    }
  }
  return true;
}

void generate_input_map_cand(std::vector<DTensor> const &tensors,
                             int3 input_map_pattern,
                             std::vector<int3> cur,
                             std::vector<std::vector<int3>> &results) {
  if (cur.size() == tensors.size()) {
    if (!is_all_replicate(cur)) {
      results.push_back(cur);
    }
    return;
  }
  DTensor const &tensor = tensors[cur.size()];
  for (unsigned int bitmap = 0; bitmap < (1 << 3); ++bitmap) {
    int3 input_map{-1, -1, -1};
    if ((bitmap & 1) && input_map.x < tensor.num_dims) {
      input_map.x = input_map_pattern.x;
    }
    if ((bitmap >> 1 & 1) && input_map.y < tensor.num_dims) {
      input_map.y = input_map_pattern.y;
    }
    if ((bitmap >> 2 & 1) && input_map.z < tensor.num_dims) {
      input_map.z = input_map_pattern.z;
    }
    cur.push_back(input_map);
    generate_input_map_cand(tensors, input_map, cur, results);
    cur.pop_back();
  }
}

std::vector<std::vector<int3>>
    get_input_map_cand(std::vector<DTensor> const &tensors) {
  // To save time to generate example
  // if (tensors.size() == 3) {
  //   return {{{0, -1, -1}, {0, 2, -1}, {0, 1, -1}}};
  // }
  // if (tensors.size() == 2) {
  //   return {{{0, -1, -1}, {0, 2, -1}}};
  // }
  std::vector<std::vector<int3>> results;
  // Assume two-dimentional inputs
  // TODO: There are invalid input maps, how to prune them out?
  for (int3 input_map_pattern : {int3{0, 2, -1} /*, int3{1, 0, -1}*/}) {
    generate_input_map_cand(tensors, input_map_pattern, {}, results);
  }
  return results;
}

void generate_forloop_dim(std::vector<DTensor> const &input_tensors,
                          std::vector<int> cur,
                          std::vector<std::vector<int>> &results) {
  if (cur.size() == input_tensors.size()) {
    bool is_none = true;
    for (int dim : cur) {
      if (dim != -1) {
        is_none = false;
        break;
      }
    }
    if (!is_none) {
      results.push_back(cur);
    }
    return;
  }

  for (int dim = -1; dim <= 2; ++dim) {
    DTensor const &tensor = input_tensors[cur.size()];
    if (dim < tensor.num_dims && tensor.dim[dim] > 1) {
      cur.push_back(dim);
      generate_forloop_dim(input_tensors, cur, results);
      cur.pop_back();
    }
  }
}

std::vector<std::vector<int>>
    get_forloop_dim_cand(std::vector<DTensor> const &input_tensors) {
  // To save time to generate example
  // if (input_tensors.size() == 2) {
  //   return {{2, 1}};
  // }
  // if (input_tensors.size() == 3) {
  //   return {{-1, 2, 1}};
  // }

  std::vector<std::vector<int>> results;
  generate_forloop_dim(input_tensors, {}, results);
  return results;
}

std::vector<int>
    get_forloop_range_cand(std::vector<DTensor> const &input_tensors,
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

  for (int x = 4; x <= 4; x *= 2) {
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
    } else {
      return results;
    }
  }
  return results;
}

} // namespace search
} // namespace aso
