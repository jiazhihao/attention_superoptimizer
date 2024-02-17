#include "aso/search/dim_strategy.h"

namespace aso {
namespace search {

unsigned int get_num_threadblock(dim3 const &grid_dim) {
  return grid_dim.x * grid_dim.y * grid_dim.z;
}

std::vector<dim3> get_grid_dim_cand(std::vector<DTensor> const &tensors,
                                    std::vector<int3> const &input_map) {
  std::vector<dim3> results;
  for (int x = 1;; x *= 2) {
    for (size_t i = 0; i < tensors.size(); ++i) {
      if (tensors[i].dim[input_map[i].x] % x != 0) {
        return results;
      }
    }
    for (int y = 1;; y *= 2) {
      bool feasible = true;
      for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].num_dims > 1 &&
            tensors[i].dim[input_map[i].y] % y != 0) {
          feasible = false;
          break;
        }
      }
      if (!feasible) {
        break;
      }
      results.push_back(dim3(x, y));
    }
  }
}

std::vector<dim3> get_block_dim_cand(std::vector<DTensor> const &tensors,
                                     std::vector<int3> const &input_map,
                                     dim3 grid_dim) {
  std::vector<dim3> results;
  for (unsigned int x : {32, 64, 128}) {
    for (DTensor const &tensor : tensors) {
      assert(tensor.data_size() % get_num_threadblock(grid_dim) == 0);
      int block_size = tensor.data_size() / get_num_threadblock(grid_dim);
      if (block_size % x == 0) {
        results.push_back(dim3{x, 1, 1});
      }
    }
  }
  return results;
}

bool is_all_replicate(std::vector<int3> const &input_maps) {
  for (int3 const &input_map : input_maps) {
    if (input_map.x != -1 || input_map.y != -1 || input_map.z != -1) {
      return false;
    }
  }
  return true;
}

void generate_input_map_cand(int num_tensors,
                             int3 input_map,
                             std::vector<int3> cur,
                             std::vector<std::vector<int3>> &results) {
  if (static_cast<int>(cur.size()) == num_tensors) {
    if (!is_all_replicate(cur)) {
      results.push_back(cur);
    }
    return;
  }
  cur.push_back(int3{-1, -1, -1});
  generate_input_map_cand(num_tensors, input_map, cur, results);
  cur.pop_back();
  cur.push_back(int3{input_map.x, -1, -1});
  generate_input_map_cand(num_tensors, input_map, cur, results);
  cur.pop_back();
  cur.push_back(int3{-1, input_map.y, -1});
  generate_input_map_cand(num_tensors, input_map, cur, results);
  cur.pop_back();
  cur.push_back(int3{input_map.x, input_map.y, -1});
  generate_input_map_cand(num_tensors, input_map, cur, results);
  cur.pop_back();
}

std::vector<std::vector<int3>>
    get_input_map_cand(std::vector<DTensor> const &tensors) {
  std::vector<std::vector<int3>> results;
  // Assume two-dimentional inputs
  // TODO: There are invalid input maps, how to prune them out?
  for (int3 input_map : {int3{0, 1, -1}, int3{1, 0, -1}}) {
    generate_input_map_cand(tensors.size(), input_map, {}, results);
  }
  return results;
}

void generate_forloop_dim(int num_tensors,
                          std::vector<int> cur,
                          std::vector<std::vector<int>> &results) {
  if (static_cast<int>(cur.size()) == num_tensors) {
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
    cur.push_back(dim);
    generate_forloop_dim(num_tensors, cur, results);
    cur.pop_back();
  }
}

std::vector<std::vector<int>>
    get_forloop_dim_cand(std::vector<DTensor> const &input_tensers) {
  std::vector<std::vector<int>> results;
  generate_forloop_dim(input_tensers.size(), {}, results);
  return results;
}

std::vector<int>
    get_forloop_range_cand(std::vector<DTensor> const &input_tensors,
                           std::vector<int3> const &input_map,
                           dim3 grid_dim,
                           dim3 block_dim,
                           std::vector<int> const &forloop_dim) {
  std::vector<int> results;

  for (int x = 1;; x *= 2) {
    for (size_t i = 0; i < input_tensors.size(); ++i) {
      if (forloop_dim[i] == -1) {
        continue;
      }
      int dim;
      switch (forloop_dim[i]) {
        case 0:
          dim = input_tensors[i].dim[input_map[i].x];
          assert(dim % grid_dim.x == 0);
          dim /= grid_dim.x;
          assert(dim % block_dim.x == 0);
          dim /= block_dim.x;
          break;
        case 1:
          dim = input_tensors[i].dim[input_map[i].y];
          assert(dim % grid_dim.y == 0);
          dim /= grid_dim.y;
          assert(dim % block_dim.y == 0);
          dim /= block_dim.y;
          break;
        case 2:
          dim = input_tensors[i].dim[input_map[i].z];
          assert(dim % grid_dim.z == 0);
          dim /= grid_dim.z;
          assert(dim % block_dim.z == 0);
          dim /= block_dim.z;
          break;
        default:
          assert(false);
      }
      if (dim % x == 0) {
        results.push_back(x);
      } else {
        return results;
      }
    }
  }
}

} // namespace search
} // namespace aso
