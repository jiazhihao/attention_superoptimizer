#include "aso/threadblock/graph.h"
#include "cutlass/cutlass.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/tensor_view_io.h"
#include <random>

// All functions in this file is for convenience and assumes there is only 1 threadblock.

namespace aso {
namespace threadblock {

inline Graph create_single_graph(unsigned int num_threads) {
  return Graph({1, 1, 1}, {num_threads, 1, 1}, 1);
}

template <typename Element, typename Layout>
inline STensor allocate_stensor(Graph &bgraph, cutlass::HostTensor<Element, Layout> const &host_tensor) {
static_assert(std::is_same_v<Element, cutlass::half_t>, "Only f16.");
  STensor tensor;
  tensor.num_dims = Layout::kRank;
  tensor.data_type = type::DT_FLOAT16;
  for (int i = 0; i < tensor.num_dims; i++) {
    tensor.dim[i] = host_tensor.extent()[i];
    tensor.stride[i] = host_tensor.stride(i);
  }
  tensor.owner_op = nullptr;
  tensor.owner_ts_idx = -1;
  tensor.smem_offset = bgraph.allocate(tensor);
  return tensor;
}

CUTLASS_DEVICE
void copy_global_to_shared(char *smem_buffer, STensor const &tensor, void *g_ptr_) {
  // Only the first thread copies. TODO: make all threads copy.
  if (cutlass::thread0()) {
    char *s_ptr = smem_buffer + tensor.smem_offset;
    char *g_ptr = reinterpret_cast<char *>(g_ptr_);
    size_t size = tensor.size();
    for (size_t i = 0; i < size; i++) {
      s_ptr[i] = g_ptr[i];
    }
  }
  __syncthreads();
}
CUTLASS_DEVICE
void copy_shared_to_global(char *smem_buffer, STensor const &tensor, void *g_ptr_) {
  if (cutlass::thread0()) {
    char *s_ptr = smem_buffer + tensor.smem_offset;
    char *g_ptr = reinterpret_cast<char *>(g_ptr_);
    size_t size = tensor.size();
    for (size_t i = 0; i < size; i++) {
      g_ptr[i] = s_ptr[i];
    }
  }
}

template <typename Element, typename Layout>
void random_fill_tensor(cutlass::HostTensor<Element, Layout> &host_tensor, size_t seed) {
  std::mt19937_64 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  size_t size = host_tensor.size();
  Element *ptr = host_tensor.host_data();
  for (size_t i = 0; i < size; ++i) {
    ptr[i] = static_cast<Element>(dist(gen));
  }
  host_tensor.sync_device();
}

} // namespace threadblock
} // namespace aso
