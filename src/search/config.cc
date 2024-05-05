#include "aso/search/config.h"

namespace aso {
namespace search {

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      {type::KN_MATMUL_OP,
       type::KN_REDUCTION_1_OP,
       type::KN_REDUCTION_2_OP,
       type::KN_EXP_OP,
       type::KN_DIV_OP,
       type::KN_CUSTOMIZED_OP},
      {type::TB_MATMUL_OP,
       type::TB_REDUCTION_1_OP,
       type::TB_REDUCTION_2_OP,
       type::TB_EXP_OP,
       type::TB_DIV_OP,
       type::TB_REDUCTION_1_TO_DIMX_OP,
       type::TB_REDUCTION_2_TO_DIMX_OP},
      {int3{0, 1, -1}, int3{0, 2, -1}, int3{0, -1, -1}, int3{0, -1, 1}} /* imap_to_explore*/,
      {int3{0, 1, -1}, int3{0, 2, -1}, int3{0, -1, -1}, int3{0, 2, 1}} /* omap_to_explore */,
      {dim3{16, 1, 1},
       dim3{16, 2, 1} , dim3{16, 4, 1}} /* grid_dim_to_explore*/,
      {dim3{128, 1, 1}} /* block_dim_to_explore */,
      {-1, 1, 2} /* fmap_to_explore */,
      {1, 4, 8, 16} /* frange_to_explore */,
      {
          layout::SmemRowMajor,
          layout::SmemColumnMajor,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
      } /* smem_layout_to_explore*/};
}

void GeneratorConfig::print_config() const {
  printf("========== Search Configuration ==========\n");
  printf("  imaps to explore:\n");
  for (const auto & imap : imap_to_explore) {
    printf("    (%d, %d, %d)\n", imap.x, imap.y, imap.z);
  }
  printf("  omaps to explore:\n");
  for (const auto & omap : omap_to_explore) {
    printf("    (%d, %d, %d)\n", omap.x, omap.y, omap.z);
  }
  printf("  grid dims to explore:\n");
  for (const auto & griddim : grid_dim_to_explore) {
    printf("    (%d, %d, %d)\n", griddim.x, griddim.y, griddim.z);
  }
  printf("  block dims to explore:\n");
  for (const auto & blockdim : block_dim_to_explore) {
    printf("    (%d, %d, %d)\n", blockdim.x, blockdim.y, blockdim.z);
  }
  printf("  fmaps to explore:");
  for (const auto & fmap : fmap_to_explore) {
    printf("%d ", fmap);
  }
  printf("\n");
  printf("  franges to explore:");
  for (const auto &frange : frange_to_explore) {
    printf("%d ", frange);
  }
  printf("\n");
}

} // namespace search
} // namespace aso
