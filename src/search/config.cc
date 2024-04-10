#include "aso/search/config.h"

namespace aso {
namespace search {

GeneratorConfig GeneratorConfig::get_default_config() {
  return {
      {
          type::KN_MATMUL_OP,
          type::KN_REDUCTION_0_OP,
          type::KN_REDUCTION_1_OP,
          type::KN_REDUCTION_2_OP,
          type::KN_EXP_OP,
          type::KN_DIV_OP,
          type::KN_CUSTOMIZED_OP,
          //  type::KN_ADD_OP,
      },
      {
          type::TB_MATMUL_OP,
          type::TB_REDUCTION_0_OP,
          type::TB_REDUCTION_1_OP,
          type::TB_REDUCTION_2_OP,
          type::TB_EXP_OP,
          type::TB_DIV_OP,
          type::TB_REDUCTION_0_TO_DIMX_OP,
          type::TB_REDUCTION_1_TO_DIMX_OP,
          type::TB_REDUCTION_2_TO_DIMX_OP,
          //  type::TB_ADD_OP,
      },
      {int3{0, 1, -1}, int3{0, 2, -1}, int3{0, -1, -1}} /* imap_to_explore*/,
      {int3{0, 1, -1}, int3{0, 2, -1}, int3{0, -1, -1}} /* omap_to_explore */,
      {dim3{16, 1, 1},
       dim3{16, 2, 1} /*, dim3{16, 4, 1}*/} /* grid_dim_to_explore*/,
      {dim3{128, 1, 1}} /* block_dim_to_explore */,
      {-1, 1, 2} /* fmap_to_explore */,
      {8, 16} /* frange_to_explore */,
      {
          layout::SmemRowMajor,
          layout::SmemColumnMajor,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise16,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise32,
          layout::SmemRowMajorTensorOpMultiplicand_Crosswise64,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise16,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise32,
          layout::SmemColumnMajorTensorOpMultiplicand_Crosswise64,
      } /* smem_layout_to_explore*/,
      64 /* reduction_dimx */};
}

} // namespace search
} // namespace aso
