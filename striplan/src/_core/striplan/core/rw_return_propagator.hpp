#ifndef E8F2FDE6_5F65_4385_B348_EB0BDC61EB9D
#define E8F2FDE6_5F65_4385_B348_EB0BDC61EB9D

#include <cstdint>

#include "striplan/core/ndarray_utils.hpp"

namespace striplan {

ndarray_utils::Float32Array1D compute_non_returning_features_flat(
    ndarray_utils::Int32Array1D row_ptr,
    ndarray_utils::Int32Array1D col_ind,
    ndarray_utils::Float32Array1D values,
    std::int32_t num_kept,
    std::int32_t k_max,
    std::int32_t block_size,
    float eps = 1e-8F
);

}  // namespace striplan


#endif /* E8F2FDE6_5F65_4385_B348_EB0BDC61EB9D */
