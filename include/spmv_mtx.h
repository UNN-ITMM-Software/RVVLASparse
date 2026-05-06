#pragma once
#include "types.h"
#include <vector>
#include <stdexcept>

namespace SparseMatrixLib
{

template<class T, template<class> class Mtx, bool simd = true, 
         sparse_matrix_mv_stage = SPARSE_MATRIX_MV_ALL_STAGES>
sparse_matrix_status sparse_mv(sparse_operation_t type_op, 
                               T alpha, 
                               const Mtx<T> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<T> &b,
                               T beta,
                               std::vector<T> &y);

}