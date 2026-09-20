/* 
*========================================================
 * Copyright (c) RVVLASparse and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVLASparse Authors (Evgeny Kozinov)
 *
 * Distributed under the MIT License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 */
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
