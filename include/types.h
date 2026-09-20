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
#include <stdint.h>

#include <iostream>

namespace SparseMatrixLib{

  struct sparse_matrix_status{
    int code = 0;
  };

  enum sparse_operation_t{
    SPARSE_OPERATION_NON_TRANSPOSE,
    SPARSE_OPERATION_TRANSPOSE
  };

  struct sparse_matrix_descr{
    
  };

  enum sparse_matrix_mv_stage{
    SPARSE_MATRIX_MV_PREPARATION,
    SPARSE_MATRIX_MV_MULT,
    SPARSE_MATRIX_MV_GET_RESULTS,
    SPARSE_MATRIX_MV_ALL_STAGES
  };

}
