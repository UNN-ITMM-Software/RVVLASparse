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
#include <string>
#include <map>
#include "CRS.h"

namespace SparseMatrixLib
{
union convertValue{
  int i;
  double d;
};

struct convertParams{
  std::map<std::string, convertValue> param;
};

template<class T, template<class> class MtxDist, bool simd = true>
void convert(MtxDist<T> &dist, const spMtxCRS<T> &src, const convertParams &params);

}
