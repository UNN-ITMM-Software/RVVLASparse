/* 
*========================================================
 * Copyright (c) RVVLASparse and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVLASparse Authors (Aleksandr Ustinov)
 *
 * Distributed under the MIT License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 */

#pragma once
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "convert.h"
#include "alignment_allocator.h"

namespace SparseMatrixLib
{

#define CSR5_SUCCESS                 0
#define CSR5_UNKNOWN_FORMAT         -1
#define CSR5_UNSUPPORTED_OMEGA      -2
#define CSR5_CONVERTION_FAILED      -3
#define CSR5_UNSUPPORTED_SPMV       -4
#define CSR5_UNSUPPORTED_VALUE_TYPE -5

#define CSR5_FORMAT_CSR  0
#define CSR5_FORMAT_CSR5 1

#define CSR5_AVX2_OMEGA 4
#define CSR5_AVX2_SIGMA 4

#define CSR5_SCALAR_OMEGA 4
#define CSR5_SCALAR_SIGMA 4

#define CSR5_CACHELINE 64

template<typename T, typename U>
std::common_type_t<T, U> iceil(T num, U den) {
    return (num + den - 1) / den;
}

inline uint32_t remove_sign(uint32_t value) {
    return value & 0x7FFFFFFF;
}


template <class ValT>
class spMtxCSR5
{
public:
  int inputCSR(int m, int n, int nnz,
               int  *csr_row_pointer_ptr,
               int  *csr_column_index_ptr,
               ValT *csr_value_ptr);
  int asCSR5(const convertParams &params);

  int _m;
  int _n;
  int _nnz;
  int _bit_y_offset;
  int _bit_seg_offset;
  int _tile_desc_len;
  int _tail_tile_start;
  int  _p;
  int _omega;
  int _sigma;
  
  std::vector<int, AlignmentAllocator<int, CSR5_CACHELINE>>  _csr_row_pointer;
  std::vector<uint32_t, AlignmentAllocator<uint32_t, CSR5_CACHELINE>>  _csr_column_index;
  std::vector<ValT, AlignmentAllocator<ValT, CSR5_CACHELINE>> _csr_value;
  
  std::vector<uint32_t, AlignmentAllocator<uint32_t, CSR5_CACHELINE>> _csr5_tile_pointer;
  std::vector<uint32_t, AlignmentAllocator<uint32_t, CSR5_CACHELINE>> _csr5_tile_descriptor;
  
  int   _num_offsets;
  std::vector<uint32_t, AlignmentAllocator<uint32_t, CSR5_CACHELINE>>  _csr5_tile_empty_offset_pointer;
  std::vector<uint32_t, AlignmentAllocator<uint32_t, CSR5_CACHELINE>>  _csr5_tile_empty_offset;
  std::vector<ValT, AlignmentAllocator<ValT, CSR5_CACHELINE>> _temp_calibrator;

  template <typename T, typename Allocator>
  void clearAndShrink(std::vector<T, Allocator> &vec) {
    vec.clear();
    vec.shrink_to_fit();
  }
  void freeMem() {
    clearAndShrink(_csr_row_pointer);
    clearAndShrink(_csr_column_index);
    clearAndShrink(_csr_value);

    clearAndShrink(_csr5_tile_pointer);
    clearAndShrink(_csr5_tile_descriptor);

    clearAndShrink(_csr5_tile_empty_offset_pointer);
    clearAndShrink(_csr5_tile_empty_offset);
    clearAndShrink(_temp_calibrator);
  }
  ~spMtxCSR5() {
    freeMem();
  }
private:
  int _format;
};

}
