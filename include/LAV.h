#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include "CRS.h"
//#include <riscv_vector.h>

namespace SparseMatrixLib
{
  template <typename type>
  struct spMtxLAV {

    size_t SIMD_Lanes = 8;
    int m = 0;
    int n = 0;
    int nz = 0;

    size_t segment_count;
    size_t chunk_count;
    double T = 0.2;
    int segment_len = 40000;

    std::vector<int> chunk_offsets;
    std::vector<int> chunk_offsets_full_mask;
    std::vector<size_t> segment_ptr;
    std::vector<uint32_t> out_order;
    std::vector<uint16_t> mask;
    std::vector<type> val;
    std::vector<uint32_t> col_id;

    spMtxCRS<type> sparse_part;

    spMtxLAV() {};

    void freeMem() {
      chunk_offsets_full_mask.resize(0); chunk_offsets_full_mask.shrink_to_fit();
      chunk_offsets.resize(0); chunk_offsets.shrink_to_fit();
      segment_ptr.resize(0); segment_ptr.shrink_to_fit();
      out_order.resize(0); out_order.shrink_to_fit();
      mask.resize(0); mask.shrink_to_fit();
      val.resize(0); val.shrink_to_fit();
      col_id.resize(0); col_id.shrink_to_fit();
      sparse_part.freeMem();
    }

    spMtxLAV(const spMtxLAV& copy)
    {
      sparse_part = copy.sparse_part;

      SIMD_Lanes = copy.SIMD_Lanes;
      m = copy.m;
      n = copy.n;
      nz = copy.nz;
      T = copy.T;
      segment_len = copy.segment_len;
      segment_count = copy.segment_count;
      chunk_count = copy.chunk_count;

      out_order = copy.out_order;
      val = copy.val;
      col_id = copy.col_id;
      mask = copy.mask;
      chunk_offsets = copy.chunk_offsets;
      segment_ptr = copy.segment_ptr;
      chunk_offsets_full_mask = copy.chunk_offsets_full_mask;
    }

    spMtxLAV& operator=(const spMtxLAV& copy) {
      if (this == &copy)
        return *this;

      sparse_part = copy.sparse_part;

      SIMD_Lanes = copy.SIMD_Lanes;
      m = copy.m;
      n = copy.n;
      nz = copy.nz;
      T = copy.T;
      segment_len = copy.segment_len;
      segment_count = copy.segment_count;
      chunk_count = copy.chunk_count;

      out_order = copy.out_order;
      val = copy.val;
      col_id = copy.col_id;
      mask = copy.mask;
      chunk_offsets = copy.chunk_offsets;
      segment_ptr = copy.segment_ptr;
      chunk_offsets_full_mask = copy.chunk_offsets_full_mask;

      return *this;
    }

  };

}