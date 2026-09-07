#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include <cstdint>
#include "CRS.h"

namespace SparseMatrixLib
{
  template <typename ValT>
  class spMtxCVR {
  public:
    size_t nItems = 0;
    size_t numRows = 0;
    size_t numCols = 0;
    size_t SIMD_LEN = 1;

    int Nthreads = 1;
    int Nthreads_avail = 1;

    std::vector<ValT> vPack_vec_vals;

    std::vector<uint32_t> vPack_vec_cols;
    std::vector<int> vPack_vec_record;
    std::vector<int> writeback_pos_record;

    spMtxCVR() {

    }

    void freeMem()
    {
      vPack_vec_vals.resize(0); vPack_vec_vals.shrink_to_fit();
      vPack_vec_cols.resize(0); vPack_vec_cols.shrink_to_fit();
      vPack_vec_record.resize(0); vPack_vec_record.shrink_to_fit();
      writeback_pos_record.resize(0); writeback_pos_record.shrink_to_fit();
    }

    spMtxCVR(const spMtxCVR& copy) : nItems(copy.nItems), numRows(copy.numRows), numCols(copy.numCols), SIMD_LEN(copy.SIMD_LEN), Nthreads(copy.Nthreads), Nthreads_avail(copy.Nthreads_avail) {
      vPack_vec_vals = copy.vPack_vec_vals;

      vPack_vec_cols = copy.vPack_vec_cols;
      vPack_vec_record = copy.vPack_vec_record;
      writeback_pos_record = copy.writeback_pos_record;
    }

    spMtxCVR(spMtxCVR&& mov) : nItems(mov.nItems), numRows(mov.numRows), numCols(mov.numCols), SIMD_LEN(mov.SIMD_LEN), Nthreads(mov.Nthreads), Nthreads_avail(mov.Nthreads_avail), vPack_vec_vals(std::move(mov.vPack_vec_vals)), vPack_vec_cols(std::move(mov.vPack_vec_cols)), vPack_vec_record(std::move(mov.vPack_vec_record)), writeback_pos_record(std::move(mov.writeback_pos_record)) {}

    spMtxCVR& operator=(const spMtxCVR& copy) {
      if (this == &copy)
        return *this;

      nItems = copy.nItems;
      numRows = copy.numRows;
      numCols = copy.numCols;
      SIMD_LEN = copy.SIMD_LEN;
      Nthreads = copy.Nthreads;
      Nthreads_avail = copy.Nthreads_avail;

      vPack_vec_vals = copy.vPack_vec_vals;

      vPack_vec_cols = copy.vPack_vec_cols;
      vPack_vec_record = copy.vPack_vec_record;
      writeback_pos_record = copy.writeback_pos_record;

      return *this;
    }

  };

} //namespace SparseMatrixLib