#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include "CRS.h"
#include <omp.h>
#include "convert.h"
#include <type_traits>

namespace SparseMatrixLib
{

template <typename ValT>
struct inCRS {
    size_t m = 0;
    size_t n = 0;
    size_t nz = 0;
    std::vector<int> row_ptr; // int16_t
    std::vector<int> col_idx; // int16_t
    std::vector<ValT> values;

    inCRS() = default;
    inCRS(const inCRS& crs) = default;
    inCRS& operator=(const inCRS& crs) = default;
};

template <typename ValT>
class spMtxHCSR {
public:
    size_t R;
    size_t C;
    size_t m = 0; // row
    size_t n = 0; // column
    size_t nz = 0; // non-zeros
    size_t b_m;
    size_t b_n;
    std::vector<int> block_ptr;    
    std::vector<uint32_t> row_offset;
    std::vector<uint32_t> row_ptr;
    std::vector<uint16_t> col_idx;
    std::vector<ValT> values;
    // my_container<int> row_ptr;
    // my_container<int> col_idx;
    // my_container<inCRS<ValT>> values; // can improve by adding bit-masks to elements

    spMtxHCSR() : R(0), C(0), m(0), n(0), nz(0), b_m(0), b_n(0), row_ptr(), col_idx(), values(), block_ptr() {}

    spMtxHCSR(size_t _m, size_t _n, size_t _nz, size_t _R, size_t _C) : m(_m), n(_n), nz(_nz), R(_R), C(_C) {
        b_m = (m + R - 1) / R;
        b_n = (n + C - 1) / C;
    }

    spMtxHCSR(const spMtxHCSR& copy) : m(copy.m), n(copy.n), nz(copy.nz), b_m(copy.b_m), b_n(copy.b_n), 
        R(copy.R), C(copy.C), row_ptr(copy.row_ptr), col_idx(copy.col_idx), values(copy.values), block_ptr(copy.block_ptr)
    {}

    spMtxHCSR(spMtxHCSR&& mov) : m(mov.m), n(mov.n), nz(mov.nz), b_m(mov.b_m), b_n(mov.b_n),
        R(mov.R), C(mov.C), row_ptr(std::move(mov.row_ptr)), col_idx(std::move(mov.col_idx)), 
        values(std::move(mov.values)), block_ptr(std::move(mov.block_ptr)) {

    }

    ~spMtxHCSR() {}

    spMtxHCSR& operator=(const spMtxHCSR& copy) {
        if (this == &copy)
            return *this;

        m = copy.m;
        n = copy.n;
        nz = copy.nz;
        b_m = copy.b_m;
        b_n = copy.b_n;
        R = copy.R;
        C = copy.C;
        row_ptr = copy.row_ptr;
        col_idx = copy.col_idx;
        values = copy.values;
        block_ptr = copy.block_ptr;

        return *this;
    }

    spMtxHCSR& operator=(spMtxHCSR&& mov) {
        if (this == &mov)
            return *this;

        m = mov.m;
        n = mov.n;
        nz = mov.nz;
        b_m = mov.b_m;
        b_n = mov.b_n;
        R = mov.R;
        C = mov.C;
        row_ptr = std::move(mov.row_ptr);
        col_idx = std::move(mov.col_idx);
        values = std::move(mov.values);
        block_ptr = std::move(mov.block_ptr);

        return *this;
    }
};

}
