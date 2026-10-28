/* 
*========================================================
 * Copyright (c) RVVLASparse and Lobachevsky State University of 
 * Nizhny Novgorod and its affiliates. All rights reserved.
 * 
 * Copyright 2026 The RVVLASparse Authors (Zagriadskov Maksim)
 *
 * Distributed under the MIT License
 * (See file LICENSE in the root directory of this 
 * source tree)
 *========================================================
 */

#include "sparse_matrix.h"
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace SparseMatrixLib
{

template <typename T>
void crs_to_HCSR(spMtxHCSR<T>& dist, const spMtxCRS<T>& src, size_t R, size_t C, double bs) {
    using HM = spMtxHCSR<T>;
    using bptr_t = typename std::decay_t<decltype(std::declval<HM&>().block_ptr)>::value_type;
    using roff_t = typename std::decay_t<decltype(std::declval<HM&>().row_offset)>::value_type;
    using rptr_t = typename std::decay_t<decltype(std::declval<HM&>().row_ptr)>::value_type;
    using cidx_t = typename std::decay_t<decltype(std::declval<HM&>().col_idx)>::value_type;

    if (R == 0 || C == 0) throw std::invalid_argument("crs_to_HCSR: R and C must be positive");

    dist = spMtxHCSR<T>(); 
    dist.R = R;
    dist.C = C;
    const double block_switch = bs; 

    dist.b_m = (src.m + dist.R - 1) / dist.R;
    dist.b_n = (src.n + dist.C - 1) / dist.C;
    dist.m = src.m;
    dist.n = src.n;
    dist.nz = src.nz;

    const size_t m = (size_t)src.m;
    const size_t b_m = (size_t)dist.b_m, b_n = (size_t)dist.b_n;
    const size_t nblocks = b_m * b_n;
    const size_t nz = (size_t)src.nz;
    if (nblocks / b_n != b_m) throw std::length_error("crs_to_HCSR: too many blocks");

    dist.block_ptr.assign(nblocks + 1, 0);
    dist.row_offset.assign(nblocks + 1, 0);
    std::vector<uint8_t> kind(nblocks, 0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (long long br = 0; br < (long long)b_m; ++br) {
        const size_t r0 = (size_t)br * R, r1 = std::min(r0 + R, m);
        const size_t blk0 = (size_t)br * b_n;
        bptr_t* cnt = dist.block_ptr.data() + blk0 + 1;
        for (size_t i = r0; i < r1; ++i)
            for (size_t k = (size_t)src.Rst[i]; k < (size_t)src.Rst[i + 1]; ++k)
                ++cnt[(size_t)src.Col[k] / C];

        const double rows = double(r1 - r0);
        for (size_t j = 0; j < b_n; ++j) {
            const size_t c = (size_t)cnt[j];
            if (c == 0) { kind[blk0 + j] = 0; dist.row_offset[blk0 + j + 1] = 0; }
            else if (rows * block_switch < double(c)) { kind[blk0 + j] = 1; dist.row_offset[blk0 + j + 1] = (roff_t)(r1 - r0 + 1); }
            else { kind[blk0 + j] = 2; dist.row_offset[blk0 + j + 1] = (roff_t)c; }
        }
    }

    size_t accNz = 0, accRows = 0;
    const size_t flagLimit = (size_t(1) << 30) - 1; 
    for (size_t b = 1; b <= nblocks; ++b) {
        accNz += (size_t)dist.block_ptr[b];
        accRows += (size_t)dist.row_offset[b];
        dist.block_ptr[b] = (bptr_t)accNz;
        dist.row_offset[b] = (roff_t)accRows;
    }
    if (accRows > flagLimit || accRows > (size_t)std::numeric_limits<roff_t>::max())
        throw std::length_error("crs_to_HCSR: row_ptr has " + std::to_string(accRows) +
                                " entries, does not fit the 30-bit row_offset field "
                                "(use larger C or larger p)");
    if (accNz != nz) throw std::logic_error("crs_to_HCSR: nnz mismatch");

    dist.col_idx.resize(nz);
    dist.values.resize(nz);
    dist.row_ptr.assign(accRows, 0);

    #pragma omp parallel
    {
        std::vector<size_t> pos(b_n);

        #pragma omp for schedule(dynamic, 1)
        for (long long br = 0; br < (long long)b_m; ++br) {
            const size_t r0 = (size_t)br * R, r1 = std::min(r0 + R, m);
            const size_t blk0 = (size_t)br * b_n;
            for (size_t j = 0; j < b_n; ++j) pos[j] = (size_t)dist.block_ptr[blk0 + j];

            for (size_t i = r0; i < r1; ++i) {
                const size_t li = i - r0;
                for (size_t k = (size_t)src.Rst[i]; k < (size_t)src.Rst[i + 1]; ++k) {
                    const size_t col = (size_t)src.Col[k];
                    const size_t j = col / C, blk = blk0 + j;
                    const size_t p = pos[j]++;
                    dist.col_idx[p] = (cidx_t)(col - j * C);
                    dist.values[p] = src.Val[k];
                    const size_t ro = (size_t)dist.row_offset[blk];
                    if (kind[blk] == 1) ++dist.row_ptr[ro + li + 1]; 
                    else dist.row_ptr[ro + (p - (size_t)dist.block_ptr[blk])] = (rptr_t)li;
                }
            }

            for (size_t j = 0; j < b_n; ++j) {
                const size_t blk = blk0 + j;
                if (kind[blk] != 1) continue;
                rptr_t* rp = dist.row_ptr.data() + (size_t)dist.row_offset[blk];
                for (size_t li = 0; li < r1 - r0; ++li) rp[li + 1] += rp[li];
            }
        }
    }

    for (size_t b = 0; b < nblocks; ++b) {
        uint32_t f = kind[b];
        f <<= 30;
        dist.row_offset[b] |= f;
    }
}

template<>
void convert<double, spMtxHCSR, false>(spMtxHCSR<double> &dist, const spMtxCRS<double> &src,
                                      const convertParams &params)
{
    int R = 2000;
    int C = 2000;
    double p = 2;
    if(params.param.count("R"))
        R = params.param.at("R").i;
    if(params.param.count("C"))
        C = params.param.at("C").i;
    if(params.param.count("p"))
        p = params.param.at("p").d;

    crs_to_HCSR(dist, src, R, C, p);
}

template<>
void convert<float, spMtxHCSR, false>(spMtxHCSR<float> &dist, const spMtxCRS<float> &src,
                                     const convertParams &params)
{
    int R = 4000;
    int C = 4000;
    double p = 2;
    if(params.param.count("R"))
        R = params.param.at("R").i;
    if(params.param.count("C"))
        C = params.param.at("C").i;
    if(params.param.count("p"))
        p = params.param.at("p").d;

    crs_to_HCSR(dist, src, R, C, p);
}

}
