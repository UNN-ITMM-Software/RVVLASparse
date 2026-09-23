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

#include "sparse_matrix.h"
#include "spmv_mtx.h"
#include <riscv_vector.h>

namespace SparseMatrixLib
{
    static int get_num_of_threads()
    {
        int num_threads;
#pragma omp parallel
        {
#pragma omp master
            {
                num_threads = omp_get_num_threads();
            }
        }
        return num_threads;
    }

#define TILE_SIMPLE_LOOP(FP, FP_SEW, FP_RVV, FP_RVV_SUF, FP_LMUL, FP_OFFSET, INT_RVV, INT_RVV_SUF, INT_LMUL) \
    inline void tile_simple_loop_rvv_##FP_RVV_SUF(const FP *d_value_tile,                                    \
                                                  const FP *d_x,                                             \
                                                  const uint32_t *d_column_index_tile,                       \
                                                  FP *d_calibrator,                                          \
                                                  FP *d_y,                                                   \
                                                  const uint32_t tile_first_row,                             \
                                                  const int tid,                                             \
                                                  const int thread_first_row,                                \
                                                  const int stride_vT,                                       \
                                                  const bool row_starts_here,                                \
                                                  const FP alpha,                                            \
                                                  const int omega,                                           \
                                                  const int sigma)                                           \
    {                                                                                                        \
        const size_t vl = omega;                                                                             \
        v##FP_RVV##FP_LMUL##_t vval, vx, vdot;                                                               \
        v##FP_RVV##m1##_t vzero, vres;                                                                       \
        v##INT_RVV##INT_LMUL##_t vcol, vcol_shift;                                                           \
                                                                                                             \
        vdot = __riscv_vfmv_v_f_##FP_RVV_SUF(0.0, vl);                                                       \
        for (int i = 0; i < sigma; ++i)                                                                      \
        {                                                                                                    \
            vcol = __riscv_vle32_v_##INT_RVV_SUF(d_column_index_tile, vl);                                   \
            vcol = __riscv_vsll_vx_##INT_RVV_SUF(vcol, FP_OFFSET, vl);                                       \
            vval = __riscv_vle##FP_SEW##_v_##FP_RVV_SUF(d_value_tile, vl);                                   \
            vx = __riscv_vloxei32_v_##FP_RVV_SUF(d_x, vcol, vl);                                             \
            vdot = __riscv_vfmacc_vv_##FP_RVV_SUF(vdot, vval, vx, vl);                                       \
                                                                                                             \
            d_column_index_tile += vl;                                                                       \
            d_value_tile += vl;                                                                              \
        }                                                                                                    \
                                                                                                             \
        vzero = __riscv_vfmv_v_f_f##FP_SEW##m1((FP)0.0, vl);                                                 \
        vres = __riscv_vfredusum_vs_##FP_RVV_SUF##_f##FP_SEW##m1(vdot, vzero, vl);                           \
                                                                                                             \
        FP sum = __riscv_vfmv_f_s_f##FP_SEW##m1##_f##FP_SEW(vres);                                           \
        sum *= alpha;                                                                                        \
                                                                                                             \
        if (tile_first_row == thread_first_row && !row_starts_here)                                          \
        {                                                                                                    \
            d_calibrator[tid * stride_vT] += sum;                                                            \
        }                                                                                                    \
        else                                                                                                 \
        {                                                                                                    \
            d_y[tile_first_row] += sum;                                                                      \
        }                                                                                                    \
    }

#define VCVT_U32_U32(FP_LMUL, VREG) VREG
#define VCVT_U32_U64(FP_LMUL, VREG) __riscv_vwcvtu_x_x_v_u64##FP_LMUL(VREG, vl)

#define SPMV_CSR5_COMPUTE_KERNEL_RVV(FP, FP_SEW, FP_RVV, FP_RVV_SUF, FP_LMUL, FP_OFFSET, INT_RVV, INT_RVV_SUF, INT_LMUL, BOOL_RVV, BOOL_RVV_SUF, SIMPLE_LOOP_KERNEL)                             \
    void spmv_csr5_compute_kernel_rvv_##FP_RVV_SUF(const uint32_t *d_column_index,                                                                                                               \
                                                   const FP *d_value,                                                                                                                            \
                                                   const int *d_row_pointer,                                                                                                                     \
                                                   const FP *d_x,                                                                                                                                \
                                                   const uint32_t *d_tile_pointer,                                                                                                               \
                                                   const uint32_t *d_tile_descriptor,                                                                                                            \
                                                   const uint32_t *d_tile_empty_offset_pointer,                                                                                                  \
                                                   const uint32_t *d_tile_empty_offset,                                                                                                          \
                                                   FP *d_calibrator,                                                                                                                             \
                                                   FP *d_y,                                                                                                                                      \
                                                   const int p,                                                                                                                                  \
                                                   const int tile_desc_len,                                                                                                                      \
                                                   const int bit_y_offset,                                                                                                                       \
                                                   const int bit_seg_offset,                                                                                                                     \
                                                   const FP alpha,                                                                                                                               \
                                                   const int omega,                                                                                                                              \
                                                   const int sigma)                                                                                                                              \
    {                                                                                                                                                                                            \
        const int num_thread = omp_get_max_threads();                                                                                                                                            \
        const int chunk = iceil(p - 1, num_thread);                                                                                                                                              \
        const int stride_vT = CSR5_CACHELINE / sizeof(FP);                                                                                                                                       \
        const int num_thread_active = iceil(p - 1, chunk);                                                                                                                                       \
                                                                                                                                                                                                 \
_Pragma("omp parallel")                                                                                                                                                                          \
        {                                                                                                                                                                                        \
            int tid = omp_get_thread_num();                                                                                                                                                      \
            int thread_first_row = tid < num_thread_active ? remove_sign(d_tile_pointer[tid * chunk]) : 0;                                                                                       \
                                                                                                                                                                                                 \
            const size_t vl = omega;                                                                                                                                                             \
                                                                                                                                                                                                 \
            v##FP_RVV##FP_LMUL##_t v_value;                                                                                                                                                      \
            v##FP_RVV##FP_LMUL##_t v_x;                                                                                                                                                          \
                                                                                                                                                                                                 \
            v##FP_RVV##FP_LMUL##_t v_sum;                                                                                                                                                        \
            v##FP_RVV##FP_LMUL##_t v_tmp_sum;                                                                                                                                                    \
            v##FP_RVV##FP_LMUL##_t v_first_sum;                                                                                                                                                  \
            v##FP_RVV##FP_LMUL##_t v_last_sum;                                                                                                                                                   \
            v##FP_RVV##FP_LMUL##_t v_y_local;                                                                                                                                                    \
                                                                                                                                                                                                 \
            v##INT_RVV##INT_LMUL##_t v_column_index;                                                                                                                                             \
            v##INT_RVV##INT_LMUL##_t v_seg_offset;                                                                                                                                               \
            v##INT_RVV##INT_LMUL##_t v_y_offset;                                                                                                                                                 \
            v##INT_RVV##INT_LMUL##_t v_y_idx;                                                                                                                                                    \
            v##INT_RVV##INT_LMUL##_t v_start;                                                                                                                                                    \
            v##INT_RVV##INT_LMUL##_t v_stop;                                                                                                                                                     \
            v##INT_RVV##INT_LMUL##_t v_descriptor;                                                                                                                                               \
            v##INT_RVV##INT_LMUL##_t v_tmp;                                                                                                                                                      \
                                                                                                                                                                                                 \
            v##BOOL_RVV##_t v_local_bit;                                                                                                                                                         \
            v##BOOL_RVV##_t v_incolumn;                                                                                                                                                          \
                                                                                                                                                                                                 \
            v##BOOL_RVV##_t v_firstbit_set;                                                                                                                                                      \
            {                                                                                                                                                                                    \
                v##BOOL_RVV##_t v_allbits = __riscv_vmset_m_##BOOL_RVV_SUF(vl);                                                                                                                  \
                v##INT_RVV##INT_LMUL##_t v_iota = __riscv_viota_m_##INT_RVV_SUF(v_allbits, vl);                                                                                                  \
                v_firstbit_set = __riscv_vmseq_vx_##INT_RVV_SUF##_##BOOL_RVV_SUF(v_iota, 0, vl);                                                                                                 \
            }                                                                                                                                                                                    \
                                                                                                                                                                                                 \
_Pragma("omp for schedule(static, chunk)")                                                                                                                                                       \
            for (int tile_id = 0; tile_id < p - 1; tile_id++)                                                                                                                                    \
            {                                                                                                                                                                                    \
                const int tile_idx = tile_id * omega * sigma;                                                                                                                                    \
                const int descriptor_idx = tile_id * omega * tile_desc_len;                                                                                                                      \
                                                                                                                                                                                                 \
                const uint32_t *d_column_index_tile = d_column_index + tile_idx;                                                                                                                 \
                const FP *d_value_tile = d_value + tile_idx;                                                                                                                                     \
                const uint32_t *d_tile_descriptor_tile = d_tile_descriptor + descriptor_idx;                                                                                                     \
                                                                                                                                                                                                 \
                uint32_t tile_first_row = d_tile_pointer[tile_id];                                                                                                                               \
                const int tile_last_row = remove_sign(d_tile_pointer[tile_id + 1]);                                                                                                              \
                                                                                                                                                                                                 \
                if (tile_first_row == tile_last_row)                                                                                                                                             \
                {                                                                                                                                                                                \
                    int first_bit_flag_offset = 31 - (bit_y_offset + bit_seg_offset);                                                                                                            \
                    bool row_starts_here = (d_tile_descriptor[tile_id * omega * tile_desc_len] >> first_bit_flag_offset) & 0x1;                                                                  \
                    SIMPLE_LOOP_KERNEL(d_value_tile, d_x, d_column_index_tile, d_calibrator, d_y,                                                                                                \
                                       tile_first_row, tid, thread_first_row, stride_vT, row_starts_here, alpha,                                                                                 \
                                       sigma, omega);                                                                                                                                            \
                }                                                                                                                                                                                \
                else                                                                                                                                                                             \
                {                                                                                                                                                                                \
                    const bool empty_rows = (tile_first_row >> 31) & 0x1;                                                                                                                        \
                    tile_first_row = remove_sign(tile_first_row);                                                                                                                                \
                                                                                                                                                                                                 \
                    FP *d_y_local = &d_y[tile_first_row + 1];                                                                                                                                    \
                    const int offset_pointer = empty_rows ? d_tile_empty_offset_pointer[tile_id] : 0;                                                                                            \
                                                                                                                                                                                                 \
                    v_first_sum = __riscv_vfmv_v_f_##FP_RVV_SUF((FP)0.0, vl);                                                                                                                    \
                    v_stop = __riscv_vmv_v_x_##INT_RVV_SUF(0, vl);                                                                                                                               \
                                                                                                                                                                                                 \
                    uint32_t tile_descriptor_offset = tile_id * omega * tile_desc_len;                                                                                                           \
                    v_descriptor = __riscv_vle32_v_##INT_RVV_SUF(d_tile_descriptor_tile, vl);                                                                                                    \
                    v_y_offset = __riscv_vsrl_vx_##INT_RVV_SUF(v_descriptor, 32 - bit_y_offset, vl);                                                                                             \
                    v_seg_offset = __riscv_vsll_vx_##INT_RVV_SUF(v_descriptor, bit_y_offset, vl);                                                                                                \
                    v_seg_offset = __riscv_vsrl_vx_##INT_RVV_SUF(v_seg_offset, 32 - bit_seg_offset, vl);                                                                                         \
                    v_descriptor = __riscv_vsll_vx_##INT_RVV_SUF(v_descriptor, bit_y_offset + bit_seg_offset, vl);                                                                               \
                                                                                                                                                                                                 \
                    v_tmp = __riscv_vsrl_vx_##INT_RVV_SUF(v_descriptor, 31, vl);                                                                                                                 \
                    v_local_bit = __riscv_vmseq_vx_##INT_RVV_SUF##_##BOOL_RVV_SUF(v_tmp, 1, vl);                                                                                                 \
                    bool first_incolumn = __riscv_vcpop_m_##BOOL_RVV_SUF(__riscv_vmand_mm_##BOOL_RVV_SUF(v_incolumn, v_firstbit_set, vl), vl) > 0;                                               \
                    bool first_all_incolumn = false;                                                                                                                                             \
                    if (tile_id == tid * chunk)                                                                                                                                                  \
                    {                                                                                                                                                                            \
                        first_all_incolumn = first_incolumn;                                                                                                                                     \
                    }                                                                                                                                                                            \
                                                                                                                                                                                                 \
                    v_local_bit = __riscv_vmor_mm_##BOOL_RVV_SUF(v_local_bit, v_firstbit_set, vl);                                                                                               \
                    v_start = __riscv_vmv_v_x_##INT_RVV_SUF(0, vl);                                                                                                                              \
                    v_start = __riscv_vadd_vx_##INT_RVV_SUF##_mu(__riscv_vmnot_m_##BOOL_RVV_SUF(v_local_bit, vl), v_start, v_start, 1, vl);                                                      \
                    v_incolumn = __riscv_vmandn_mm_##BOOL_RVV_SUF(v_local_bit, v_firstbit_set, vl);                                                                                              \
                                                                                                                                                                                                 \
                    v_value = __riscv_vle##FP_SEW##_v_##FP_RVV_SUF(d_value_tile, vl);                                                                                                            \
                    v_column_index = __riscv_vle32_v_##INT_RVV_SUF(d_column_index_tile, vl);                                                                                                     \
                    v_column_index = __riscv_vsll_vx_##INT_RVV_SUF(v_column_index, FP_OFFSET, vl);                                                                                               \
                    v_x = __riscv_vloxei32_v_##FP_RVV_SUF(d_x, v_column_index, vl);                                                                                                              \
                    v_sum = __riscv_vfmul_vv_##FP_RVV_SUF(v_value, v_x, vl);                                                                                                                     \
                    d_column_index_tile += vl;                                                                                                                                                   \
                    d_value_tile += vl;                                                                                                                                                          \
                                                                                                                                                                                                 \
                    int ly = 0;                                                                                                                                                                  \
                    for (int i = 1; i < sigma; ++i)                                                                                                                                              \
                    {                                                                                                                                                                            \
                        v_column_index = __riscv_vle32_v_##INT_RVV_SUF(d_column_index_tile, vl);                                                                                                 \
                        v_column_index = __riscv_vsll_vx_##INT_RVV_SUF(v_column_index, FP_OFFSET, vl);                                                                                           \
                        v_x = __riscv_vloxei32_v_##FP_RVV_SUF(d_x, v_column_index, vl);                                                                                                          \
                        d_column_index_tile += vl;                                                                                                                                               \
                                                                                                                                                                                                 \
                        int norm_i = i - (32 - bit_y_offset - bit_seg_offset);                                                                                                                   \
                        bool end_of_dword = (norm_i % 32 == 0);                                                                                                                                  \
                        if (end_of_dword)                                                                                                                                                        \
                        {                                                                                                                                                                        \
                            ly++;                                                                                                                                                                \
                            d_tile_descriptor_tile += vl;                                                                                                                                        \
                            v_descriptor = __riscv_vle32_v_##INT_RVV_SUF(d_tile_descriptor_tile, vl);                                                                                            \
                        }                                                                                                                                                                        \
                        norm_i = (ly == 0) ? i : norm_i % 32;                                                                                                                                    \
                        const int offset = 31 - norm_i;                                                                                                                                          \
                                                                                                                                                                                                 \
                        v_local_bit = __riscv_vmseq_vx_##INT_RVV_SUF##_##BOOL_RVV_SUF(__riscv_vand_vx_##INT_RVV_SUF(__riscv_vsrl_vx_##INT_RVV_SUF(v_descriptor, offset, vl), 0x1, vl), 0x1, vl); \
                        int store_green = __riscv_vcpop_m_##BOOL_RVV_SUF(v_local_bit, vl);                                                                                                  \
                        if (store_green)                                                                                                                                                    \
                        {                                                                                                                                                                        \
                            v_y_idx = empty_rows                                                                                                                                                 \
                                          ? __riscv_vloxei32_v_##INT_RVV_SUF(d_tile_empty_offset + offset_pointer,                                                                               \
                                                                             __riscv_vsll_vx_##INT_RVV_SUF(v_y_offset, 2, vl), vl)                                                               \
                                          : v_y_offset;                                                                                                                                          \
                                                                                                                                                                                                 \
                            v##BOOL_RVV##_t v_green_segment = __riscv_vmand_mm_##BOOL_RVV_SUF(v_incolumn, v_local_bit, vl);                                                                      \
                            v_y_idx = __riscv_vsll_vx_##INT_RVV_SUF(v_y_idx, FP_OFFSET, vl);                                                                                                     \
                            v_y_local = __riscv_vloxei32_v_##FP_RVV_SUF##_m(v_green_segment, d_y_local, v_y_idx, vl);                                                                            \
                            v_y_local = __riscv_vfmacc_vf_##FP_RVV_SUF(v_y_local, alpha, v_sum, vl);                                                                       \
                            __riscv_vsoxei32_v_##FP_RVV_SUF##_m(v_green_segment, d_y_local, v_y_idx, v_y_local, vl);                                                                             \
                            v_y_offset = __riscv_vadd_vx_##INT_RVV_SUF##_mu(v_green_segment, v_y_offset, v_y_offset, 1, vl);                                                                     \
                                                                                                                                                                                                 \
                            v##BOOL_RVV##_t v_red_segment = __riscv_vmandn_mm_##BOOL_RVV_SUF(v_local_bit, v_incolumn, vl);                                                                       \
                            v_first_sum = __riscv_vmerge_vvm_##FP_RVV_SUF(v_first_sum, v_sum, v_red_segment, vl);                                                                                \
                            v_sum = __riscv_vfmerge_vfm_##FP_RVV_SUF(v_sum, 0.0, v_local_bit, vl);                                                                                               \
                            v_incolumn = __riscv_vmor_mm_##BOOL_RVV_SUF(v_incolumn, v_local_bit, vl);                                                                                            \
                            v_stop = __riscv_vadd_vx_##INT_RVV_SUF##_mu(v_local_bit, v_stop, v_stop, 1, vl);                                                                                     \
                        }                                                                                                                                                                        \
                                                                                                                                                                                                 \
                        v_value = __riscv_vle##FP_SEW##_v_##FP_RVV_SUF(d_value_tile, vl);                                                                                                        \
                        v_sum = __riscv_vfmacc_vv_##FP_RVV_SUF(v_sum, v_value, v_x, vl);                                                                                                         \
                        d_value_tile += vl;                                                                                                                                                      \
                    }                                                                                                                                                                            \
                                                                                                                                                                                                 \
                    v_first_sum = __riscv_vmerge_vvm_##FP_RVV_SUF(v_sum, v_first_sum, v_incolumn, vl);                                                                                           \
                    v_last_sum = __riscv_vmv_v_v_##FP_RVV_SUF(v_sum, vl);                                                                                                                        \
                                                                                                                                                                                                 \
                    v_sum = __riscv_vfmerge_vfm_##FP_RVV_SUF(v_first_sum, (FP)0.0, __riscv_vmseq_vx_##INT_RVV_SUF##_##BOOL_RVV_SUF(v_start, 0x0, vl), vl);                                       \
                                                                                                                                                                                                 \
                    v_sum = __riscv_vfslide1down_vf_##FP_RVV_SUF(v_sum, (FP)0.0, vl);                                                                                                            \
                    v_tmp_sum = __riscv_vmv_v_v_##FP_RVV_SUF(v_sum, vl);                                                                                                                         \
                    for (int offset = 1; offset < vl; offset <<= 1)                                                                                                                              \
                    {                                                                                                                                                                            \
                        v##FP_RVV##FP_LMUL##_t v_tmp_slideup = __riscv_vfmv_v_f_##FP_RVV_SUF((FP)0.0, vl);                                                                                       \
                        v_tmp_slideup = __riscv_vslideup_vx_##FP_RVV_SUF(v_tmp_slideup, v_sum, offset, vl);                                                                                      \
                        v_sum = __riscv_vfadd_vv_##FP_RVV_SUF(v_sum, v_tmp_slideup, vl);                                                                                                         \
                    }                                                                                                                                                                            \
                                                                                                                                                                                                 \
                    const v##INT_RVV##INT_LMUL##_t v_seg_add = __riscv_viota_m_##INT_RVV_SUF(__riscv_vmset_m_##BOOL_RVV_SUF(vl), vl);                                                            \
                    v_seg_offset = __riscv_vadd_vv_##INT_RVV_SUF(v_seg_offset, v_seg_add, vl);                                                                                                   \
                    v##FP_RVV##FP_LMUL##_t v_sum_perm = __riscv_vrgather_vv_##FP_RVV_SUF(v_sum, VCVT_U32_U##FP_SEW(FP_LMUL, v_seg_offset), vl);                                                  \
                                                                                                                                                                                                 \
                    v_sum = __riscv_vfsub_vv_##FP_RVV_SUF(v_sum_perm, v_sum, vl);                                                                                                                \
                    v_sum = __riscv_vfadd_vv_##FP_RVV_SUF(v_sum, v_tmp_sum, vl);                                                                                                                 \
                    v##BOOL_RVV##_t v_add_mask = __riscv_vmsleu_vv_##INT_RVV_SUF##_##BOOL_RVV_SUF(v_start, v_stop, vl);                                                                          \
                    v_last_sum = __riscv_vfadd_vv_##FP_RVV_SUF##_mu(v_add_mask, v_last_sum, v_last_sum, v_sum, vl);                                                                              \
                                                                                                                                                                                                 \
                    v_y_idx = empty_rows                                                                                                                                                         \
                                  ? __riscv_vloxei32_v_##INT_RVV_SUF(d_tile_empty_offset + offset_pointer,                                                                                       \
                                                                     __riscv_vsll_vx_##INT_RVV_SUF(v_y_offset, 2, vl), vl)                                                                       \
                                  : v_y_offset;                                                                                                                                                  \
                    v_y_idx = __riscv_vsll_vx_##INT_RVV_SUF(v_y_idx, FP_OFFSET, vl);                                                                                                             \
                    v_y_local = __riscv_vloxei32_v_##FP_RVV_SUF##_m(v_incolumn, d_y_local, v_y_idx, vl);                                                                                         \
                    v_y_local = __riscv_vfmacc_vf_##FP_RVV_SUF(v_y_local, alpha, v_last_sum, vl);                                                                               \
                    __riscv_vsoxei32_v_##FP_RVV_SUF##_m(v_incolumn, d_y_local, v_y_idx, v_y_local, vl);                                                                                          \
                                                                                                                                                                                                 \
                    bool use_first_sum = __riscv_vcpop_m_##BOOL_RVV_SUF(__riscv_vmand_mm_##BOOL_RVV_SUF(v_firstbit_set, v_incolumn, vl), vl) > 0;                                                \
                    FP add = __riscv_vfmv_f_s_##FP_RVV_SUF##_f##FP_SEW(use_first_sum ? v_first_sum : v_last_sum);                                                                                \
                    add *= alpha;                                                                                                                                                                \
                    if (tile_first_row == thread_first_row && !first_all_incolumn)                                                                                                               \
                        d_calibrator[tid * stride_vT] += add;                                                                                                                                    \
                    else                                                                                                                                                                         \
                    {                                                                                                                                                                            \
                        d_y[tile_first_row] += add;                                                                                                                                              \
                    }                                                                                                                                                                            \
                }                                                                                                                                                                                \
            }                                                                                                                                                                                    \
        }                                                                                                                                                                                        \
    }

    TILE_SIMPLE_LOOP(double, 64, float64, f64m1, m1, 3, uint32, u32mf2, mf2)
    TILE_SIMPLE_LOOP(double, 64, float64, f64m2, m2, 3, uint32, u32m1, m1)
    TILE_SIMPLE_LOOP(double, 64, float64, f64m4, m4, 3, uint32, u32m2, m2)
    TILE_SIMPLE_LOOP(double, 64, float64, f64m8, m8, 3, uint32, u32m4, m4)
    TILE_SIMPLE_LOOP(float, 32, float32, f32m1, m1, 2, uint32, u32m1, m1)
    TILE_SIMPLE_LOOP(float, 32, float32, f32m2, m2, 2, uint32, u32m2, m2)
    TILE_SIMPLE_LOOP(float, 32, float32, f32m4, m4, 2, uint32, u32m4, m4)
    TILE_SIMPLE_LOOP(float, 32, float32, f32m8, m8, 2, uint32, u32m8, m8)
    
    SPMV_CSR5_COMPUTE_KERNEL_RVV(double, 64, float64, f64m1, m1, 3, uint32, u32mf2, mf2, bool64, b64, tile_simple_loop_rvv_f64m1)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(double, 64, float64, f64m2, m2, 3, uint32, u32m1, m1, bool32, b32, tile_simple_loop_rvv_f64m2)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(double, 64, float64, f64m4, m4, 3, uint32, u32m2, m2, bool16, b16, tile_simple_loop_rvv_f64m4)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(double, 64, float64, f64m8, m8, 3, uint32, u32m4, m4, bool8, b8, tile_simple_loop_rvv_f64m8)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(float, 32, float32, f32m1, m1, 2, uint32, u32m1, m1, bool32, b32, tile_simple_loop_rvv_f32m1)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(float, 32, float32, f32m2, m2, 2, uint32, u32m2, m2, bool16, b16, tile_simple_loop_rvv_f32m2)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(float, 32, float32, f32m4, m4, 2, uint32, u32m4, m4, bool8, b8, tile_simple_loop_rvv_f32m4)
    SPMV_CSR5_COMPUTE_KERNEL_RVV(float, 32, float32, f32m8, m8, 2, uint32, u32m8, m8, bool4, b4, tile_simple_loop_rvv_f32m8)

    using compute_kernel_function_double_ptr = decltype(&spmv_csr5_compute_kernel_rvv_f64m1);
    using compute_kernel_function_float_ptr = decltype(&spmv_csr5_compute_kernel_rvv_f32m1);
    template <typename vT>
    void spmv_csr5_calibrate_kernel_rvv(const uint32_t *d_tile_pointer,
                                        vT *d_calibrator,
                                        vT *d_y,
                                        const int p)
    {
        int num_thread = omp_get_max_threads();
        int chunk = iceil(p - 1, num_thread);
        int stride_vT = CSR5_CACHELINE / sizeof(vT);
        int num_thread_active = iceil(p - 1, chunk);
        int num_cali = num_thread_active < num_thread ? num_thread_active : num_thread;

        for (int i = 0; i < num_cali; i++)
        {
            int tile_first_row = remove_sign(d_tile_pointer[i * chunk]);
            d_y[tile_first_row] += d_calibrator[i * stride_vT];
        }
    }

    template <typename vT>
    void spmv_csr5_tail_tile_kernel_rvv(const int *d_row_pointer,
                                        const uint32_t *d_column_index,
                                        const vT *d_value,
                                        const vT *d_x,
                                        vT *d_y,
                                        const int tail_tile_start,
                                        const int p,
                                        const int m,
                                        const vT alpha,
                                        const int omega,
                                        const int sigma)
    {
        const int index_first_element_tail = (p - 1) * omega * sigma;

        for (int row_id = tail_tile_start; row_id < m; row_id++)
        {
            const int idx_start = row_id == tail_tile_start ? (p - 1) * omega * sigma : d_row_pointer[row_id];
            const int idx_stop = d_row_pointer[row_id + 1];

            vT sum = 0;
            for (int idx = idx_start; idx < idx_stop; idx++)
            {
                sum += d_value[idx] * d_x[d_column_index[idx]];
            }
            sum *= alpha;
            d_y[row_id] += sum;
        }
    }

    template <typename VT>
    int csr5_spmv_rvv(const int p,
                      const int m,
                      const int bit_y_offset,
                      const int bit_seg_offset,
                      const int tile_desc_len,
                      const int *row_pointer,
                      const uint32_t *column_index,
                      const VT *value,
                      const uint32_t *tile_pointer,
                      const uint32_t *tile_descriptor,
                      const uint32_t *tile_empty_offset_pointer,
                      const uint32_t *tile_empty_offset,
                      VT *calibrator,
                      const int tail_tile_start,
                      const VT alpha,
                      const VT *x,
                      VT *y,
                      const int omega,
                      const int sigma);

    template <>
    int csr5_spmv_rvv<double>(const int p,
                              const int m,
                              const int bit_y_offset,
                              const int bit_seg_offset,
                              const int tile_desc_len,
                              const int *row_pointer,
                              const uint32_t *column_index,
                              const double *value,
                              const uint32_t *tile_pointer,
                              const uint32_t *tile_descriptor,
                              const uint32_t *tile_empty_offset_pointer,
                              const uint32_t *tile_empty_offset,
                              double *calibrator,
                              const int tail_tile_start,
                              const double alpha,
                              const double *x,
                              double *y,
                              const int omega,
                              const int sigma)
    {
        int err = CSR5_SUCCESS;

        const int num_thread = omp_get_max_threads();
        memset(calibrator, 0, CSR5_CACHELINE * num_thread);

        // Choose kernel based on LMUL
        compute_kernel_function_double_ptr compute_kernel_rvv;
        if (omega == __riscv_vsetvlmax_e64m1()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f64m1;
        } else if (omega == __riscv_vsetvlmax_e64m2()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f64m2;
        } else if (omega == __riscv_vsetvlmax_e64m4()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f64m4;
        } else if (omega == __riscv_vsetvlmax_e64m8()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f64m8;
        } else {
            assert(0);
        }
        compute_kernel_rvv(column_index, value, row_pointer, x,
                           tile_pointer, tile_descriptor,
                           tile_empty_offset_pointer, tile_empty_offset,
                           calibrator, y, p,
                           tile_desc_len, bit_y_offset, bit_seg_offset, alpha,
                           omega, sigma);

        spmv_csr5_calibrate_kernel_rvv(tile_pointer, calibrator, y, p);

        spmv_csr5_tail_tile_kernel_rvv(row_pointer, column_index, value, x, y,
                                       tail_tile_start, p, m, alpha,
                                       omega, sigma);

        return err;
    }

    template <>
    int csr5_spmv_rvv<float>(const int p,
                             const int m,
                             const int bit_y_offset,
                             const int bit_seg_offset,
                             const int tile_desc_len,
                             const int *row_pointer,
                             const uint32_t *column_index,
                             const float *value,
                             const uint32_t *tile_pointer,
                             const uint32_t *tile_descriptor,
                             const uint32_t *tile_empty_offset_pointer,
                             const uint32_t *tile_empty_offset,
                             float *calibrator,
                             const int tail_tile_start,
                             const float alpha,
                             const float *x,
                             float *y,
                             const int omega,
                             const int sigma)
    {
        int err = CSR5_SUCCESS;

        const int num_thread = omp_get_max_threads();
        memset(calibrator, 0, CSR5_CACHELINE * num_thread);

        // Choose kernel based on LMUL
        compute_kernel_function_float_ptr compute_kernel_rvv;
        if (omega == __riscv_vsetvlmax_e32m1()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f32m1;
        } else if (omega == __riscv_vsetvlmax_e32m2()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f32m2;
        } else if (omega == __riscv_vsetvlmax_e32m4()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f32m4;
        } else if (omega == __riscv_vsetvlmax_e32m8()) {
            compute_kernel_rvv = spmv_csr5_compute_kernel_rvv_f32m8;
        } else {
            assert(0);
        }
        compute_kernel_rvv(column_index, value, row_pointer, x,
                           tile_pointer, tile_descriptor,
                           tile_empty_offset_pointer, tile_empty_offset,
                           calibrator, y, p,
                           tile_desc_len, bit_y_offset, bit_seg_offset, alpha,
                           omega, sigma);

        spmv_csr5_calibrate_kernel_rvv(tile_pointer, calibrator, y, p);

        spmv_csr5_tail_tile_kernel_rvv(row_pointer, column_index, value, x, y,
                                       tail_tile_start, p, m, alpha,
                                       omega, sigma);

        return err;
    }

    template <>
    sparse_matrix_status sparse_mv<double, spMtxCSR5, true, SPARSE_MATRIX_MV_ALL_STAGES>(
        sparse_operation_t type_op,
        double alpha,
        const spMtxCSR5<double> &mat,
        sparse_matrix_descr descr,
        const std::vector<double> &b,
        double beta,
        std::vector<double> &y)
    {
        sparse_matrix_status status;

        auto calibrator = mat._temp_calibrator;
        int num_of_threads = get_num_of_threads();

#pragma omp parallel for
        for (int i = 0; i < mat._m; ++i)
        {
            y[i] *= beta;
        }

        status.code = csr5_spmv_rvv(
            mat._p,
            mat._m,
            mat._bit_y_offset,
            mat._bit_seg_offset,
            mat._tile_desc_len,
            mat._csr_row_pointer.data(),
            mat._csr_column_index.data(),
            mat._csr_value.data(),
            mat._csr5_tile_pointer.data(),
            mat._csr5_tile_descriptor.data(),
            mat._csr5_tile_empty_offset_pointer.data(),
            mat._csr5_tile_empty_offset.data(),
            calibrator.data(),
            mat._tail_tile_start,
            alpha,
            b.data(),
            y.data(),
            mat._omega,
            mat._sigma);

        return status;
    }

    template <>
    sparse_matrix_status sparse_mv<float, spMtxCSR5, true, SPARSE_MATRIX_MV_ALL_STAGES>(
        sparse_operation_t type_op,
        float alpha,
        const spMtxCSR5<float> &mat,
        sparse_matrix_descr descr,
        const std::vector<float> &b,
        float beta,
        std::vector<float> &y)
    {
        sparse_matrix_status status;

        auto calibrator = mat._temp_calibrator;
        int num_of_threads = get_num_of_threads();

#pragma omp parallel for
        for (int i = 0; i < mat._m; ++i)
        {
            y[i] *= beta;
        }

        status.code = csr5_spmv_rvv(
            mat._p,
            mat._m,
            mat._bit_y_offset,
            mat._bit_seg_offset,
            mat._tile_desc_len,
            mat._csr_row_pointer.data(),
            mat._csr_column_index.data(),
            mat._csr_value.data(),
            mat._csr5_tile_pointer.data(),
            mat._csr5_tile_descriptor.data(),
            mat._csr5_tile_empty_offset_pointer.data(),
            mat._csr5_tile_empty_offset.data(),
            calibrator.data(),
            mat._tail_tile_start,
            alpha,
            b.data(),
            y.data(),
            mat._omega,
            mat._sigma);

        return status;
    }

}
