#include "sparse_matrix.h"
#include "spmv_mtx.h"

#include <riscv_vector.h>

namespace SparseMatrixLib
{

#define SPMV_CSR_RVV(                                                          \
    FP_TYPE,                                                                   \
    FP_SEW,                                                                    \
    FP_RVV,                                                                    \
    FP_RVV_SUF,                                                                \
    FP_LMUL,                                                                   \
    FP_OFFSET,                                                                 \
    INT_TYPE,                                                                  \
    INT_RVV_SUF,                                                               \
    INT_LMUL,                                                                  \
    FUNC_NAME)                                                                 \
                                                                               \
void FUNC_NAME(const spMtxCRS<FP_TYPE>& mat,                                   \
               const std::vector<FP_TYPE>& b,                                  \
               std::vector<FP_TYPE>& y,                                        \
               FP_TYPE alpha,                                                  \
               FP_TYPE beta)                                                   \
{                                                                              \
    const int vlmax = __riscv_vsetvlmax_e##FP_SEW##FP_LMUL();                  \
                                                                               \
    _Pragma("omp parallel for")                                                \
    for (int i = 0; i < mat.m; ++i)                                            \
    {                                                                          \
        int vl = vlmax;                                                        \
                                                                               \
        v##FP_RVV##FP_LMUL##_t res =                                           \
            __riscv_vfmv_v_f_##FP_RVV_SUF((FP_TYPE)0, vlmax);                  \
                                                                               \
        for (int j = mat.Rst[i]; j < mat.Rst[i + 1]; j += vl)                  \
        {                                                                      \
            vl = __riscv_vsetvl_e##FP_SEW##FP_LMUL(mat.Rst[i + 1] - j);        \
                                                                               \
            v##FP_RVV##FP_LMUL##_t val =                                       \
                __riscv_vle##FP_SEW##_v_##FP_RVV_SUF(mat.Val + j, vl);         \
                                                                               \
            v##INT_TYPE##INT_LMUL##_t index =                                  \
                __riscv_vle32_v_##INT_RVV_SUF(                                 \
                    reinterpret_cast<const uint32_t*>(mat.Col + j), vl);       \
                                                                               \
            v##INT_TYPE##INT_LMUL##_t index_shift =                            \
            __riscv_vsll_vx_##INT_RVV_SUF(index, FP_OFFSET, vl);               \
                                                                               \
            v##FP_RVV##FP_LMUL##_t x =                                         \
                __riscv_vloxei32_v_##FP_RVV_SUF(b.data(), index_shift, vl);    \
                                                                               \
            res = __riscv_vfmacc_vv_##FP_RVV_SUF##_tu(res, val, x, vl);        \
        }                                                                      \
                                                                               \
        vl = __riscv_vsetvl_e##FP_SEW##FP_LMUL(vlmax);                         \
                                                                               \
        v##FP_RVV##m1_t red_zero =                                             \
            __riscv_vfmv_v_f_f##FP_SEW##m1((FP_TYPE)0, 1);                     \
                                                                               \
        v##FP_RVV##m1_t sum =                                                  \
            __riscv_vfredusum_vs_##FP_RVV_SUF##_f##FP_SEW##m1(                 \
                res, red_zero, vl);                                            \
                                                                               \
        FP_TYPE tmp = __riscv_vfmv_f_s_f##FP_SEW##m1_f##FP_SEW(sum);           \
                                                                               \
        y[i] = alpha * tmp + beta * y[i];                                      \
    }                                                                          \
}

SPMV_CSR_RVV(double, 64, float64, f64m1, m1, 3, uint32, u32mf2, mf2, spmv_crs_f64m1)
SPMV_CSR_RVV(double, 64, float64, f64m2, m2, 3, uint32, u32m1, m1, spmv_crs_f64m2)
SPMV_CSR_RVV(double, 64, float64, f64m4, m4, 3, uint32, u32m2, m2, spmv_crs_f64m4)
SPMV_CSR_RVV(double, 64, float64, f64m8, m8, 3, uint32, u32m4, m4, spmv_crs_f64m8)
SPMV_CSR_RVV(float, 32, float32, f32m1, m1, 2, uint32, u32m1, m1, spmv_crs_f32m1)
SPMV_CSR_RVV(float, 32, float32, f32m2, m2, 2, uint32, u32m2, m2, spmv_crs_f32m2)
SPMV_CSR_RVV(float, 32, float32, f32m4, m4, 2, uint32, u32m4, m4, spmv_crs_f32m4)
SPMV_CSR_RVV(float, 32, float32, f32m8, m8, 2, uint32, u32m8, m8, spmv_crs_f32m8)

template<>
sparse_matrix_status sparse_mv<double, spMtxCRS, true>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxCRS<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
    sparse_matrix_status status;
    spmv_crs_f64m4(mat, b, y, alpha, beta);
    return status;
}

template<>
sparse_matrix_status sparse_mv<float, spMtxCRS, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxCRS<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
    sparse_matrix_status status;
    spmv_crs_f32m1(mat, b, y, alpha, beta);
    return status;
}

}
