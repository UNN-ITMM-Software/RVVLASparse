#include "sparse_matrix.h"
#include "spmv_mtx.h"

#include <riscv_vector.h>

namespace SparseMatrixLib
{

inline void HCSR_spmv_ker_crs_double(double alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const double* values, const double* b, double* y, int m) {
    int vlmax = __riscv_vsetvlmax_e64m1();
    for (int i = 0; i < m; ++i) {
        double tmp = 0.0;
        int vl = vlmax;
        vfloat64m1_t res = __riscv_vfmv_v_f_f64m1(0.0, vl);

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j += vlmax) {
            vl = __riscv_vsetvl_e64m1(row_ptr[i + 1] - j);
            vfloat64m1_t val = __riscv_vle64_v_f64m1(values + j, vl);
            vuint16mf4_t index = __riscv_vle16_v_u16mf4(col_idx + j, vl); // reinterpret cast
            vuint16mf4_t index_shift = __riscv_vsll_vx_u16mf4(index, 3, vl);
            vfloat64m1_t b_ = __riscv_vloxei16_v_f64m1(b, index_shift, vl);
            res = __riscv_vfmacc_vv_f64m1_tu(res, val, b_, vl);
        }
        vl = __riscv_vsetvl_e64m1(vlmax);
        vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0.0, 1);
        sum = __riscv_vfredosum_vs_f64m1_f64m1(res, sum, vl);
        tmp = __riscv_vfmv_f_s_f64m1_f64(sum);

        y[i] = alpha * tmp + y[i];
    }
}

inline void HCSR_spmv_ker_crs_float(float alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const float* values, const float* b, float* y, int m) {
    int vlmax = __riscv_vsetvlmax_e32m1();
    for (int i = 0; i < m; ++i) {
        float tmp = 0.0f;
        int vl = vlmax;
        vfloat32m1_t res = __riscv_vfmv_v_f_f32m1(0.0, vl);

        for (int j = row_ptr[i]; j < row_ptr[i + 1]; j += vlmax) {
            vl = __riscv_vsetvl_e32m1(row_ptr[i + 1] - j);
            vfloat32m1_t val = __riscv_vle32_v_f32m1(values + j, vl);
            vuint16mf2_t index = __riscv_vle16_v_u16mf2(col_idx + j, vl); // reinterpret cast
            vuint16mf2_t index_shift = __riscv_vsll_vx_u16mf2(index, 2, vl);
            vfloat32m1_t b_ = __riscv_vloxei16_v_f32m1(b, index_shift, vl);
            res = __riscv_vfmacc_vv_f32m1_tu(res, val, b_, vl);
        }
        vl = __riscv_vsetvl_e32m1(vlmax);
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0, 1);
        sum = __riscv_vfredosum_vs_f32m1_f32m1(res, sum, vl);
        tmp = __riscv_vfmv_f_s_f32m1_f32(sum);

        y[i] = alpha * tmp + y[i];
    }
}

inline void HCSR_spmv_ker_coo_double(double alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const double* values, const double* b, double* y, int m, int nz) {
    for (int i = 0; i < nz; ++i) {
        y[row_ptr[i]] += alpha * values[i] * b[col_idx[i]];
    }
}

inline void HCSR_spmv_ker_coo_float(float alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const float* values, const float* b, float* y, int m, int nz) {
    for (int i = 0; i < nz; ++i) {
        y[row_ptr[i]] += alpha * values[i] * b[col_idx[i]];
    }
}

inline void HCSR_spmv_double(double alpha, const spMtxHCSR<double>& mat, const std::vector<double>& b, double beta, std::vector<double>& y) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < mat.m; i += mat.R) {
        int maxr = std::min(mat.R, mat.m - i);
        for (int r = 0; r < maxr; ++r)
            y[i + r] *= beta;

        for (int j = 0; j < mat.n; j += mat.C) { // for every block in row: 
            int block = j / mat.C + (i / mat.R) * mat.b_n;
            int bptr = mat.block_ptr[block];
            uint32_t row_ptr_offset = mat.row_offset[block];

            uint32_t mask = 0x3FFF'FFFFu;
            uint32_t key = row_ptr_offset >> 30;
            row_ptr_offset &= mask;

            int nz = mat.block_ptr[block + 1] - bptr;
            if (key == 1)
                HCSR_spmv_ker_crs_double(alpha, mat.row_ptr.data() + row_ptr_offset, mat.col_idx.data() + bptr, mat.values.data() + bptr, b.data() + j, y.data() + i, maxr);
            else
                HCSR_spmv_ker_coo_double(alpha, mat.row_ptr.data() + row_ptr_offset, mat.col_idx.data() + bptr, mat.values.data() + bptr, b.data() + j, y.data() + i, maxr, nz);
        }

    }
}

inline void HCSR_spmv_float(float alpha, const spMtxHCSR<float>& mat, const std::vector<float>& b, float beta, std::vector<float>& y) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < mat.m; i += mat.R) {
        int maxr = std::min(mat.R, mat.m - i);
        for (int r = 0; r < maxr; ++r)
            y[i + r] *= beta;

        for (int j = 0; j < mat.n; j += mat.C) { // for every block in row: 
            int block = j / mat.C + (i / mat.R) * mat.b_n;
            int bptr = mat.block_ptr[block];
            uint32_t row_ptr_offset = mat.row_offset[block];

            uint32_t mask = 0x3FFF'FFFFu;
            uint32_t key = row_ptr_offset >> 30;
            row_ptr_offset &= mask;

            int nz = mat.block_ptr[block + 1] - bptr;
            if (key == 1)
                HCSR_spmv_ker_crs_float(alpha, mat.row_ptr.data() + row_ptr_offset, mat.col_idx.data() + bptr, mat.values.data() + bptr, b.data() + j, y.data() + i, maxr);
            else
                HCSR_spmv_ker_coo_float(alpha, mat.row_ptr.data() + row_ptr_offset, mat.col_idx.data() + bptr, mat.values.data() + bptr, b.data() + j, y.data() + i, maxr, nz);
        }

    }
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxHCSR<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  return status;
}


template<> 
sparse_matrix_status sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxHCSR<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  HCSR_spmv_double(alpha, mat, b, beta, y);

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxHCSR<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_ALL_STAGES>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxHCSR<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxHCSR, true, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxHCSR<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxHCSR<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  HCSR_spmv_float(alpha, mat, b, beta, y);

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxHCSR<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  return status;
}


template<> 
sparse_matrix_status sparse_mv<float, spMtxHCSR, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxHCSR<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxHCSR, true, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

}