#include "sparse_matrix.h"

#include <iomanip>
#include <cmath>

namespace SparseMatrixLib
{

inline void HCSR_spmv_ker_crs_double(double alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const double* values, const double* b, double* y, int m) {
    for (int i = 0; i < m; ++i) {
        double tmp = 0.0;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            tmp += values[j] * b[col_idx[j]];
        y[i] += alpha * tmp;
    }
}

inline void HCSR_spmv_ker_crs_float(float alpha, const uint32_t* row_ptr, const uint16_t* col_idx, const float* values, const float* b, float* y, int m) {
    for (int i = 0; i < m; ++i) {
        float tmp = 0.0f;
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j)
            tmp += values[j] * b[col_idx[j]];
        y[i] += alpha * tmp;
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
#pragma omp parallel for
    for (int i = 0; i < mat.m; i += mat.R) {
        int maxr = std::min(mat.R, mat.m - i);
        for (int r = 0; r < maxr; ++r)
            y[i + r] *= beta;

        for (int j = 0; j < mat.n; j += mat.C) { // for every block in row: 
            int block = j / mat.C + (i / mat.R) * mat.b_n;
            int bptr = mat.block_ptr[block];
            uint32_t row_ptr_offset = mat.row_offset[block];

            uint32_t mask = ~(0xC000'0000u);
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
#pragma omp parallel for
    for (int i = 0; i < mat.m; i += mat.R) {
        int maxr = std::min(mat.R, mat.m - i);
        for (int r = 0; r < maxr; ++r)
            y[i + r] *= beta;

        for (int j = 0; j < mat.n; j += mat.C) { // for every block in row: 
            int block = j / mat.C + (i / mat.R) * mat.b_n;
            int bptr = mat.block_ptr[block];
            uint32_t row_ptr_offset = mat.row_offset[block];

            uint32_t mask = ~(0xC000'0000u);
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
sparse_matrix_status sparse_mv<double, spMtxHCSR, false>(
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
sparse_matrix_status sparse_mv<float, spMtxHCSR, false>(
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

}