#include "sparse_matrix.h"
#include <utility>
#include <algorithm>
#include <vector>

namespace SparseMatrixLib
{

  template<>
  sparse_matrix_status sparse_mv<double, spMtxCVR, false>(
    sparse_operation_t type_op,
    double alpha,
    const spMtxCVR<double>& mat,
    sparse_matrix_descr descr,
    const std::vector<double>& b,
    double beta,
    std::vector<double>& y) {
    sparse_matrix_status status;

    for (int iter_y = 0; iter_y < y.size(); iter_y++) {
      y[iter_y] *= beta;
    }

    std::vector<double> v_res(mat.SIMD_LEN, 0);
    double *res = v_res.data();

    int iter_record = 0, iter_record_max = (mat.vPack_vec_record.size() - 2) / 2, left_bound = 0;

    while (iter_record < iter_record_max)
    {
      for (; left_bound + mat.SIMD_LEN <= mat.vPack_vec_record[2 * iter_record]; left_bound += mat.SIMD_LEN)
      {
        for (int i = 0; i < mat.SIMD_LEN; i++)
        {
          res[i] += mat.vPack_vec_vals[left_bound + i] * b[mat.vPack_vec_cols[left_bound + i]];
        }
      }

      int pos = mat.vPack_vec_record[2 * iter_record] % mat.SIMD_LEN;

      y[mat.vPack_vec_record[2 * iter_record + 1]] += alpha * res[pos];
      res[pos] = .0;
      iter_record++;
    }

    for (; left_bound + mat.SIMD_LEN <= mat.nItems; left_bound += mat.SIMD_LEN)
      for (int i = 0; i < mat.SIMD_LEN; i++)
        res[i] += mat.vPack_vec_vals[left_bound + i] * b[mat.vPack_vec_cols[left_bound + i]];

    const int* tail = mat.writeback_pos_record.data() + (mat.Nthreads_avail - 1) * (mat.SIMD_LEN + 2) + 2;

    for (int i = 0; i < mat.SIMD_LEN; i++) {
      y[tail[i]] += alpha * res[i];
    }
    return status;
  }


  template<>
  sparse_matrix_status sparse_mv<float, spMtxCVR, false>(
    sparse_operation_t type_op,
    float alpha,
    const spMtxCVR<float>& mat,
    sparse_matrix_descr descr,
    const std::vector<float>& b,
    float beta,
    std::vector<float>& y) {
    sparse_matrix_status status;

    for (int iter_y = 0; iter_y < y.size(); iter_y++) {
      y[iter_y] *= beta;
    }

    std::vector<float> v_res(mat.SIMD_LEN, 0);
    float *res = v_res.data();

    int iter_record = 0, iter_record_max = (mat.vPack_vec_record.size() - 2) / 2, left_bound = 0;

    while (iter_record < iter_record_max)
    {
      for (; left_bound + mat.SIMD_LEN <= mat.vPack_vec_record[2 * iter_record]; left_bound += mat.SIMD_LEN)
      {
        for (int i = 0; i < mat.SIMD_LEN; i++)
        {
          res[i] += mat.vPack_vec_vals[left_bound + i] * b[mat.vPack_vec_cols[left_bound + i]];
        }
      }

      int pos = mat.vPack_vec_record[2 * iter_record] % mat.SIMD_LEN;

      y[mat.vPack_vec_record[2 * iter_record + 1]] += alpha * res[pos];
      res[pos] = .0;
      iter_record++;
    }

    for (; left_bound + mat.SIMD_LEN <= mat.nItems; left_bound += mat.SIMD_LEN)
      for (int i = 0; i < mat.SIMD_LEN; i++)
        res[i] += mat.vPack_vec_vals[left_bound + i] * b[mat.vPack_vec_cols[left_bound + i]];

    const int* tail = mat.writeback_pos_record.data() + (mat.Nthreads_avail - 1) * (mat.SIMD_LEN + 2) + 2;

    for (int i = 0; i < mat.SIMD_LEN; i++) {
      y[tail[i]] += alpha * res[i];
    }
    return status;
  }
}