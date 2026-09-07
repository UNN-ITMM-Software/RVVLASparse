#include "sparse_matrix.h"
#include "spmv_mtx.h"
#include "omp.h"

#include <riscv_vector.h>

namespace SparseMatrixLib
{
  template<>
  sparse_matrix_status sparse_mv<double, spMtxCVR, true>(
    sparse_operation_t type_op,
    double alpha,
    const spMtxCVR<double>& mat,
    sparse_matrix_descr descr,
    const std::vector<double>& b,
    double beta,
    std::vector<double>& y) {
    sparse_matrix_status status;
    unsigned int VL;
    VL = __riscv_vsetvlmax_e64m2();

    int iter_y = 0;
    vfloat64m2_t y_copy;

    for (; iter_y + VL < y.size(); iter_y += VL) {
      y_copy = __riscv_vle64_v_f64m2(y.data() + iter_y, VL);
      y_copy = __riscv_vfmul_vf_f64m2(y_copy, beta, VL);
      __riscv_vse64_v_f64m2(y.data() + iter_y, y_copy, VL);
    }

    if (y.size() - iter_y) {
      y_copy = __riscv_vle64_v_f64m2(y.data() + iter_y, y.size() - iter_y);
      y_copy = __riscv_vfmul_vf_f64m2(y_copy, beta, y.size() - iter_y);
      __riscv_vse64_v_f64m2(y.data() + iter_y, y_copy, y.size() - iter_y);
    }

    int iter_record = 0;

#pragma omp parallel num_threads(mat.Nthreads)
    {
      int thread_num = omp_get_thread_num();
      int left_bound = mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2)];
      int right_bound = mat.writeback_pos_record[(thread_num + 1) * (mat.SIMD_LEN + 2)];
      int iter_record = mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2) + 1];
      int iter_record_max = mat.writeback_pos_record[(thread_num + 1) * (mat.SIMD_LEN + 2) + 1];

      double res[mat.SIMD_LEN] = { .0 };
      vfloat64m2_t res_vec = __riscv_vfmv_v_f_f64m2(0.0, VL);

      while (left_bound < right_bound && iter_record < iter_record_max)
      {
        for (; left_bound + VL <= mat.vPack_vec_record[2 * iter_record] && left_bound < right_bound; left_bound += VL)
        {
          vfloat64m2_t val = __riscv_vle64_v_f64m2(mat.vPack_vec_vals.data() + left_bound, VL);

          vuint32m1_t col_index = __riscv_vle32_v_u32m1(mat.vPack_vec_cols.data() + left_bound, VL);
          vuint32m1_t col_index_shift = __riscv_vsll_vx_u32m1(col_index, 3, VL);

          vfloat64m2_t b_ = __riscv_vloxei32_v_f64m2(b.data(), col_index_shift, VL);

          res_vec = __riscv_vfmadd_vv_f64m2(val, b_, res_vec, VL);
        }
        __riscv_vse64_v_f64m2(res, res_vec, VL);

#pragma omp atomic
        y[mat.vPack_vec_record[2 * iter_record + 1]] += alpha * res[mat.vPack_vec_record[2 * iter_record] % VL];

        res[mat.vPack_vec_record[2 * iter_record] % VL] = 0.0;
        res_vec = __riscv_vle64_v_f64m2(res, VL);

        iter_record++;

      }

      for (; left_bound + VL <= right_bound; left_bound += VL)
      {
        vfloat64m2_t val = __riscv_vle64_v_f64m2(mat.vPack_vec_vals.data() + left_bound, VL);

        vuint32m1_t col_index = __riscv_vle32_v_u32m1(mat.vPack_vec_cols.data() + left_bound, VL);
        vuint32m1_t col_index_shift = __riscv_vsll_vx_u32m1(col_index, 3, VL);

        vfloat64m2_t b_ = __riscv_vloxei32_v_f64m2(b.data(), col_index_shift, VL);

        res_vec = __riscv_vfmadd_vv_f64m2(val, b_, res_vec, VL);
      }

      res_vec = __riscv_vfmul_vf_f64m2(res_vec, alpha, VL);
      __riscv_vse64_v_f64m2(res, res_vec, VL);

      for (int i = 0; i < VL; i++) {
#pragma omp atomic
        y[mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2) + 2 + i]] += res[i];
      }
    }

    return status;
  }

  template<>
  sparse_matrix_status sparse_mv<float, spMtxCVR, true>(
    sparse_operation_t type_op,
    float alpha,
    const spMtxCVR<float>& mat,
    sparse_matrix_descr descr,
    const std::vector<float>& b,
    float beta,
    std::vector<float>& y) {
    sparse_matrix_status status;
    unsigned int VL;
    VL = __riscv_vsetvlmax_e32m1();

    int iter_y = 0;
    vfloat32m1_t y_copy;

    for (; iter_y + VL < y.size(); iter_y += VL) {
      y_copy = __riscv_vle32_v_f32m1(y.data() + iter_y, VL);
      y_copy = __riscv_vfmul_vf_f32m1(y_copy, beta, VL);
      __riscv_vse32_v_f32m1(y.data() + iter_y, y_copy, VL);
    }

    if (y.size() - iter_y) {
      y_copy = __riscv_vle32_v_f32m1(y.data() + iter_y, y.size() - iter_y);
      y_copy = __riscv_vfmul_vf_f32m1(y_copy, beta, y.size() - iter_y);
      __riscv_vse32_v_f32m1(y.data() + iter_y, y_copy, y.size() - iter_y);
    }

    int iter_record = 0;

#pragma omp parallel num_threads(mat.Nthreads)
    {
      int thread_num = omp_get_thread_num();
      int left_bound = mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2)];
      int right_bound = mat.writeback_pos_record[(thread_num + 1) * (mat.SIMD_LEN + 2)];
      int iter_record = mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2) + 1];
      int iter_record_max = mat.writeback_pos_record[(thread_num + 1) * (mat.SIMD_LEN + 2) + 1];

      float res[mat.SIMD_LEN] = { .0 };
      vfloat32m1_t res_vec = __riscv_vfmv_v_f_f32m1(0.0, VL);

      while (left_bound < right_bound && iter_record < iter_record_max)
      {
        for (; left_bound + VL <= mat.vPack_vec_record[2 * iter_record] && left_bound < right_bound; left_bound += VL)
        {
          vfloat32m1_t val = __riscv_vle32_v_f32m1(mat.vPack_vec_vals.data() + left_bound, VL);

          vuint32m1_t col_index = __riscv_vle32_v_u32m1(mat.vPack_vec_cols.data() + left_bound, VL);
          vuint32m1_t col_index_shift = __riscv_vsll_vx_u32m1(col_index, 2, VL);

          vfloat32m1_t b_ = __riscv_vloxei32_v_f32m1(b.data(), col_index_shift, VL);

          res_vec = __riscv_vfmadd_vv_f32m1(val, b_, res_vec, VL);
        }
        __riscv_vse32_v_f32m1(res, res_vec, VL);

#pragma omp atomic
        y[mat.vPack_vec_record[2 * iter_record + 1]] += alpha * res[mat.vPack_vec_record[2 * iter_record] % VL];

        res[mat.vPack_vec_record[2 * iter_record] % VL] = 0.0;
        res_vec = __riscv_vle32_v_f32m1(res, VL);

        iter_record++;

      }

      for (; left_bound + VL <= right_bound; left_bound += VL)
      {
        vfloat32m1_t val = __riscv_vle32_v_f32m1(mat.vPack_vec_vals.data() + left_bound, VL);

        vuint32m1_t col_index = __riscv_vle32_v_u32m1(mat.vPack_vec_cols.data() + left_bound, VL);
        vuint32m1_t col_index_shift = __riscv_vsll_vx_u32m1(col_index, 2, VL);

        vfloat32m1_t b_ = __riscv_vloxei32_v_f32m1(b.data(), col_index_shift, VL);

        res_vec = __riscv_vfmadd_vv_f32m1(val, b_, res_vec, VL);
      }
      res_vec = __riscv_vfmul_vf_f32m1(res_vec, alpha, VL);
      __riscv_vse32_v_f32m1(res, res_vec, VL);

      for (int i = 0; i < VL; i++) {
#pragma omp atomic
        y[mat.writeback_pos_record[thread_num * (mat.SIMD_LEN + 2) + 2 + i]] += res[i];
      }
    }

    return status;
  }
}
