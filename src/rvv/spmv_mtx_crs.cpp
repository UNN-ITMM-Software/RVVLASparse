#include "sparse_matrix.h"
#include "spmv_mtx.h"

#include <riscv_vector.h>

namespace SparseMatrixLib
{

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
  unsigned int gvl_max;
  
  gvl_max = __riscv_vsetvlmax_e64m4();
  
#pragma omp parallel for
  for (int i = 0; i < mat.m; i++) {
    double tmp = 0.0;
    int j = mat.Rst[i];
    unsigned int gvl = gvl_max; // __riscv_vsetvl_e64m4(mat.Rst[i + 1] - mat.Rst[i]); //  
    vfloat64m4_t res = __riscv_vfmv_v_f_f64m4(0.0, gvl);

    while (j + gvl <= mat.Rst[i + 1]) {
      vfloat64m4_t val  = __riscv_vle64_v_f64m4(mat.Val + j, gvl);
      vuint32m2_t  index = __riscv_vle32_v_u32m2(reinterpret_cast<uint32_t *>(mat.Col + j), gvl);    
      vuint32m2_t index_shift = __riscv_vsll_vx_u32m2(index, 3, gvl);    
      vfloat64m4_t b_   = __riscv_vloxei32_v_f64m4(b.data(), index_shift, gvl);
      res = __riscv_vfmadd_vv_f64m4(val, b_, res, gvl);
      j += gvl;
    }
    
    vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0.0, gvl);
    sum = __riscv_vfredosum_vs_f64m4_f64m1(res, sum, gvl);
    tmp = __riscv_vfmv_f_s_f64m1_f64(sum);

    for (; j < mat.Rst[i + 1]; j++)
      tmp += mat.Val[j] * b[mat.Col[j]];
    
    y[i] = tmp * alpha + beta * y[i];
  }

  return status;
}



template<> sparse_matrix_status sparse_mv<float, spMtxCRS, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxCRS<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;
  unsigned int gvl_max;

  gvl_max = __riscv_vsetvlmax_e32m4();

#pragma omp parallel for
  for (int i = 0; i < mat.m; i++) {
    double tmp = 0.0;
    int j = mat.Rst[i];
    vfloat32m4_t val;
    vuint32m4_t  index;
    vuint32m4_t index_shift;
    vfloat32m4_t b_;
    vfloat32m1_t sum;
    unsigned int gvl = gvl_max;  
    vfloat32m4_t res = __riscv_vfmv_v_f_f32m4(0.0, gvl);

    while (j + gvl <= mat.Rst[i + 1]) {
      val  = __riscv_vle32_v_f32m4(mat.Val + j, gvl);
      index = __riscv_vle32_v_u32m4(reinterpret_cast<uint32_t *>(mat.Col + j), gvl);    
      index_shift = __riscv_vsll_vx_u32m4(index, 2, gvl);    
      b_   = __riscv_vloxei32_v_f32m4(b.data(), index_shift, gvl);
      res = __riscv_vfmadd_vv_f32m4(val, b_, res, gvl);
      j += gvl;
    }

    sum = __riscv_vfmv_v_f_f32m1(0.0, gvl);
    sum = __riscv_vfredosum_vs_f32m4_f32m1(res, sum, gvl);
    tmp = __riscv_vfmv_f_s_f32m1_f32(sum);

    for (; j < mat.Rst[i + 1]; j++)
     tmp += mat.Val[j] * b[mat.Col[j]];
    y[i] = tmp * alpha + beta * y[i];
  }

  
  return status;
}

}
