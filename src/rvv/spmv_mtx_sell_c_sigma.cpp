#include "sparse_matrix.h"

#include <riscv_vector.h>

#include <iomanip>
#include <cmath>

namespace SparseMatrixLib
{

template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  return status;
}


template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  double *py = y.data();
  const double * pb = b.data();
  const int *Cl = mat.vCl.data();
  const int *Cs = mat.vCs.data();
  const int *Bs = mat.vBs.data();
  const int *SBs = mat.vSBs.data();
  const uint32_t* Col  = mat.vCol.data();
  const uint32_t* Perm = mat.vPerm.data();
  const double* Val = mat.vVal.data();

#pragma omp parallel for
  for(int i = 0; i < mat.cnt_b; i++)
  {
    int cur_pos = SBs[i];
    uint32_t bs = __riscv_vsetvlmax_e64m2();
    for (int k = 0; k < Bs[i]; k+= bs)
    {
      bs = __riscv_vsetvl_e64m2(Bs[i] - k);
      vfloat64m2_t v_py = __riscv_vfmv_v_f_f64m2(0.0, bs);
      for (int j = 0; j < Cl[i]; j++)
      {
        vuint32m1_t index = __riscv_vle32_v_u32m1(Col + Cs[i]+ j * Bs[i] + k, bs);    
        vuint32m1_t index_shift = __riscv_vsll_vx_u32m1(index, 3, bs);    
        vfloat64m2_t v_val = __riscv_vle64_v_f64m2(Val + Cs[i]+ j * Bs[i] + k, bs);
        vfloat64m2_t v_b        = __riscv_vloxei32_v_f64m2(pb, index_shift, bs);
        v_py = __riscv_vfmacc_vv_f64m2(v_py, v_val, v_b, bs);
      }

      vuint32m1_t index_perm = __riscv_vle32_v_u32m1(Perm + cur_pos + k, bs);    
      vuint32m1_t index_perm_shift = __riscv_vsll_vx_u32m1(index_perm, 3, bs);    
      vfloat64m2_t v_py_c = __riscv_vloxei32_v_f64m2(py, index_perm_shift, bs);
      v_py_c = __riscv_vfmul_vf_f64m2(v_py_c, beta, bs);
      v_py = __riscv_vfmadd_vf_f64m2(v_py, alpha, v_py_c, bs);
      __riscv_vsoxei32_v_f64m2(py, index_perm_shift, v_py, bs);
    }
  }

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_ALL_STAGES>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  float *py = y.data();
  const float * pb = b.data();
  const int *Cl = mat.vCl.data();
  const int *Cs = mat.vCs.data();
  const int *Bs = mat.vBs.data();
  const int *SBs = mat.vSBs.data();
  const uint32_t* Col  = mat.vCol.data();
  const uint32_t* Perm = mat.vPerm.data();
  const float* Val = mat.vVal.data();

#pragma omp parallel for
  for(int i = 0; i < mat.cnt_b; i++)
  {
    int cur_pos = SBs[i];
    uint32_t bs = __riscv_vsetvlmax_e32m2();
    for (int k = 0; k < Bs[i]; k+= bs)
    {
      bs = __riscv_vsetvl_e32m2(Bs[i] - k);
      vfloat32m2_t v_py = __riscv_vfmv_v_f_f32m2(0.0f, bs);
      for (int j = 0; j < Cl[i]; j++)
      {
        vuint32m2_t  index = __riscv_vle32_v_u32m2(Col + Cs[i]+ j * Bs[i] + k, bs);    
        vuint32m2_t index_shift = __riscv_vsll_vx_u32m2(index, 2, bs);    
        vfloat32m2_t v_val = __riscv_vle32_v_f32m2(Val + Cs[i]+ j * Bs[i] + k, bs);
        vfloat32m2_t v_b        = __riscv_vloxei32_v_f32m2(pb, index_shift, bs);
        v_py = __riscv_vfmacc_vv_f32m2(v_py, v_val, v_b, bs);
      }
      vuint32m2_t index_perm = __riscv_vle32_v_u32m2(Perm + cur_pos + k, bs);    
      vuint32m2_t index_perm_shift = __riscv_vsll_vx_u32m2(index_perm, 2, bs);    
      vfloat32m2_t v_py_c = __riscv_vloxei32_v_f32m2(py, index_perm_shift, bs);
      v_py_c = __riscv_vfmul_vf_f32m2(v_py_c, beta, bs);
      v_py = __riscv_vfmadd_vf_f32m2(v_py, alpha, v_py_c, bs);
      __riscv_vsoxei32_v_f32m2(py, index_perm_shift, v_py, bs);

    }
  }

  return status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  return status;
}


template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxSELL_C_Sigma, true, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

}