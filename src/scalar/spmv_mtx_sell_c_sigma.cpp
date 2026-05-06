#include "sparse_matrix.h"

#include <iomanip>
#include <cmath>
#include <vector>

namespace SparseMatrixLib
{

template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  std::vector<double> tmp_y(mat.m);
  double *p_tmp_y = tmp_y.data();
  double *py = y.data();
  const uint32_t *perm = mat.vPerm.data();

  for(int i = 0; i < mat.m; i++)
  {
    p_tmp_y[i] = py[perm[i]];
  }

  for(int i = 0; i < mat.m; i++)
  {
    py[i] = p_tmp_y[i];
  }

  return status;
}


template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_MULT>(
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
  std::vector<double> v_y_tmp(Bs[0]); 
  double *y_tmp = v_y_tmp.data();

  for(int i = 0; i < mat.cnt_b; i++)
  {
    int cur_pos = SBs[i];
    for (int k = 0; k < Bs[i]; k++)
    {
      y_tmp[k] = 0.0;
    }
    for (int j = 0; j < Cl[i]; j++)
    {
      for (int k = 0; k < Bs[i]; k++)
      {
        y_tmp[k] += Val[ Cs [i]+ j * Bs[i] + k] *
                    pb[Col[ Cs [i]+ j * Bs[i] + k]];
      }
    }
    for (int k = 0; k < Bs[i]; k++)
    {
      py[cur_pos + k] = y_tmp[k] * alpha + beta * py[cur_pos + k];
    }
  }

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  std::vector<double> tmp_y(mat.m);
  double *p_tmp_y = tmp_y.data();
  double *py = y.data();
  const uint32_t *perm = mat.vPerm.data();

  for(int i = 0; i < mat.m; i++)
  {
    p_tmp_y[i] = py[i];
  }

  for(int i = 0; i < mat.m; i++)
  {
    py[perm[i]] = p_tmp_y[i];
  }

  return status;
}


template<> 
sparse_matrix_status sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_ALL_STAGES>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxSELL_C_Sigma<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  std::vector<float> tmp_y(mat.m);
  float *p_tmp_y = tmp_y.data();
  float *py = y.data();
  const uint32_t *perm = mat.vPerm.data();

  for(int i = 0; i < mat.m; i++)
  {
    p_tmp_y[i] = py[perm[i]];
  }

  for(int i = 0; i < mat.m; i++)
  {
    py[i] = p_tmp_y[i];
  }

  return status;
}
template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_MULT>(
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
  std::vector<float> v_y_tmp(Bs[0]); 
  float *y_tmp = v_y_tmp.data();

  for(int i = 0; i < mat.cnt_b; i++)
  {
    int cur_pos = SBs[i];
    for (int k = 0; k < Bs[i]; k++)
    {
      y_tmp[k] = 0.0f;
    }
    for (int j = 0; j < Cl[i]; j++)
    {
      for (int k = 0; k < Bs[i]; k++)
      {
        y_tmp[k] += Val[ Cs [i]+ j * Bs[i] + k] *
                    pb[Col[ Cs [i]+ j * Bs[i] + k]];
      }
    }
    for (int k = 0; k < Bs[i]; k++)
    {
      py[cur_pos + k] = y_tmp[k] * alpha + beta * py[cur_pos + k];
    }
  }

  return status;
}
template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  std::vector<float> tmp_y(mat.m);
  float *p_tmp_y = tmp_y.data();
  float *py = y.data();
  const uint32_t *perm = mat.vPerm.data();

  for(int i = 0; i < mat.m; i++)
  {
    p_tmp_y[i] = py[i];
  }

  for(int i = 0; i < mat.m; i++)
  {
    py[perm[i]] = p_tmp_y[i];
  }

  return status;
}


template<> 
sparse_matrix_status sparse_mv<float, spMtxSELL_C_Sigma, false>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxSELL_C_Sigma<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<float, spMtxSELL_C_Sigma, false, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}

}