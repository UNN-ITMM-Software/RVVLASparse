#include "sparse_matrix.h"

#include <iomanip>
#include <cmath>

namespace SparseMatrixLib
{

template<> 
sparse_matrix_status sparse_mv<double, spMtxCRS, false>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxCRS<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  double * p_y = y.data();
  const double * p_b = b.data();

  
#pragma omp parallel for schedule(dynamic, 256)
  for (int i = 0; i < mat.m; i++) {
    double tmp = 0.0;
    for (int j = mat.Rst[i]; j < mat.Rst[i + 1]; j++) 
      tmp += mat.Val[j] * p_b[mat.Col[j]];
    p_y[i] = tmp * alpha + beta * p_y[i];
  }

  return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxCRS, false>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxCRS<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
  sparse_matrix_status status;

  float * p_y = y.data();
  const float * p_b = b.data();
  
#pragma omp parallel for schedule(dynamic, 256)
  for (int i = 0; i < mat.m; i++) {
    float tmp = 0.0;
    for (int j = mat.Rst[i]; j < mat.Rst[i + 1]; j++) 
      tmp += mat.Val[j] * p_b[mat.Col[j]];
    y[i] = alpha * tmp + beta * p_y[i];
  }

  return status;
}

}