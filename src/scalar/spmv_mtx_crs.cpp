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
  
#pragma omp parallel for
  for (int i = 0; i < mat.m; i++) {
    double tmp = 0.0;
    for (int j = mat.Rst[i]; j < mat.Rst[i + 1]; j++) 
      tmp += mat.Val[j] * b[mat.Col[j]];
    y[i] = tmp * alpha + beta * y[i];
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
  
#pragma omp parallel for
  for (int i = 0; i < mat.m; i++) {
    float tmp = 0.0;
    for (int j = mat.Rst[i]; j < mat.Rst[i + 1]; j++) 
      tmp += mat.Val[j] * b[mat.Col[j]];
    y[i] = alpha * tmp + beta * y[i];
  }

  return status;
}

}