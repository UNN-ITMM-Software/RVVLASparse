#include "sparse_matrix.h"
#include "spmv_mtx.h"
#include "omp.h"
#include <cmath>
#include <riscv_vector.h>

namespace SparseMatrixLib
{
  
namespace VHCC {
  
typedef uint32_t index_t;
  
int write_result_f64_uint8(double* result, const index_t* row_arr, vfloat64m2_t res,  
                            vbool32_t mask, double betta) {
                              
  unsigned int gvl1 = __riscv_vsetvlmax_e8m1();
  unsigned int gvl = __riscv_vsetvlmax_e64m2();  
  vfloat64m2_t elems_wr, y, res_wr, zero;
  
  int bcnt = __riscv_vcpop_m_b32(mask, gvl1);
  elems_wr = __riscv_vcompress_vm_f64m2(res, mask, gvl);

  vuint32m1_t index = __riscv_vle32_v_u32m1(row_arr, bcnt);    
  vuint32m1_t index_shift = __riscv_vsll_vx_u32m1(index, 3, bcnt);    
  y = __riscv_vloxei32_v_f64m2(result, index_shift, bcnt);
  y = __riscv_vfadd_vv_f64m2(y, elems_wr, bcnt);
  __riscv_vsoxei32_v_f64m2(result, index_shift, y, bcnt);
  
  return bcnt;  
}

int write_result_f32_uint16(float* result, const index_t* row_arr, vfloat32m2_t res,  
                            vbool16_t mask) {
                              
  unsigned int gvl1 = __riscv_vsetvlmax_e16m1();
  unsigned int gvl = __riscv_vsetvlmax_e32m2();  
  vfloat32m2_t elems_wr, y;
  
  int bcnt = __riscv_vcpop_m_b16(mask, gvl1);
  elems_wr = __riscv_vcompress_vm_f32m2(res, mask, gvl);

  vuint32m2_t index = __riscv_vle32_v_u32m2(row_arr, bcnt);    
  vuint32m2_t index_shift = __riscv_vsll_vx_u32m2(index, 2, bcnt);    
  y = __riscv_vloxei32_v_f32m2(result, index_shift, bcnt);
  y = __riscv_vfadd_vv_f32m2(y, elems_wr, bcnt);
  __riscv_vsoxei32_v_f32m2(result, index_shift, y, bcnt);
  
  return bcnt;  
}

}


///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace SparseMatrixLib::VHCC;


void vhcc_spmv_1panel_riscv(const spMtxVHCC<double>& mat, const double *input, 
                            double *result, int n_threads, int vec_len, double alpha,
                            double betta)
{
  const thr_info_t *thr_info       = mat.get_thr_info();
  const index_t    *veceor_ptr     = mat.get_veceor_ptr();
  const uint8_t    *scan_mask      = mat.get_scan_mask();
  const index_t    *row_arr        = mat.get_row_arr();
  const index_t    *col_arr        = mat.get_col_arr();
  const double     *vals_arr      = mat.get_vals_arr();
    
#pragma omp parallel default(shared) num_threads(n_threads)
  {
    int id = omp_get_thread_num();
    index_t start_vec = thr_info[id].start_vec;
    index_t end_vec   = thr_info[id].end_vec;
 
    unsigned int gvl = __riscv_vsetvlmax_e64m2();
    unsigned int gvl1 = __riscv_vsetvlmax_e8m1();
    
    vfloat64m2_t res, tmp, val, v_tmp_slideup;
    vuint32m1_t  index, index_shift;
    vbool32_t maskwr, mask_nrval, mask;
    unsigned int bcnt;

    index_t cidx       = thr_info[id].vbase;
    index_t veceor_idx = thr_info[id].vbase;
    index_t scan_idx   = thr_info[id].vbase * 4;
    index_t ridx       = thr_info[id].rbase;
    index_t vec_idx    = start_vec * gvl;
 
    vuint8m1_t m, last;
    vfloat64m2_t zero = __riscv_vfmv_v_f_f64m2(0.0, gvl);

    double nrval = 0;
    index_t eor_vec = veceor_ptr[veceor_idx++];
    
    res = __riscv_vfmv_v_f_f64m2(0.0, gvl);

    for (index_t v = start_vec; v < end_vec; ++v) {
      val  = __riscv_vle64_v_f64m2(vals_arr + vec_idx, gvl); 
      index = __riscv_vle32_v_u32m1(col_arr + vec_idx, gvl);    
      index_shift = __riscv_vsll_vx_u32m1(index, 3, gvl);    
      tmp   = __riscv_vloxei32_v_f64m2(input, index_shift, gvl);
      res = __riscv_vfmadd_vv_f64m2(val, tmp, res, gvl);
    
      vec_idx += gvl;
      nrval = 0.0;

      if (v == eor_vec) {

        for (int offset = 1; offset < gvl; offset <<= 1) {
          m = __riscv_vle8_v_u8m1(scan_mask + scan_idx, 1);
          mask = __riscv_vreinterpret_v_u8m1_b32(m);
          v_tmp_slideup = __riscv_vfmv_v_f_f64m2(0.0, gvl);
          v_tmp_slideup = __riscv_vslideup_vx_f64m2(v_tmp_slideup, res, offset, gvl);
          res = __riscv_vfadd_vv_f64m2_mu(mask, res, res, v_tmp_slideup, gvl);
          scan_idx++;
        }
        m = __riscv_vle8_v_u8m1(scan_mask + scan_idx, 1);
        maskwr = __riscv_vreinterpret_v_u8m1_b32(m);
        scan_idx++;

        bcnt = write_result_f64_uint8(result, row_arr + ridx, res, maskwr, betta);
        ridx += bcnt;
        
        eor_vec = veceor_ptr[veceor_idx++];

        uint8_t z = !(scan_mask[scan_idx - 1] & (1 << (gvl - 1)));
        last = __riscv_vle8_v_u8m1(&z, 1);
        mask_nrval = __riscv_vreinterpret_v_u8m1_b32(last);
        res = __riscv_vrgather_vx_f64m2_mu(mask_nrval, zero, res, gvl - 1, gvl); 
      } 
      else {
        vfloat64m1_t sum = __riscv_vfmv_v_f_f64m1(0.0, gvl);
        sum = __riscv_vfredusum_vs_f64m2_f64m1(res, sum, gvl);
        nrval = __riscv_vfmv_f_s_f64m1_f64(sum);
        res = __riscv_vfmv_v_f_f64m2(0.0, gvl);
        res = __riscv_vfmv_s_f_f64m2_tu(res, nrval, gvl);
      }
    }

#pragma omp barrier

    index_t nridx = thr_info[id].last_row;
    nrval =  result[thr_info[id].overflow_row];
#pragma omp atomic update
    result[nridx] += nrval;
  }
}

void vhcc_spmv_1panel_riscv(const spMtxVHCC<float>& mat, const float *input, 
                            float *result, int n_threads, int vec_size)
{
  const thr_info_t *thr_info   = mat.get_thr_info();
  const index_t    *veceor_ptr = mat.get_veceor_ptr();
  const uint16_t   *scan_mask  = reinterpret_cast<const uint16_t*>(mat.get_scan_mask());
  const index_t    *row_arr    = mat.get_row_arr();
  const index_t    *col_arr    = mat.get_col_arr();
  const float      *vals_arr   = mat.get_vals_arr();
  int              n_masks     = mat.get_n_masks();

#pragma omp parallel default(shared) num_threads(n_threads)
  {

    int id = omp_get_thread_num();
    index_t start_vec = thr_info[id].start_vec;
    index_t end_vec   = thr_info[id].end_vec;
  
    unsigned int gvlm1 = __riscv_vsetvlmax_e32m1();
    unsigned int gvl  = __riscv_vsetvlmax_e32m2();
    unsigned int gvl1 = __riscv_vsetvlmax_e16m1();
    
    vfloat32m2_t res, tmp, val, v_tmp_slideup;
    vuint32m2_t  index, index_shift;
    vbool16_t    maskwr, mask_nrval, mask;
    vuint16m1_t  m, last;
    vfloat32m2_t zero = __riscv_vfmv_v_f_f32m2(0.0f, gvl);

    index_t cidx       = thr_info[id].vbase;
    index_t veceor_idx = thr_info[id].vbase;
    index_t scan_idx   = thr_info[id].sbase / (vec_size / 8);
    index_t ridx       = thr_info[id].rbase;
    index_t vec_idx    = start_vec * gvl;
    int bcnt;
    float nrval = 0.0f;
    index_t eor_vec = veceor_ptr[veceor_idx++];
    float* elems = new float[gvl];
    res = __riscv_vfmv_v_f_f32m2(0.0f, gvl);
    
    for (index_t v = start_vec; v < end_vec; ++v) {

      val  = __riscv_vle32_v_f32m2(vals_arr + vec_idx, gvl);
      index = __riscv_vle32_v_u32m2(col_arr + vec_idx, gvl);    
      index_shift = __riscv_vsll_vx_u32m2(index, 2, gvl);    
      tmp   = __riscv_vloxei32_v_f32m2(input, index_shift, gvl);
      res = __riscv_vfmadd_vv_f32m2(val, tmp, res, gvl);
          
      vec_idx += gvl;
      nrval = 0.0f;
      if (v == eor_vec) {

        for (int offset = 1; offset < gvl; offset <<= 1) {
          m = __riscv_vle16_v_u16m1(scan_mask + scan_idx, 1);
          mask = __riscv_vreinterpret_v_u16m1_b16(m);
          v_tmp_slideup = __riscv_vslideup_vx_f32m2(zero, res, offset, gvl);
          res = __riscv_vfadd_vv_f32m2_mu(mask, res, res, v_tmp_slideup, gvl);
          scan_idx++;
        }
        m = __riscv_vle16_v_u16m1(scan_mask + scan_idx, 1);
        maskwr = __riscv_vreinterpret_v_u16m1_b16(m);
        scan_idx++;

        __riscv_vse32_v_f32m2(elems, res, gvl);
        nrval = elems[gvl-1] * !(scan_mask[scan_idx - 1] & (1 << (gvl - 1)));

        bcnt = write_result_f32_uint16(result, row_arr + ridx, res, maskwr);
        ridx += bcnt;
        eor_vec = veceor_ptr[veceor_idx++];
      } 
      else {        
        vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0, gvlm1);
        sum = __riscv_vfredusum_vs_f32m2_f32m1(res, sum, gvl);
        nrval = __riscv_vfmv_f_s_f32m1_f32(sum);
      }
      res = __riscv_vfmv_s_f_f32m2_tu(zero, nrval, gvl);
    }

#pragma omp barrier

    index_t nridx = thr_info[id].last_row;
    nrval =  result[thr_info[id].overflow_row];
#pragma omp atomic update
    result[nridx] += nrval;
    delete [] elems;
  }

}
  
template<> 
sparse_matrix_status sparse_mv<float, spMtxVHCC, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxVHCC<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y) {
                                                                  
  sparse_matrix_status status;
  std::vector<float> tmp_y( mat.get_pad_rows());
  float* result_y = y.data();
  float* tmp_y_ptr = tmp_y.data();
  vhcc_params params = mat.get_params(); 
  
  #pragma omp parallel num_threads(params.num_threads)
  {
    int id = omp_get_thread_num();
    int count = tmp_y.size() / params.num_threads;
    int start = count * id;
    if (id == params.num_threads - 1)
        count += tmp_y.size() % params.num_threads;
    std::memset(tmp_y_ptr + start, 0, sizeof(float)*count);
  }

  vhcc_spmv_1panel_riscv(mat, b.data(), tmp_y_ptr, 
    params.num_threads, params.vec_size);

  if ((fabs(beta) > 0.0f) || (fabs(alpha - 1.0f) > 1e-8)) 
  {
    unsigned int gvl = __riscv_vsetvlmax_e32m2();
    vfloat32m2_t res, tmp;
    int tail = y.size() & (-gvl);
    
#pragma omp parallel for num_threads(params.num_threads)
    for (int i = 0; i < tail; i+= gvl) {
      vfloat32m2_t res, tmp;
      res = __riscv_vle32_v_f32m2(result_y + i, gvl);
      tmp = __riscv_vle32_v_f32m2(tmp_y_ptr + i, gvl);
      tmp = __riscv_vfmul_vf_f32m2(tmp, alpha, gvl);
      res = __riscv_vfmadd_vf_f32m2(res, beta, tmp, gvl);
      __riscv_vse32_v_f32m2(result_y + i, res, gvl);
    }
    gvl = __riscv_vsetvl_e32m2(y.size() - tail);
    res = __riscv_vle32_v_f32m2(result_y + tail, gvl);
    tmp = __riscv_vle32_v_f32m2(tmp_y_ptr + tail, gvl);
    tmp = __riscv_vfmul_vf_f32m2(tmp, alpha, gvl);
    res = __riscv_vfmadd_vf_f32m2(res, beta, tmp, gvl);
    __riscv_vse32_v_f32m2(result_y + tail, res, gvl);
  }
  else {
    #pragma omp parallel num_threads(params.num_threads)
    {
      int id = omp_get_thread_num();
      int count = y.size() / params.num_threads;
      int start = count * id;
      if (id == params.num_threads - 1)
        count += y.size() % params.num_threads;
      std::memcpy(y.data() + start, tmp_y.data() + start, sizeof(float)*count);
    }
  }

  return status;
}

//////////////////////////////////////////////////////////////////////////////

template<> 
sparse_matrix_status sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVHCC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVHCC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;
  std::vector<double> tmp_y( mat.get_pad_rows());
  double* result_y = y.data();
  double* tmp_y_ptr = tmp_y.data();
  vhcc_params params = mat.get_params();
  
  #pragma omp parallel num_threads(params.num_threads)
  {
    int id = omp_get_thread_num();
    int count = tmp_y.size() / params.num_threads;
    int start = count * id;
    if (id == params.num_threads - 1)
        count += tmp_y.size() % params.num_threads;
    std::memset(tmp_y_ptr + start, 0, sizeof(double)*count);
  }

  vhcc_spmv_1panel_riscv(mat, b.data(), tmp_y_ptr, 
    params.num_threads, params.vec_size, alpha, beta);

  if ((fabs(beta) > 0.0) || (fabs(alpha - 1.0) > 1e-15)) 
  {
    unsigned int gvl = __riscv_vsetvlmax_e64m2();
    vfloat64m2_t res, tmp;
    int tail = y.size() & (-gvl);
    
#pragma omp parallel for 
  for (int i = 0; i < tail; i+= gvl) {
    vfloat64m2_t res, tmp;
    res = __riscv_vle64_v_f64m2(result_y + i, gvl);
    tmp = __riscv_vle64_v_f64m2(tmp_y_ptr + i, gvl);
    tmp = __riscv_vfmul_vf_f64m2(tmp, alpha, gvl);
    res = __riscv_vfmadd_vf_f64m2(res, beta, tmp, gvl);
    __riscv_vse64_v_f64m2(result_y + i, res, gvl);
  }
    gvl = __riscv_vsetvl_e64m2(y.size() - tail);
    res = __riscv_vle64_v_f64m2(result_y + tail, gvl);
    tmp = __riscv_vle64_v_f64m2(tmp_y_ptr + tail, gvl);
    tmp = __riscv_vfmul_vf_f64m2(tmp, alpha, gvl);
    res = __riscv_vfmadd_vf_f64m2(res, beta, tmp, gvl);
    __riscv_vse64_v_f64m2(result_y + tail, res, gvl);
  }
  else {
    #pragma omp parallel
    {
      int id = omp_get_thread_num();
      int count = y.size() / params.num_threads;
      int start = count * id;
      if (id == params.num_threads - 1)
        count += y.size() % params.num_threads;
      std::memcpy(y.data() + start, tmp_y.data() + start, sizeof(double)*count);
    }
  }

  return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVHCC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;
  
  return status;
}

//////////////////////////////////////////////////////////////////////////////

template<> 
sparse_matrix_status sparse_mv<double, spMtxVHCC, true>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVHCC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
  sparse_matrix_status status;

  sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_PREPARATION>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_MULT>
    (type_op, alpha, mat, descr, b, beta, y);
  sparse_mv<double, spMtxVHCC, true, SPARSE_MATRIX_MV_GET_RESULTS>
    (type_op, alpha, mat, descr, b, beta, y);

  return status;
}



}