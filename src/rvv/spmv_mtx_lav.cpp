#include "sparse_matrix.h"

#include <riscv_vector.h>
#include <iomanip>
#include <cmath>
#include "omp.h"


namespace SparseMatrixLib
{
  template<>
  sparse_matrix_status sparse_mv<double, spMtxLAV, true>(
  sparse_operation_t double_op,
  double alpha,
  const spMtxLAV<double>& mat,
  sparse_matrix_descr descr,
  const std::vector<double>& b,
  double beta,
  std::vector<double>& y) {
    sparse_matrix_status status;

    uint32_t SIMD_Lanes = mat.SIMD_Lanes;

    sparse_mv<double, spMtxCRS, true>(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat.sparse_part, descr, b, beta, y);

    const double* ptr_b = b.data();
    double* ptr_y = y.data();

    const double* ptr_val = mat.val.data();
    const uint32_t* ptr_col_id = mat.col_id.data();
    const uint16_t* ptr_mask = mat.mask.data();
    const int* ptr_chunk_offsets = mat.chunk_offsets.data();
    const int* ptr_chunk_offsets_full_mask = mat.chunk_offsets_full_mask.data();
    const uint32_t* ptr_out_order = mat.out_order.data();

    const int segment_count = mat.segment_count;
    const int chunk_count = mat.chunk_count;


    for (int num_seg = 0; num_seg < segment_count; ++num_seg) {
      const double* chunk_val = ptr_val + (mat.segment_ptr[num_seg] * SIMD_Lanes);
      const uint32_t* chunk_col_id = ptr_col_id + (mat.segment_ptr[num_seg] * SIMD_Lanes);
      const uint16_t* chunk_mask = ptr_mask + mat.segment_ptr[num_seg];
      const uint32_t offsets_shift = ((chunk_count + 1) * num_seg);

#pragma omp parallel for schedule(static, 1)
      for (int c = 0; c < chunk_count; ++c) {
        vfloat64m4_t v_tmp_y = __riscv_vfmv_v_f_f64m4(0.0, SIMD_Lanes);

        size_t end = ptr_chunk_offsets[offsets_shift + c + 1];
        size_t full_mask_end = ptr_chunk_offsets_full_mask[(num_seg * chunk_count) + c];
        uint16_t bs;

        //without mask
        for (size_t lane = ptr_chunk_offsets[offsets_shift + c]; lane < full_mask_end; ++lane) {
          vuint32m2_t v_col_id = __riscv_vle32_v_u32m2(chunk_col_id + (lane * SIMD_Lanes), SIMD_Lanes);

          v_col_id = __riscv_vsll_vx_u32m2(v_col_id, 3, SIMD_Lanes);

          vfloat64m4_t v_x = __riscv_vloxei32_v_f64m4(ptr_b, v_col_id, SIMD_Lanes);

          vfloat64m4_t v_val = __riscv_vle64_v_f64m4(chunk_val + ((lane)*SIMD_Lanes), SIMD_Lanes);

          v_tmp_y = __riscv_vfmacc_vv_f64m4(v_tmp_y, v_val, v_x, SIMD_Lanes);
        }

        //with mask
        for (size_t lane = full_mask_end; lane < end; ++lane) {
          bs = chunk_mask[lane];

          vuint32m2_t v_col_id = __riscv_vle32_v_u32m2(chunk_col_id + (lane * SIMD_Lanes), bs);

          v_col_id = __riscv_vsll_vx_u32m2(v_col_id, 3, bs);

          vfloat64m4_t v_x = __riscv_vloxei32_v_f64m4(ptr_b, v_col_id, bs);

          vfloat64m4_t v_val = __riscv_vle64_v_f64m4(chunk_val + ((lane)*SIMD_Lanes), bs);

          v_tmp_y = __riscv_vfmacc_vv_f64m4_tu(v_tmp_y, v_val, v_x, bs);
        }

        vuint32m2_t v_out = __riscv_vle32_v_u32m2(ptr_out_order + (SIMD_Lanes * (c + chunk_count * num_seg)), SIMD_Lanes);
        v_out = __riscv_vsll_vx_u32m2(v_out, 3, SIMD_Lanes);
        vfloat64m4_t v_y = __riscv_vloxei32_v_f64m4(ptr_y, v_out, SIMD_Lanes);

        v_y = __riscv_vfmadd_vf_f64m4(v_tmp_y, alpha, v_y, SIMD_Lanes);
        __riscv_vsoxei32_v_f64m4(ptr_y, v_out, v_y, SIMD_Lanes);
      }
    }

    return status;
  }





  template<>
  sparse_matrix_status sparse_mv<float, spMtxLAV, true>(
  sparse_operation_t float_op,
  float alpha,
  const spMtxLAV<float>& mat,
  sparse_matrix_descr descr,
  const std::vector<float>& b,
  float beta,
  std::vector<float>& y) {
    sparse_matrix_status status;

    uint32_t SIMD_Lanes = mat.SIMD_Lanes;

    sparse_mv<float, spMtxCRS, true>(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat.sparse_part, descr, b, beta, y);

    const float* ptr_b = b.data();
    float* ptr_y = y.data();

    const float* ptr_val = mat.val.data();
    const uint32_t* ptr_col_id = mat.col_id.data();
    const uint16_t* ptr_mask = mat.mask.data();
    const int* ptr_chunk_offsets = mat.chunk_offsets.data();
    const int* ptr_chunk_offsets_full_mask = mat.chunk_offsets_full_mask.data();
    const uint32_t* ptr_out_order = mat.out_order.data();

    const int segment_count = mat.segment_count;
    const int chunk_count = mat.chunk_count;


    for (int num_seg = 0; num_seg < segment_count; ++num_seg) {
      const float* chunk_val = ptr_val + (mat.segment_ptr[num_seg] * SIMD_Lanes);
      const uint32_t* chunk_col_id = ptr_col_id + (mat.segment_ptr[num_seg] * SIMD_Lanes);
      const uint16_t* chunk_mask = ptr_mask + mat.segment_ptr[num_seg];
      const uint32_t offsets_shift = ((chunk_count + 1) * num_seg);

#pragma omp parallel for schedule(static, 1)
      for (int c = 0; c < chunk_count; ++c) {
        vfloat32m2_t v_tmp_y = __riscv_vfmv_v_f_f32m2(0.0f, SIMD_Lanes);

        size_t end = ptr_chunk_offsets[offsets_shift + c + 1];
        size_t full_mask_end = ptr_chunk_offsets_full_mask[(num_seg * chunk_count) + c];
        uint16_t bs;

        //without mask
        for (size_t lane = ptr_chunk_offsets[offsets_shift + c]; lane < full_mask_end; ++lane) {
          vuint32m2_t v_col_id = __riscv_vle32_v_u32m2(chunk_col_id + (lane * SIMD_Lanes), SIMD_Lanes);

          v_col_id = __riscv_vsll_vx_u32m2(v_col_id, 2, SIMD_Lanes);

          vfloat32m2_t v_x = __riscv_vloxei32_v_f32m2(ptr_b, v_col_id, SIMD_Lanes);

          vfloat32m2_t v_val = __riscv_vle32_v_f32m2(chunk_val + ((lane)*SIMD_Lanes), SIMD_Lanes);

          v_tmp_y = __riscv_vfmacc_vv_f32m2(v_tmp_y, v_val, v_x, SIMD_Lanes);
        }

        //with mask
        for (size_t lane = full_mask_end; lane < end; ++lane) {
          bs = chunk_mask[lane];

          vuint32m2_t v_col_id = __riscv_vle32_v_u32m2(chunk_col_id + (lane * SIMD_Lanes), bs);

          v_col_id = __riscv_vsll_vx_u32m2(v_col_id, 2, bs);

          vfloat32m2_t v_x = __riscv_vloxei32_v_f32m2(ptr_b, v_col_id, bs);

          vfloat32m2_t v_val = __riscv_vle32_v_f32m2(chunk_val + ((lane)*SIMD_Lanes), bs);

          v_tmp_y = __riscv_vfmacc_vv_f32m2_tu(v_tmp_y, v_val, v_x, bs);
        }

        vuint32m2_t v_out = __riscv_vle32_v_u32m2(ptr_out_order + (SIMD_Lanes * (c + chunk_count * num_seg)), SIMD_Lanes);
        v_out = __riscv_vsll_vx_u32m2(v_out, 2, SIMD_Lanes);
        vfloat32m2_t v_y = __riscv_vloxei32_v_f32m2(ptr_y, v_out, SIMD_Lanes);

        v_y = __riscv_vfmadd_vf_f32m2(v_tmp_y, alpha, v_y, SIMD_Lanes);
        __riscv_vsoxei32_v_f32m2(ptr_y, v_out, v_y, SIMD_Lanes);
      }
    }

    return status;
  }

}