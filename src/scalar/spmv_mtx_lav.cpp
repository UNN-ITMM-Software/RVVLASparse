#include "sparse_matrix.h"

#include <iomanip>
#include <cmath>
#include "omp.h"


namespace SparseMatrixLib
{
  template<>
  sparse_matrix_status sparse_mv<double, spMtxLAV, false>(
    sparse_operation_t double_op,
    double alpha,
    const spMtxLAV<double>& mat,
    sparse_matrix_descr descr,
    const std::vector<double>& b,
    double beta,
    std::vector<double>& y) {
    sparse_matrix_status status;

    uint32_t SIMD_Lanes = mat.SIMD_Lanes;

    sparse_mv<double, spMtxCRS, false>(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat.sparse_part, descr, b, beta, y);

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

#pragma omp parallel for schedule(static,1)
      for (int c = 0; c < chunk_count; ++c) {

        size_t end = ptr_chunk_offsets[offsets_shift + c + 1];
        size_t full_mask_end = ptr_chunk_offsets_full_mask[(num_seg * chunk_count) + c];
        uint16_t bs;

        for (size_t lane = ptr_chunk_offsets[offsets_shift + c]; lane < full_mask_end; ++lane) {
          for (size_t i = 0; i < SIMD_Lanes; ++i) {
            ptr_y[ptr_out_order[i + (c + chunk_count * num_seg) * SIMD_Lanes]] += alpha * chunk_val[((lane)*SIMD_Lanes) + i] * ptr_b[chunk_col_id[lane * SIMD_Lanes + i]];
          }
        }

        for (size_t lane = full_mask_end; lane < end; ++lane) {
          bs = chunk_mask[lane];
          for (size_t i = 0; i < bs; ++i) {
            ptr_y[ptr_out_order[i + (c + chunk_count * num_seg) * SIMD_Lanes]] += alpha * chunk_val[((lane)*SIMD_Lanes) + i] * ptr_b[chunk_col_id[lane * SIMD_Lanes + i]];
          }
        }

      }
    }

    return status;
  }



  template<>
  sparse_matrix_status sparse_mv<float, spMtxLAV, false>(
    sparse_operation_t float_op,
    float alpha,
    const spMtxLAV<float>& mat,
    sparse_matrix_descr descr,
    const std::vector<float>& b,
    float beta,
    std::vector<float>& y) {
    sparse_matrix_status status;

    uint32_t SIMD_Lanes = mat.SIMD_Lanes;

    sparse_mv<float, spMtxCRS, false>(SPARSE_OPERATION_NON_TRANSPOSE, alpha, mat.sparse_part, descr, b, beta, y);

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

#pragma omp parallel for schedule(static,1)
      for (int c = 0; c < chunk_count; ++c) {

        size_t end = ptr_chunk_offsets[offsets_shift + c + 1];
        size_t full_mask_end = ptr_chunk_offsets_full_mask[(num_seg * chunk_count) + c];
        uint16_t bs;

        for (size_t lane = ptr_chunk_offsets[offsets_shift + c]; lane < full_mask_end; ++lane) {
          for (size_t i = 0; i < SIMD_Lanes; ++i) {
            ptr_y[ptr_out_order[i + (c + chunk_count * num_seg) * SIMD_Lanes]] += alpha * chunk_val[((lane)*SIMD_Lanes) + i] * ptr_b[chunk_col_id[lane * SIMD_Lanes + i]];
          }
        }

        for (size_t lane = full_mask_end; lane < end; ++lane) {
          bs = chunk_mask[lane];
          for (size_t i = 0; i < bs; ++i) {
            ptr_y[ptr_out_order[i + (c + chunk_count * num_seg) * SIMD_Lanes]] += alpha * chunk_val[((lane)*SIMD_Lanes) + i] * ptr_b[chunk_col_id[lane * SIMD_Lanes + i]];
          }
        }
      }
    }

    return status;
  }
}