#include "sparse_matrix.h"
#include <vector>
#include <riscv_vector.h>

namespace SparseMatrixLib
{

always_inline void RVV100_spvv_cross_row_fp32(const int *rows, int *row_begin, int *row_end, int *col, float *nnz, float *x, float *y, float alpha, float beta)
{
    size_t vl32 = __riscv_vsetvlmax_e32m2();
    vuint32m2_t rs = __riscv_vle32_v_u32m2((const uint32_t*)(rows), vl32);
    rs = __riscv_vmul_vx_u32m2(rs, (uint32_t)sizeof(float), vl32);
    vfloat32m2_t acc = __riscv_vfmv_v_f_f32m2(0.0f, vl32); 

    int rowlen = *row_end - *row_begin;
    int base = *row_begin;
    for (int c = 0; c < rowlen; c++)
    {
        int offset = base + c * nLanes_f32;
        vuint32m2_t cc = __riscv_vle32_v_u32m2((const uint32_t*)(col + offset), vl32);
        cc = __riscv_vmul_vx_u32m2(cc, (uint32_t)sizeof(float), vl32);
        vfloat32m2_t nz = __riscv_vle32_v_f32m2(nnz + offset, vl32);
        vfloat32m2_t xx = __riscv_vloxei32_v_f32m2(x, cc, vl32);
        acc = __riscv_vfmadd_vv_f32m2(nz, xx, acc, vl32);
    }
    vfloat32m2_t yy = __riscv_vloxei32_v_f32m2(y, rs, vl32); // yy = y
    yy = __riscv_vfmul_vf_f32m2(yy, beta, vl32); // yy = yy * beta
    yy = __riscv_vfmadd_vf_f32m2(acc, alpha, yy, vl32); // yy = alpha * acc + yy
    __riscv_vsoxei32_v_f32m2(y, rs, yy, vl32); // y = yy
}

always_inline float RVV100_spvv_in_row_fp32(int *col, float *nnz, int rowlen, float *x, float alpha, float beta)
{
    int limit = rowlen - (nLanes_f32 - 1);
    int *col_p;
    float *nnz_p;
    float sum = 0.0f;
    size_t vl32 = __riscv_vsetvlmax_e32m2();
    vuint32m2_t c1;
    vfloat32m2_t v1, v2, s;
    vfloat32m1_t tmpsum = __riscv_vle32_v_f32m1(&sum, 1);
    s = __riscv_vfmv_v_f_f32m2(0.0f, vl32);
    int i = 0;
    for (i = 0; i < limit; i += nLanes_f32)
    {
        col_p = col + i;
        nnz_p = nnz + i;
        c1 = __riscv_vle32_v_u32m2((const uint32_t*)(col_p), vl32);
        c1 = __riscv_vmul_vx_u32m2(c1, (uint32_t)sizeof(float), vl32);
        v2 = __riscv_vloxei32_v_f32m2(x, c1, vl32);
        v1 = __riscv_vle32_v_f32m2(nnz_p, vl32);
        s = __riscv_vfmadd_vv_f32m2(v1, v2, s, vl32);
    }
    tmpsum = __riscv_vfredosum_vs_f32m2_f32m1(s, tmpsum, vl32);
    __riscv_vse32_v_f32m1(&sum, tmpsum, 1);
    for (; i < rowlen; i++)
    {
        sum += nnz[i] * x[col[i]];
    }
    return sum;
}

always_inline double RVV100_spvv_in_row_fp64(int *col, double *nnz, int rowlen, double *x, double alpha, double beta)
{
    int limit = rowlen - (nLanes_f64 - 1);
    int *col_p;
    double *nnz_p;
    double sum = 0.0;
    size_t vl32 = __riscv_vsetvlmax_e32m1();
    size_t vl64 = __riscv_vsetvlmax_e64m2();
    vuint32m1_t c1;
    vfloat64m2_t v1, v2, s;
    vfloat64m1_t tmpsum = __riscv_vle64_v_f64m1(&sum, 1);

    s = __riscv_vfmv_v_f_f64m2(0.0, vl64);
    int i = 0;
    for (i = 0; i < limit; i += nLanes_f64)
    {
        col_p = col + i;
        nnz_p = nnz + i;

        c1 = __riscv_vle32_v_u32m1((const uint32_t*)(col_p), vl32);
        c1 = __riscv_vmul_vx_u32m1(c1, (uint32_t)sizeof(double), vl32);
        v2 = __riscv_vloxei32_v_f64m2(x, c1, vl64);
        v1 = __riscv_vle64_v_f64m2(nnz_p, vl64);
        s = __riscv_vfmadd_vv_f64m2(v1, v2, s, vl64);
    }
    tmpsum = __riscv_vfredosum_vs_f64m2_f64m1(s, tmpsum, vl64);
    __riscv_vse64_v_f64m1(&sum, tmpsum, 1);

    for (; i < rowlen; i++)
    {
        sum += nnz[i] * x[col[i]];
    }
    return sum;
}

always_inline void RVV100_spvv_cross_row_fp64(const int *rows, int *row_begin, int *row_end, int *col, double *nnz, double *x, double *y, double alpha, double beta)
{
    size_t vl32 = __riscv_vsetvlmax_e32m1();
    size_t vl64 = __riscv_vsetvlmax_e64m2();
    vuint32m1_t rs = __riscv_vle32_v_u32m1((const uint32_t*)(rows), vl32);
    rs = __riscv_vmul_vx_u32m1(rs, (uint32_t)sizeof(double), vl32);
    vfloat64m2_t acc = __riscv_vfmv_v_f_f64m2(0.0, vl64);

    int rowlen = *row_end - *row_begin;
    int base = *row_begin;
    for (int c = 0; c < rowlen; c++)
    {
        int offset = base + c * nLanes_f64;
        vuint32m1_t cc = __riscv_vle32_v_u32m1((const uint32_t*)(col + offset), vl32);
        cc = __riscv_vmul_vx_u32m1(cc, (uint32_t)sizeof(double), vl32);
        vfloat64m2_t nz = __riscv_vle64_v_f64m2(nnz + offset, vl64);
        vfloat64m2_t xx = __riscv_vloxei32_v_f64m2(x, cc, vl64);
        acc = __riscv_vfmadd_vv_f64m2(nz, xx, acc, vl64);
    }
    vfloat64m2_t yy = __riscv_vloxei32_v_f64m2(y, rs, vl64); // yy = y
    yy = __riscv_vfmul_vf_f64m2(yy, beta, vl64); // yy = yy * beta
    yy = __riscv_vfmadd_vf_f64m2(acc, alpha, yy, vl64); // yy = alpha * acc + yy
    __riscv_vsoxei32_v_f64m2(y, rs, yy, vl64); // y = yy
}
void SpMV_VNEC_S_FP32(std::vector<float>& y, const std::vector<float>& x, const spMtxVNEC<float> *mat_thd, float alpha, float beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        size_t vl32 = __riscv_vsetvlmax_e32m2();
        int *local_use_x_indices = mat_thd->use_x_indices[tid];
        float *local_ecr_xx_val = mat_thd->ecr_xx_val[tid];
        int NEC_NUM = mat_thd->NEC_NUM[tid];

        // x-vector preprocessing
        int j = 0;
        for (; j < NEC_NUM - nLanes_f32; j += nLanes_f32)
        {
            vuint32m2_t id_avx = __riscv_vle32_v_u32m2((const uint32_t*)(local_use_x_indices + j), vl32);
            id_avx = __riscv_vmul_vx_u32m2(id_avx, (uint32_t)sizeof(float), vl32);
            vfloat32m2_t val_avx = __riscv_vloxei32_v_f32m2(x.data(), id_avx, vl32);
            __riscv_vse32_v_f32m2(local_ecr_xx_val + j, val_avx, vl32);
        }
        for (; j < NEC_NUM; j++)
        {
            local_ecr_xx_val[j] = *(x.data() + local_use_x_indices[j]);
        }

        int *rows = mat_thd->tasks[tid];
        spMtxVNEC<float>::csr_f *mat = &(mat_thd->reorder_mat);
        int T_start = mat->task_start[tid];
        int T_end = mat->task_end[tid];
        int limit = mat_thd->spvv_len[tid];
        int p, c;
        for (p = T_start, c = 0; c < limit; p += nLanes_f32, c += nLanes_f32)
        {
            RVV100_spvv_cross_row_fp32(rows + c, mat->row_begin + p, mat->row_end + p, mat->col, mat->nnz, local_ecr_xx_val, y.data(), alpha, beta);
        }
        for (; p < T_end; p++)
        {
            int r = rows[p - T_start];
            int r_begin = mat->row_begin[p];
            int rowlen = mat->row_end[p] - r_begin;
            y[r] = beta * y[r] + alpha * RVV100_spvv_in_row_fp32(mat->col + r_begin, mat->nnz + r_begin, rowlen, local_ecr_xx_val, alpha, beta);
        }
    }
}
void SpMV_VNEC_S_FP64(std::vector<double>& y, const std::vector<double>& x, const spMtxVNEC<double> *mat_thd, double alpha, double beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        size_t vl32 = __riscv_vsetvlmax_e32m1();
        size_t vl64 = __riscv_vsetvlmax_e64m2();
        int *local_use_x_indices = mat_thd->use_x_indices[tid];
        double *local_ecr_xx_val = mat_thd->ecr_xx_val[tid];
        int NEC_NUM = mat_thd->NEC_NUM[tid];
        // x-vector preprocessing
        int j = 0;
        for (; j < NEC_NUM - nLanes_f64; j += nLanes_f64)
        {
            vuint32m1_t id_avx = __riscv_vle32_v_u32m1((const uint32_t*)(local_use_x_indices + j), vl32);
            id_avx = __riscv_vmul_vx_u32m1(id_avx, (uint32_t)sizeof(double), vl32);
            vfloat64m2_t val_avx = __riscv_vloxei32_v_f64m2(x.data(), id_avx, vl64);
            __riscv_vse64_v_f64m2(local_ecr_xx_val + j, val_avx, vl64);
        }
        for (; j < NEC_NUM; j++)
        {
            local_ecr_xx_val[j] = *(x.data() + local_use_x_indices[j]);
        }
        int *rows = mat_thd->tasks[tid];
        spMtxVNEC<double>::csr_f *mat = &(mat_thd->reorder_mat);
        int T_start = mat->task_start[tid];
        int T_end = mat->task_end[tid];
        int limit = mat_thd->spvv_len[tid];
        int p, c;
        for (p = T_start, c = 0; c < limit; p += nLanes_f64, c += nLanes_f64)
        {
            RVV100_spvv_cross_row_fp64(rows + c, mat->row_begin + p, mat->row_end + p, mat->col, mat->nnz, local_ecr_xx_val, y.data(), alpha, beta);
        }

        for (; p < T_end; p++)
        {
            int r = rows[p - T_start];
            int r_begin = mat->row_begin[p];
            int rowlen = mat->row_end[p] - r_begin;
            y[r] = beta * y[r] + alpha * RVV100_spvv_in_row_fp64(mat->col + r_begin, mat->nnz + r_begin, rowlen, local_ecr_xx_val, alpha, beta);
        }

    }
}

void SpMV_VNEC_D_FP32(std::vector<float>& y, const std::vector<float>& x, const spMtxVNEC<float> *mat_thd, float alpha, float beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
    const spMtxVNEC<float>::SpB_Matrix *A = &(mat_thd->M), *a = &(mat_thd->M);
    int *row_end_offsets = a->ptr + 1;
    int row_carry_out[256];
    float value_carry_out[256];
    size_t vl32 = __riscv_vsetvlmax_e32m2();
    size_t vl32m4 = __riscv_vsetvlmax_e32m4();
    size_t elem_for_thread = y.size() / NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++) {
            size_t shift = elem_for_thread * tid;
            for (int i = 0; i < elem_for_thread; i += vl32m4) {
                size_t vl = __riscv_vsetvl_e32m4(elem_for_thread - i);
                vfloat32m4_t val = __riscv_vle32_v_f32m4(y.data() + shift + i, vl);
                val = __riscv_vfmul_vf_f32m4(val, beta, vl);
                __riscv_vse32_v_f32m4(y.data() + shift + i, val, vl);
            }
        }
        for (int i = y.size() - y.size() % NUM_THREADS_VNEC; i < y.size(); i += vl32m4) {
            size_t vl = __riscv_vsetvl_e32m4(y.size() - i);
            vfloat32m4_t val = __riscv_vle32_v_f32m4(y.data() + i, vl);
            val = __riscv_vfmul_vf_f32m4(val, beta, vl);
            __riscv_vse32_v_f32m4(y.data() + i, val, vl);
        }
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        int *local_use_x_indices = mat_thd->use_x_indices[tid];
        float *local_ecr_xx_val = mat_thd->ecr_xx_val[tid];
        int NEC_NUM = mat_thd->NEC_NUM[tid];

        // x-vector preprocessing
        int j = 0;
        for (; j < NEC_NUM - nLanes_f32; j += nLanes_f32)
        {
            vuint32m2_t id_avx = __riscv_vle32_v_u32m2((const uint32_t*)(local_use_x_indices + j), vl32);
            id_avx = __riscv_vmul_vx_u32m2(id_avx, (uint32_t)sizeof(float), vl32);
            vfloat32m2_t val_avx = __riscv_vloxei32_v_f32m2(x.data(), id_avx, vl32);
            __riscv_vse32_v_f32m2(local_ecr_xx_val + j, val_avx, vl32);
        }
        for (; j < NEC_NUM; j++)
        {
            local_ecr_xx_val[j] = *(x.data() + local_use_x_indices[j]);
        }
        coord thread_coord_start = mat_thd->thread_coord_start[tid];
        coord thread_coord_end = mat_thd->thread_coord_end[tid];
        for (int i = thread_coord_start.x; i < thread_coord_end.x; ++i)
        {
            SpB_Index ptr_start = mat_thd->v_row_ptr[i];
            SpB_Index n_one_line = mat_thd->v_row_ptr[i + 1] - ptr_start;
            vfloat32m2_t v_tmp = __riscv_vfmv_v_f_f32m2(0.0f, vl32);
            vfloat32m1_t red_tmp = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            for (SpB_Index j = 0; j < n_one_line; j++)
            {
                SpB_Index col = ptr_start + j;
                vfloat32m2_t matrix = __riscv_vle32_v_f32m2(mat_thd->val_align + col * nLanes_f32, vl32);
                vfloat32m2_t vector = __riscv_vle32_v_f32m2(local_ecr_xx_val + mat_thd->col_start[col], vl32);
                v_tmp = __riscv_vfmadd_vv_f32m2(matrix, vector, v_tmp, vl32);
            }
            red_tmp = __riscv_vfredosum_vs_f32m2_f32m1(v_tmp, red_tmp, vl32);
            float tmpy; 
            __riscv_vse32_v_f32m1(&tmpy, red_tmp, 1); 
            y[i] = y[i] + alpha * tmpy;
        }
        float running_total = 0.0;
        thread_coord_start.y = row_end_offsets[thread_coord_end.x - 1];
        for (; thread_coord_start.y < thread_coord_end.y; ++thread_coord_start.y)
        {
            running_total += ((float *)a->val)[thread_coord_start.y] * local_ecr_xx_val[mat_thd->ecr_indices[thread_coord_start.y]];
        }
        row_carry_out[tid] = thread_coord_end.x;
        value_carry_out[tid] = running_total;
    }
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC - 1; ++tid)
    {
        y[row_carry_out[tid]] += alpha * value_carry_out[tid];
    }
}

void SpMV_VNEC_D_FP64(std::vector<double>& y, const std::vector<double>& x, const spMtxVNEC<double> *mat_thd, double alpha, double beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
    const spMtxVNEC<double>::SpB_Matrix *A = &(mat_thd->M), *a = &(mat_thd->M);
    int *row_end_offsets = a->ptr + 1; // Merge list A
    int row_carry_out[256];
    double value_carry_out[256];
    size_t vl32 = __riscv_vsetvlmax_e32m1();
    size_t vl64 = __riscv_vsetvlmax_e64m2();
    size_t vl64m4 = __riscv_vsetvlmax_e64m4();
    size_t elem_for_thread = y.size() / NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++) {
            size_t shift = elem_for_thread * tid;
            for (int i = 0; i < elem_for_thread; i += vl64m4) {
                size_t vl = __riscv_vsetvl_e64m4(elem_for_thread - i);
                vfloat64m4_t val = __riscv_vle64_v_f64m4(y.data() + shift + i, vl);
                val = __riscv_vfmul_vf_f64m4(val, beta, vl);
                __riscv_vse64_v_f64m4(y.data() + shift + i, val, vl);
            }
        }
        for (int i = y.size() - y.size() % NUM_THREADS_VNEC; i < y.size(); i += vl64m4) {
            size_t vl = __riscv_vsetvl_e64m4(y.size() - i);
            vfloat64m4_t val = __riscv_vle64_v_f64m4(y.data() + i, vl);
            val = __riscv_vfmul_vf_f64m4(val, beta, vl);
            __riscv_vse64_v_f64m4(y.data() + i, val, vl);
        }
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        int *local_use_x_indices = mat_thd->use_x_indices[tid];
        double *local_ecr_xx_val = mat_thd->ecr_xx_val[tid];
        int NEC_NUM = mat_thd->NEC_NUM[tid];

        // x-vector preprocessing
        int j = 0;
        for (; j < NEC_NUM - nLanes_f64; j += nLanes_f64)
        {
            vuint32m1_t id_avx = __riscv_vle32_v_u32m1((const uint32_t*)(local_use_x_indices + j), vl32);
            id_avx = __riscv_vmul_vx_u32m1(id_avx, (uint32_t)sizeof(double), vl32);            
            vfloat64m2_t val_avx = __riscv_vloxei32_v_f64m2(x.data(), id_avx, vl64);
            __riscv_vse64_v_f64m2(local_ecr_xx_val + j, val_avx, vl64);
        }
        for (; j < NEC_NUM; j++)
        {
            local_ecr_xx_val[j] = *(x.data() + local_use_x_indices[j]);
        }
        coord thread_coord_start = mat_thd->thread_coord_start[tid];
        coord thread_coord_end = mat_thd->thread_coord_end[tid];
        for (int i = thread_coord_start.x; i < thread_coord_end.x; ++i)
        {
            SpB_Index ptr_start = mat_thd->v_row_ptr[i];
            SpB_Index n_one_line = mat_thd->v_row_ptr[i + 1] - ptr_start;
            vfloat64m2_t v_tmp = __riscv_vfmv_v_f_f64m2(0.0, vl64);
            vfloat64m1_t red_tmp = __riscv_vfmv_v_f_f64m1(0.0, 1);
            for (SpB_Index j = 0; j < n_one_line; j++)
            {
                SpB_Index col = ptr_start + j;
                vfloat64m2_t matrix = __riscv_vle64_v_f64m2(mat_thd->val_align + col * nLanes_f64, vl64);
                vfloat64m2_t vector = __riscv_vle64_v_f64m2(local_ecr_xx_val + mat_thd->col_start[col], vl64);
                v_tmp = __riscv_vfmadd_vv_f64m2(matrix, vector, v_tmp, vl64);
            }
            red_tmp = __riscv_vfredosum_vs_f64m2_f64m1(v_tmp, red_tmp, vl64);
            double tmpy;
            __riscv_vse64_v_f64m1(&tmpy, red_tmp, 1);
            y[i] = y[i] + alpha * tmpy;
        }
        double running_total = 0.0;
        thread_coord_start.y = row_end_offsets[thread_coord_end.x - 1];
        for (; thread_coord_start.y < thread_coord_end.y; ++thread_coord_start.y)
        {
            running_total += ((double *)a->val)[thread_coord_start.y] * local_ecr_xx_val[mat_thd->ecr_indices[thread_coord_start.y]];
        }
        row_carry_out[tid] = thread_coord_end.x;
        value_carry_out[tid] = running_total;
    }
// update the values in y for rows that span multiple threads
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC - 1; ++tid)
    {
        y[row_carry_out[tid]] += alpha * value_carry_out[tid];
    }
}

void SpMV_VNEC_L_FP32(std::vector<float>& y, const std::vector<float>& x, const spMtxVNEC<float> *mat_thd, float alpha, float beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
    const spMtxVNEC<float>::SpB_Matrix *A = &(mat_thd->M), *a = &(mat_thd->M);
    int *row_end_offsets = a->ptr + 1; // Merge list A
    int row_carry_out[NUM_THREADS_VNEC];
    float value_carry_out[NUM_THREADS_VNEC];
    size_t vl32 = __riscv_vsetvlmax_e32m2();
        size_t vl32m4 = __riscv_vsetvlmax_e32m4();
        size_t elem_for_thread = y.size() / NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++) {
            size_t shift = elem_for_thread * tid;
            for (int i = 0; i < elem_for_thread; i += vl32m4) {
                size_t vl = __riscv_vsetvl_e32m4(elem_for_thread - i);
                vfloat32m4_t val = __riscv_vle32_v_f32m4(y.data() + shift + i, vl);
                val = __riscv_vfmul_vf_f32m4(val, beta, vl);
                __riscv_vse32_v_f32m4(y.data() + shift + i, val, vl);
            }
        }
        for (int i = y.size() - y.size() % NUM_THREADS_VNEC; i < y.size(); i += vl32m4) {
            size_t vl = __riscv_vsetvl_e32m4(y.size() - i);
            vfloat32m4_t val = __riscv_vle32_v_f32m4(y.data() + i, vl);
            val = __riscv_vfmul_vf_f32m4(val, beta, vl);
            __riscv_vse32_v_f32m4(y.data() + i, val, vl);
        }

#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        int ind[vl32];
        coord thread_coord_start = mat_thd->thread_coord_start[tid];
        coord thread_coord_end = mat_thd->thread_coord_end[tid];
        for (; thread_coord_start.x < thread_coord_end.x; ++thread_coord_start.x)
        {
            float running_total = 0.0;
            vfloat32m2_t v_running_total = __riscv_vfmv_v_f_f32m2(0.0f, vl32);
            vfloat32m1_t tmp_running_total = __riscv_vfmv_v_f_f32m1(0.0f, 1);
            int loop_end = (int)row_end_offsets[thread_coord_start.x] - 16;
            for (; thread_coord_start.y < loop_end; thread_coord_start.y += 16)
            {
                for (size_t i = 0; i < vl32; ++i) ind[i] = a->indices[thread_coord_start.y + i];
                vuint32m2_t v_indices = __riscv_vle32_v_u32m2((const uint32_t*)(ind), vl32);
                v_indices = __riscv_vmul_vx_u32m2(v_indices, (uint32_t)sizeof(float), vl32);
                vfloat32m2_t v_xtmp = __riscv_vloxei32_v_f32m2(x.data(), v_indices, vl32);

                vfloat32m2_t v_atmp = __riscv_vle32_v_f32m2((float *)(a->val) + thread_coord_start.y, vl32);
                v_running_total = __riscv_vfmadd_vv_f32m2(v_xtmp, v_atmp, v_running_total, vl32);
            }
            tmp_running_total = __riscv_vfredosum_vs_f32m2_f32m1(v_running_total, tmp_running_total, vl32);
            __riscv_vse32_v_f32m1(&running_total, tmp_running_total, 1);

            for (; thread_coord_start.y < (int)row_end_offsets[thread_coord_start.x]; ++thread_coord_start.y)
            {
                running_total += ((float *)a->val)[thread_coord_start.y] * x[a->indices[thread_coord_start.y]];
            } // End of this line

              y[thread_coord_start.x] += alpha * running_total;
        }

        //  finish one row, calculate the partial sum of the next row
        //  Consume partial portion of thread's last row (accumulate any nonzeros for a partial row shared with the next thread)
        float running_total = 0.0;
        for (; thread_coord_start.y < thread_coord_end.y; ++thread_coord_start.y)
        {
            running_total += ((float *)a->val)[thread_coord_start.y] * x[a->indices[thread_coord_start.y]];
        }
        // save the thread's running total and row-id for subsequent fix-up
        row_carry_out[tid] = thread_coord_end.x; // The value of this row is incomplete
        value_carry_out[tid] = running_total;    // This row would be a little bit more complete with this. Each thread might have one of these
    }
// update the values in y for rows that span multiple threads
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC - 1; ++tid)
    {
        y[row_carry_out[tid]] += alpha * value_carry_out[tid];
    }
}

void SpMV_VNEC_L_FP64(std::vector<double>& y, const std::vector<double>& x, const spMtxVNEC<double> *mat_thd, double alpha, double beta)
{
    int NUM_THREADS_VNEC = mat_thd->NUM_THREADS_VNEC;
    const spMtxVNEC<double>::SpB_Matrix *A = &(mat_thd->M), *a = &(mat_thd->M);
    int *row_end_offsets = a->ptr + 1; // Merge list A
    int row_carry_out[NUM_THREADS_VNEC];
    double value_carry_out[NUM_THREADS_VNEC];

        size_t vl64m4 = __riscv_vsetvlmax_e64m4();
        size_t elem_for_thread = y.size() / NUM_THREADS_VNEC;
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++) {
            size_t shift = elem_for_thread * tid;
            for (int i = 0; i < elem_for_thread; i += vl64m4) {
                size_t vl = __riscv_vsetvl_e64m4(elem_for_thread - i);
                vfloat64m4_t val = __riscv_vle64_v_f64m4(y.data() + shift + i, vl);
                val = __riscv_vfmul_vf_f64m4(val, beta, vl);
                __riscv_vse64_v_f64m4(y.data() + shift + i, val, vl);
            }
        }
        for (int i = y.size() - y.size() % NUM_THREADS_VNEC; i < y.size(); i += vl64m4) {
            size_t vl = __riscv_vsetvl_e64m4(y.size() - i);
            vfloat64m4_t val = __riscv_vle64_v_f64m4(y.data() + i, vl);
            val = __riscv_vfmul_vf_f64m4(val, beta, vl);
            __riscv_vse64_v_f64m4(y.data() + i, val, vl);
        }

#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
    {
        coord thread_coord_start = mat_thd->thread_coord_start[tid];
        coord thread_coord_end = mat_thd->thread_coord_end[tid];
        for (; thread_coord_start.x < thread_coord_end.x; ++thread_coord_start.x)
        {
            double running_total = 0.0;
            for (; thread_coord_start.y < (int)row_end_offsets[thread_coord_start.x]; ++thread_coord_start.y)
            {
                running_total += ((double *)a->val)[thread_coord_start.y] * x[a->indices[thread_coord_start.y]];
            } // End of this line
            y[thread_coord_start.x] += alpha * running_total;
        }

        //  finish one row, calculate the partial sum of the next row
        //  Consume partial portion of thread's last row (accumulate any nonzeros for a partial row shared with the next thread)
        double running_total = 0.0;
        for (; thread_coord_start.y < thread_coord_end.y; ++thread_coord_start.y)
        {
            running_total += ((double *)a->val)[thread_coord_start.y] * x[a->indices[thread_coord_start.y]];
        }
        // save the thread's running total and row-id for subsequent fix-up
        row_carry_out[tid] = thread_coord_end.x; // The value of this row is incomplete
        value_carry_out[tid] = running_total;    // This row would be a little bit more complete with this. Each thread might have one of these
    }
// update the values in y for rows that span multiple threads
#pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
    for (int tid = 0; tid < NUM_THREADS_VNEC - 1; ++tid)
    {
            y[row_carry_out[tid]] += alpha * value_carry_out[tid];
    }
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVNEC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
    sparse_matrix_status status;

    return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVNEC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
    sparse_matrix_status status;

    if (mat.vnec_t == SpB_VNEC_S) {
        SpMV_VNEC_S_FP64(y, b, &mat, alpha, beta);
    }
    
    else if (mat.vnec_t == SpB_VNEC_D) {
        SpMV_VNEC_D_FP64(y, b, &mat, alpha, beta);
    }

    else if (mat.vnec_t == SpB_VNEC_L) {
        SpMV_VNEC_L_FP64(y, b, &mat, alpha, beta);
    }

    return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVNEC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
    sparse_matrix_status status;

    return status;
}

template<> 
sparse_matrix_status sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_ALL_STAGES>(
                               sparse_operation_t type_op, 
                               double alpha, 
                               const spMtxVNEC<double> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<double> &b,
                               double beta,
                               std::vector<double> &y){
    sparse_matrix_status status;

    sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_PREPARATION>
        (type_op, alpha, mat, descr, b, beta, y);
    sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_MULT>
        (type_op, alpha, mat, descr, b, beta, y);
    sparse_mv<double, spMtxVNEC, true, SPARSE_MATRIX_MV_GET_RESULTS>
        (type_op, alpha, mat, descr, b, beta, y);

    return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_PREPARATION>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxVNEC<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
    sparse_matrix_status status;

    return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_MULT>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxVNEC<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
    sparse_matrix_status status;

    if (mat.vnec_t == SpB_VNEC_S) {
        SpMV_VNEC_S_FP32(y, b, &mat, alpha, beta);
    }

    else if (mat.vnec_t == SpB_VNEC_D) {
        SpMV_VNEC_D_FP32(y, b, &mat, alpha, beta);
    }

    else if (mat.vnec_t == SpB_VNEC_L) {
        SpMV_VNEC_L_FP32(y, b, &mat, alpha, beta);
    }

    return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_GET_RESULTS>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxVNEC<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
    sparse_matrix_status status;

    

    return status;
}

template<> 
sparse_matrix_status sparse_mv<float, spMtxVNEC, true>(
                               sparse_operation_t type_op, 
                               float alpha, 
                               const spMtxVNEC<float> &mat, 
                               sparse_matrix_descr descr, 
                               const std::vector<float> &b,
                               float beta,
                               std::vector<float> &y){
    sparse_matrix_status status;

    sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_PREPARATION>
        (type_op, alpha, mat, descr, b, beta, y);
    sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_MULT>
        (type_op, alpha, mat, descr, b, beta, y);
    sparse_mv<float, spMtxVNEC, true, SPARSE_MATRIX_MV_GET_RESULTS>
        (type_op, alpha, mat, descr, b, beta, y);

    return status;
}

}
