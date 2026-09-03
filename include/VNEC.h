#pragma once
#include <iostream>
#include <cstring>
#include <vector>
#include "CRS.h"
#include <omp.h>
#include "SpB_mxv_VNEC.h"
#include "convert.h"
#include <type_traits>

namespace SparseMatrixLib 
{

    void MergePathDivide(
        int diagonal,
        int *a,
        int *b,
        int a_len,
        int b_len,
        coord *path_coordinate);

template <typename ValT>
class spMtxVNEC{
public:
    struct csr_f {
        ValT *nnz;
        int *col, *row_begin, *row_end;
        int *task_start;
        int *task_end;
        csr_f(): nnz(nullptr), col(nullptr), row_begin(nullptr),
        row_end(nullptr), task_start(nullptr), task_end(nullptr) {}
    };
    struct SpB_Matrix {
        int row;
        int col;
        int nnz;
        int ptr_len;
        int *ptr;
        int *indices;
        ValT *val;
    };
    int NUM_THREADS_VNEC;
    const spMtxCRS<ValT> *G;
    SpB_Matrix M;
    coord *thread_coord_start;
    coord *thread_coord_end;
    int *ecr_indices;
    int *NEC_NUM;
    int **use_x_indices;
    ValT **ecr_xx_val;
    int *v_row_ptr;
    int *col_start;
    ValT *val_align;
    int *spvv_len;
    int **tasks;
    mutable csr_f reorder_mat;
    int num_merge_items;
    int items_per_thread;
    int *diagonal_start;
    int *diagonal_end;
    int *nz_indices;
    float IRD_mat;
    SpB_VNEC_type vnec_t;

    float IRD_VNEC(SpB_Type type)
    {
        int nLanes = (type == SpB_FP32) ? nLanes_f32 : nLanes_f64;
        SpB_Matrix *a = &M, *A = &M;
        int num_merge_items = a->nnz + A->row;
        int items_per_thread = (num_merge_items + NUM_THREADS_VNEC - 1) / NUM_THREADS_VNEC;
        int *nz_indices = (int *)malloc((a->nnz) * sizeof(int));
        int *diagonal_start = (int *)malloc((NUM_THREADS_VNEC) * sizeof(int));
        int *diagonal_end = (int *)malloc((NUM_THREADS_VNEC) * sizeof(int));
        coord *thread_coord_start = (coord *)malloc((NUM_THREADS_VNEC) * sizeof(coord));
        coord *thread_coord_end = (coord *)malloc((NUM_THREADS_VNEC) * sizeof(coord));
        for (SpB_Index i = 0; i < a->nnz; i++)
        {
            nz_indices[i] = i;
        }
    #pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
        {
            diagonal_start[tid] = std::min(items_per_thread * tid, num_merge_items);
            diagonal_end[tid] = std::min(diagonal_start[tid] + items_per_thread, num_merge_items);
            MergePathDivide(diagonal_start[tid], (a->ptr + 1), nz_indices, A->row, a->nnz, thread_coord_start + tid);
            MergePathDivide(diagonal_end[tid], (a->ptr + 1), nz_indices, A->row, a->nnz, thread_coord_end + tid);
        }
        SpB_Index A_COLS = A->col;
        int *ecr_indices = (int *)malloc((a->nnz + 10) * sizeof(int));
        memset(ecr_indices, 0, (a->nnz + 10) * sizeof(int));
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
        {
            int *not_null_col_flag = (int *)malloc(A_COLS * sizeof(int));
            int *IDX_MAP = (int *)malloc(A_COLS * sizeof(int));
            int *IDX_OFFSET = (int *)malloc(A_COLS * sizeof(int));
            coord thread_coord_start_ = thread_coord_start[tid];
            coord thread_coord_end_ = thread_coord_end[tid];
            for (SpB_Index col = 0; col < A_COLS; col++)
            {
                IDX_MAP[col] = col;
                not_null_col_flag[col] = 1;
                IDX_OFFSET[col] = 1;
            }

            for (int j = thread_coord_start_.y; j < thread_coord_end_.y; ++j)
            {
                not_null_col_flag[a->indices[j]] = 0;
            }
            IDX_OFFSET[0] = not_null_col_flag[0];
            for (SpB_Index col = 1; col < A_COLS; col++)
            {
                IDX_OFFSET[col] = IDX_OFFSET[col - 1] + not_null_col_flag[col];
            }
            for (SpB_Index col = 0; col < A_COLS; col++)
            {
                IDX_MAP[col] = IDX_MAP[col] - IDX_OFFSET[col];
            }
            {
                for (int j = thread_coord_start_.y; j < thread_coord_end_.y; ++j)
                {
                    ecr_indices[j] = IDX_MAP[a->indices[j]];
                }
            }
            free(not_null_col_flag);
            free(IDX_MAP);
            free(IDX_OFFSET);
        }
        int *col_start = (int *)malloc((a->nnz + 1) * sizeof(int));
        int *v_row_ptr = (int *)malloc((a->ptr_len + 1) * sizeof(int));

        memset(col_start, 0, (a->nnz + 1) * sizeof(int));
        memset(v_row_ptr, 0, (a->ptr_len + 1) * sizeof(int));

        int group_index = 0;
        v_row_ptr[0] = 0;
        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS_VNEC)
        for (int tid = 0; tid < NUM_THREADS_VNEC; tid++)
        {
            coord thread_coord_start_ = thread_coord_start[tid];
            coord thread_coord_end_ = thread_coord_end[tid];
            for (int i = thread_coord_start_.x; i < thread_coord_end_.x; ++i)
            {
                int ptr_start = ((int)(a->ptr[i]) > thread_coord_start_.y) ? a->ptr[i] : thread_coord_start_.y;
                int n_one_line = a->ptr[i + 1] - ptr_start;
                col_start[group_index] = ecr_indices[ptr_start];
                for (int j = 1; j < n_one_line; j++)
                {
                        int dist = ecr_indices[ptr_start + j] - col_start[group_index];
                        if (dist < nLanes)
                        {
                        }
                        else
                        {
                            group_index++;
                            col_start[group_index] = ecr_indices[ptr_start + j];
                        }
                    }
                    if (n_one_line != 0)
                    {
                        group_index++;
                    }
                v_row_ptr[i + 1] = group_index;
            }
        }
        int _nnz_ = v_row_ptr[A->row] * nLanes;
        float _IRD_mat = (float)a->nnz / (float)_nnz_;

        free(diagonal_start);
        free(diagonal_end);
        free(nz_indices);
        free(thread_coord_start);
        free(thread_coord_end);
        free(ecr_indices);
        free(col_start);
        free(v_row_ptr);
        return _IRD_mat;
    }

    spMtxVNEC():
                thread_coord_start(nullptr), thread_coord_end(nullptr), ecr_indices(nullptr),
                NEC_NUM(nullptr), use_x_indices(nullptr), ecr_xx_val(nullptr),
                v_row_ptr(nullptr), col_start(nullptr), val_align(nullptr),
                spvv_len(nullptr), tasks(nullptr), diagonal_start(nullptr), diagonal_end(nullptr),
                nz_indices(nullptr), G(nullptr), NUM_THREADS_VNEC(get_num_threads())
                {}

    spMtxVNEC(const spMtxCRS<ValT>& _G):
                thread_coord_start(nullptr), thread_coord_end(nullptr), ecr_indices(nullptr),
                NEC_NUM(nullptr), use_x_indices(nullptr), ecr_xx_val(nullptr),
                v_row_ptr(nullptr), col_start(nullptr), val_align(nullptr),
                spvv_len(nullptr), tasks(nullptr), diagonal_start(nullptr), diagonal_end(nullptr),
                nz_indices(nullptr), G(nullptr), NUM_THREADS_VNEC(get_num_threads())
    {
        G = &_G;
        M.row = G->m;
        M.col = G->n;
        M.nnz = G->nz;
        M.ptr_len = G->m + 1;
        M.ptr = G->Rst;
        M.indices = G->Col;
        M.val = G->Val;
        SpB_Type type;
        if (std::is_same<ValT, float>::value) type = SpB_FP32;
        else if (std::is_same<ValT, double>::value) type = SpB_FP64;
        else type = SpB_Type_N;
        float IRD_thr = (type == SpB_FP32) ? IRD_thr_fp32 : IRD_thr_fp64;
        IRD_mat = IRD_VNEC(type);

        if (G->m < 2'000'000) {
            if (IRD_mat >= IRD_thr) vnec_t = SpB_VNEC_D;
            else vnec_t = SpB_VNEC_S;
        }
        else vnec_t = SpB_VNEC_L;
    }

    spMtxVNEC(const spMtxVNEC &copy):
                vnec_t(copy.vnec_t), G(copy.G), IRD_mat(copy.IRD_mat), NUM_THREADS_VNEC(copy.NUM_THREADS_VNEC)
    {
        M.row = G->m;
        M.col = G->n;
        M.nnz = G->nz;
        M.ptr_len = G->m + 1;
        M.ptr = G->Rst;
        M.indices = G->Col;
        M.val = G->Val;
        convert<ValT, spMtxVNEC, false>(*this, *G, convertParams());
    }

    spMtxVNEC(spMtxVNEC &&mov):
                    vnec_t(mov.vnec_t), IRD_mat(mov.IRD_mat), NUM_THREADS_VNEC(mov.NUM_THREADS_VNEC)
    {
        G = mov.G;
        mov.G = nullptr;
        M.row = G->m;
        M.col = G->n;
        M.nnz = G->nz;
        M.ptr_len = G->m + 1;
        M.ptr = G->Rst;
        M.indices = G->Col;
        M.val = G->Val;

    }

    spMtxVNEC& operator=(const spMtxVNEC &copy)
    {
        if (this == &copy)
            return *this;

        vnec_t = copy.vnec_t;
        IRD_mat = copy.IRD_mat;
        G = copy.G;
        M.row = G->m;
        M.col = G->n;
        M.nnz = G->nz;
        M.ptr_len = G->m + 1;
        M.ptr = G->Rst;
        M.indices = G->Col;
        M.val = G->Val;
        NUM_THREADS_VNEC = copy.NUM_THREADS_VNEC;
        convert<ValT, spMtxVNEC, false>(*this, *G, convertParams());
        return *this;
    }

    spMtxVNEC& operator=(spMtxVNEC &&mov)
    {
        vnec_t = mov.vnec_t;
        IRD_mat = mov.IRD_mat;
        G = mov.G;
        mov.G = nullptr;
        M.row = G->m;
        M.col = G->n;
        M.nnz = G->nz;
        M.ptr_len = G->m + 1;
        M.ptr = G->Rst;
        M.indices = G->Col;
        M.val = G->Val;
        NUM_THREADS_VNEC = mov.NUM_THREADS_VNEC;
        return *this;
    }

    void freeMem() {
                free(NEC_NUM);
        if (use_x_indices != nullptr) 
            for (size_t i = 0; i < NUM_THREADS_VNEC + 1; ++i) 
                free(use_x_indices[i]);
        free(use_x_indices);
        if (ecr_xx_val != nullptr) 
            for (size_t i = 0; i < NUM_THREADS_VNEC + 1; ++i) 
                free(ecr_xx_val[i]);
        free(ecr_xx_val);
        free(val_align);
        free(spvv_len);
        if (tasks != nullptr) 
            for (size_t i = 0; i < NUM_THREADS_VNEC; ++i) 
                free(tasks[i]);
        free(tasks);

        free(thread_coord_start);
        free(thread_coord_end);
        free(ecr_indices);
        free(v_row_ptr);
        free(col_start);
        free(diagonal_start);
        free(diagonal_end);
        free(nz_indices);

        free(reorder_mat.nnz);
        free(reorder_mat.col);
        free(reorder_mat.row_begin);
        free(reorder_mat.row_end);
        free(reorder_mat.task_start);
        free(reorder_mat.task_end);

        thread_coord_start = nullptr;
        thread_coord_end = nullptr;
        ecr_indices = nullptr;
        NEC_NUM = nullptr;
        use_x_indices = nullptr;
        ecr_xx_val = nullptr;
        v_row_ptr = nullptr;
        col_start = nullptr;
        val_align = nullptr;
        spvv_len = nullptr;
        tasks = nullptr;
        diagonal_start = nullptr;
        diagonal_end = nullptr;
        nz_indices = nullptr;
        reorder_mat.nnz = nullptr;
        reorder_mat.col = nullptr;
        reorder_mat.row_begin = nullptr;
        reorder_mat.row_end = nullptr;
        reorder_mat.task_start = nullptr;
        reorder_mat.task_end = nullptr;
    }

    ~spMtxVNEC()
    {
        freeMem();
    }
};

}
