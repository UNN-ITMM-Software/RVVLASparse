#include "sparse_matrix.h"
#include <cstring>
#include <algorithm>

namespace SparseMatrixLib
{

template <typename T>
void crs_to_HCSR(spMtxHCSR<T>& dist, const spMtxCRS<T>& src, size_t R, size_t C, double bs) {
    uint32_t num_threads = static_cast<uint32_t>(omp_get_max_threads());
    if (num_threads * R > src.m) {
        if (src.m >= num_threads)
            R = (src.m + num_threads - 1) / num_threads;
        else
            num_threads = 1;
    }

    dist = spMtxHCSR<T>();
    dist.R = R;
    dist.C = C;
    double block_switch = bs;

    dist.b_m = (src.m + dist.R - 1) / dist.R;
    dist.b_n = (src.n + dist.C - 1) / dist.C;
    dist.m = src.m;
    dist.n = src.n;
    dist.nz = src.nz;

    dist.col_idx.assign(dist.nz, 0);
    dist.values.assign(dist.nz, 0.0);
    dist.block_ptr.assign(dist.b_m * dist.b_n + 1, 0);
    dist.row_offset.assign(dist.b_m * dist.b_n + 1, 0);
    std::vector<uint32_t> crs_or_coo(dist.b_m * dist.b_n, 0);

    std::vector<inCRS<T>> block_row;
    size_t non_empty_blocks_rows_ptr = 0;
    size_t block_ind = 0;
    
    for (int br = 0; br < dist.b_m; ++br) {

        block_row = std::vector<inCRS<T>>(dist.b_n);
        int start_row = br * dist.R;
        int end_row = std::min((br + 1) * dist.R, dist.m);
        for (int i = start_row; i < end_row; ++i) {
            for (int k = src.Rst[i]; k < src.Rst[i + 1]; ++k) {
                int c = src.Col[k] / dist.C;
                if (block_row[c].n == 0) {
                    block_row[c].m = end_row - start_row;
                    block_row[c].n = dist.C;
                    block_row[c].row_ptr.assign(block_row[c].m + 1, 0);
                }
                ++block_row[c].row_ptr[i - start_row + 1];
                block_row[c].col_idx.push_back(src.Col[k] % dist.C);
                block_row[c].values.push_back(src.Val[k]);
            }
        }

        for (int j = 0; j < dist.b_n; ++j) {
            inCRS<T>& crs = block_row[j];
            crs.nz = crs.values.size();
            if (crs.n != 0) {
                if (double(crs.m) * block_switch < crs.nz) {
                    crs_or_coo[block_ind] = 1;
                    dist.row_offset[block_ind + 1] = dist.row_offset[block_ind] + crs.m + 1;
                    non_empty_blocks_rows_ptr += (crs.m + 1);
                    std::copy(crs.col_idx.begin(), crs.col_idx.end(), dist.col_idx.begin() + dist.block_ptr[block_ind]);
                    std::copy(crs.values.begin(), crs.values.end(), dist.values.begin() + dist.block_ptr[block_ind]);
                }
                else {
                    crs_or_coo[block_ind] = 2;
                    dist.row_offset[block_ind + 1] = dist.row_offset[block_ind] + crs.nz;
                    non_empty_blocks_rows_ptr += (crs.nz);
                    std::copy(crs.col_idx.begin(), crs.col_idx.end(), dist.col_idx.begin() + dist.block_ptr[block_ind]);
                    std::copy(crs.values.begin(), crs.values.end(), dist.values.begin() + dist.block_ptr[block_ind]);
                }
            }
            else {
                dist.row_offset[block_ind + 1] = dist.row_offset[block_ind];
                crs_or_coo[block_ind] = 0;
            }
            dist.block_ptr[block_ind + 1] = dist.block_ptr[block_ind] + crs.nz;
            ++block_ind;
        }
    }
    dist.row_ptr.assign(non_empty_blocks_rows_ptr, 0);

    non_empty_blocks_rows_ptr = 0;
    for (int br = 0; br < dist.b_m; ++br) {
        block_row = std::vector<inCRS<T>>(dist.b_n);
        int start_row = br * dist.R;
        int end_row = std::min((br + 1) * dist.R, dist.m);
        for (int i = start_row; i < end_row; ++i) {
            for (int k = src.Rst[i]; k < src.Rst[i + 1]; ++k) {
                int c = src.Col[k] / dist.C;
                if (block_row[c].n == 0) {
                    block_row[c].m = end_row - start_row;
                    block_row[c].n = dist.C;
                    block_row[c].row_ptr.assign(block_row[c].m + 1, 0);
                }
                ++block_row[c].row_ptr[i - start_row + 1];
                block_row[c].col_idx.push_back(src.Col[k] % dist.C);
                block_row[c].values.push_back(src.Val[k]);
            }
        }

        for (int j = 0; j < dist.b_n; ++j) {
            inCRS<T>& crs = block_row[j];
            crs.nz = crs.values.size();
            if (crs.n != 0) { 
                if (double(crs.m) * block_switch < crs.nz) {
                    for (int i = 0; i < crs.m; ++i) crs.row_ptr[i + 1] += crs.row_ptr[i];
                    std::copy(crs.row_ptr.begin(), crs.row_ptr.end(), dist.row_ptr.begin() + non_empty_blocks_rows_ptr);
                    non_empty_blocks_rows_ptr += (crs.m + 1);
                }
                else {
                    int nz_counter = non_empty_blocks_rows_ptr;
                    for (int i = 0; i <= crs.m; ++i) {
                        for (int _ = 0; _ < crs.row_ptr[i]; ++_) {
                            dist.row_ptr[nz_counter++] = i - 1;
                        }
                    }
                    non_empty_blocks_rows_ptr += (crs.nz);
                }
            }
        }
    }

    for (int i = 0; i < crs_or_coo.size(); ++i) {
        crs_or_coo[i] <<= 30;
        dist.row_offset[i] |= crs_or_coo[i];
    }
}

template<> 
void convert<double, spMtxHCSR, false>(spMtxHCSR<double> &dist, const spMtxCRS<double> &src, 
                                      const convertParams &params)
{
    int R = 2000;
    int C = 2000;
    double p = 2;
    if(params.param.count("R"))
        R = params.param.at("R").i;
    if(params.param.count("C"))
        C = params.param.at("C").i;
    if(params.param.count("p"))
        p = params.param.at("p").d; 

    crs_to_HCSR(dist, src, R, C, p);
}

template<> 
void convert<float, spMtxHCSR, false>(spMtxHCSR<float> &dist, const spMtxCRS<float> &src,
                                     const convertParams &params)
{
    int R = 4000;
    int C = 4000;
    double p = 2;
    if(params.param.count("R"))
        R = params.param.at("R").i;
    if(params.param.count("C"))
        C = params.param.at("C").i;
    if(params.param.count("p"))
        p = params.param.at("p").d; 
        
    crs_to_HCSR(dist, src, R, C, p);
}

}
