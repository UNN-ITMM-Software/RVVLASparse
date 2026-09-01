#pragma once

// The converter and variable names from the original implementation were partially used 
// https://github.com/vhccspmv/vhcc/tree/master

#include "CRS.h"
#include "malloc.h"
#include <stdint.h>
#include "omp.h"
#include <vector>
#include <cmath>

namespace SparseMatrixLib {

struct vhcc_params {
  size_t num_threads; 
  size_t num_panels;
  size_t tile_row;
  size_t tile_col;
  size_t vec_size;
  size_t n_masks;
  
  vhcc_params(size_t nt = 1, size_t np = 1, size_t row_tile = 512, size_t col_tile = 512, size_t vec = 8) :
              num_threads(nt), num_panels(np), tile_row(row_tile), tile_col(col_tile), 
              vec_size(vec), n_masks(log2(vec_size)) {}
};

struct thr_info_t {
  int vbase;
  int rbase;
  int sbase;
  int last_row;
  int overflow_row;
  int start_vec;
  int end_vec;
  int panel_id;
  int merge_start;
  int merge_end;
} ;

  typedef uint32_t index_t;

template <typename ValT>
class spMtxVHCC {

  size_t _num_rows;
  size_t _num_cols;
  size_t _num_entries;
  size_t _num_vectors;
  size_t _nnz_per_panel;
  size_t _pad_entries;
  size_t _extended_rows;
  size_t _pad_rows;
  
  std::vector<thr_info_t> _thr_info;
  std::vector<uint32_t> _veceor_ptr;
  std::vector<uint8_t> _scan_mask;
  std::vector<uint32_t> _row_arr;
  std::vector<uint32_t> _col_arr;
  std::vector<ValT> _vals_arr;

  vhcc_params params;

  public:
  
  spMtxVHCC() {  }
  
  size_t  get_n_masks() const { return params.n_masks; }
  int get_pad_rows() const    { return _pad_rows; }
  
  const thr_info_t *get_thr_info() const { return _thr_info.data(); }
  const uint32_t *get_veceor_ptr() const { return _veceor_ptr.data(); }
  const uint8_t  *get_scan_mask() const  { return _scan_mask.data(); }
  const uint32_t *get_row_arr() const    { return _row_arr.data(); }
  const uint32_t *get_col_arr() const    { return _col_arr.data(); }
  const ValT     *get_vals_arr() const   { return _vals_arr.data(); }

  vhcc_params     get_params() const     { return params; }  
    
  std::vector<thr_info_t>& get_thr_info_reference()  { return _thr_info; };
  std::vector<uint32_t>&   get_veceor_reference()    { return _veceor_ptr; }
  std::vector<uint8_t>&    get_scan_mask_reference() { return _scan_mask; } 
  std::vector<uint32_t>&   get_row_arr_reference()   { return _row_arr; }
  std::vector<uint32_t>&   get_col_arr_reference()   { return _col_arr; }
  std::vector<ValT>&       get_vals_arr_reference()  { return _vals_arr; }
  
  void freeMem() {
    _thr_info.resize(0); _thr_info.shrink_to_fit();
    _veceor_ptr.resize(0); _veceor_ptr.shrink_to_fit();
    _scan_mask.resize(0); _scan_mask.shrink_to_fit();
    _row_arr.resize(0); _row_arr.shrink_to_fit();
    _col_arr.resize(0); _col_arr.shrink_to_fit();
    _vals_arr.resize(0); _vals_arr.shrink_to_fit();
  }
  
  ~spMtxVHCC() {
    freeMem();
  }  

  void set_params(const vhcc_params& info) {
    params = info;
  }  

  void set_properties(size_t new_num_rows, size_t new_num_cols, size_t new_num_entries, 
                      size_t new_num_vectors, size_t new_pad_entries, size_t new_pad_rows) {
    _num_rows = new_num_rows; 
    _num_cols = new_num_cols;
    _num_entries = new_num_entries; 
    _num_vectors = new_num_vectors;
    _pad_entries = new_pad_entries;
    _pad_rows = new_pad_rows;
  }
  

};




}