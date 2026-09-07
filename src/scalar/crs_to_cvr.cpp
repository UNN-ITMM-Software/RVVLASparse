#include "sparse_matrix.h"
#include "omp.h"
#include <vector>
namespace SparseMatrixLib
{
  template <typename T_CONVERT>
  void convert_body(spMtxCVR<T_CONVERT>& dist, const spMtxCRS<T_CONVERT>& src,
    const convertParams& params)
  {
    dist.freeMem();
    if (params.param.count("SIMD_LEN"))
      dist.SIMD_LEN = params.param.at("SIMD_LEN").i;
    else
      throw std::invalid_argument("Invalid converter parameter SIMD_LEN. SIMD_LEN must be equal to 8 for both single and double precision");
    dist.nItems = (src.nz + dist.SIMD_LEN - 1) / dist.SIMD_LEN * dist.SIMD_LEN;
    dist.numRows = src.m;
    dist.numCols = src.n;

    dist.vPack_vec_vals.resize(dist.nItems);

    dist.vPack_vec_cols.resize(dist.nItems);
    dist.vPack_vec_record.resize(2);

    std::vector<int> v_tracker_iter(2 * dist.SIMD_LEN, 0); //pairs (position where current row starts, position where current row ends) for every SIMD line
    std::vector<int> v_tracker_row(dist.SIMD_LEN, 0);
    int *tracker_iter = v_tracker_iter.data();
    int *tracker_row = v_tracker_row.data();
    int cvr_iter = 0;
    int record_iter = 0;
    int last_row = 0;
    int conversion_iter = 0;

#pragma omp parallel
    {
#pragma omp master
      {
        dist.Nthreads_avail = omp_get_num_threads();
      }
    }

    if (dist.nItems / dist.Nthreads_avail >= dist.SIMD_LEN) {
      dist.Nthreads = dist.Nthreads_avail;
    }
    else
      dist.Nthreads = 1;

    dist.writeback_pos_record.resize(dist.Nthreads * (dist.SIMD_LEN + 2) + 2);

    int elems_by_thread = (dist.nItems / dist.SIMD_LEN) / dist.Nthreads * dist.SIMD_LEN, writeback_record = 0;

    for (int i = 0; i < dist.Nthreads; i++)
      dist.writeback_pos_record[i * (dist.SIMD_LEN + 2)] = elems_by_thread * i;

    dist.writeback_pos_record[dist.Nthreads * (dist.SIMD_LEN + 2)] = dist.nItems;
    dist.writeback_pos_record[0 * (dist.SIMD_LEN + 2) + 1] = record_iter / 2;

    int iter_start = 0;
    bool feeding_start = true;

    //searching for non-zero lines
    for (; last_row < dist.numRows && iter_start < dist.SIMD_LEN; last_row++) {
      if (src.Rst[last_row] != src.Rst[last_row + 1]) {
        tracker_iter[2 * iter_start] = src.Rst[last_row];
        tracker_iter[2 * iter_start + 1] = src.Rst[last_row + 1];
        tracker_row[iter_start] = last_row;
        iter_start++;
      }
    }

    while (cvr_iter < dist.nItems) {
      if (cvr_iter == dist.writeback_pos_record[(writeback_record + 1) * (dist.SIMD_LEN + 2)])      //write thread writeback info
      {
        dist.writeback_pos_record[(writeback_record + 1) * (dist.SIMD_LEN + 2) + 1] = record_iter / 2;
        std::copy(tracker_row, tracker_row + dist.SIMD_LEN, dist.writeback_pos_record.data() + writeback_record * (dist.SIMD_LEN + 2) + 2);
        writeback_record++;
      }
      for (int i = 0; i < dist.SIMD_LEN; i++) {
        if (tracker_iter[2 * i] < tracker_iter[2 * i + 1])              //can write to cvr format directly
        {
          dist.vPack_vec_cols[cvr_iter] = src.Col[tracker_iter[2 * i]];
          dist.vPack_vec_vals[cvr_iter++] = src.Val[tracker_iter[2 * i]];

          tracker_iter[2 * i]++;
        }
        else {                                      //tracker is empty
          for (; last_row < dist.numRows; last_row++) {
            if (src.Rst[last_row] != src.Rst[last_row + 1])               break;
          }
          if (last_row < dist.numRows)                      //can refill the tracker 
          {
            dist.vPack_vec_record[record_iter + 1] = tracker_row[i];

            tracker_row[i] = last_row++;
            tracker_iter[2 * i] = src.Rst[tracker_row[i]];
            tracker_iter[2 * i + 1] = src.Rst[tracker_row[i] + 1];

            dist.vPack_vec_record[record_iter] = conversion_iter * dist.SIMD_LEN + i;

            record_iter += 2;

            dist.vPack_vec_cols[cvr_iter] = tracker_iter[2 * i] <= src.nz ? src.Col[tracker_iter[2 * i]] : 0;      //write to cvr format
            dist.vPack_vec_vals[cvr_iter++] = tracker_iter[2 * i] <= src.nz ? src.Val[tracker_iter[2 * i]] : .0;

            tracker_iter[2 * i]++;
            dist.vPack_vec_record.resize(dist.vPack_vec_record.size() + 2);
          }
          else {                                    //can't refill the tracker, feeding finished, stealing started
            int quan_steal = dist.nItems - cvr_iter + i;
            int conv_sequences_left = quan_steal / dist.SIMD_LEN;

            int candi = -1, max = 0;
            for (int j = 0; j < dist.SIMD_LEN; j++)
            {
              if ((tracker_iter[2 * j + 1] - tracker_iter[2 * j]) - (j > i ? 1 : 0) > max)
              {
                max = tracker_iter[2 * j + 1] - tracker_iter[2 * j];
                candi = j;
              }
            }

            if (candi == -1 || std::min(conv_sequences_left, max - (conv_sequences_left - 1)) == 0) { //no candidate to steal from, tail start, filling the tracker with zero
              dist.vPack_vec_record[record_iter++] = conversion_iter * dist.SIMD_LEN + i;
              dist.vPack_vec_record[record_iter++] = tracker_row[i];

              dist.vPack_vec_cols[cvr_iter] = dist.numCols - 1;
              dist.vPack_vec_vals[cvr_iter++] = .0;

              dist.vPack_vec_record.resize(dist.vPack_vec_record.size() + 2);

              continue;
            }
            //found candidate to steal from, stealing
            dist.vPack_vec_record[record_iter + 1] = tracker_row[i];

            tracker_row[i] = tracker_row[candi];

            tracker_iter[2 * i] = tracker_iter[2 * candi + 1] - std::min(conv_sequences_left, max - (conv_sequences_left - 1));
            tracker_iter[2 * i + 1] = tracker_iter[2 * candi + 1];
            tracker_iter[2 * candi + 1] = tracker_iter[2 * i];

            dist.vPack_vec_record[record_iter] = conversion_iter * dist.SIMD_LEN + i;

            record_iter += 2;

            dist.vPack_vec_cols[cvr_iter] = tracker_iter[2 * i] <= src.nz ? src.Col[tracker_iter[2 * i]] : 0;
            dist.vPack_vec_vals[cvr_iter++] = tracker_iter[2 * i] <= src.nz ? src.Val[tracker_iter[2 * i]] : .0;

            tracker_iter[2 * i]++;
            dist.vPack_vec_record.resize(dist.vPack_vec_record.size() + 2);
          }
        }
      }
      conversion_iter++;
    }

    dist.writeback_pos_record[(writeback_record + 1) * (dist.SIMD_LEN + 2) + 1] = record_iter / 2;
    std::copy(tracker_row, tracker_row + dist.SIMD_LEN, dist.writeback_pos_record.data() + writeback_record * (dist.SIMD_LEN + 2) + 2);

    return;
  }


  template<>
  void convert<double, spMtxCVR, false>(spMtxCVR<double>& dist, const spMtxCRS<double>& src,
    const convertParams& params)
  {
    convert_body(dist, src, params);
    return;
  }

  template<>
  void convert<float, spMtxCVR, false>(spMtxCVR<float>& dist, const spMtxCRS<float>& src,
    const convertParams& params)
  {
    convert_body(dist, src, params);
    return;
  }

}