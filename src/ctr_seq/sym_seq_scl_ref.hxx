/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#ifndef __SYM_SEQ_SCL_REF_HXX__
#define __SYM_SEQ_SCL_REF_HXX__

#include "../shared/util.h"
#include <limits.h>

/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 2*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             int *                  ndim_tot,
             int **                 idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  dim_max++;
  *ndim_tot = dim_max;
  *idx_arr = (int*)malloc(sizeof(int)*dim_max);
  std::fill((*idx_arr), (*idx_arr)+dim_max, -1);  

  for (i=0; i<ndim_A; i++){
    (*idx_arr)[idx_A[i]] = i;
  }
}

/**
 * \brief performs symmetric contraction
 */
template<typename dtype>
int sym_seq_scl_ref( dtype const        alpha,
                     dtype *      A,
                     int const          ndim_A,
                     int const *        edge_len_A,
                     int const *        _lda_A,
                     int const *        sym_A,
                     int const *        idx_map_A){
  TAU_FSTART(sym_seq_sum_ref);
  int idx, i, idx_max, imin, imax, idx_A, iA, j, k;
  int off_idx, off_lda, sym_pass;
  int * idx_glb, * rev_idx_map;
  int * dlen_A;

  inv_idx(ndim_A,       idx_map_A,
          &idx_max,     &rev_idx_map);

  dlen_A = (int*)malloc(sizeof(int)*ndim_A);
  memcpy(dlen_A, edge_len_A, sizeof(int)*ndim_A);

  idx_glb = (int*)malloc(sizeof(int)*idx_max);
  memset(idx_glb, 0, sizeof(int)*idx_max);


  idx_A = 0;
  sym_pass = 1;
  for (;;){
    if (sym_pass){
      A[idx_A] = alpha*A[idx_A];
    }

    for (idx=0; idx<idx_max; idx++){
      imin = 0, imax = INT_MAX;

#define GET_MIN_MAX(__X)                                                                  \
do{                                                                                                         \
      i##__X = rev_idx_map[idx];                                                      \
      if (i##__X != -1){                                                                    \
        imax = MIN(imax, edge_len_##__X[i##__X]);       \
      }                                                                                           \
} while (0);
      GET_MIN_MAX(A);
#undef GET_MIN_MAX

      LIBT_ASSERT(idx_glb[idx] >= imin && idx_glb[idx] < imax);

      idx_glb[idx]++;

      if (idx_glb[idx] >= imax){
              idx_glb[idx] = imin;
      }
      if (idx_glb[idx] != imin) {
              break;
      }
    }
    if (idx == idx_max) break;

#ifdef SEQ
#define CHECK_SYM(__X)                              \
do {                                                \
        sym_pass = 1;                               \
        for (i=0; i<ndim_##__X; i++){               \
          if ((sym_##__X[i] & 0x2) == 0x2){         \
            if (idx_glb[idx_map_##__X[i+1]] <=      \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
          if (sym_##__X[i] == SY){                  \
            if (idx_glb[idx_map_##__X[i+1]] <       \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
        }                                           \
} while(0)
#else
#define CHECK_SYM(__X)                              \
do {                                                \
        sym_pass = 1;                               \
        for (i=0; i<ndim_##__X; i++){               \
          if (sym_##__X[i] != NS){                  \
            if (idx_glb[idx_map_##__X[i+1]] <       \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
        }                                           \
} while(0)
#endif
    CHECK_SYM(A);
    if (!sym_pass) continue;
    

#define RESET_IDX(__X)                                                                        \
do {                                                                                                        \
        idx_##__X = idx_glb[idx_map_##__X[0]];                                    \
        off_idx = 0, off_lda = 1;                                                             \
        for (i=1; i<ndim_##__X; i++){                                                     \
          if (sym_##__X[i-1] == NS){                                                      \
            off_idx = i;                                                                            \
            off_lda = sy_packed_size(i, dlen_##__X, sym_##__X); \
            idx_##__X += off_lda*idx_glb[idx_map_##__X[i]];               \
          } else if (idx_glb[idx_map_##__X[i]]!=0) {                          \
            k = 1;                                                                                      \
            dlen_##__X[i] = idx_glb[idx_map_##__X[i]];                      \
            do {                                                                                          \
              dlen_##__X[i-k] = idx_glb[idx_map_##__X[i]];                \
              k++;                                                                                      \
            } while (i>=k && sym_##__X[i-k] != NS);                           \
              idx_##__X += off_lda*sy_packed_size(i+1-off_idx,  \
                                    dlen_##__X+off_idx,sym_##__X+off_idx);      \
            for (j=0; j<k; j++){                                                              \
              dlen_##__X[i-j] = edge_len_##__X[i-j];                          \
            }                                                                                               \
          }                                                                                                   \
        }                                                                                                       \
} while (0)
    if (ndim_A > 0)
      RESET_IDX(A);
#undef RESET_IDX
  }
  free(dlen_A);
  free(idx_glb);
  TAU_FSTOP(sym_seq_sum_ref);
  return 0;
}






#endif
