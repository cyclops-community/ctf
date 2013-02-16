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

#ifndef __SYM_SEQ_SUM_INNER_HXX__
#define __SYM_SEQ_SUM_INNER_HXX__

#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_sum_ref.hxx"

/**
 * \brief performs symmetric summation
 */
template<typename dtype>
int sym_seq_sum_inr( dtype const        alpha,
                     dtype const *      A,
                     int const          ndim_A,
                     int const *        edge_len_A,
                     int const *        _lda_A,
                     int const *        sym_A,
                     int const *        idx_map_A,
                     dtype const        beta,
                     dtype *            B,
                     int const          ndim_B,
                     int const *        edge_len_B,
                     int const *        _lda_B,
                     int const *        sym_B,
                     int const *        idx_map_B,
                     int const          inr_stride){
  TAU_FSTART(sym_seq_sum_inr);
  int idx, i, idx_max, imin, imax, idx_A, idx_B, iA, iB, j, k;
  int off_idx, off_lda, sym_pass;
  int * idx_glb, * rev_idx_map;
  int * dlen_A, * dlen_B;

  inv_idx(ndim_A,       idx_map_A,
          ndim_B,       idx_map_B,
          &idx_max,     &rev_idx_map);

  dlen_A = (int*)malloc(sizeof(int)*ndim_A);
  dlen_B = (int*)malloc(sizeof(int)*ndim_B);
  memcpy(dlen_A, edge_len_A, sizeof(int)*ndim_A);
  memcpy(dlen_B, edge_len_B, sizeof(int)*ndim_B);

  idx_glb = (int*)malloc(sizeof(int)*idx_max);
  memset(idx_glb, 0, sizeof(int)*idx_max);


  /* Scale B immediately. FIXME: wrong for iterators over subset of B */
/*  if (beta != 1.0) {
    sz = sy_packed_size(ndim_B, edge_len_B, sym_B, NULL);
    for (i=0; i<sz; i++){
      B[i] = B[i]*beta;
    }
  }*/
  idx_A = 0, idx_B = 0;
  sym_pass = 1;
  for (;;){
    if (sym_pass){
  /*    printf("B[%d] = %lf*(A[%d]=%lf)+%lf*(B[%d]=%lf\n",
              idx_B,alpha,idx_A,A[idx_A],beta,idx_B,B[idx_B]);*/
    //  B[idx_B] = alpha*A[idx_A] + beta*B[idx_B];
      if (beta != 1.0){
        cxaxpy<dtype>(inr_stride, beta-1.0, B+idx_B*inr_stride, 1, B+idx_B*inr_stride, 1);
      }
      cxaxpy<dtype>(inr_stride, alpha, A+idx_A*inr_stride, 1, B+idx_B*inr_stride, 1); 
    }

    for (idx=0; idx<idx_max; idx++){
      imin = 0, imax = INT_MAX;

#define GET_MIN_MAX(__X,nr)                                                     \
do{                                                                             \
      i##__X = rev_idx_map[2*idx+nr];                                           \
      if (i##__X != -1){                                                        \
        imax = MIN(imax, edge_len_##__X[i##__X]);                               \
        /*if (sym_##__X[i##__X] > -1){                                          \
          imax = MIN(imax, idx_glb[idx_map_##__X[sym_##__X[i##__X]]]+1);        \
        }                                                                       \
        if (i##__X > 0 && sym_##__X[i##__X-1] > -1){                            \
          imin = MAX(imin, idx_glb[idx_map_##__X[i##__X-1]]);                   \
        }*/                                                                     \
      }                                                                 \
} while (0);
      GET_MIN_MAX(A,0);
      GET_MIN_MAX(B,1);
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
    CHECK_SYM(B);
    if (!sym_pass) continue;
    

#define RESET_IDX(__X)                                                  \
do {                                                                    \
        idx_##__X = idx_glb[idx_map_##__X[0]];                          \
        off_idx = 0, off_lda = 1;                                       \
        for (i=1; i<ndim_##__X; i++){                                   \
          if (sym_##__X[i-1] == NS){                                    \
            off_idx = i;                                                \
            off_lda = sy_packed_size(i, dlen_##__X, sym_##__X);         \
            idx_##__X += off_lda*idx_glb[idx_map_##__X[i]];             \
          } else if (idx_glb[idx_map_##__X[i]]!=0) {                    \
            k = 1;                                                      \
            dlen_##__X[i] = idx_glb[idx_map_##__X[i]];                  \
            do {                                                        \
              dlen_##__X[i-k] = idx_glb[idx_map_##__X[i]];              \
              k++;                                                      \
            } while (i>=k && sym_##__X[i-k] != NS);                     \
            idx_##__X += off_lda*sy_packed_size(i+1-off_idx,            \
                          dlen_##__X+off_idx,sym_##__X+off_idx);        \
            for (j=0; j<k; j++){                                        \
              dlen_##__X[i-j] = edge_len_##__X[i-j];                    \
            }                                                           \
          }                                                             \
        }                                                               \
} while (0)
    if (ndim_A > 0)
      RESET_IDX(A);
    if (ndim_B > 0)
      RESET_IDX(B);
#undef RESET_IDX
  }
  free(dlen_A);
  free(dlen_B);
  free(idx_glb);
  free(rev_idx_map);
  TAU_FSTOP(sym_seq_sum_inr);
  return 0;
}






#endif
