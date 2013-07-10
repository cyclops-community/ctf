/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SYM_SEQ_SUM_REF_HXX__
#define __SYM_SEQ_SUM_REF_HXX__

#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_shared.hxx"

/**
 * \brief performs symmetric contraction
 */
template<typename dtype>
int sym_seq_sum_ref( dtype const        alpha,
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
                     int const *        idx_map_B){
  TAU_FSTART(sym_seq_sum_ref);
  int idx, i, idx_max, imin, imax, idx_A, idx_B, iA, iB, j, k;
  int off_idx, off_lda, sym_pass;
  int * idx_glb, * rev_idx_map;
  int * dlen_A, * dlen_B;

  inv_idx(ndim_A,       idx_map_A,
          ndim_B,       idx_map_B,
          &idx_max,     &rev_idx_map);

  dlen_A = (int*)CTF_alloc(sizeof(int)*ndim_A);
  dlen_B = (int*)CTF_alloc(sizeof(int)*ndim_B);
  memcpy(dlen_A, edge_len_A, sizeof(int)*ndim_A);
  memcpy(dlen_B, edge_len_B, sizeof(int)*ndim_B);

  idx_glb = (int*)CTF_alloc(sizeof(int)*idx_max);
  memset(idx_glb, 0, sizeof(int)*idx_max);


  idx_A = 0, idx_B = 0;
  sym_pass = 1;
  for (;;){
    if (sym_pass){
  /*    printf("B[%d] = %lf*(A[%d]=%lf)+%lf*(B[%d]=%lf\n",
              idx_B,alpha,idx_A,A[idx_A],beta,idx_B,B[idx_B]);*/
      B[idx_B] = alpha*A[idx_A] + beta*B[idx_B];
    }

    for (idx=0; idx<idx_max; idx++){
      imin = 0, imax = INT_MAX;

      GET_MIN_MAX(A,0,2);
      GET_MIN_MAX(B,1,2);

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

    CHECK_SYM(A);
    if (!sym_pass) continue;
    CHECK_SYM(B);
    if (!sym_pass) continue;
    
    if (ndim_A > 0)
      RESET_IDX(A);
    if (ndim_B > 0)
      RESET_IDX(B);
  }
  CTF_free(dlen_A);
  CTF_free(dlen_B);
  CTF_free(idx_glb);
  CTF_free(rev_idx_map);
  TAU_FSTOP(sym_seq_sum_ref);
  return 0;
}






#endif
