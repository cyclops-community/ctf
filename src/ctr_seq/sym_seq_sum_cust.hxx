/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SYM_SEQ_SUM_CUST_HXX__
#define __SYM_SEQ_SUM_CUST_HXX__

#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_shared.hxx"
#include "../dist_tensor/cyclopstf.hpp"

/**
 * \brief performs symmetric summation
 */
template<typename dtype>
int sym_seq_sum_cust(dtype const          alpha,
                     dtype const *        A,
                     int const            ndim_A,
                     int const *          edge_len_A,
                     int const *          _lda_A,
                     int const *          sym_A,
                     int const *          idx_map_A,
                     dtype const          beta,
                     dtype *              B,
                     int const            ndim_B,
                     int const *          edge_len_B,
                     int const *          _lda_B,
                     int const *          sym_B,
                     int const *          idx_map_B,
                     fseq_elm_sum<dtype>* prm){
  TAU_FSTART(sym_seq_sum_cust);
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
      B[idx_B] = beta*B[idx_B] 
                  + (*(prm->func_ptr))(alpha, A[idx_A], B[idx_B]);
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
  TAU_FSTOP(sym_seq_sum_cust);
  return 0;
}






#endif
