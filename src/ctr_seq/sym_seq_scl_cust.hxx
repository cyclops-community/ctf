/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SYM_SEQ_SCL_CUST_HXX__
#define __SYM_SEQ_SCL_CUST_HXX__

#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_shared.hxx"
#include "../dist_tensor/cyclopstf.hpp"


/**
 * \brief performs symmetric contraction
 */
template<typename dtype>
int sym_seq_scl_cust(dtype const          alpha,
                     dtype *              A,
                     int const            ndim_A,
                     int const *          edge_len_A,
                     int const *          _lda_A,
                     int const *          sym_A,
                     int const *          idx_map_A,
                     fseq_elm_scl<dtype>* prm){
  TAU_FSTART(sym_seq_sum_cust)
  int idx, i, idx_max, imin, imax, idx_A, iA, j, k;
  int off_idx, off_lda, sym_pass;
  int * idx_glb, * rev_idx_map;
  int * dlen_A;

  inv_idx(ndim_A,       idx_map_A,
          &idx_max,     &rev_idx_map);

  dlen_A = (int*)CTF_alloc(sizeof(int)*ndim_A);
  memcpy(dlen_A, edge_len_A, sizeof(int)*ndim_A);

  idx_glb = (int*)CTF_alloc(sizeof(int)*idx_max);
  memset(idx_glb, 0, sizeof(int)*idx_max);


  idx_A = 0;
  sym_pass = 1;
  for (;;){
    if (sym_pass){
      (*(prm->func_ptr))(alpha, A[idx_A]);
    }

    for (idx=0; idx<idx_max; idx++){
      imin = 0, imax = INT_MAX;

      GET_MIN_MAX(A,0,1);

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
    
    if (ndim_A > 0)
      RESET_IDX(A);
  }
  CTF_free(dlen_A);
  CTF_free(idx_glb);
  TAU_FSTOP(sym_seq_sum_cust);
  return 0;
}






#endif
