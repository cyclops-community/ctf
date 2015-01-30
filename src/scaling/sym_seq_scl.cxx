/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_scl.h"

namespace CTF_int {
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int *              order_tot,
               int **             idx_arr){
    int i, dim_max;

    dim_max = -1;
    for (i=0; i<order_A; i++){
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    dim_max++;
    *order_tot = dim_max;
    *idx_arr = (int*)CTF_int::alloc(sizeof(int)*dim_max);
    std::fill((*idx_arr), (*idx_arr)+dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[idx_A[i]] = i;
    }
  }


  int sym_seq_scl_ref(char const * alpha,
                      char *       A,
                      semiring const &  sr_A,
                      int          order_A,
                      int const *  edge_len_A,
                      int const *  _lda_A,
                      int const *  sym_A,
                      int const *  idx_map_A){
    TAU_FSTART(sym_seq_sum_ref);
    int idx, i, idx_max, imin, imax, idx_A, iA, j, k;
    int off_idx, off_lda, sym_pass;
    int * idx_glb, * rev_idx_map;
    int * dlen_A;

    inv_idx(order_A,       idx_map_A,
            &idx_max,     &rev_idx_map);

    dlen_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(dlen_A, edge_len_A, sizeof(int)*order_A);

    idx_glb = (int*)CTF_int::alloc(sizeof(int)*idx_max);
    memset(idx_glb, 0, sizeof(int)*idx_max);


    idx_A = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
        //A[idx_A] = alpha*A[idx_A];
        sr_A.mul(A+idx_A*sr_A.el_size, alpha, A+idx_A*sr_A.el_size);
        CTF_FLOPS_ADD(1);
      }

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,1);

        ASSERT(idx_glb[idx] >= imin && idx_glb[idx] < imax);

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
      
      if (order_A > 0)
        RESET_IDX(A);
    }
    CTF_int::cfree(dlen_A);
    CTF_int::cfree(idx_glb);
    CTF_int::cfree(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_ref);
    return 0;
  }


  int sym_seq_scl_cust(char *               A,
                       semiring const &     sr_A,
                       int const            order_A,
                       int const *          edge_len_A,
                       int const *          _lda_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       endomorphism        func){
    TAU_FSTART(sym_seq_sum_cust)
    int idx, i, idx_max, imin, imax, idx_A, iA, j, k;
    int off_idx, off_lda, sym_pass;
    int * idx_glb, * rev_idx_map;
    int * dlen_A;

    inv_idx(order_A,       idx_map_A,
            &idx_max,     &rev_idx_map);

    dlen_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(dlen_A, edge_len_A, sizeof(int)*order_A);

    idx_glb = (int*)CTF_int::alloc(sizeof(int)*idx_max);
    memset(idx_glb, 0, sizeof(int)*idx_max);


    idx_A = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
        func.apply_f(A+idx_A*sr_A.el_size);
        CTF_FLOPS_ADD(1);
      }

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,1);

        ASSERT(idx_glb[idx] >= imin && idx_glb[idx] < imax);

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
      
      if (order_A > 0)
        RESET_IDX(A);
    }
    CTF_int::cfree(dlen_A);
    CTF_int::cfree(idx_glb);
    TAU_FSTOP(sym_seq_sum_cust);
    return 0;
  }


}
