/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_scl.h"
#include "scaling.h"
#include "../interface/idx_tensor.h"

namespace CTF_int {

  void endomorphism::operator()(Term const & A) const { 
    CTF::Idx_Tensor op_A = A.execute(A.get_uniq_inds());
    scaling s(op_A.parent, op_A.idx_map, op_A.scale, this);
    s.execute();
  }


  void inv_idx(int const   order_A,
               int const * idx_A,
               int *       order_tot,
               int **      idx_arr){
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


  int sym_seq_scl_ref(char const *     alpha,
                      char *           A,
                      algstrct const * sr_A,
                      int              order_A,
                      int64_t const *  edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A){
    TAU_FSTART(sym_seq_sum_ref);
    int idx, i, idx_max, imin, imax, iA, j, k;
    int off_idx, sym_pass;
    int64_t * idx_glb;
    int * rev_idx_map;
    int64_t * dlen_A;
    int64_t idx_A, off_lda;

    inv_idx(order_A,       idx_map_A,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);


    idx_A = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
        //A[idx_A] = alpha*A[idx_A];
        sr_A->mul(A+idx_A*sr_A->el_size, alpha, A+idx_A*sr_A->el_size);
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
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_ref);
    return 0;
  }


  int sym_seq_scl_cust(char const *         alpha,
                       char *               A,
                       algstrct const *     sr_A,
                       int const            order_A,
                       int64_t const *      edge_len_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       endomorphism const * func){
    TAU_FSTART(sym_seq_sum_cust)
    int imin, iA, j, k;
    int idx, idx_max, off_idx, sym_pass;
    int * rev_idx_map;
    int64_t * idx_glb, * dlen_A;
    int64_t idx_A, off_lda, i, imax;

    inv_idx(order_A,       idx_map_A,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);


    idx_A = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
        if (alpha != NULL)
          sr_A->mul(A+idx_A*sr_A->el_size, alpha, A+idx_A*sr_A->el_size);
        func->apply_f(A+idx_A*sr_A->el_size);
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
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_cust);
    return 0;
  }


}
