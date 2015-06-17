/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include <limits.h>
#include "sym_seq_ctr.h"
#include "../shared/offload.h"
#include "../shared/util.h"

namespace CTF_int{
  int sym_seq_ctr_ref(char const *     alpha,
                      char const *     A,
                      algstrct const * sr_A,
                      int              order_A,
                      int const *      edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int const *      edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int const *      edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C){
    TAU_FSTART(sym_seq_ctr_ref);
    int idx, i, idx_max, imin, imax, sz, iA, iB, iC, j, k;
    int off_idx, sym_pass;
    int * idx_glb, * rev_idx_map;
    int * dlen_A, * dlen_B, * dlen_C;
    int64_t idx_A, idx_B, idx_C, off_lda;

    inv_idx(order_A,  idx_map_A,
            order_B,  idx_map_B,
            order_C,  idx_map_C,
            &idx_max, &rev_idx_map);

    dlen_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    dlen_B = (int*)CTF_int::alloc(sizeof(int)*order_B);
    dlen_C = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int)*order_C);

    idx_glb = (int*)CTF_int::alloc(sizeof(int)*idx_max);
    memset(idx_glb, 0, sizeof(int)*idx_max);


    /* Scale C immediately. FIXME: wrong for iterators over subset of C */
    if (!sr_C->isequal(beta, sr_C->mulid())){
      sz = sy_packed_size(order_C, edge_len_C, sym_C);
      if (sr_C->isequal(beta, sr_C->addid())){
        sr_C->set(C, sr_C->addid(), sz);
      } else {
        for (i=0; i<sz; i++){
          sr_C->mul(C+i*sr_C->el_size, beta, 
                    C+i*sr_C->el_size);
        }
      }
    }
    idx_A = 0, idx_B = 0, idx_C = 0;
    sym_pass = 1;
    for (;;){
      //printf("[%d] <- [%d]*[%d]\n",idx_C, idx_A, idx_B);
      if (sym_pass){
        if (alpha == NULL && beta == NULL){
          sr_C->mul(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                   C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(1);
        } else  if (alpha == NULL){
          char tmp[sr_C->el_size];
          sr_C->mul(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                   tmp);
          sr_C->add(tmp, C+idx_C*sr_C->el_size, C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(2);
        } else {
          char tmp[sr_C->el_size];
          sr_C->mul(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                   tmp);
          sr_C->mul(tmp, alpha, tmp);
          sr_C->add(tmp, C+idx_C*sr_C->el_size, C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(3);
        }
      }
      //printf("[%lf] <- [%lf]*[%lf]\n",C[idx_C],A[idx_A],B[idx_B]);

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,3);
        GET_MIN_MAX(B,1,3);
        GET_MIN_MAX(C,2,3);

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
      CHECK_SYM(B);
      if (!sym_pass) continue;
      CHECK_SYM(C);
      if (!sym_pass) continue;
      
      if (order_A > 0)
        RESET_IDX(A);
      if (order_B > 0)
        RESET_IDX(B);
      if (order_C > 0)
        RESET_IDX(C);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(dlen_C);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_ctr_ref);
    return 0;
  }

  int sym_seq_ctr_cust(char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int const *      edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     B,
                       algstrct const * sr_B,
                       int              order_B,
                       int const *      edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       char const *     beta,
                       char *           C,
                       algstrct const * sr_C,
                       int              order_C,
                       int const *      edge_len_C,
                       int const *      sym_C,
                       int const *      idx_map_C,
                       bivar_function * func){
    TAU_FSTART(sym_seq_ctr_cust);
    int idx, i, idx_max, imin, imax, iA, iB, iC, j, k;
    int off_idx, sym_pass;
    int * idx_glb, * rev_idx_map;
    int * dlen_A, * dlen_B, * dlen_C;
    //int64_t sz;
    int64_t idx_A, idx_B, idx_C, off_lda;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            order_C,       idx_map_C,
            &idx_max,     &rev_idx_map);

    dlen_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    dlen_B = (int*)CTF_int::alloc(sizeof(int)*order_B);
    dlen_C = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int)*order_C);

    idx_glb = (int*)CTF_int::alloc(sizeof(int)*idx_max);
    memset(idx_glb, 0, sizeof(int)*idx_max);

    /* Scale C immediately. FIXME: wrong for iterators over subset of C */
    /*if (beta != get_one<dtype>()) {
      sz = sy_packed_size(order_C, edge_len_C, sym_C);
      for (i=0; i<sz; i++){
        C[i] = C[i]*beta;
      }
    }*/
    if (!sr_C->isequal(beta, sr_C->mulid())){
      int64_t sz = sy_packed_size(order_C, edge_len_C, sym_C);
      if (sr_C->isequal(beta, sr_C->addid())){
        sr_C->set(C, sr_C->addid(), sz);
      } else {
        for (i=0; i<sz; i++){
          sr_C->mul(C+i*sr_C->el_size, beta, 
                    C+i*sr_C->el_size);
        }
      }
    }

    idx_A = 0, idx_B = 0, idx_C = 0;
    sym_pass = 1;
    for (;;){
      //printf("[%d] <- [%d]*[%d]\n",idx_C, idx_A, idx_B);
      if (sym_pass){
        if (alpha == NULL && beta == NULL){
          func->apply_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                        C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(1);
        } else  if (alpha == NULL){
          char tmp[sr_C->el_size];
          func->apply_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                        tmp);
          sr_C->add(tmp, C+idx_C*sr_C->el_size, C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(2);
        } else {
          char tmp[sr_C->el_size];
          func->apply_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                        tmp);
          sr_C->mul(tmp, alpha, tmp);
          sr_C->add(tmp, C+idx_C*sr_C->el_size, C+idx_C*sr_C->el_size);
          CTF_FLOPS_ADD(3);
        }
      }

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,3);
        GET_MIN_MAX(B,1,3);
        GET_MIN_MAX(C,2,3);

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
      CHECK_SYM(B);
      if (!sym_pass) continue;
      CHECK_SYM(C);
      if (!sym_pass) continue;
      

      if (order_A > 0)
        RESET_IDX(A);
      if (order_B > 0)
        RESET_IDX(B);
      if (order_C > 0)
        RESET_IDX(C);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(dlen_C);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_ctr_cust);
    return 0;
  }

  int sym_seq_ctr_inr(char const *     alpha,
                      char const *     A,
                      algstrct const * sr_A,
                      int              order_A,
                      int const *      edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int const *      edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int const *      edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C,
                      iparam const *   prm){
    TAU_FSTART(sym_seq_ctr_inner);
    int idx, i, idx_max, imin, imax, iA, iB, iC, j, k;
    int off_idx, sym_pass, stride_A, stride_B, stride_C;
    int * idx_glb, * rev_idx_map;
    int * dlen_A, * dlen_B, * dlen_C;
    int64_t idx_A, idx_B, idx_C, off_lda;

    stride_A = prm->m*prm->k;
    stride_B = prm->k*prm->n;
    stride_C = prm->m*prm->n;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            order_C,       idx_map_C,
            &idx_max,     &rev_idx_map);

    dlen_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    dlen_B = (int*)CTF_int::alloc(sizeof(int)*order_B);
    dlen_C = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int)*order_C);

    idx_glb = (int*)CTF_int::alloc(sizeof(int)*idx_max);
    memset(idx_glb, 0, sizeof(int)*idx_max);


    /* Scale C immediately. FIXME: wrong for iterators over subset of C */
  #ifndef OFFLOAD
    if (!sr_C->isequal(beta, sr_C->mulid())){
      CTF_FLOPS_ADD(prm->sz_C);
  /*    for (i=0; i<prm->sz_C; i++){
        C[i] = C[i]*beta;
      }*/
      if (sr_C->isequal(beta, sr_C->addid())){
        sr_C->set(C, sr_C->addid(), prm->sz_C);
      } else {
        sr_C->scal(prm->sz_C, beta, C, 1);
      }
    }
  #endif
    idx_A = 0, idx_B = 0, idx_C = 0;
    sym_pass = 1;

   // int cntr=0;  
    for (;;){
      if (sym_pass){
  //      C[idx_C] += alpha*A[idx_A]*B[idx_B];
        TAU_FSTART(gemm);
  #ifdef OFFLOAD
  //      if (prm->m*prm->n*prm->k > 1000){
        offload_gemm<dtype>(prm->tA, prm->tB, prm->m, prm->n, prm->k, alpha, 
                            A+idx_A*stride_A*sr_A->el_size, prm->k,
                            B+idx_B*stride_B*sr_B->el_size, prm->k, sr_C->mulid(),
                            C+idx_C*stride_C*sr_C->el_size, prm->m);
  #else
        //printf("[%d] <- [%d]*[%d] (%d)\n",idx_C, idx_A, idx_B, cntr++);
        sr_C->gemm(prm->tA, prm->tB, prm->m, prm->n, prm->k, alpha, 
                  A+idx_A*stride_A*sr_A->el_size, 
                  B+idx_B*stride_B*sr_B->el_size, sr_C->mulid(),
                  C+idx_C*stride_C*sr_C->el_size);
  #endif
        /*printf("multiplying %lf by %lf and got %lf\n", 
((double*)(A+idx_A*stride_A*sr_A->el_size))[0],
((double*)(B+idx_B*stride_B*sr_B->el_size))[0],
((double*)(C+idx_C*stride_C*sr_C->el_size))[0]);*/
        TAU_FSTOP(gemm);
        // count n^2 FLOPS too
        CTF_FLOPS_ADD((2 * (int64_t)prm->n * (int64_t)prm->m * (int64_t)(prm->k+1)));
      }
      //printf("[%ld] <- [%ld]*[%ld] (%d <- %d, %d)\n",idx_C,idx_A,idx_B,stride_C,stride_A,stride_B);

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,3);
        GET_MIN_MAX(B,1,3);
        GET_MIN_MAX(C,2,3);

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
      CHECK_SYM(B);
      if (!sym_pass) continue;
      CHECK_SYM(C);
      if (!sym_pass) continue;
      

      if (order_A > 0)
        RESET_IDX(A);
      if (order_B > 0)
        RESET_IDX(B);
      if (order_C > 0)
        RESET_IDX(C);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(dlen_C);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_ctr_inner);
    return 0;
  }
}
