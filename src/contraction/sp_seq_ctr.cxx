/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include <limits.h>
#include "sp_seq_ctr.h"
#include "sym_seq_ctr.h"
#include "../shared/offload.h"
#include "../shared/util.h"

namespace CTF_int{
  template<int idim>
  void spA_dnB_dnC_ctrloop(char const *           alpha,
                           ConstPairIterator &    A,
                           int64_t &              size_A,
                           algstrct const *       sr_A,
                           int                    order_A,
                           int64_t const *        edge_len_A,
                           int64_t const *        lda_A,
                           int const *            sym_A,
                           int const *            idx_map_A,
                           char const *           B,
                           algstrct const *       sr_B,
                           int                    order_B,
                           int64_t const *        edge_len_B,
                           int const *            sym_B,
                           int const *            idx_map_B,
                           uint64_t *const*       offsets_B,
                           char const *           beta,
                           char *                 C,
                           algstrct const *       sr_C,
                           int                    order_C,
                           int64_t const *        edge_len_C,
                           int const *            sym_C,
                           int const *            idx_map_C,
                           uint64_t *const*       offsets_C,
                           bivar_function const * func,
                           int64_t const *        idx,
                           int const *            rev_idx_map,
                           int                    idx_max){
    int64_t imax=0;
    int rA = rev_idx_map[3*idim+0];
    int rB = rev_idx_map[3*idim+1];
    int rC = rev_idx_map[3*idim+2];

    ASSERT(!(rA != -1 && rB == -1 && rC == -1));
  
    if (rB != -1)
      imax = edge_len_B[rB];
    else if (rC != -1)
      imax = edge_len_C[rC];

    if (rA != -1 && sym_A[rA] != NS){
      int rrA = rA;
      do {
        if (idx_map_A[rrA+1] > idim)
          imax = idx[idx_map_A[rrA+1]]+1;
        rrA++;
      } while (sym_A[rrA] != NS && idx_map_A[rrA] < idim);
    }

    if (rB != -1 && sym_B[rB] != NS){
      int rrB = rB;
      do {
        if (idx_map_B[rrB+1] > idim)
          imax = std::min(imax,idx[idx_map_B[rrB+1]]+1);
        rrB++;
      } while (sym_B[rrB] != NS && idx_map_B[rrB] < idim);
    }

    if (rC != -1 && sym_C[rC] != NS){
      int rrC = rC;
      do {
        if (idx_map_C[rrC+1] > idim)
          imax = std::min(imax,idx[idx_map_C[rrC+1]]+1);
        rrC++;
      } while (sym_C[rrC] != NS && idx_map_C[rrC] < idim);
    }

    int64_t imin = 0;

    if (rA > 0 && sym_A[rA-1] != NS){
      int rrA = rA;
      do {
        if (idx_map_A[rrA-1] > idim)
          imin = idx[idx_map_A[rrA-1]];
        rrA--;
      } while (rrA>0 && sym_A[rrA-1] != NS && idx_map_A[rrA] < idim);
    }

    if (rB > 0 && sym_B[rB-1] != NS){
      int rrB = rB;
      do {
        if (idx_map_B[rrB-1] > idim)
          imin = std::max(imin,idx[idx_map_B[rrB-1]]);
        rrB--;
      } while (rrB>0 && sym_B[rrB-1] != NS && idx_map_B[rrB] < idim);
    }

    if (rC > 0 && sym_C[rC-1] != NS){
      int rrC = rC;
      do {
        if (idx_map_C[rrC-1] > idim)
          imin = std::max(imin,idx[idx_map_C[rrC-1]]);
        rrC--;
      } while (rrC>0 && sym_C[rrC-1] != NS && idx_map_C[rrC] < idim);
    }
    int64_t key_offset = 0;
    for (int i=0; i<order_A; i++){
      if (i != rA){
        key_offset += idx[idx_map_A[i]]*lda_A[i];
      }
    }

    //FIXME: if rC != -1 && rA == -1, thread
    for (int64_t i=imin; i<imax; i++){
/*      if (size_A != 0){
        printf("lda_A[rA]=%ld\n",lda_A[rA]);
        printf("A[0].k() == %ld\n",A[0].k());
      }*/
      if (size_A == 0 ||
          (rA == -1 && A[0].k()!=key_offset) ||
          (rA != -1 && (A[0].k()/lda_A[rA]/edge_len_A[rA])!=key_offset/lda_A[rA]/edge_len_A[rA])){ 
        //printf("i = %d idim = %d breaking rA = %d k=%ld\n", i, idim, rA, A[0].k());
        break;
      }
      // if we should be iterating over A and the next sparse value of A does not match this iteration number
      // there will be no actual work to do in the inner iterations, so continue
      if (rA != -1 && (A[0].k()/lda_A[rA])%edge_len_A[rA] != i){
        ASSERT((A[0].k()/lda_A[rA])%edge_len_A[rA] > i);
        continue;
      }
      ConstPairIterator cpiA(sr_A, A.ptr);
      int64_t new_size_A = size_A;
      int64_t nidx[idx_max];
      memcpy(nidx, idx, idx_max*sizeof(int64_t));
      nidx[idim] = i;
      spA_dnB_dnC_ctrloop<idim-1>(alpha, cpiA, new_size_A, sr_A, order_A, edge_len_A, lda_A, sym_A, idx_map_A, B+offsets_B[idim][nidx[idim]], sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C+offsets_C[idim][nidx[idim]], sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, nidx, rev_idx_map, idx_max);
      if (rA != -1) {
        if (size_A == new_size_A){
          ASSERT(rA==0);
          //if we did not advance, in recursive loops, it means all rA=-1 for lower idim, and we now want to advance by 1
          size_A = new_size_A-1;
          A.ptr = cpiA[1].ptr;
        } else {
          size_A = new_size_A;
          A.ptr = cpiA.ptr;
        }
      }
    }
  }

  template<>
  void spA_dnB_dnC_ctrloop<0>(char const *           alpha,
                              ConstPairIterator &    A,
                              int64_t &              size_A,
                              algstrct const *       sr_A,
                              int                    order_A,
                              int64_t const *        edge_len_A,
                              int64_t const *        lda_A,
                              int const *            sym_A,
                              int const *            idx_map_A,
                              char const *           B,
                              algstrct const *       sr_B,
                              int                    order_B,
                              int64_t const *        edge_len_B,
                              int const *            sym_B,
                              int const *            idx_map_B,
                              uint64_t *const*       offsets_B,
                              char const *           beta,
                              char *                 C,
                              algstrct const *       sr_C,
                              int                    order_C,
                              int64_t const *        edge_len_C,
                              int const *            sym_C,
                              int const *            idx_map_C,
                              uint64_t *const*       offsets_C,
                              bivar_function const * func,
                              int64_t const *        idx,
                              int const *            rev_idx_map,
                              int                    idx_max){


    int64_t imax=0;
    int rA = rev_idx_map[0];
    int rB = rev_idx_map[1];
    int rC = rev_idx_map[2];
    
    ASSERT(!(rA != -1 && rB == -1 && rC == -1));
  
    if (rB != -1)
      imax = edge_len_B[rB];
    else if (rC != -1)
      imax = edge_len_C[rC];

    if (rA != -1 && sym_A[rA] != NS)
      imax = idx[idx_map_A[rA+1]]+1;
    if (rB != -1 && sym_B[rB] != NS)
      imax = std::min(imax,idx[idx_map_B[rB+1]]+1);
    if (rC != -1 && sym_C[rC] != NS)
      imax = std::min(imax,idx[idx_map_C[rC+1]]+1);

    int64_t imin = 0;

    if (rA > 0 && sym_A[rA-1] != NS)
      imin = idx[idx_map_A[rA-1]];
    if (rB > 0 && sym_B[rB-1] != NS)
      imin = std::max(imin,idx[idx_map_B[rB-1]]);
    if (rC > 0 && sym_C[rC-1] != NS)
      imin = std::max(imin,idx[idx_map_C[rC-1]]);

/*    int tid, ntd;
    tid = omp_get_thread_num();
    ntd = omp_get_num_threads();
    printf("-> %d/%d %d %d %d\n",tid,ntd,func==NULL, alpha==NULL,beta==NULL);*/
    if (rA == -1){
      if (func == NULL){
/*        if (alpha == NULL && beta == NULL){
          for (int i=imin; i<imax; i++){
            sr_C->mul(A[0].d(),
                      B+offsets_B[0][i], 
                      C+offsets_C[0][i]);
          }
          CTF_FLOPS_ADD(imax-imin);
        } else if (alpha == NULL)*/
        if (alpha == NULL || sr_C->isequal(alpha,sr_C->mulid())){
          for (int64_t i=imin; i<imax; i++){
            char tmp[sr_C->el_size];
            sr_C->mul(A[0].d(), 
                      B+offsets_B[0][i], 
                      tmp);
            sr_C->add(tmp, 
                      C+offsets_C[0][i], 
                      C+offsets_C[0][i]);
          }
          CTF_FLOPS_ADD(2*(imax-imin));
        } else {
          for (int64_t i=imin; i<imax; i++){
            char tmp[sr_C->el_size];
            sr_C->mul(A[0].d(), 
                      B+offsets_B[0][i], 
                      tmp);
            sr_C->mul(tmp,
                      alpha,
                      tmp);
            sr_C->add(tmp, 
                      C+offsets_C[0][i], 
                      C+offsets_C[0][i]);
          }
          CTF_FLOPS_ADD(3*(imax-imin));
        }
      } else {
/*        if (alpha == NULL && beta == NULL){
          for (int i=imin; i<imax; i++){
            func->apply_f(A[0].d(),
                          B+offsets_B[0][i], 
                          C+offsets_C[0][i]);
          }
          CTF_FLOPS_ADD(imax-imin);
        } else if (alpha == NULL)*/
        if (alpha == NULL || sr_C->isequal(alpha,sr_C->mulid())){
          for (int64_t i=imin; i<imax; i++){
            func->acc_f(A[0].d(), 
                        B+offsets_B[0][i], 
                        C+offsets_C[0][i],
                        sr_C);
/*
            char tmp[sr_C->el_size];
            func->apply_f(A[0].d(), 
                          B+offsets_B[0][i], 
                          tmp);
            sr_C->add(tmp, 
                      C+offsets_C[0][i], 
                      C+offsets_C[0][i]);*/
          }
          CTF_FLOPS_ADD(2*(imax-imin));
        } else {
          ASSERT(0);
          assert(0);
          for (int64_t i=imin; i<imax; i++){
            char tmp[sr_C->el_size];
            func->apply_f(A[0].d(), 
                          B+offsets_B[0][i], 
                          tmp);
            sr_C->mul(tmp,
                      alpha,
                      tmp);
            sr_C->add(tmp, 
                      C+offsets_C[0][i], 
                      C+offsets_C[0][i]);
          }
          CTF_FLOPS_ADD(3*(imax-imin));
        }
      }
    } else {
      int64_t key_offset = 0;
      for (int i=0; i<order_A; i++){
        if (i != rA){
          key_offset += idx[idx_map_A[i]]*lda_A[i];
        }
      }
      ASSERT(func == NULL && alpha != NULL && beta != NULL);
      assert(func == NULL && alpha != NULL && beta != NULL);
      do {
        int64_t sk = A[0].k()-key_offset;
        ASSERT(sk >= 0);
        int64_t i = sk/lda_A[rA];
        if (i>=imax) break;
        if (i>=imin){
          char tmp[sr_C->el_size];
          sr_C->mul(A[0].d(), 
                    B+offsets_B[0][i], 
                    tmp);
          sr_C->mul(tmp,
                    alpha,
                    tmp);
          sr_C->add(tmp, 
                    C+offsets_C[0][i], 
                    C+offsets_C[0][i]);
        }
        A = A[1];
        size_A--;
      } while (size_A > 0);
    }
  }

  template<>
  void spA_dnB_dnC_ctrloop< MAX_ORD >
                          (char const *           alpha,
                           ConstPairIterator &    A,
                           int64_t &              size_A,
                           algstrct const *       sr_A,
                           int                    order_A,
                           int64_t const *        edge_len_A,
                           int64_t const *        lda_A,
                           int const *            sym_A,
                           int const *            idx_map_A,
                           char const *           B,
                           algstrct const *       sr_B,
                           int                    order_B,
                           int64_t const *        edge_len_B,
                           int const *            sym_B,
                           int const *            idx_map_B,
                           uint64_t *const*       offsets_B,
                           char const *           beta,
                           char *                 C,
                           algstrct const *       sr_C,
                           int                    order_C,
                           int64_t const *        edge_len_C,
                           int const *            sym_C,
                           int const *            idx_map_C,
                           uint64_t *const*       offsets_C,
                           bivar_function const * func,
                           int64_t const *        idx,
                           int const *            rev_idx_map,
                           int                    idx_max);


  void spA_dnB_dnC_seq_ctr(char const *            alpha,
                           char  const *           A,
                           int64_t                 size_A,
                           algstrct const *        sr_A,
                           int                     order_A,
                           int64_t const *         edge_len_A,
                           int const *             sym_A,
                           int const *             idx_map_A,
                           char const *            B,
                           algstrct const *        sr_B,
                           int                     order_B,
                           int64_t const *         edge_len_B,
                           int const *             sym_B,
                           int const *             idx_map_B,
                           char const *            beta,
                           char *                  C,
                           algstrct const *        sr_C,
                           int                     order_C,
                           int64_t const *         edge_len_C,
                           int const *             sym_C,
                           int const *             idx_map_C,
                           bivar_function const *  func){
    TAU_FSTART(spA_dnB_dnC_seq_ctr);
    int idx_max;
    int64_t sz;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B, * dlen_C;

    inv_idx(order_A,  idx_map_A,
            order_B,  idx_map_B,
            order_C,  idx_map_C,
            &idx_max, &rev_idx_map);

    if (idx_max == 0){
      if (alpha == NULL && beta == NULL){
        sr_C->mul(A, B, C);
        CTF_FLOPS_ADD(1);
      } else  if (alpha == NULL){
        char tmp[sr_C->el_size];
        sr_C->mul(A, B, tmp);
        sr_C->mul(C, beta, C);
        sr_C->add(tmp, C, C);
        CTF_FLOPS_ADD(2);
      } else {
        char tmp[sr_C->el_size];
        sr_C->mul(A, B, tmp);
        sr_C->mul(tmp, alpha, tmp);
        sr_C->mul(C, beta, C);
        sr_C->add(tmp, C, C);
        CTF_FLOPS_ADD(3);
      }
      TAU_FSTOP(spA_dnB_dnC_seq_ctr);
      return;
    }
    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    dlen_C = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int64_t)*order_C);



    /* Scale C immediately. FIXME: wrong for iterators over subset of C */
    if (!sr_C->isequal(beta, sr_C->mulid())){
      sz = sy_packed_size(order_C, edge_len_C, sym_C);
      if (sr_C->isequal(beta, sr_C->addid()) || sr_C->isequal(beta, NULL)){
        sr_C->set(C, sr_C->addid(), sz);
      } else {
        sr_C->scal(sz, beta, C, 1);
        /*for (i=0; i<sz; i++){
          sr_C->mul(C+i*sr_C->el_size, beta, 
                    C+i*sr_C->el_size);
        }*/
      }
    }
    ASSERT(idx_max <= MAX_ORD);
    uint64_t ** offsets_A;
    uint64_t ** offsets_B;
    uint64_t ** offsets_C;
    compute_syoffs(sr_A, order_A, edge_len_A, sym_A, idx_map_A, sr_B, order_B, edge_len_B, sym_B, idx_map_B, sr_C, order_C, edge_len_C, sym_C, idx_map_C, idx_max, rev_idx_map, offsets_A, offsets_B, offsets_C);

    //if we have something to parallelize without needing to replicate C
    //FIXME spawn threads when
//    if (order_C > 1 || (order_C > 0 && idx_map_C[0] != 0)){
    {
      int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
      memset(idx_glb, 0, sizeof(int64_t)*idx_max);


      int64_t lda_A[order_A];
      for (int i=0; i<order_A; i++){
        if (i==0) lda_A[i] = 1;
        else      lda_A[i] = lda_A[i-1]*edge_len_A[i-1];
      }

      ASSERT(idx_max<=MAX_ORD);

      ConstPairIterator pA(sr_A, A);


      SWITCH_ORD_CALL(spA_dnB_dnC_ctrloop, idx_max-1, alpha, pA, size_A, sr_A, order_A, edge_len_A, lda_A, sym_A, idx_map_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C, sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, idx_glb, rev_idx_map, idx_max);
      cdealloc(idx_glb);
    }
    for (int l=0; l<idx_max; l++){
      cdealloc(offsets_A[l]);
      cdealloc(offsets_B[l]);
      cdealloc(offsets_C[l]);
    }
    cdealloc(offsets_A);
    cdealloc(offsets_B);
    cdealloc(offsets_C);

    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(dlen_C);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(spA_dnB_dnC_seq_ctr);
  }
}
