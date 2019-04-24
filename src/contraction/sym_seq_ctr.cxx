/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include <limits.h>
#include "sym_seq_ctr.h"
#include "../shared/offload.h"
#include "../shared/util.h"

namespace CTF_int{
  
  template <int idim>
  void sym_seq_ctr_loop(char const *     alpha,
                        char const *     A,
                        algstrct const * sr_A,
                        int              order_A,
                        int64_t const *  edge_len_A,
                        int const *      sym_A,
                        int const *      idx_map_A,
                        uint64_t *const* offsets_A,
                        char const *     B,
                        algstrct const * sr_B,
                        int              order_B,
                        int64_t const *  edge_len_B,
                        int const *      sym_B,
                        int const *      idx_map_B,
                        uint64_t *const* offsets_B,
                        char const *     beta,
                        char *           C,
                        algstrct const * sr_C,
                        int              order_C,
                        int64_t const *  edge_len_C,
                        int const *      sym_C,
                        int const *      idx_map_C,
                        uint64_t *const* offsets_C,
                        bivar_function const * func,
                        int64_t const *  idx,
                        int const *      rev_idx_map,
                        int              idx_max){
    int64_t imax=0;
    int rA = rev_idx_map[3*idim+0];
    int rB = rev_idx_map[3*idim+1];
    int rC = rev_idx_map[3*idim+2];
  
    if (rA != -1)
      imax = edge_len_A[rA];
    else if (rB != -1)
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

    if (rC != -1){
#ifdef USE_OMP    
      #pragma omp for
#endif
      for (int64_t i=imin; i<imax; i++){
#ifdef USE_OMP    
        #pragma omp parallel 
#endif
        {
          int64_t nidx[idx_max];
          memcpy(nidx, idx, idx_max*sizeof(int64_t));
          nidx[idim] = i;
          sym_seq_ctr_loop<idim-1>(alpha, A+offsets_A[idim][nidx[idim]], sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B+offsets_B[idim][nidx[idim]], sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C+offsets_C[idim][nidx[idim]], sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, nidx, rev_idx_map, idx_max);
        }
      }
    } else {
      for (int64_t i=imin; i<imax; i++){
        int64_t nidx[idx_max];
        memcpy(nidx, idx, idx_max*sizeof(int64_t));
        nidx[idim] = i;
        sym_seq_ctr_loop<idim-1>(alpha, A+offsets_A[idim][nidx[idim]], sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B+offsets_B[idim][nidx[idim]], sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C+offsets_C[idim][nidx[idim]], sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, nidx, rev_idx_map, idx_max);
      }

    }
//    idx[idim] = 0;
  }


  template <>
  void sym_seq_ctr_loop<0>
                       (char const *     alpha,
                        char const *     A,
                        algstrct const * sr_A,
                        int              order_A,
                        int64_t const *  edge_len_A,
                        int const *      sym_A,
                        int const *      idx_map_A,
                        uint64_t *const* offsets_A,
                        char const *     B,
                        algstrct const * sr_B,
                        int              order_B,
                        int64_t const *  edge_len_B,
                        int const *      sym_B,
                        int const *      idx_map_B,
                        uint64_t *const* offsets_B,
                        char const *     beta,
                        char *           C,
                        algstrct const * sr_C,
                        int              order_C,
                        int64_t const *  edge_len_C,
                        int const *      sym_C,
                        int const *      idx_map_C,
                        uint64_t *const* offsets_C,
                        bivar_function const * func,
                        int64_t const *  idx,
                        int const *      rev_idx_map,
                        int              idx_max){
    int64_t imax=0;
    int rA = rev_idx_map[0];
    int rB = rev_idx_map[1];
    int rC = rev_idx_map[2];
  
    if (rA != -1)
      imax = edge_len_A[rA];
    else if (rB != -1)
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

    if (func == NULL){
      /*if (alpha == NULL && beta == NULL){
        for (int i=imin; i<imax; i++){
          sr_C->mul(A+offsets_A[0][i],
                    B+offsets_B[0][i], 
                    C+offsets_C[0][i]);
        }
        CTF_FLOPS_ADD(imax-imin);
      } else*/ 
      if (alpha == NULL || sr_C->isequal(alpha,sr_C->mulid())){
        for (int64_t i=imin; i<imax; i++){
          char tmp[sr_C->el_size];
          sr_C->mul(A+offsets_A[0][i], 
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
          sr_C->mul(A+offsets_A[0][i], 
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
      /*if (alpha == NULL && beta == NULL){
        for (int i=imin; i<imax; i++){
          func->apply_f(A+offsets_A[0][i],
                        B+offsets_B[0][i], 
                        C+offsets_C[0][i]);
        }
        CTF_FLOPS_ADD(imax-imin);
      } else*/ 
      if (alpha == NULL || sr_C->isequal(alpha,sr_C->mulid())){
        for (int64_t i=imin; i<imax; i++){
          func->acc_f(A+offsets_A[0][i], 
                      B+offsets_B[0][i], 
                      C+offsets_C[0][i],
                      sr_C);
        }
        CTF_FLOPS_ADD(2*(imax-imin));
      } else {
        //ASSERT(0);
        //assert(0);
        //printf("HERTE alpha = %d\n",*(int*)alpha);
        for (int64_t i=imin; i<imax; i++){
          char tmp[sr_C->el_size];
          func->apply_f(A+offsets_A[0][i], 
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
  }

  template 
  void sym_seq_ctr_loop< MAX_ORD >
                       (char const *     alpha,
                        char const *     A,
                        algstrct const * sr_A,
                        int              order_A,
                        int64_t const *  edge_len_A,
                        int const *      sym_A,
                        int const *      idx_map_A,
                        uint64_t *const* offsets_A,
                        char const *     B,
                        algstrct const * sr_B,
                        int              order_B,
                        int64_t const *  edge_len_B,
                        int const *      sym_B,
                        int const *      idx_map_B,
                        uint64_t *const* offsets_B,
                        char const *     beta,
                        char *           C,
                        algstrct const * sr_C,
                        int              order_C,
                        int64_t const *  edge_len_C,
                        int const *      sym_C,
                        int const *      idx_map_C,
                        uint64_t *const* offsets_C,
                        bivar_function const * func,
                        int64_t const *  idx,
                        int const *      rev_idx_map,
                        int              idx_max);


  void compute_syoff(int              r,
                     int64_t          len,
                     algstrct const * sr,
                     int64_t const *  edge_len,
                     int const *      sym,
                     uint64_t *       offsets){
    if (r == -1){
      std::fill(offsets, offsets+len, 0);
    } else if (r == 0){
      for (int64_t i=0; i<len; i++){
        offsets[i] = i*sr->el_size;
      }
    } else if (sym[r-1] == NS){
      int64_t sz = sy_packed_size(r, edge_len, sym)*sr->el_size;
      for (int64_t i=0; i<len; i++){
        offsets[i] = i*sz;
      }
    } else {
      int64_t medge_len[r+1];
      memcpy(medge_len, edge_len, r*sizeof(int64_t));
      int rr = r-1;
      while (rr>0 && sym[rr-1] != NS) rr--;
      for (int64_t i=0; i<len; i++){
        std::fill(medge_len+rr,medge_len+r+1, i);
        int64_t sz = sy_packed_size(r+1, medge_len, sym)*sr->el_size;
        offsets[i] = sz;
      }
    }
  }


  void compute_syoffs(algstrct const * sr_A,
                      int              order_A,
                      int64_t const *  edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      algstrct const * sr_B,
                      int              order_B,
                      int64_t const *  edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      algstrct const * sr_C,
                      int              order_C,
                      int64_t const *  edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C,
                      int              tot_order,
                      int const *      rev_idx_map,
                      uint64_t **&     offsets_A,
                      uint64_t **&     offsets_B,
                      uint64_t **&     offsets_C){
    TAU_FSTART(compute_syoffs);
    offsets_A = (uint64_t**)CTF_int::alloc(sizeof(uint64_t*)*tot_order);
    offsets_B = (uint64_t**)CTF_int::alloc(sizeof(uint64_t*)*tot_order);
    offsets_C = (uint64_t**)CTF_int::alloc(sizeof(uint64_t*)*tot_order);

    for (int idim=0; idim<tot_order; idim++){
      int64_t len=0;

      int rA = rev_idx_map[3*idim+0];
      int rB = rev_idx_map[3*idim+1];
      int rC = rev_idx_map[3*idim+2];
  
      if (rA != -1)
        len = edge_len_A[rA];
      else if (rB != -1)
        len = edge_len_B[rB];
      else if (rC != -1)
        len = edge_len_C[rC];

      offsets_A[idim] = (uint64_t*)CTF_int::alloc(sizeof(uint64_t)*len);
      offsets_B[idim] = (uint64_t*)CTF_int::alloc(sizeof(uint64_t)*len);
      offsets_C[idim] = (uint64_t*)CTF_int::alloc(sizeof(uint64_t)*len);
      compute_syoff(rA, len, sr_A, edge_len_A, sym_A, offsets_A[idim]);
      compute_syoff(rB, len, sr_B, edge_len_B, sym_B, offsets_B[idim]);
      compute_syoff(rC, len, sr_C, edge_len_C, sym_C, offsets_C[idim]);
    }
    TAU_FSTOP(compute_syoffs);
  }

  int sym_seq_ctr_ref(char const *     alpha,
                      char const *     A,
                      algstrct const * sr_A,
                      int              order_A,
                      int64_t const *  edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int64_t const *  edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int64_t const *  edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C){
    TAU_FSTART(sym_seq_ctr_ref);
    int idx, i, idx_max, imin, imax, iA, iB, iC, j, k;
    int64_t sz;
    int off_idx, sym_pass;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B, * dlen_C;
    int64_t idx_A, idx_B, idx_C, off_lda;

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
      TAU_FSTOP(sym_seq_ctr_ref);
      return 0;
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
    if (idx_max <= MAX_ORD){
      uint64_t ** offsets_A;
      uint64_t ** offsets_B;
      uint64_t ** offsets_C;
      compute_syoffs(sr_A, order_A, edge_len_A, sym_A, idx_map_A, sr_B, order_B, edge_len_B, sym_B, idx_map_B, sr_C, order_C, edge_len_C, sym_C, idx_map_C, idx_max, rev_idx_map, offsets_A, offsets_B, offsets_C);

      //if we have something to parallelize without needing to replicate C
      if (order_C > 1 || (order_C > 0 && idx_map_C[0] != 0)){
#ifdef USE_OMP    
        #pragma omp parallel
#endif
        {
          int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
          memset(idx_glb, 0, sizeof(int64_t)*idx_max);

          SWITCH_ORD_CALL(sym_seq_ctr_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C, sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, NULL, idx_glb, rev_idx_map, idx_max);
          cdealloc(idx_glb);
        }
      } else {
        {
          int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
          memset(idx_glb, 0, sizeof(int64_t)*idx_max);

          SWITCH_ORD_CALL(sym_seq_ctr_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C, sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, NULL, idx_glb, rev_idx_map, idx_max);
          cdealloc(idx_glb);
        }
      }
      for (int l=0; l<idx_max; l++){
        cdealloc(offsets_A[l]);
        cdealloc(offsets_B[l]);
        cdealloc(offsets_C[l]);
      }
      cdealloc(offsets_A);
      cdealloc(offsets_B);
      cdealloc(offsets_C);
    } else {
      int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
      memset(idx_glb, 0, sizeof(int64_t)*idx_max);

      idx_A = 0, idx_B = 0, idx_C = 0;
      sym_pass = 1;
      for (;;){
        //printf("[%d] <- [%d]*[%d]\n",idx_C, idx_A, idx_B);
        if (sym_pass){
          /*if (alpha == NULL && beta == NULL){
            sr_C->mul(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                     C+idx_C*sr_C->el_size);
            CTF_FLOPS_ADD(1);
          } else*/  if (alpha == NULL){
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
      CTF_int::cdealloc(idx_glb);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(dlen_C);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_ctr_ref);
    return 0;
  }

  int sym_seq_ctr_cust(char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int64_t const *  edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     B,
                       algstrct const * sr_B,
                       int              order_B,
                       int64_t const *  edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       char const *     beta,
                       char *           C,
                       algstrct const * sr_C,
                       int              order_C,
                       int64_t const *  edge_len_C,
                       int const *      sym_C,
                       int const *      idx_map_C,
                       bivar_function const * func){
    TAU_FSTART(sym_seq_ctr_cust);
    int idx, i, idx_max, imin, imax, iA, iB, iC, j, k;
    int off_idx, sym_pass;
    int64_t * idx_glb;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B, * dlen_C;
    //int64_t sz;
    int64_t idx_A, idx_B, idx_C, off_lda;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            order_C,       idx_map_C,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    dlen_C = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int64_t)*order_C);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);

    /* Scale C immediately. FIXME: wrong for iterators over subset of C */
    /*if (beta != get_one<dtype>()) {
      sz = sy_packed_size(order_C, edge_len_C, sym_C);
      for (i=0; i<sz; i++){
        C[i] = C[i]*beta;
      }
    }*/
/*    if (beta != NULL && !sr_C->isequal(beta, sr_C->mulid())){
      int64_t sz = sy_packed_size(order_C, edge_len_C, sym_C);
      if (sr_C->isequal(beta, sr_C->addid())){
        sr_C->set(C, sr_C->addid(), sz);
      } else {
        for (i=0; i<sz; i++){
          sr_C->mul(C+i*sr_C->el_size, beta, 
                    C+i*sr_C->el_size);
        }
      }
    }*/
    if (!sr_C->isequal(beta, sr_C->mulid())){
      int64_t sz = sy_packed_size(order_C, edge_len_C, sym_C);
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


    if (idx_max <= MAX_ORD){
      uint64_t ** offsets_A;
      uint64_t ** offsets_B;
      uint64_t ** offsets_C;
      compute_syoffs(sr_A, order_A, edge_len_A, sym_A, idx_map_A, sr_B, order_B, edge_len_B, sym_B, idx_map_B, sr_C, order_C, edge_len_C, sym_C, idx_map_C, idx_max, rev_idx_map, offsets_A, offsets_B, offsets_C);

      //if we have something to parallelize without needing to replicate C
      if (order_C > 1 || (order_C > 0 && idx_map_C[0] != 0)){
#ifdef USE_OMP    
        #pragma omp parallel
#endif
        {
          int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
          memset(idx_glb, 0, sizeof(int64_t)*idx_max);

          SWITCH_ORD_CALL(sym_seq_ctr_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C, sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, idx_glb, rev_idx_map, idx_max);
          cdealloc(idx_glb);
        }
      } else {
        {
          int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
          memset(idx_glb, 0, sizeof(int64_t)*idx_max);

          SWITCH_ORD_CALL(sym_seq_ctr_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, beta, C, sr_C, order_C, edge_len_C, sym_C, idx_map_C, offsets_C, func, idx_glb, rev_idx_map, idx_max);
          cdealloc(idx_glb);
        }
      }
      for (int l=0; l<idx_max; l++){
        cdealloc(offsets_A[l]);
        cdealloc(offsets_B[l]);
        cdealloc(offsets_C[l]);
      }
      cdealloc(offsets_A);
      cdealloc(offsets_B);
      cdealloc(offsets_C);
    } else {
  
  
      idx_A = 0, idx_B = 0, idx_C = 0;
      sym_pass = 1;
      for (;;){
        //printf("[%d] <- [%d]*[%d]\n",idx_C, idx_A, idx_B);
        if (sym_pass){
          /*if (alpha == NULL && beta == NULL){
            func->apply_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, 
                          C+idx_C*sr_C->el_size);
            CTF_FLOPS_ADD(1);
          } else */ if (alpha == NULL){
            func->acc_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, C+idx_C*sr_C->el_size, sr_C);
            CTF_FLOPS_ADD(2);
          } else {
            char tmp[sr_C->el_size];
            sr_C->mul(A+idx_A*sr_A->el_size, alpha, tmp);
            func->acc_f(tmp, B+idx_B*sr_B->el_size, C+idx_C*sr_C->el_size, sr_C);
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
                      int64_t const *  edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int64_t const *  edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int64_t const *  edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C,
                      iparam const *   prm,
                      bivar_function const * func){
    TAU_FSTART(sym_seq_ctr_inner);
    int idx, i, idx_max, imin, imax, iA, iB, iC, j, k;
    int off_idx, sym_pass, stride_A, stride_B, stride_C;
    int64_t * idx_glb;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B, * dlen_C;
    int64_t idx_A, idx_B, idx_C, off_lda;

    stride_A = prm->m*prm->k*prm->l;
    stride_B = prm->k*prm->n*prm->l;
    stride_C = prm->m*prm->n*prm->l;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            order_C,       idx_map_C,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    dlen_C = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_C);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);
    memcpy(dlen_C, edge_len_C, sizeof(int64_t)*order_C);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);


    /* Scale C immediately. WARNING: wrong for iterators over subset of C */
    if (!prm->offload){
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
    }
    idx_A = 0, idx_B = 0, idx_C = 0;
    sym_pass = 1;
   // int cntr=0;  
    for (;;){
      if (sym_pass){
        TAU_FSTART(gemm);
        if (prm->tC == 'N'){
          if (prm->offload){
            //FIXME: Add GPU batched gemm support
            ASSERT(prm->l == 1);
            if (func == NULL){
              sr_C->offload_gemm(prm->tA, prm->tB, prm->m, prm->n, prm->k, alpha, 
                                 A+idx_A*stride_A*sr_A->el_size,
                                 B+idx_B*stride_B*sr_B->el_size, sr_C->mulid(),
                                 C+idx_C*stride_C*sr_C->el_size);
            } else {
              ASSERT(sr_C->isequal(alpha,sr_C->mulid()));
              func->coffload_gemm(prm->tA, prm->tB, prm->m, prm->n, prm->k, 
                                  A+idx_A*stride_A*sr_A->el_size,
                                  B+idx_B*stride_B*sr_B->el_size, 
                                  C+idx_C*stride_C*sr_C->el_size);
            }
          } else {
            if (func == NULL){
              sr_C->gemm_batch(prm->tA, prm->tB, prm->l, prm->m, prm->n, prm->k, alpha, 
                         A+idx_A*stride_A*sr_A->el_size, 
                         B+idx_B*stride_B*sr_B->el_size, sr_C->mulid(),
                         C+idx_C*stride_C*sr_C->el_size);
            } else {
              ASSERT(prm->l == 1);
              ASSERT(sr_C->isequal(alpha,sr_C->mulid()));
              func->cgemm(prm->tA, prm->tB, prm->m, prm->n, prm->k, 
                           A+idx_A*stride_A*sr_A->el_size, 
                           B+idx_B*stride_B*sr_B->el_size, 
                           C+idx_C*stride_C*sr_C->el_size);
            }
          }
        } else {
          if (prm->offload){
            ASSERT(prm->l == 1);
            if (func == NULL){
              sr_C->offload_gemm(prm->tB, prm->tA, prm->n, prm->m, prm->k, alpha, 
                                 B+idx_B*stride_B*sr_B->el_size,
                                 A+idx_A*stride_A*sr_A->el_size, sr_C->mulid(),
                                 C+idx_C*stride_C*sr_C->el_size);
            } else {
              ASSERT(sr_C->isequal(alpha,sr_C->mulid()));
              func->coffload_gemm(prm->tB, prm->tA, prm->n, prm->m, prm->k, 
                                  B+idx_B*stride_B*sr_B->el_size,
                                  A+idx_A*stride_A*sr_A->el_size,
                                  C+idx_C*stride_C*sr_C->el_size);
            }
          } else {
            if (func == NULL){ 
              sr_C->gemm_batch(prm->tB, prm->tA, prm->l, prm->n, prm->m, prm->k, alpha, 
                         B+idx_B*stride_B*sr_B->el_size,
                         A+idx_A*stride_A*sr_A->el_size, sr_C->mulid(), 
                         C+idx_C*stride_C*sr_C->el_size);
            } else {
              ASSERT(sr_C->isequal(alpha,sr_C->mulid()));
              ASSERT(prm->l == 1);
              func->cgemm(prm->tB, prm->tA, prm->n, prm->m, prm->k, 
                           B+idx_B*stride_B*sr_B->el_size,
                           A+idx_A*stride_A*sr_A->el_size, 
                           C+idx_C*stride_C*sr_C->el_size);

            }
          }
        }
        //printf("[%d] <- [%d]*[%d] (%d)\n",idx_C, idx_A, idx_B, cntr++);
        //printf("%c %c %c %d %d %d\n", prm->tC, prm->tA, prm->tB, prm->m, prm->n, prm->k);
        /*printf("multiplying %lf by %lf and got %lf\n", 
    ((double*)(A+idx_A*stride_A*sr_A->el_size))[0],
    ((double*)(B+idx_B*stride_B*sr_B->el_size))[0],
    ((double*)(C+idx_C*stride_C*sr_C->el_size))[0]);*/
        TAU_FSTOP(gemm);
        // count n^2 FLOPS too
        CTF_FLOPS_ADD((2 * (int64_t)prm->l * (int64_t)prm->n * (int64_t)prm->m * (int64_t)(prm->k+1)));
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
