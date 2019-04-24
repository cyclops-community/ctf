/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include "../shared/util.h"
#include <limits.h>
#include "sym_seq_sum.h"

namespace CTF_int {
  
  template <int idim>
  void sym_seq_sum_loop(char const *            alpha,
                        char const *            A,
                        algstrct const *        sr_A,
                        int                     order_A,
                        int64_t const *         edge_len_A,
                        int const *             sym_A,
                        int const *             idx_map_A,
                        uint64_t *const*        offsets_A,
                        char *                  B,
                        algstrct const *        sr_B,
                        int                     order_B,
                        int64_t const *         edge_len_B,
                        int const *             sym_B,
                        int const *             idx_map_B,
                        uint64_t *const*        offsets_B,
                        univar_function const * func,
                        int64_t const *         idx,
                        int const *             rev_idx_map,
                        int                     idx_max){
    int64_t imax=0;
    int rA = rev_idx_map[2*idim+0];
    int rB = rev_idx_map[2*idim+1];
  
    if (rA != -1)
      imax = edge_len_A[rA];
    else if (rB != -1)
      imax = edge_len_B[rB];

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

    if (rB != -1){
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
          sym_seq_sum_loop<idim-1>(alpha, A+offsets_A[idim][nidx[idim]], sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B+offsets_B[idim][nidx[idim]], sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, func, nidx, rev_idx_map, idx_max);
        }
      }
    } else {
      for (int64_t i=imin; i<imax; i++){
        int64_t nidx[idx_max];
        memcpy(nidx, idx, idx_max*sizeof(int64_t));
        nidx[idim] = i;
        sym_seq_sum_loop<idim-1>(alpha, A+offsets_A[idim][nidx[idim]], sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B+offsets_B[idim][nidx[idim]], sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, func, nidx, rev_idx_map, idx_max);
      }

    }
//    idx[idim] = 0;
  }


  template <>
  void sym_seq_sum_loop<0>
                       (char const *            alpha,
                        char const *            A,
                        algstrct const *        sr_A,
                        int                     order_A,
                        int64_t const *         edge_len_A,
                        int const *             sym_A,
                        int const *             idx_map_A,
                        uint64_t *const*        offsets_A,
                        char *                  B,
                        algstrct const *        sr_B,
                        int                     order_B,
                        int64_t const *         edge_len_B,
                        int const *             sym_B,
                        int const *             idx_map_B,
                        uint64_t *const*        offsets_B,
                        univar_function const * func,
                        int64_t const *         idx,
                        int const *             rev_idx_map,
                        int                     idx_max){
    int64_t imax=0;
    int rA = rev_idx_map[0];
    int rB = rev_idx_map[1];
  
    if (rA != -1)
      imax = edge_len_A[rA];
    else if (rB != -1)
      imax = edge_len_B[rB];

    if (rA != -1 && sym_A[rA] != NS)
      imax = idx[idx_map_A[rA+1]]+1;
    if (rB != -1 && sym_B[rB] != NS)
      imax = std::min(imax,idx[idx_map_B[rB+1]]+1);

    int64_t imin = 0;
    char * tmp = (char*)malloc(sr_A->el_size);

    if (rA > 0 && sym_A[rA-1] != NS)
      imin = idx[idx_map_A[rA-1]];
    if (rB > 0 && sym_B[rB-1] != NS)
      imin = std::max(imin,idx[idx_map_B[rB-1]]);

    if (func == NULL){
      if (alpha == NULL){
        for (int64_t i=imin; i<imax; i++){
          sr_B->add(A+offsets_A[0][i],
                    B+offsets_B[0][i], 
                    B+offsets_B[0][i]);
        }
        CTF_FLOPS_ADD(imax-imin);
      } else {
        for (int64_t i=imin; i<imax; i++){
          sr_A->mul(A+offsets_A[0][i], 
                    alpha,
                    tmp);
          sr_B->add(tmp, 
                    B+offsets_B[0][i], 
                    B+offsets_B[0][i]);
        }
        CTF_FLOPS_ADD(2*(imax-imin));
      }
    } else assert(0); //FIXME else 
    free(tmp);
  }

  template 
  void sym_seq_sum_loop< MAX_ORD >
                       (char const *            alpha,
                        char const *            A,
                        algstrct const *        sr_A,
                        int                     order_A,
                        int64_t const *         edge_len_A,
                        int const *             sym_A,
                        int const *             idx_map_A,
                        uint64_t *const*        offsets_A,
                        char *                  B,
                        algstrct const *        sr_B,
                        int                     order_B,
                        int64_t const *         edge_len_B,
                        int const *             sym_B,
                        int const *             idx_map_B,
                        uint64_t *const*        offsets_B,
                        univar_function const * func,
                        int64_t const *         idx,
                        int const *             rev_idx_map,
                        int                     idx_max);


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
                      int              tot_order,
                      int const *      rev_idx_map,
                      uint64_t **&     offsets_A,
                      uint64_t **&     offsets_B){
    TAU_FSTART(compute_syoffs);
    offsets_A = (uint64_t**)CTF_int::alloc(sizeof(uint64_t*)*tot_order);
    offsets_B = (uint64_t**)CTF_int::alloc(sizeof(uint64_t*)*tot_order);
          
    for (int idim=0; idim<tot_order; idim++){
      int len=0;

      int rA = rev_idx_map[2*idim+0];
      int rB = rev_idx_map[2*idim+1];
  
      if (rA != -1)
        len = edge_len_A[rA];
      else if (rB != -1)
        len = edge_len_B[rB];

      offsets_A[idim] = (uint64_t*)CTF_int::alloc(sizeof(uint64_t)*len);
      offsets_B[idim] = (uint64_t*)CTF_int::alloc(sizeof(uint64_t)*len);
      compute_syoff(rA, len, sr_A, edge_len_A, sym_A, offsets_A[idim]);
      compute_syoff(rB, len, sr_B, edge_len_B, sym_B, offsets_B[idim]);
    }
    TAU_FSTOP(compute_syoffs);
  }


  #define SCAL_B do {                                                      \
    if (!sr_B->isequal(beta, sr_B->mulid())){                                 \
      memset(idx_glb, 0, sizeof(int64_t)*idx_max);                             \
      idx_A = 0, idx_B = 0;                                                \
      sym_pass = 1;                                                        \
      for (;;){                                                            \
        if (sym_pass){                                                     \
          sr_B->mul(beta, B+idx_B*sr_B->el_size, B+idx_B*sr_B->el_size);      \
          CTF_FLOPS_ADD(1);                                                \
        }                                                                  \
        for (idx=0; idx<idx_max; idx++){                                   \
          imin = 0, imax = INT_MAX;                                        \
          GET_MIN_MAX(B,1,2);                                              \
          if (rev_idx_map[2*idx+1] == -1) imax = imin+1;                   \
          idx_glb[idx]++;                                                  \
          if (idx_glb[idx] >= imax){                                       \
             idx_glb[idx] = imin;                                          \
          }                                                                \
          if (idx_glb[idx] != imin) {                                      \
             break;                                                        \
          }                                                                \
        }                                                                  \
        if (idx == idx_max) break;                                         \
        CHECK_SYM(B);                                                      \
        if (!sym_pass) continue;                                           \
        if (order_B > 0)                                                   \
          RESET_IDX(B);                                                    \
      }                                                                    \
    } } while (0)


  #define SCAL_B_inr do {                                                  \
    if (!sr_B->isequal(beta, sr_B->mulid())){                                 \
      memset(idx_glb, 0, sizeof(int64_t)*idx_max);                             \
      idx_A = 0, idx_B = 0;                                                \
      sym_pass = 1;                                                        \
      for (;;){                                                            \
        if (sym_pass){                                                     \
          sr_B->scal(inr_stride, beta, B+idx_B*inr_stride*sr_B->el_size, 1); \
          CTF_FLOPS_ADD(inr_stride);                                       \
        }                                                                  \
        for (idx=0; idx<idx_max; idx++){                                   \
          imin = 0, imax = INT_MAX;                                        \
          GET_MIN_MAX(B,1,2);                                              \
          if (rev_idx_map[2*idx+1] == -1) imax = imin+1;                   \
          idx_glb[idx]++;                                                  \
          if (idx_glb[idx] >= imax){                                       \
             idx_glb[idx] = imin;                                          \
          }                                                                \
          if (idx_glb[idx] != imin) {                                      \
             break;                                                        \
          }                                                                \
        }                                                                  \
        if (idx == idx_max) break;                                         \
        CHECK_SYM(B);                                                      \
        if (!sym_pass) continue;                                           \
        if (order_B > 0)                                                   \
          RESET_IDX(B);                                                    \
      }                                                                    \
    } } while (0)

  int sym_seq_sum_ref( char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int64_t const *  edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     beta,
                       char *           B,
                       algstrct const * sr_B,
                       int              order_B,
                       int64_t const *  edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B){
    TAU_FSTART(sym_seq_sum_ref);
    int idx, i, idx_max, imin, imax, iA, iB, j, k;
    int off_idx, sym_pass;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B;
    int64_t idx_A, idx_B, off_lda;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            &idx_max,     &rev_idx_map);

    bool rep_idx = false;
    for (i=0; i<order_A; i++){
      for (j=0; j<order_A; j++){
        if (i!=j && idx_map_A[i] == idx_map_A[j]) rep_idx = true;
      }
    }
    for (i=0; i<order_B; i++){
      for (j=0; j<order_B; j++){
        if (i!=j && idx_map_B[i] == idx_map_B[j]) rep_idx = true;
      }
    }

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);

    int64_t * idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);
    //FIXME do via scal()
    TAU_FSTART(SCAL_B);
    if (rep_idx)
      SCAL_B;
    else {
      int64_t sz_B = sy_packed_size(order_B, edge_len_B, sym_B);
      if (beta != NULL || sr_B->mulid() != NULL){
        if (beta == NULL || sr_B->isequal(beta, sr_B->addid()))
            sr_B->set(B, sr_B->addid(), sz_B);
        else if (!sr_B->isequal(beta, sr_B->mulid()))
          sr_B->scal(sz_B, beta, B, 1);
      }
    }
    TAU_FSTOP(SCAL_B);

    memset(idx_glb, 0, sizeof(int)*idx_max);
    if (!rep_idx && idx_max>0 && idx_max <= MAX_ORD){
      uint64_t ** offsets_A;
      uint64_t ** offsets_B;
      compute_syoffs(sr_A, order_A, edge_len_A, sym_A, idx_map_A, sr_B, order_B, edge_len_B, sym_B, idx_map_B, idx_max, rev_idx_map, offsets_A, offsets_B);
      if (order_B > 1 || (order_B > 0 && idx_map_B[0] != 0)){
#ifdef USE_OMP    
        #pragma omp parallel
#endif
        {
          int64_t * nidx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
          memset(nidx_glb, 0, sizeof(int64_t)*idx_max);

          SWITCH_ORD_CALL(sym_seq_sum_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, NULL, nidx_glb, rev_idx_map, idx_max);
          cdealloc(nidx_glb);
        }
      } else {
        SWITCH_ORD_CALL(sym_seq_sum_loop, idx_max-1, alpha, A, sr_A, order_A, edge_len_A, sym_A, idx_map_A, offsets_A, B, sr_B, order_B, edge_len_B, sym_B, idx_map_B, offsets_B, NULL, idx_glb, rev_idx_map, idx_max);
      }
      for (int l=0; l<idx_max; l++){
        cdealloc(offsets_A[l]);
        cdealloc(offsets_B[l]);
      }
      cdealloc(offsets_A);
      cdealloc(offsets_B);
    } else {
      idx_A = 0, idx_B = 0;
      sym_pass = 1;
      for (;;){
        if (sym_pass){
      /*    printf("B[%d] = %lf*(A[%d]=%lf)+%lf*(B[%d]=%lf\n",
                  idx_B,alpha,idx_A,A[idx_A],beta,idx_B,B[idx_B]);*/
//        printf("adding to %d ",idx_B); sr_B->print(B+sr_B->el_size*idx_B); printf("\n");
          if (alpha != NULL){
            char tmp[sr_A->el_size];
            sr_A->mul(A+sr_A->el_size*idx_A, alpha, tmp);
            sr_B->add(tmp, B+sr_B->el_size*idx_B, B+sr_B->el_size*idx_B);
            CTF_FLOPS_ADD(2);
          } else {
            sr_B->add(A+sr_A->el_size*idx_A, B+sr_B->el_size*idx_B, B+sr_B->el_size*idx_B);
            CTF_FLOPS_ADD(1);
          }
//        printf("computed %d ",idx_B); sr_B->print(B+sr_B->el_size*idx_B); printf("\n");
        }

        for (idx=0; idx<idx_max; idx++){
          imin = 0, imax = INT_MAX;

          GET_MIN_MAX(A,0,2);
          GET_MIN_MAX(B,1,2);

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
        
        if (order_A > 0)
          RESET_IDX(A);
        if (order_B > 0)
          RESET_IDX(B);
      }
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_ref);
    return 0;
  }

  int sym_seq_sum_inr( char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int64_t const *  edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     beta,
                       char *           B,
                       algstrct const * sr_B,
                       int              order_B,
                       int64_t const *  edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       int              inr_stride){
    TAU_FSTART(sym_seq_sum_inr);
    int idx, i, idx_max, imin, imax, iA, iB, j, k;
    int off_idx, sym_pass;
    int64_t * idx_glb;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B;
    int64_t idx_A, idx_B, off_lda;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);

    SCAL_B_inr;

    memset(idx_glb, 0, sizeof(int64_t)*idx_max);
   
    idx_A = 0, idx_B = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
    /*    printf("B[%d] = %lf*(A[%d]=%lf)+%lf*(B[%d]=%lf\n",
                idx_B,alpha,idx_A,A[idx_A],beta,idx_B,B[idx_B]);*/
      //  B[idx_B] = alpha*A[idx_A] + beta*B[idx_B];
    /*    if (beta != 1.0){
          cxaxpy<dtype>(inr_stride, beta-1.0, B+idx_B*inr_stride, 1, B+idx_B*inr_stride, 1);
          CTF_FLOPS_ADD(2*inr_stride);
        }*/
        //cxaxpy<dtype>(inr_stride, alpha, A+idx_A*inr_stride, 1, B+idx_B*inr_stride, 1); 
        sr_B->axpy(inr_stride, alpha, A+idx_A*sr_A->el_size*inr_stride, 1, B+idx_B*sr_B->el_size*inr_stride, 1); 
        CTF_FLOPS_ADD(2*inr_stride);
      }

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,2);
        GET_MIN_MAX(B,1,2);

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
      
      if (order_A > 0)
        RESET_IDX(A);
      if (order_B > 0)
        RESET_IDX(B);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_inr);
    return 0;
  }

  int sym_seq_sum_cust(char const *            alpha,
                       char const *            A,
                       algstrct const *        sr_A,
                       int                     order_A,
                       int64_t const *         edge_len_A,
                       int const *             sym_A,
                       int const *             idx_map_A,
                       char const *            beta,
                       char *                  B,
                       algstrct const *        sr_B,
                       int                     order_B,
                       int64_t const *         edge_len_B,
                       int const *             sym_B,
                       int const *             idx_map_B,
                       univar_function const * func){
    TAU_FSTART(sym_seq_sum_cust);
    int idx, i, idx_max, imin, imax, iA, iB, j, k;
    int off_idx, sym_pass;
    int64_t * idx_glb;
    int * rev_idx_map;
    int64_t * dlen_A, * dlen_B;
    int64_t idx_A, idx_B, off_lda;

    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            &idx_max,     &rev_idx_map);

    dlen_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    dlen_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    memcpy(dlen_A, edge_len_A, sizeof(int64_t)*order_A);
    memcpy(dlen_B, edge_len_B, sizeof(int64_t)*order_B);

    idx_glb = (int64_t*)CTF_int::alloc(sizeof(int64_t)*idx_max);
    memset(idx_glb, 0, sizeof(int64_t)*idx_max);

    SCAL_B;

    idx_A = 0, idx_B = 0;
    sym_pass = 1;
    for (;;){
      if (sym_pass){
        if (alpha != NULL){
          char tmp_A[sr_A->el_size];
          sr_A->mul(A+sr_A->el_size*idx_A, alpha, tmp_A);
          func->acc_f(tmp_A, B+idx_B*sr_B->el_size, sr_B);
//          func->apply_f(tmp_A, tmp_B);
  //        sr_B->add(B+idx_B*sr_B->el_size, tmp_B, B+sr_B->el_size*idx_B);
          CTF_FLOPS_ADD(2);
        } else {
          func->acc_f(A+idx_A*sr_A->el_size, B+idx_B*sr_B->el_size, sr_B);
          //func->apply_f(A+idx_A*sr_A->el_size, tmp_B);
          //sr_B->add(B+idx_B*sr_B->el_size, tmp_B, B+idx_B*sr_B->el_size);
          CTF_FLOPS_ADD(1);
        }
      }

      for (idx=0; idx<idx_max; idx++){
        imin = 0, imax = INT_MAX;

        GET_MIN_MAX(A,0,2);
        GET_MIN_MAX(B,1,2);

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
      
      if (order_A > 0)
        RESET_IDX(A);
      if (order_B > 0)
        RESET_IDX(B);
    }
    CTF_int::cdealloc(dlen_A);
    CTF_int::cdealloc(dlen_B);
    CTF_int::cdealloc(idx_glb);
    CTF_int::cdealloc(rev_idx_map);
    TAU_FSTOP(sym_seq_sum_cust);
    return 0;
  }
}
