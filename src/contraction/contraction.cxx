#include "contraction.h"
#include "../redistribution/nosym_transp.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_ctr.h"
#include "ctr_tsr.h"
#include "ctr_comm.h"
#include "ctr_2d_general.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include <cfloat>
#include <limits>

namespace CTF_int {

  using namespace CTF;

  contraction::~contraction(){
    if (idx_A != NULL) cdealloc(idx_A);
    if (idx_B != NULL) cdealloc(idx_B);
    if (idx_C != NULL) cdealloc(idx_C);
  }

  contraction::contraction(contraction const & other){
    A     = other.A;
    idx_A = (int*)alloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B     = other.B;
    idx_B = (int*)alloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    C     = other.C;
    idx_C = (int*)alloc(sizeof(int)*other.C->order);
    memcpy(idx_C, other.idx_C, sizeof(int)*other.C->order);
    if (other.is_custom){
      func      = other.func;
      is_custom = 1;
    } else is_custom = 0;
    alpha = other.alpha;
    beta  = other.beta;
  }
 
  contraction::contraction(tensor *         A_,
                           int const *      idx_A_,
                           tensor *         B_,
                           int const *      idx_B_,
                           char const *     alpha_,
                           tensor *         C_,
                           int const *      idx_C_,
                           char const *     beta_,
                           bivar_function * func_){
    A = A_;
    B = B_;
    C = C_;
    if (func_ == NULL) is_custom = 0;
    else { 
      is_custom = 1;
      func = func_;
    }
    alpha = alpha_;
    beta  = beta_;
    
    idx_A = (int*)alloc(sizeof(int)*A->order);
    idx_B = (int*)alloc(sizeof(int)*B->order);
    idx_C = (int*)alloc(sizeof(int)*C->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    memcpy(idx_C, idx_C_, sizeof(int)*C->order);
  }
 
  contraction::contraction(tensor *         A_,
                           char const *     cidx_A,
                           tensor *         B_,
                           char const *     cidx_B,
                           char const *     alpha_,
                           tensor *         C_,
                           char const *     cidx_C,
                           char const *     beta_,
                           bivar_function * func_){
    A = A_;
    B = B_;
    C = C_;
    if (func_ == NULL) is_custom = 0;
    else { 
      is_custom = 1;
      func = func_;
    }
    alpha = alpha_;
    beta  = beta_;
    
    conv_idx(A->order, cidx_A, &idx_A, B->order, cidx_B, &idx_B, C->order, cidx_C, &idx_C);
  }

  void contraction::execute(){
#if DEBUG >= 2
    if (A->wrld->cdt.rank == 0) printf("Contraction::execute (head):\n");
    print();
#endif
    
    int stat = home_contract();
    assert(stat == SUCCESS); 
  }
  
  double contraction::estimate_time(){
    assert(0); //FIXME
    return 0.0;
  }

  int contraction::is_equal(contraction const & os){
    if (this->A != os.A) return 0;
    if (this->B != os.B) return 0;
    if (this->C != os.C) return 0;
    
    for (int i=0; i<A->order; i++){
      if (idx_A[i] != os.idx_A[i]) return 0;
    }
    for (int i=0; i<B->order; i++){
      if (idx_B[i] != os.idx_B[i]) return 0;
    }
    for (int i=0; i<C->order; i++){
      if (idx_C[i] != os.idx_C[i]) return 0;
    }
    return 1;
  }

  void contraction::calc_fold_nmk(
                      int const *    ordering_A,
                      int const *    ordering_B,
                      iparam *       inner_prm){
    int i, num_ctr, num_tot;
    int * idx_arr;
    iparam prm;
      
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    num_ctr = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        num_ctr++;
      } 
    }
    prm.m = 1;
    prm.n = 1;
    prm.k = 1;
    for (i=0; i<A->order; i++){
      if (i >= num_ctr)
        prm.m = prm.m * A->pad_edge_len[ordering_A[i]];
      else 
        prm.k = prm.k * A->pad_edge_len[ordering_A[i]];
    }
    for (i=0; i<B->order; i++){
      if (i >= num_ctr)
        prm.n = prm.n * B->pad_edge_len[ordering_B[i]];
    }
    /* This gets set later */
    prm.sz_C = 0;
    CTF_int::cdealloc(idx_arr);
    *inner_prm = prm;  
  }

  void contraction::get_fold_indices(int *  num_fold,
                                     int ** fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, iC, inA, inB, inC, iiA, iiB, iiC;
    int * idx_arr, * idx;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    CTF_int::alloc_ptr(num_tot*sizeof(int), (void**)&idx);

    for (i=0; i<num_tot; i++){
      idx[i] = 1;
    }

    for (iA=0; iA<A->order; iA++){
      i = idx_A[iA];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      broken = 0;
      inA = iA;
      do {
        in = idx_A[inA];
        inB = idx_arr[3*in+1];
        inC = idx_arr[3*in+2];
        if (((inA>=0) + (inB>=0) + (inC>=0) != 2) ||
            ((inB == -1) ^ (iB == -1)) ||
            ((inC == -1) ^ (iC == -1)) ||
            (iB != -1 && inB - iB != in-i) ||
            (iC != -1 && inC - iC != in-i) ||
            (iB != -1 && A->sym[inA] != B->sym[inB]) ||
            (iC != -1 && A->sym[inA] != C->sym[inC])){
          broken = 1;
        }
        inA++;
      } while (A->sym[inA-1] != NS);
      if (broken){
        for (iiA=iA;iiA<inA;iiA++){
          idx[idx_A[iiA]] = 0;
        }
      }
    }
    
    for (iC=0; iC<C->order; iC++){
      i = idx_C[iC];
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      broken = 0;
      inC = iC;
      do {
        in = idx_C[inC];
        inA = idx_arr[3*in+0];
        inB = idx_arr[3*in+1];
        if (((inC>=0) + (inA>=0) + (inB>=0) != 2) ||
            ((inA == -1) ^ (iA == -1)) ||
            ((inB == -1) ^ (iB == -1)) ||
            (iA != -1 && inA - iA != in-i) ||
            (iB != -1 && inB - iB != in-i) ||
            (iA != -1 && C->sym[inC] != A->sym[inA]) ||
            (iB != -1 && C->sym[inC] != B->sym[inB])){
          broken = 1;
        }
        inC++;
      } while (C->sym[inC-1] != NS);
      if (broken){
        for (iiC=iC;iiC<inC;iiC++){
          idx[idx_C[iiC]] = 0;
        }
      }
    }
    
    for (iB=0; iB<B->order; iB++){
      i = idx_B[iB];
      iC = idx_arr[3*i+2];
      iA = idx_arr[3*i+0];
      broken = 0;
      inB = iB;
      do {
        in = idx_B[inB];
        inC = idx_arr[3*in+2];
        inA = idx_arr[3*in+0];
        if (((inB>=0) + (inC>=0) + (inA>=0) != 2) ||
            ((inC == -1) ^ (iC == -1)) ||
            ((inA == -1) ^ (iA == -1)) ||
            (iC != -1 && inC - iC != in-i) ||
            (iA != -1 && inA - iA != in-i) ||
            (iC != -1 && B->sym[inB] != C->sym[inC]) ||
            (iA != -1 && B->sym[inB] != A->sym[inA])){
              broken = 1;
            }
        inB++;
      } while (B->sym[inB-1] != NS);
      if (broken){
        for (iiB=iB;iiB<inB;iiB++){
          idx[idx_B[iiB]] = 0;
        }
      }
    }

    nfold = 0;
    for (i=0; i<num_tot; i++){
      if (idx[i] == 1){
        idx[nfold] = i;
        nfold++;
      }
    }
    *num_fold = nfold;
    *fold_idx = idx;
    CTF_int::cdealloc(idx_arr);

  }

  int contraction::can_fold(){
    int nfold, * fold_idx, i, j;
    for (i=0; i<A->order; i++){
      for (j=i+1; j<A->order; j++){
        if (idx_A[i] == idx_A[j]) return 0;
      }
    }
    for (i=0; i<B->order; i++){
      for (j=i+1; j<B->order; j++){
        if (idx_B[i] == idx_B[j]) return 0;
      }
    }
    for (i=0; i<C->order; i++){
      for (j=i+1; j<C->order; j++){
        if (idx_C[i] == idx_C[j]) return 0;
      }
    }
    get_fold_indices(&nfold, &fold_idx);
    CTF_int::cdealloc(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }


  void contraction::get_len_ordering(
            int ** new_ordering_A,
            int ** new_ordering_B,
            int ** new_ordering_C){
    int i, num_tot, num_ctr, idx_ctr, num_no_ctr_A;
    int idx_no_ctr_A, idx_no_ctr_B;
    int * ordering_A, * ordering_B, * ordering_C, * idx_arr;
    
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&ordering_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&ordering_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&ordering_C);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    num_ctr = 0, num_no_ctr_A = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        num_ctr++;
      } else if (idx_arr[3*i] != -1){
        num_no_ctr_A++;
      }
    }
    /* Put all contraction indices up front, put A indices in front for C */
    idx_ctr = 0, idx_no_ctr_A = 0, idx_no_ctr_B = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        ordering_A[idx_ctr] = idx_arr[3*i];
        ordering_B[idx_ctr] = idx_arr[3*i+1];
        idx_ctr++;
      } else {
        if (idx_arr[3*i] != -1){
          ordering_A[num_ctr+idx_no_ctr_A] = idx_arr[3*i];
          ordering_C[idx_no_ctr_A] = idx_arr[3*i+2];
          idx_no_ctr_A++;
        }
        if (idx_arr[3*i+1] != -1){
          ordering_B[num_ctr+idx_no_ctr_B] = idx_arr[3*i+1];
          ordering_C[num_no_ctr_A+idx_no_ctr_B] = idx_arr[3*i+2];
          idx_no_ctr_B++;
        }
      }
    }
    CTF_int::cdealloc(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
    *new_ordering_C = ordering_C;
  }

  iparam contraction::map_fold(){
    int i, j, nfold, nf, all_fdim_A, all_fdim_B, all_fdim_C;
    int nvirt_A, nvirt_B, nvirt_C;
    int * fold_idx, * fidx_A, * fidx_B, * fidx_C;
    int * fnew_ord_A, * fnew_ord_B, * fnew_ord_C;
    int * all_flen_A, * all_flen_B, * all_flen_C;
    tensor * fA, * fB, * fC;
    iparam iprm;

    get_fold_indices(&nfold, &fold_idx);
    if (nfold == 0) {
      CTF_int::cdealloc(fold_idx);
      assert(0); //return ERROR;
    }

    /* overestimate this space to not bother with it later */
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_A);
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_B);
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_C);

    A->fold(nfold, fold_idx, idx_A,
            &all_fdim_A, &all_flen_A);
    B->fold(nfold, fold_idx, idx_B,
            &all_fdim_B, &all_flen_B);
    C->fold(nfold, fold_idx, idx_C,
            &all_fdim_C, &all_flen_C);

//    printf("rec tsr C order is %d\n",C->rec_tsr->order);

    nf = 0;
    for (i=0; i<A->order; i++){
      for (j=0; j<nfold; j++){
        if (A->sym[i] == NS && idx_A[i] == fold_idx[j]){
          fidx_A[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<B->order; i++){
      for (j=0; j<nfold; j++){
        if (B->sym[i] == NS && idx_B[i] == fold_idx[j]){
          fidx_B[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<C->order; i++){
      for (j=0; j<nfold; j++){
        if (C->sym[i] == NS && idx_C[i] == fold_idx[j]){
          fidx_C[nf] = j;
          nf++;
        }
      }
    }

    fA = A->rec_tsr;
    fB = B->rec_tsr;
    fC = C->rec_tsr;
/*
    fold_ctr.A = fA; 
    fold_ctr.B = fB; 
    fold_ctr.C = fC; */

    int * sidx_A, * sidx_B, * sidx_C;
    CTF_int::conv_idx<int>(fA->order, fidx_A, &sidx_A,
                       fB->order, fidx_B, &sidx_B,
                       fC->order, fidx_C, &sidx_C);

    contraction fold_ctr(fA, sidx_A, fB, sidx_B, alpha, fC, sidx_C, beta);

    free(sidx_A);
    free(sidx_B);
    free(sidx_C);
  #if DEBUG>=2
    CommData global_comm = A->wrld->cdt;
    if (global_comm.rank == 0){
      printf("Folded contraction type:\n");
    }
    fold_ctr.print();
  #endif
    
    //for type order 1 to 3 
    fold_ctr.get_len_ordering(&fnew_ord_A, &fnew_ord_B, &fnew_ord_C); 
    
    //permute_target(fA->order, fnew_ord_A, cpy_A_inner_ordering);
    //permute_target(fB->order, fnew_ord_B, cpy_B_inner_ordering);

    //get nosym_transpose_estimate cost estimate an save best

    permute_target(fA->order, fnew_ord_A, A->inner_ordering);
    permute_target(fB->order, fnew_ord_B, B->inner_ordering);
    permute_target(fC->order, fnew_ord_C, C->inner_ordering);
    

    nvirt_A = A->calc_nvirt();
    for (i=0; i<nvirt_A; i++){
      nosym_transpose(all_fdim_A, A->inner_ordering, all_flen_A,
                      A->data + A->sr->el_size*i*(A->size/nvirt_A), 1, A->sr);
    }
    nvirt_B = B->calc_nvirt();
    for (i=0; i<nvirt_B; i++){
      nosym_transpose(all_fdim_B, B->inner_ordering, all_flen_B,
                      B->data + B->sr->el_size*i*(B->size/nvirt_B), 1, B->sr);
    }
    nvirt_C = C->calc_nvirt();
    for (i=0; i<nvirt_C; i++){
      nosym_transpose(all_fdim_C, C->inner_ordering, all_flen_C,
                      C->data + C->sr->el_size*i*(C->size/nvirt_C), 1, C->sr);
    }

    fold_ctr.calc_fold_nmk(fnew_ord_A, fnew_ord_B, &iprm);

    //FIXME: try all possibilities
    iprm.tA = 'T';
    iprm.tB = 'N';

    CTF_int::cdealloc(fidx_A);
    CTF_int::cdealloc(fidx_B);
    CTF_int::cdealloc(fidx_C);
    CTF_int::cdealloc(fnew_ord_A);
    CTF_int::cdealloc(fnew_ord_B);
    CTF_int::cdealloc(fnew_ord_C);
    CTF_int::cdealloc(all_flen_A);
    CTF_int::cdealloc(all_flen_B);
    CTF_int::cdealloc(all_flen_C);
    CTF_int::cdealloc(fold_idx);

    return iprm;
  }

  int contraction::unfold_broken_sym(contraction ** new_contraction){
    int i, num_tot, iA, iB, iC, iA2, iB2;
    int * idx_arr;
    tensor * nA, * nB, * nC;
   
    contraction * nctr; 
    
    if (new_contraction != NULL){
      nA = new tensor(A, 0, 0);
      nB = new tensor(B, 0, 0);
      nC = new tensor(C, 0, 0);
      nctr = new contraction(nA, idx_A, nB, idx_B, alpha, nC, idx_C, beta);
      *new_contraction = nctr;
    } else {
      nA = NULL;
      nB = NULL;
      nC = NULL;
    }

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    for (i=0; i<A->order; i++){
      if (A->sym[i] != NS){
        iA = idx_A[i];
        if (idx_arr[3*iA+1] != -1){
          if (B->sym[idx_arr[3*iA+1]] == NS ||
              idx_A[i+1] != idx_B[idx_arr[3*iA+1]+1]){
            if (new_contraction != NULL)
              nA->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*idx_A[i+1]+1] != -1){
            if (new_contraction != NULL)
              nA->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }       
        }
        if (idx_arr[3*iA+2] != -1){
          if (C->sym[idx_arr[3*iA+2]] == NS ||
              idx_A[i+1] != idx_C[idx_arr[3*iA+2]+1]){
            if (new_contraction != NULL)
              nA->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*idx_A[i+1]+2] != -1){
            if (new_contraction != NULL)
              nA->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }       
        }
      }
    }

   
    for (i=0; i<B->order; i++){
      if (B->sym[i] != NS){
        iB = idx_B[i];
        if (idx_arr[3*iB+0] != -1){
          if (A->sym[idx_arr[3*iB+0]] == NS ||
              idx_B[i+1] != idx_A[idx_arr[3*iB+0]+1]){
            if (new_contraction != NULL)
              nB->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*idx_B[i+1]+0] != -1){
            if (new_contraction != NULL)
              nB->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }       
        }
        if (idx_arr[3*iB+2] != -1){
          if (C->sym[idx_arr[3*iB+2]] == NS || 
              idx_B[i+1] != idx_C[idx_arr[3*iB+2]+1]){
            if (new_contraction != NULL)
              nB->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*idx_B[i+1]+2] != -1){
            if (new_contraction != NULL)
              nB->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }       
        }
      }
    } 
    //if A=B, output symmetry may still be preserved, so long as all indices in A and B are proper
    bool is_preserv = true;
    if (A != B) is_preserv = false; 
    else {
      for (int j=0; j<A->order; j++){
        if (idx_A[j] != idx_B[j]){
          iA = idx_A[j];
          iB = idx_B[j];
          if (idx_arr[3*iA+2] == -1 || idx_arr[3*iB+2] == -1) is_preserv = false;
          else {
            for (int k=MIN(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k<MAX(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k++){
               if (C->sym[k] != SY) is_preserv = false;
            }
          }
        }
      }
    }
    if (!is_preserv){
      for (i=0; i<C->order; i++){
        if (C->sym[i] != NS){
          iC = idx_C[i];
          if (idx_arr[3*iC+1] != -1){
            if (B->sym[idx_arr[3*iC+1]] == NS ||
                idx_C[i+1] != idx_B[idx_arr[3*iC+1]+1]){
              if (new_contraction != NULL)
                nC->sym[i] = NS;
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*idx_C[i+1]+1] != -1){
            if (new_contraction != NULL)
              nC->sym[i] = NS;
            CTF_int::cdealloc(idx_arr); 
            return 3*i+2;
          }       
          if (idx_arr[3*iC+0] != -1){
            if (A->sym[idx_arr[3*iC+0]] == NS ||
                idx_C[i+1] != idx_A[idx_arr[3*iC+0]+1]){
              if (new_contraction != NULL)
                nC->sym[i] = NS;
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*iC+0] == -1){
            if (idx_arr[3*idx_C[i+1]] != -1){
              if (new_contraction != NULL)
                nC->sym[i] = NS;
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }       
          }
        }
      }
    }
    for (i=0; i<A->order; i++){
      if (A->sym[i] == SY){
        iA = idx_A[i];
        iA2 = idx_A[i+1];
        if (idx_arr[3*iA+2] == -1 &&
            idx_arr[3*iA2+2] == -1){
          if (new_contraction != NULL)
            nA->sym[i] = NS;
          CTF_int::cdealloc(idx_arr); 
          return 3*i;
        }
      }
    }
    for (i=0; i<B->order; i++){
      if (B->sym[i] == SY){
        iB = idx_B[i];
        iB2 = idx_B[i+1];
        if (idx_arr[3*iB+2] == -1 &&
            idx_arr[3*iB2+2] == -1){
          if (new_contraction != NULL)
            nB->sym[i] = NS;
          CTF_int::cdealloc(idx_arr); 
          return 3*i+1;
        }
      }
    }

    CTF_int::cdealloc(idx_arr);
    return -1;
  }

  void contraction::check_consistency(){
    int i, num_tot, len;
    int iA, iB, iC;
    int * idx_arr;
       
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    for (i=0; i<num_tot; i++){
      len = -1;
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      if (iA != -1){
        len = A->lens[iA];
      }
      if (len != -1 && iB != -1 && len != B->lens[iB]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s does not",
                  iA, A->name);
          printf("match the %dth edge length of tensor %s.\n",
                  iB, B->name);
        }
        ABORT;
      }
      if (len != -1 && iC != -1 && len != C->lens[iC]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s (%d) does not",
                  iA, A->name, len);
          printf("match the %dth edge length of tensor %s (%d).\n",
                  iC, C->name, C->lens[iC]);
        }
        ABORT;
      }
      if (iB != -1){
        len = B->lens[iB];
      }
      if (len != -1 && iC != -1 && len != C->lens[iC]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s does not",
                  iB, B->name);
          printf("match the %dth edge length of tensor %s.\n",
                  iC, C->name);
        }
        ABORT;
      }
    }
    CTF_int::cdealloc(idx_arr);
  }

    
  int contraction::check_mapping(){

    int num_tot, i, ph_A, ph_B, iA, iB, iC, pass, order, topo_order;
    int * idx_arr;
    int * phys_mismatched, * phys_mapped;
    mapping * map;
    tensor * pA, * pB;

    pass = 1;

    if (A->is_mapped == 0) pass = 0;
    if (B->is_mapped == 0) pass = 0;
    if (C->is_mapped == 0) pass = 0;
    ASSERT(pass==1);
    
    if (A->is_folded == 1) pass = 0;
    if (B->is_folded == 1) pass = 0;
    if (C->is_folded == 1) pass = 0;
    
    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    if (A->topo != B->topo) pass = 0;
    if (A->topo != C->topo) pass = 0;

    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    topo_order = A->topo->order;
    CTF_int::alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mismatched);
    CTF_int::alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mapped);
    memset(phys_mismatched, 0, sizeof(int)*topo_order);
    memset(phys_mapped, 0, sizeof(int)*topo_order);


    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    
    if (!check_self_mapping(A, idx_A))
      pass = 0;
    if (!check_self_mapping(B, idx_B))
      pass = 0;
    if (!check_self_mapping(C, idx_C))
      pass = 0;
    if (pass == 0){
      DPRINTF(3,"failed confirmation here\n");
    }


    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] != -1 &&
          idx_arr[3*i+1] != -1 &&
          idx_arr[3*i+2] != -1){
        iA = idx_arr[3*i+0];
        iB = idx_arr[3*i+1];
        iC = idx_arr[3*i+2];
  //      printf("A[%d].np = %d\n", iA, A->edge_map[iA].np);
        //printf("B[%d].np = %d\n", iB, B->edge_map[iB].np);
        //printf("C[%d].np = %d\n", iC, C->edge_map[iC].np);
        if (0 == comp_dim_map(&B->edge_map[iB], &A->edge_map[iA]) || 
            0 == comp_dim_map(&B->edge_map[iB], &C->edge_map[iC])){
          DPRINTF(3,"failed confirmation here %d %d %d\n",iA,iB,iC);
          pass = 0;
          break;
        } else {
          map = &A->edge_map[iA];
          for (;;){
            if (map->type == PHYSICAL_MAP){
              if (phys_mapped[map->cdt] == 1){
                DPRINTF(3,"failed confirmation here %d\n",iA);
                pass = 0;
                break;
              } else {
                phys_mapped[map->cdt] = 1;
                phys_mismatched[map->cdt] = 1;
              }
            } else break;
            if (map->has_child) map = map->child;
            else break;
          } 
        }
      }
    }
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] == -1 ||
          idx_arr[3*i+1] == -1 ||
          idx_arr[3*i+2] == -1){
        for (order=0; order<3; order++){
          switch (order){
            case 0:
              pA = A;
              pB = B;
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+2];
              break;
            case 1:
              pA = A;
              pB = C;
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+2];
              iC = idx_arr[3*i+1];
              break;
            case 2:
              pA = C;
              pB = B;
              iA = idx_arr[3*i+2];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+0];
              break;
          }
          if (iC == -1){
            if (iB == -1){
              if (iA != -1) {
                map = &pA->edge_map[iA];
                for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mapped[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here %d\n",iA);
                      pass = 0;
                      break;
                    } else
                      phys_mapped[map->cdt] = 1;
                  } else break;
                  if (map->has_child) map = map->child;
                  else break;
                } 
              }
            } else if (iA == -1){
              map = &pB->edge_map[iB];
              for (;;){
                if (map->type == PHYSICAL_MAP){
                if (phys_mapped[map->cdt] == 1){
                  DPRINTF(3,"failed confirmation here %d\n",iA);
                  pass = 0;
                  break;
                } else
                  phys_mapped[map->cdt] = 1;
              } else break;
              if (map->has_child) map = map->child;
              else break;
              } 
            } else { 
              /* Confirm that the phases of A and B 
                 over which we are contracting are the same */
              ph_A = pA->edge_map[iA].calc_phase();
              ph_B = pB->edge_map[iB].calc_phase();

              if (ph_A != ph_B){
                //if (global_comm.rank == 0) 
                  DPRINTF(3,"failed confirmation here iA=%d iB=%d\n",iA,iB);
                pass = 0;
                break;
              }
              /* If the mapping along this dimension is the same make sure
                 its mapped to a onto a unique free dimension */
              if (comp_dim_map(&pB->edge_map[iB], &pA->edge_map[iA])){
                map = &pB->edge_map[iB];
              if (map->type == PHYSICAL_MAP){
                if (phys_mapped[map->cdt] == 1){
                  DPRINTF(3,"failed confirmation here %d\n",iB);
                  pass = 0;
                } else
                  phys_mapped[map->cdt] = 1;
                } 
                /*if (map->has_child) {
                  if (map->child->type == PHYSICAL_MAP){
                    DPRINTF(3,"failed confirmation here %d, matched and folded physical mapping not allowed\n",iB);
                    pass = 0;
                  }
                }*/
              } else {
                /* If the mapping along this dimension is different, make sure
                   the mismatch is mapped onto unqiue physical dimensions */
                map = &pA->edge_map[iA];
                for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mismatched[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here i=%d iA=%d iB=%d\n",i,iA,iB);
                      pass = 0;
                      break;
                    } else
                      phys_mismatched[map->cdt] = 1;
                    if (map->has_child) 
                      map = map->child;
                    else break;
                  } else break;
                      }
                      map = &pB->edge_map[iB];
                            for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mismatched[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here i=%d iA=%d iB=%d\n",i,iA,iB);
                      pass = 0;
                      break;
                    } else
                      phys_mismatched[map->cdt] = 1;
                    if (map->has_child) 
                      map = map->child;
                    else break;
                  } else break;
                }
              }
            }
          }
        }
      }
    }
    for (i=0; i<topo_order; i++){
      if (phys_mismatched[i] == 1 && phys_mapped[i] == 0){
        DPRINTF(3,"failed confirmation here i=%d\n",i);
        pass = 0;
        break;
      }
  /*   if (phys_mismatched[i] == 0 && phys_mapped[i] == 0){
        DPRINTF(3,"failed confirmation here i=%d\n",i);
        pass = 0;
        break;
      }    */
    }


    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(phys_mismatched);
    CTF_int::cdealloc(phys_mapped);
    return pass;
  }

  /**
   * \brief map the indices over which we will be weighing
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C corresponding to every global index
   * \param idx_weigh specification of which indices are being contracted
   * \param num_tot total number of indices
   * \param num_weigh number of indices being contracted over
   * \param topo topology to map to
   */
  static int
      map_weigh_indices(int const *      idx_arr,
                        int const *      idx_weigh,
                        int              num_tot,
                        int              num_weigh,
                        topology const * topo,
                        tensor *         A,
                        tensor *         B,
                        tensor *         C){
    int tsr_order, iweigh, iA, iB, iC, i, j, k, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * weigh_map;

    TAU_FSTART(map_weigh_indices);

    tsr_order = num_weigh;

    
    for (i=0; i<num_weigh; i++){
      iweigh = idx_weigh[i];
      iA = idx_arr[iweigh*3+0];
      iB = idx_arr[iweigh*3+1];
      iC = idx_arr[iweigh*3+2];

      if (A->edge_map[iA].type == PHYSICAL_MAP ||
          B->edge_map[iB].type == PHYSICAL_MAP ||
          C->edge_map[iC].type == PHYSICAL_MAP)
        return NEGATIVE; 
    }  
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_int::alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_int::alloc_ptr(tsr_order*sizeof(mapping),            (void**)&weigh_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));
    extract_free_comms(topo, A->order, A->edge_map,
                             B->order, B->edge_map,
                       num_sub_phys_dims, &sub_phys_comm, &comm_idx);

    for (i=0; i<tsr_order; i++){ 
      weigh_map[i].type             = VIRTUAL_MAP; 
      weigh_map[i].has_child        = 0; 
      weigh_map[i].np               = 1; 
    }
    for (i=0; i<num_weigh; i++){
      iweigh = idx_weigh[i];
      iA = idx_arr[iweigh*3+0];
      iB = idx_arr[iweigh*3+1];
      iC = idx_arr[iweigh*3+2];

      
      weigh_map[i].np = lcm(weigh_map[i].np,A->edge_map[iA].np);
      weigh_map[i].np = lcm(weigh_map[i].np,B->edge_map[iB].np);
      weigh_map[i].np = lcm(weigh_map[i].np,C->edge_map[iC].np);

      tsr_edge_len[i] = A->pad_edge_len[iA];

      for (j=i+1; j<num_weigh; j++){
        jX = idx_arr[idx_weigh[j]*3+0];

        for (k=MIN(iA,jX); k<MAX(iA,jX); k++){
          if (A->sym[k] == NS)
            break;
        }
        if (k==MAX(iA,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+1];

        for (k=MIN(iB,jX); k<MAX(iB,jX); k++){
          if (B->sym[k] == NS)
            break;
        }
        if (k==MAX(iB,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+2];

        for (k=MIN(iC,jX); k<MAX(iC,jX); k++){
          if (C->sym[k] == NS)
            break;
        }
        if (k==MAX(iC,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }
      }
    }
    stat = map_tensor(num_sub_phys_dims,  tsr_order,
                      tsr_edge_len,       tsr_sym_table,
                      restricted,         sub_phys_comm,
                      comm_idx,           0,
                      weigh_map);

    if (stat == ERROR)
      return ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == SUCCESS){
      for (i=0; i<num_weigh; i++){
        iweigh = idx_weigh[i];
        iA = idx_arr[iweigh*3+0];
        iB = idx_arr[iweigh*3+1];
        iC = idx_arr[iweigh*3+2];

        copy_mapping(1, &weigh_map[i], &A->edge_map[iA]);
        copy_mapping(1, &weigh_map[i], &B->edge_map[iB]);
        copy_mapping(1, &weigh_map[i], &C->edge_map[iC]);
      }
    }
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(tsr_edge_len);
    CTF_int::cdealloc(tsr_sym_table);
    for (i=0; i<num_weigh; i++){
      weigh_map[i].clear();
    }
    CTF_int::cdealloc(weigh_map);
    //if (num_sub_phys_dims > 0)
    CTF_int::cdealloc(sub_phys_comm);
    CTF_int::cdealloc(comm_idx);

    TAU_FSTOP(map_weigh_indices);
    return stat;
  }
  /**
   * \brief map the indices over which we will be contracting
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_ctr specification of which indices are being contracted
   * \param num_tot total number of indices
   * \param num_ctr number of indices being contracted over
   * \param topo topology to map to
   */
  static int
  map_ctr_indices(int const *      idx_arr,
                  int const *      idx_ctr,
                  int              num_tot,
                  int              num_ctr,
                  topology const * topo,
                  tensor *         A,
                  tensor *         B){
    int tsr_order, ictr, iA, iB, i, j, jctr, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * ctr_map;

    TAU_FSTART(map_ctr_indices);

    tsr_order = num_ctr*2;

    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_int::alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_int::alloc_ptr(tsr_order*sizeof(mapping),            (void**)&ctr_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));

    for (i=0; i<tsr_order; i++){ 
      ctr_map[i].type             = VIRTUAL_MAP; 
      ctr_map[i].has_child        = 0; 
      ctr_map[i].np               = 1; 
    }
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

      copy_mapping(1, &A->edge_map[iA], &ctr_map[2*i+0]);
      copy_mapping(1, &B->edge_map[iB], &ctr_map[2*i+1]);
    }
  /*  for (i=0; i<tsr_order; i++){ 
      if (ctr_map[i].type == PHYSICAL_MAP) is_premapped = 1;
    }*/

    extract_free_comms(topo, A->order, A->edge_map,
                             B->order, B->edge_map,
                       num_sub_phys_dims, &sub_phys_comm, &comm_idx);
    

    /* Map a tensor of dimension 2*num_ctr, with symmetries among each pair.
     * Set the edge lengths and symmetries according to those in ctr dims of A and B.
     * This gives us a mapping for the contraction dimensions of tensors A and B. */
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

      tsr_edge_len[2*i+0] = A->pad_edge_len[iA];
      tsr_edge_len[2*i+1] = A->pad_edge_len[iA];

      tsr_sym_table[2*i*tsr_order+2*i+1] = 1;
      tsr_sym_table[(2*i+1)*tsr_order+2*i] = 1;

      /* Check if A has symmetry among the dimensions being contracted over.
       * Ignore symmetry with non-contraction dimensions.
       * FIXME: this algorithm can be more efficient but should not be a bottleneck */
      if (A->sym[iA] != NS){
        for (j=0; j<num_ctr; j++){
          jctr = idx_ctr[j];
          jX = idx_arr[jctr*3+0];
          if (jX == iA+1){
            tsr_sym_table[2*i*tsr_order+2*j] = 1;
            tsr_sym_table[2*i*tsr_order+2*j+1] = 1;
            tsr_sym_table[2*j*tsr_order+2*i] = 1;
            tsr_sym_table[2*j*tsr_order+2*i+1] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j+1] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i+1] = 1;
          }
        }
      }
      if (B->sym[iB] != NS){
        for (j=0; j<num_ctr; j++){
          jctr = idx_ctr[j];
          jX = idx_arr[jctr*3+1];
          if (jX == iB+1){
            tsr_sym_table[2*i*tsr_order+2*j] = 1;
            tsr_sym_table[2*i*tsr_order+2*j+1] = 1;
            tsr_sym_table[2*j*tsr_order+2*i] = 1;
            tsr_sym_table[2*j*tsr_order+2*i+1] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j+1] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i+1] = 1;
          }
        }
      }
    }
    /* Run the mapping algorithm on this construct */
    /*if (is_premapped){
      stat = map_symtsr(tsr_order, tsr_sym_table, ctr_map);
    } else {*/
      stat = map_tensor(num_sub_phys_dims,  tsr_order,
                        tsr_edge_len,       tsr_sym_table,
                        restricted,         sub_phys_comm,
                        comm_idx,           0,
                        ctr_map);

    //}
    if (stat == ERROR)
      return ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == SUCCESS){
      for (i=0; i<num_ctr; i++){
        ictr = idx_ctr[i];
        iA = idx_arr[ictr*3+0];
        iB = idx_arr[ictr*3+1];

  /*      A->edge_map[iA] = ctr_map[2*i+0];
        B->edge_map[iB] = ctr_map[2*i+1];*/
        copy_mapping(1, &ctr_map[2*i+0], &A->edge_map[iA]);
        copy_mapping(1, &ctr_map[2*i+1], &B->edge_map[iB]);
      }
    }
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(tsr_edge_len);
    CTF_int::cdealloc(tsr_sym_table);
    for (i=0; i<2*num_ctr; i++){
      ctr_map[i].clear();
    }
    CTF_int::cdealloc(ctr_map);
    CTF_int::cdealloc(sub_phys_comm);
    CTF_int::cdealloc(comm_idx);

    TAU_FSTOP(map_ctr_indices);
    return stat;
  }

  /**
   * \brief map the indices over which we will not be contracting
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_noctr specification of which indices are not being contracted
   * \param num_tot total number of indices
   * \param num_noctr number of indices not being contracted over
   * \param topo topology to map to
   */
  static int
  map_no_ctr_indices(int const *      idx_arr,
                     int const *      idx_no_ctr,
                     int              num_tot,
                     int              num_no_ctr,
                     topology const * topo,
                     tensor *         A,
                     tensor *         B,
                     tensor *         C){
    int stat, i, inoctr, iA, iB, iC;

    TAU_FSTART(map_noctr_indices);

  /*  for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iC != -1 && iA != -1){
        copy_mapping(1, C->edge_map + iC, A->edge_map + iA); 
      } 
      if (iB != -1 && iA != -1){
        copy_mapping(1, C->edge_map + iB, A->edge_map + iA); 
      }
    }*/
    /* Map remainders of A and B to remainders of phys grid */
    stat = A->map_tensor_rem(topo->order,  topo->dim_comm, 1);
    if (stat != SUCCESS){
      if (A->order != 0 || B->order != 0 || C->order != 0){
        TAU_FSTOP(map_noctr_indices);
        return stat;
      }
    }
    for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iA != -1 && iC != -1){
        copy_mapping(1, A->edge_map + iA, C->edge_map + iC); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, B->edge_map + iB, C->edge_map + iC); 
      } 
    }
    stat = C->map_tensor_rem(topo->order,  topo->dim_comm, 0);
    if (stat != SUCCESS){
      TAU_FSTOP(map_noctr_indices);
      return stat;
    }
    for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iA != -1 && iC != -1){
        copy_mapping(1, C->edge_map + iC, A->edge_map + iA); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, C->edge_map + iC, B->edge_map + iB); 
      }
    }
    TAU_FSTOP(map_noctr_indices);

    return SUCCESS;
  }


  /**
   * \brief map the indices which are indexed only for A or B or C
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_extra specification of which indices are not being contracted
   * \param num_extra number of indices not being contracted over
   */
  static int
      map_extra_indices(int const * idx_arr,
                        int const * idx_extra,
                        int         num_extra,
                        tensor *    A,
                        tensor *    B,
                        tensor *    C){
    int i, iA, iB, iC, iextra;


    for (i=0; i<num_extra; i++){
      iextra = idx_extra[i];
      iA = idx_arr[3*iextra+0];
      iB = idx_arr[3*iextra+1];
      iC = idx_arr[3*iextra+2];

      if (iA != -1){
        //FIXME handle extra indices via reduction
        if (A->edge_map[iA].type == PHYSICAL_MAP)
          return NEGATIVE;
        if (A->edge_map[iA].type == NOT_MAPPED){
          A->edge_map[iA].type = VIRTUAL_MAP;
          A->edge_map[iA].np = 1;
          A->edge_map[iA].has_child = 0;
        }
      } else {
        if (iB != -1) {
          if (B->edge_map[iB].type == PHYSICAL_MAP)
            return NEGATIVE;
          if (B->edge_map[iB].type == NOT_MAPPED){
            B->edge_map[iB].type = VIRTUAL_MAP;
            B->edge_map[iB].np = 1;
            B->edge_map[iB].has_child = 0;
          }
        } else {
          ASSERT(iC != -1);
          if (C->edge_map[iC].type == PHYSICAL_MAP)
            return NEGATIVE;
          if (C->edge_map[iC].type == NOT_MAPPED){
            C->edge_map[iC].type = VIRTUAL_MAP;
            C->edge_map[iC].np = 1;
            C->edge_map[iC].has_child = 0;
          }
        }
      }
    }
    return SUCCESS;
  }


  int contraction::
      map_to_topology(topology * topo,
                      int        order){
      /*                int *      idx_ctr,
                      int *      idx_extra,
                      int *      idx_no_ctr,
                      int *      idx_weigh){*/
    int num_tot, num_ctr, num_no_ctr, num_weigh, num_extra, i, ret;
    int const * tidx_A, * tidx_B, * tidx_C;
    int * idx_arr, * idx_extra, * idx_no_ctr, * idx_weigh, * idx_ctr;

    tensor * tA, * tB, * tC;
    switch (order){
      case 0:
        tA = A;
        tB = B;
        tC = C;
        tidx_A = idx_A;
        tidx_B = idx_B;
        tidx_C = idx_C;
        break;
      case 1:
        tA = A;
        tB = C;
        tC = B;
        tidx_A = idx_A;
        tidx_B = idx_C;
        tidx_C = idx_B;
        break;
      case 2:
        tA = B;
        tB = A;
        tC = C;
        tidx_A = idx_B;
        tidx_B = idx_A;
        tidx_C = idx_C;
        break;
      case 3:
        tA = B;
        tB = C;
        tC = A;
        tidx_A = idx_B;
        tidx_B = idx_C;
        tidx_C = idx_A;
        break;
      case 4:
        tA = C;
        tB = A;
        tC = B;
        tidx_A = idx_C;
        tidx_B = idx_A;
        tidx_C = idx_B;
        break;
      case 5:
        tA = C;
        tB = B;
        tC = A;
        tidx_A = idx_C;
        tidx_B = idx_B;
        tidx_C = idx_A;
        break;
      default:
        return ERROR;
        break;
    }
   
    inv_idx(tA->order, tidx_A,
            tB->order, tidx_B,
            tC->order, tidx_C,
            &num_tot, &idx_arr);

    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_no_ctr);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_extra);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_weigh);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_ctr);
    num_ctr = 0, num_no_ctr = 0, num_extra = 0, num_weigh = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1 && idx_arr[3*i+2] != -1){
        idx_weigh[num_weigh] = i;
        num_weigh++;
      } else if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        idx_ctr[num_ctr] = i;
        num_ctr++;
      } else if (idx_arr[3*i+2] != -1 &&  
                  ((idx_arr[3*i+0] != -1) || (idx_arr[3*i+1] != -1))){
        idx_no_ctr[num_no_ctr] = i;
        num_no_ctr++;
      } else {
        idx_extra[num_extra] = i;
        num_extra++;
      }
    }
    tA->topo = topo;
    tB->topo = topo;
    tC->topo = topo;
    
    /* Map the weigh indices of A, B, and C*/
    ret = map_weigh_indices(idx_arr, idx_weigh, num_tot, num_weigh, topo, tA, tB, tC);
    int stat;
    do {
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }

      
      /* Map the contraction indices of A and B */
      ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, topo, tA, tB);
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }


    /*  ret = map_self_indices(tA, tidx_A);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }
      ret = map_self_indices(tB, tidx_B);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }
      ret = map_self_indices(tC, tidx_C);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }*/
      ret = map_extra_indices(idx_arr, idx_extra, num_extra, tA, tB, tC);
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }


      /* Map C or equivalently, the non-contraction indices of A and B */
      ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, topo, tA, tB, tC);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }
      ret = map_symtsr(tA->order, tA->sym_table, tA->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tB->order, tB->sym_table, tB->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tC->order, tC->sym_table, tC->edge_map);
      if (ret!=SUCCESS) return ret;

      /* Do it again to make sure everything is properly mapped. FIXME: loop */
      ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, topo, tA ,tB);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }
      ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, topo, tA, tB, tC);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }

      /*ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr,
                                tA, tB, topo);*/
      /* Map C or equivalently, the non-contraction indices of A and B */
      /*ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr,
                                    tA, tB, tC, topo);*/
      ret = map_symtsr(tA->order, tA->sym_table, tA->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tB->order, tB->sym_table, tB->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tC->order, tC->sym_table, tC->edge_map);
      if (ret!=SUCCESS) return ret;
      

      stat = SUCCESS;
    } while(0);

    cdealloc(idx_arr); cdealloc(idx_ctr); cdealloc(idx_extra); cdealloc(idx_no_ctr); cdealloc(idx_weigh);
    return stat;
  }

  int contraction::try_topo_morph(){
    topology * tA, * tB, * tC;
    int ret;
    tensor * tsr_keep, * tsr_change_A, * tsr_change_B;
    
    tA = A->topo;
    tB = B->topo;
    tC = C->topo;

    if (tA == tB && tB == tC){
      return SUCCESS;
    }

    if (tA->order >= tB->order){
      if (tA->order >= tC->order){
        tsr_keep = A;
        tsr_change_A = B;
        tsr_change_B = C;
      } else {
        tsr_keep = C;
        tsr_change_A = A;
        tsr_change_B = B;
      } 
    } else {
      if (tB->order >= tC->order){
        tsr_keep = B;
        tsr_change_A = A;
        tsr_change_B = C;
      } else {
        tsr_keep = C;
        tsr_change_A = A;
        tsr_change_B = B;
      }
    }
    
    tA = tsr_change_A->topo;
    tB = tsr_change_B->topo;
    tC = tsr_keep->topo;

    if (tA != tC){
      ret = can_morph(tC, tA);
      if (!ret)
        return NEGATIVE;
    }
    if (tB != tC){
      ret = can_morph(tC, tB);
      if (!ret)
        return NEGATIVE;
    }
    
    if (tA != tC){
      morph_topo(tC, tA,
                 tsr_change_A->order, tsr_change_A->edge_map);
      tsr_change_A->topo = tC;
    }
    if (tB != tC){
      morph_topo(tC, tB,
                 tsr_change_B->order, tsr_change_B->edge_map);
      tsr_change_B->topo = tC;
    }
    return SUCCESS;

  }

  int contraction::map(ctr ** ctrf, bool do_remap){
    int ret, j, need_remap, d;
    int need_remap_A, need_remap_B, need_remap_C;
    int64_t memuse;//, bmemuse;
    double est_time, best_time;
    int btopo;
    int64_t nvirt;
    //int * idx_arr, * idx_ctr, * idx_no_ctr, * idx_extra, * idx_weigh;
    int * old_phase_A, * old_phase_B, * old_phase_C;
    topology * old_topo_A, * old_topo_B, * old_topo_C;
    ctr * sctr;
    distribution * dA, * dB, * dC;

    ASSERT(A->wrld == B->wrld && B->wrld == C->wrld);
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;
    
    old_topo_A = NULL;
    old_topo_B = NULL;
    old_topo_C = NULL;

    TAU_FSTART(map_tensors);
  #if BEST_VOL
    CTF_int::alloc_ptr(sizeof(int)*A->order,     (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order,     (void**)&virt_blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order,     (void**)&virt_blk_len_C);
  #endif
    
    mapping * old_map_A = new mapping[A->order];
    mapping * old_map_B = new mapping[B->order];
    mapping * old_map_C = new mapping[C->order];
    copy_mapping(A->order, A->edge_map, old_map_A);
    copy_mapping(B->order, B->edge_map, old_map_B);
    copy_mapping(C->order, C->edge_map, old_map_C);
    old_topo_A = A->topo;
    old_topo_B = B->topo;
    old_topo_C = C->topo;
    if (do_remap){
      ASSERT(A->is_mapped);
      ASSERT(B->is_mapped);
      ASSERT(C->is_mapped);
    #if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("Initial mappings:\n");
      A->print_map();
      B->print_map();
      C->print_map();
    #endif
      A->unfold();
      B->unfold();
      C->unfold();
      A->set_padding();
      B->set_padding();
      C->set_padding();
      /* Save the current mappings of A, B, C */
      dA = new distribution(A);
      dB = new distribution(B);
      dC = new distribution(C);
      /*save_mapping(A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A,
                   &old_size_A, &was_cyclic_A, &old_padding_A,
                   &old_edge_len_A, A->topo);
      save_mapping(B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B,
                   &old_size_B, &was_cyclic_B, &old_padding_B,
                   &old_edge_len_B, B->topo);
      save_mapping(C, &old_phase_C, &old_rank_C, &old_virt_dim_C, &old_pe_lda_C,
                   &old_size_C, &was_cyclic_C, &old_padding_C,
                   &old_edge_len_C, C->topo);*/
    //} else {
    } else {
      dA = NULL;
      dB = NULL;
      dC = NULL;
    }
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&old_phase_A);
    for (j=0; j<A->order; j++){
      old_phase_A[j]   = A->edge_map[j].calc_phase();
    }
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&old_phase_B);
    for (j=0; j<B->order; j++){
      old_phase_B[j]   = B->edge_map[j].calc_phase();
    }
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&old_phase_C);
    for (j=0; j<C->order; j++){
      old_phase_C[j]   = C->edge_map[j].calc_phase();
    }
    //}
    btopo = -1;
    best_time = DBL_MAX;
    //bmemuse = UINT64_MAX;

    for (j=0; j<6; j++){
      /* Attempt to map to all possible permutations of processor topology */
  #if DEBUG < 3 
      for (int t=global_comm.rank; t<(int)wrld->topovec.size()+3; t+=global_comm.np){
  #else
      for (int t=global_comm.rank*(wrld->topovec.size()+3); t<(int)wrld->topovec.size()+3; t++){
  #endif
        A->clear_mapping();
        B->clear_mapping();
        C->clear_mapping();
        A->set_padding();
        B->set_padding();
        C->set_padding();
      
        topology * topo_i = NULL;
        if (t < 3){
          switch (t){
            case 0:
            if (old_topo_A == NULL) continue;
            topo_i = old_topo_A;
            copy_mapping(A->order, old_map_A, A->edge_map);
            break;
          
            case 1:
            if (old_topo_B == NULL) continue;
            topo_i = old_topo_B;
            copy_mapping(B->order, old_map_B, B->edge_map);
            break;

            case 2:
            if (old_topo_C == NULL) continue;
            topo_i = old_topo_C;
            copy_mapping(C->order, old_map_C, C->edge_map);
            break;
          }
        } else topo_i = wrld->topovec[t-3];
      

        ret = map_to_topology(topo_i, j);
        

        if (ret == ERROR) {
          TAU_FSTOP(map_tensors);
          return ERROR;
        }
        if (ret == NEGATIVE){
          //printf("map_to_topology returned negative\n");
          continue;
        }
    
        A->is_mapped = 1;
        B->is_mapped = 1;
        C->is_mapped = 1;
        A->topo = topo_i;
        B->topo = topo_i;
        C->topo = topo_i;
  #if DEBUG >= 4
        printf("\nTest mappings:\n");
        A->print_map(stdout, 0);
        B->print_map(stdout, 0);
        C->print_map(stdout, 0);
  #endif
        
        if (check_mapping() == 0) continue;
        est_time = 0.0;
        
  #if 0
        nvirt_all = -1;
        old_nvirt_all = -2;
        while (nvirt_all < MIN_NVIRT){
          old_nvirt_all = nvirt_all;
          A->set_padding();
          B->set_padding();
          C->set_padding();
          sctr = construct_contraction(type, buffer, buffer_len, func_ptr,
                                        alpha, beta, 0, NULL, &nvirt_all);
          /* If this cannot be stretched */
          if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
            A->clear_mapping();
            B->clear_mapping();
            C->clear_mapping();
            A->set_padding();
            B->set_padding();
            C->set_padding();

            ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, idx_A,
                                  idx_B, idx_C, i, j,
                                  idx_arr, idx_ctr, idx_extra, idx_no_ctr);
            A->is_mapped = 1;
            B->is_mapped = 1;
            C->is_mapped = 1;
            A->itopo = i;
            B->itopo = i;
            C->itopo = i;
            break;

          }
          if (nvirt_all < MIN_NVIRT){
            stretch_virt(A->order, 2, A->edge_map);
            stretch_virt(B->order, 2, B->edge_map);
            stretch_virt(C->order, 2, C->edge_map);
          }
        }
  #endif
        A->set_padding();
        B->set_padding();
        C->set_padding();
        sctr = construct_ctr();
       
        est_time = sctr->est_time_rec(sctr->num_lyr);
        //sctr->print();
  #if DEBUG >= 4
        printf("mapping passed contr est_time = %E sec\n", est_time);
  #endif 
        ASSERT(est_time > 0.0);
        memuse = 0;
        need_remap_A = 0;
        need_remap_B = 0;
        need_remap_C = 0;
        if (topo_i == old_topo_A){
          for (d=0; d<A->order; d++){
            if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
              need_remap_A = 1;
          }
        } else
          need_remap_A = 1;
        if (need_remap_A) {
          nvirt = (int64_t)A->calc_nvirt();
          est_time += global_comm.estimate_alltoallv_time(A->sr->el_size*A->size);
          if (can_block_reshuffle(A->order, old_phase_A, A->edge_map)){
            memuse = MAX(memuse,(int64_t)A->sr->el_size*A->size);
          } else {
            est_time += 5.*COST_MEMBW*A->sr->el_size*A->size+global_comm.estimate_alltoall_time(1);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*A->sr->el_size*A->size;
            memuse = MAX(memuse,(int64_t)A->sr->el_size*A->size*2.5);
          }
        } else
          memuse = 0;
        if (topo_i == old_topo_B){
          for (d=0; d<B->order; d++){
            if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
              need_remap_B = 1;
          }
        } else
          need_remap_B = 1;
        if (need_remap_B) {
          nvirt = (int64_t)B->calc_nvirt();
          est_time += global_comm.estimate_alltoallv_time(B->sr->el_size*B->size);
          if (can_block_reshuffle(B->order, old_phase_B, B->edge_map)){
            memuse = MAX(memuse,(int64_t)B->sr->el_size*B->size);
          } else {
            est_time += 5.*COST_MEMBW*B->sr->el_size*B->size+global_comm.estimate_alltoall_time(1);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*B->sr->el_size*B->size;
            memuse = MAX(memuse,(int64_t)B->sr->el_size*B->size*2.5);
          }
        }
        if (topo_i == old_topo_C){
          for (d=0; d<C->order; d++){
            if (!comp_dim_map(&C->edge_map[d],&old_map_C[d]))
              need_remap_C = 1;
          }
        } else
          need_remap_C = 1;
        if (need_remap_C) {
          nvirt = (int64_t)C->calc_nvirt();
          est_time += global_comm.estimate_alltoallv_time(B->sr->el_size*B->size);
          if (can_block_reshuffle(C->order, old_phase_C, C->edge_map)){
            memuse = MAX(memuse,(int64_t)C->sr->el_size*C->size);
          } else {
            est_time += 5.*COST_MEMBW*C->sr->el_size*C->size+global_comm.estimate_alltoall_time(1);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*C->sr->el_size*C->size;
            memuse = MAX(memuse,(int64_t)C->sr->el_size*C->size*2.5);
          }
        }
        memuse = MAX((int64_t)sctr->mem_rec(), memuse);
  #if DEBUG >= 4
        printf("total (with redistribution) est_time = %E\n", est_time);
  #endif
        ASSERT(est_time > 0.0);

        if ((int64_t)memuse >= proc_bytes_available()){
          DPRINTF(2,"Not enough memory available for topo %d with order %d\n", t, j);
          delete sctr;
          continue;
        } 

        /* be careful about overflow */
  /*      nvirt = (int64_t)A->calc_nvirt();
        tnvirt = nvirt*(int64_t)B->calc_nvirt();
        if (tnvirt < nvirt) nvirt = UINT64_MAX;
        else {
          nvirt = tnvirt;
          tnvirt = nvirt*(int64_t)C->calc_nvirt();
          if (tnvirt < nvirt) nvirt = UINT64_MAX;
          else nvirt = tnvirt;
        }*/
        //if (btopo == -1 || (nvirt < bnvirt  || 
        //((nvirt == bnvirt || nvirt <= ALLOW_NVIRT) && est_time < best_time))) {
        if (est_time < best_time) {
          best_time = est_time;
          //bmemuse = memuse;
          btopo = 6*t+j;      
        }  
        delete sctr;
  /*#else
    #if BEST_COMM
        est_time = sctr->comm_rec(sctr->num_lyr);
        if (est_time < best_time){
          best_time = est_time;
          btopo = 6*i+j;
        }
    #endif
  #endif*/
      }
    }
  #if DEBUG>=3
    MPI_Barrier(A->wrld->comm);
  #endif
  /*#if BEST_VOL
    ALLREDUCE(&bnvirt, &gnvirt, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, global_comm);
    if (bnvirt != gnvirt){
      btopo = INT_MAX;
    }
    ALLREDUCE(&btopo, &gtopo, 1, MPI_INT, MPI_MIN, global_comm);
  #endif
  #if BEST_VIRT
    if (btopo == -1){
      bnvirt = UINT64_MAX;
      btopo = INT_MAX;
    }
    DEBUG_PRINTF("bnvirt = " PRIu64 "\n", (int64_t)bnvirt);
    // pick lower dimensional mappings, if equivalent 
  #if BEST_COMM
    if (bnvirt >= ALLOW_NVIRT)
      gtopo = get_best_topo(bnvirt+1-ALLOW_NVIRT, btopo, global_comm, best_time, bmemuse);
    else
      gtopo = get_best_topo(1, btopo, global_comm, best_time, bmemuse);
  #else
    gtopo = get_best_topo(bnvirt, btopo, global_comm);
  #endif
  #endif*/
    double gbest_time;
    MPI_Allreduce(&best_time, &gbest_time, 1, MPI_DOUBLE, MPI_MIN, global_comm.cm);
    if (best_time != gbest_time){
      btopo = INT_MAX;
    }
    int ttopo;
    MPI_Allreduce(&btopo, &ttopo, 1, MPI_INT, MPI_MIN, global_comm.cm);


    A->clear_mapping();
    B->clear_mapping();
    C->clear_mapping();
    A->set_padding();
    B->set_padding();
    C->set_padding();
    
    if (!do_remap || ttopo == INT_MAX || ttopo == -1){
      CTF_int::cdealloc(old_phase_A);
      CTF_int::cdealloc(old_phase_B);
      CTF_int::cdealloc(old_phase_C);
      delete [] old_map_A;
      delete [] old_map_B;
      delete [] old_map_C;
/*      for (i=0; i<A->order; i++)
        old_map_A[i].clear();
      for (i=0; i<B->order; i++)
        old_map_B[i].clear();
      for (i=0; i<C->order; i++)
        old_map_C[i].clear();
      CTF_int::cdealloc(old_map_A);
      CTF_int::cdealloc(old_map_B);
      CTF_int::cdealloc(old_map_C);
*/
      TAU_FSTOP(map_tensors);
      if (ttopo == INT_MAX || ttopo == -1){
        printf("ERROR: Failed to map contraction!\n");
        //ABORT;
        return ERROR;
      }
      return SUCCESS;
    }
    topology * topo_g;
    int j_g = ttopo%6;
    if (ttopo < 18){
      switch (ttopo/6){
        case 0:
        topo_g = old_topo_A;
        copy_mapping(A->order, old_map_A, A->edge_map);
        break;
      
        case 1:
        topo_g = old_topo_B;
        copy_mapping(B->order, old_map_B, B->edge_map);
        break;

        case 2:
        topo_g = old_topo_C;
        copy_mapping(C->order, old_map_C, C->edge_map);
        break;
        
        default:
        topo_g = NULL;
        assert(0);
        break;
      }
    } else topo_g = wrld->topovec[(ttopo-18)/6];
   

    A->topo = topo_g;
    B->topo = topo_g;
    C->topo = topo_g;
    
    ret = map_to_topology(topo_g, j_g);


    if (ret == NEGATIVE || ret == ERROR) {
      printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
      TAU_FSTOP(map_tensors);
      return ERROR;
    }
    A->is_mapped = 1;
    B->is_mapped = 1;
    C->is_mapped = 1;
  #if DEBUG > 2
    if (!check_mapping())
      printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
  //  else if (global_comm.rank == 0) printf("Mapping successful estimated execution time = %lf sec\n",best_time);
  #endif
    ASSERT(check_mapping());
  
#if 0
    nvirt_all = -1;
    old_nvirt_all = -2;
    while (nvirt_all < MIN_NVIRT){
      old_nvirt_all = nvirt_all;
      A->set_padding();
      B->set_padding();
      C->set_padding();
      *ctrf = construct_ctr();
      delete *ctrf;
      /* If this cannot be stretched */
      if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
        A->clear_mapping();
        B->clear_mapping();
        C->clear_mapping();
        A->set_padding();
        B->set_padding();
        C->set_padding();
        A->topo = topo_g;
        B->topo = topo_g;
        C->topo = topo_g;

          ret = map_to_topology(topo_g, j_g);
          A->is_mapped = 1;
          B->is_mapped = 1;
          C->is_mapped = 1;
          break;
        }
        if (nvirt_all < MIN_NVIRT){
          stretch_virt(A->order, 2, A->edge_map);
          stretch_virt(B->order, 2, B->edge_map);
          stretch_virt(C->order, 2, C->edge_map);
        }
      }
#endif
      A->set_padding();
      B->set_padding();
      C->set_padding();
      *ctrf = construct_ctr();
    #if DEBUG > 2
      if (global_comm.rank == 0)
        printf("New mappings:\n");
      A->print_map(stdout);
      B->print_map(stdout);
      C->print_map(stdout);
    #endif
     
         
      //FIXME: adhoc? 
      memuse = MAX((int64_t)(*ctrf)->mem_rec(), (int64_t)(A->size*A->sr->el_size+B->size*B->sr->el_size+C->size*C->sr->el_size)*3);
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      VPRINTF(1,"Contraction will use %E bytes per processor out of %E available memory and take an estimated of %lf sec\n",
              (double)memuse,(double)proc_bytes_available(),gbest_time);
  #endif          

    if (A->is_cyclic == 0 &&
        B->is_cyclic == 0 &&
        C->is_cyclic == 0){
      A->is_cyclic = 0;
      B->is_cyclic = 0;
      C->is_cyclic = 0;
    } else {
      A->is_cyclic = 1;
      B->is_cyclic = 1;
      C->is_cyclic = 1;
    }
    TAU_FSTOP(map_tensors);
    /* redistribute tensor data */
    TAU_FSTART(redistribute_for_contraction);
    need_remap = 0;
    if (A->topo == old_topo_A){
      for (d=0; d<A->order; d++){
        if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      A->redistribute(*dA);
    need_remap = 0;
    if (B->topo == old_topo_B){
      for (d=0; d<B->order; d++){
        if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      B->redistribute(*dB);
    need_remap = 0;
    if (C->topo == old_topo_C){
      for (d=0; d<C->order; d++){
        if (!comp_dim_map(&C->edge_map[d],&old_map_C[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      C->redistribute(*dC);
                   
    TAU_FSTOP(redistribute_for_contraction);
    
    (*ctrf)->A    = A->data;
    (*ctrf)->B    = B->data;
    (*ctrf)->C    = C->data;

    CTF_int::cdealloc( old_phase_A );
    CTF_int::cdealloc( old_phase_B );
    CTF_int::cdealloc( old_phase_C );
    
    delete [] old_map_A;
    delete [] old_map_B;
    delete [] old_map_C;

    
    delete dA;
    delete dB;
    delete dC;

    return SUCCESS;
  }


  ctr * contraction::construct_ctr(int            is_inner,
                                   iparam const * inner_params,
                                   int *          nvirt_all,
                                   int            is_used){
    int num_tot, i, i_A, i_B, i_C, is_top, j, nphys_dim,  k;
    int64_t nvirt;
    int64_t blk_sz_A, blk_sz_B, blk_sz_C;
    int64_t vrt_sz_A, vrt_sz_B, vrt_sz_C;
    int sA, sB, sC, need_rep;
    int * blk_len_A, * virt_blk_len_A, * blk_len_B;
    int * virt_blk_len_B, * blk_len_C, * virt_blk_len_C;
    int * idx_arr, * virt_dim, * phys_mapped;
    strp_tsr * str_A, * str_B, * str_C;
    mapping * map;
    ctr * hctr = NULL;
    ctr ** rec_ctr = NULL;
    ASSERT(A->wrld == B->wrld && B->wrld == C->wrld);
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;

    TAU_FSTART(construct_contraction);
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    nphys_dim = A->topo->order;

    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&virt_blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&virt_blk_len_C);

    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&blk_len_C);
    CTF_int::alloc_ptr(sizeof(int)*num_tot, (void**)&virt_dim);
    CTF_int::alloc_ptr(sizeof(int)*nphys_dim*3, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*3);


    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = A->size;
    blk_sz_B = B->size;
    blk_sz_C = C->size;
    calc_dim(A->order, blk_sz_A, A->pad_edge_len, A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(B->order, blk_sz_B, B->pad_edge_len, B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);
    calc_dim(C->order, blk_sz_C, C->pad_edge_len, C->edge_map,
             &vrt_sz_C, virt_blk_len_C, blk_len_C);

    /* Strip out the relevant part of the tensor if we are contracting over diagonal */
    sA = strip_diag( A->order, num_tot, idx_A, vrt_sz_A,
                     A->edge_map, A->topo, A->sr,
                     blk_len_A, &blk_sz_A, &str_A);
    sB = strip_diag( B->order, num_tot, idx_B, vrt_sz_B,
                     B->edge_map, B->topo, B->sr,
                     blk_len_B, &blk_sz_B, &str_B);
    sC = strip_diag( C->order, num_tot, idx_C, vrt_sz_C,
                     C->edge_map, C->topo, C->sr,
                     blk_len_C, &blk_sz_C, &str_C);

    is_top = 1;
    if (sA || sB || sC){
      ASSERT(0);//this is always done via sum now
      if (global_comm.rank == 0)
        DPRINTF(1,"Stripping tensor\n");
      strp_ctr * sctr = new strp_ctr;
      hctr = sctr;
      hctr->num_lyr = 1;
      hctr->idx_lyr = 0;
      is_top = 0;
      rec_ctr = &sctr->rec_ctr;

      sctr->rec_strp_A = str_A;
      sctr->rec_strp_B = str_B;
      sctr->rec_strp_C = str_C;
      sctr->strip_A = sA;
      sctr->strip_B = sB;
      sctr->strip_C = sC;
    }

    for (i=0; i<A->order; i++){
      map = &A->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+0] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+0] = 1;
        }
      }
    }
    for (i=0; i<B->order; i++){
      map = &B->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+1] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+1] = 1;
        }
      }
    }
    for (i=0; i<C->order; i++){
      map = &C->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+2] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+2] = 1;
        }
      }
    }
    need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[3*i+0] == 0 ||
        phys_mapped[3*i+1] == 0 ||
        phys_mapped[3*i+2] == 0){
        /*ASSERT((phys_mapped[3*i+0] == 0 && phys_mapped[3*i+1] == 0) ||
        (phys_mapped[3*i+0] == 0 && phys_mapped[3*i+2] == 0) ||
        (phys_mapped[3*i+1] == 0 && phys_mapped[3*i+2] == 0));*/
        need_rep = 1;
        break;
      }
    }
    if (need_rep){
      if (global_comm.rank == 0)
        DPRINTF(1,"Replicating tensor\n");

      ctr_replicate * rctr = new ctr_replicate;
      if (is_top){
        hctr = rctr;
        is_top = 0;
      } else {
        *rec_ctr = rctr;
      }
      rec_ctr = &rctr->rec_ctr;
      hctr->idx_lyr = 0;
      hctr->num_lyr = 1;
      rctr->idx_lyr = 0;
      rctr->num_lyr = 1;
      rctr->ncdt_A = 0;
      rctr->ncdt_B = 0;
      rctr->ncdt_C = 0;
      rctr->size_A = blk_sz_A;
      rctr->size_B = blk_sz_B;
      rctr->size_C = blk_sz_C;
      rctr->cdt_A = NULL;
      rctr->cdt_B = NULL;
      rctr->cdt_C = NULL;
      rctr->sr_A = A->sr;
      rctr->sr_B = B->sr;
      rctr->sr_C = C->sr;
      for (i=0; i<nphys_dim; i++){
        if (phys_mapped[3*i+0] == 0 &&
            phys_mapped[3*i+1] == 0 &&
            phys_mapped[3*i+2] == 0){
  /*        printf("ERROR: ALL-TENSOR REPLICATION NO LONGER DONE\n");
          ABORT;
          ASSERT(rctr->num_lyr == 1);
          hctr->idx_lyr = A->topo->dim_comm[i].rank;
          hctr->num_lyr = A->topo->dim_comm[i]->np;
          rctr->idx_lyr = A->topo->dim_comm[i].rank;
          rctr->num_lyr = A->topo->dim_comm[i]->np;*/
        } else {
          if (phys_mapped[3*i+0] == 0){
            rctr->ncdt_A++;
          }
          if (phys_mapped[3*i+1] == 0){
            rctr->ncdt_B++;
          }
          if (phys_mapped[3*i+2] == 0){
            rctr->ncdt_C++;
          }
        }
      }
      if (rctr->ncdt_A > 0)
        CTF_int::alloc_ptr(sizeof(CommData*)*rctr->ncdt_A, (void**)&rctr->cdt_A);
      if (rctr->ncdt_B > 0)
        CTF_int::alloc_ptr(sizeof(CommData*)*rctr->ncdt_B, (void**)&rctr->cdt_B);
      if (rctr->ncdt_C > 0)
        CTF_int::alloc_ptr(sizeof(CommData*)*rctr->ncdt_C, (void**)&rctr->cdt_C);
      rctr->ncdt_A = 0;
      rctr->ncdt_B = 0;
      rctr->ncdt_C = 0;
      for (i=0; i<nphys_dim; i++){
        if (!(phys_mapped[3*i+0] == 0 &&
              phys_mapped[3*i+1] == 0 &&
              phys_mapped[3*i+2] == 0)){
          if (phys_mapped[3*i+0] == 0){
            rctr->cdt_A[rctr->ncdt_A] = &A->topo->dim_comm[i];
        /*    if (is_used && rctr->cdt_A[rctr->ncdt_A].alive == 0)
              rctr->cdt_A[rctr->ncdt_A].activate(global_comm.cm);*/
            rctr->ncdt_A++;
          }
          if (phys_mapped[3*i+1] == 0){
            rctr->cdt_B[rctr->ncdt_B] = &B->topo->dim_comm[i];
/*            if (is_used && rctr->cdt_B[rctr->ncdt_B].alive == 0)
              rctr->cdt_B[rctr->ncdt_B].activate(global_comm.cm);*/
            rctr->ncdt_B++;
          }
          if (phys_mapped[3*i+2] == 0){
            rctr->cdt_C[rctr->ncdt_C] = &C->topo->dim_comm[i];
/*            if (is_used && rctr->cdt_C[rctr->ncdt_C].alive == 0)
              rctr->cdt_C[rctr->ncdt_C].activate(global_comm.cm);*/
            rctr->ncdt_C++;
          }
        }
      }
    }

  //#ifdef OFFLOAD
    int total_iter = 1;
    int upload_phase_A = 1;
    int upload_phase_B = 1;
    int download_phase_C = 1;
  //#endif
    nvirt = 1;

    ctr_2d_general * bottom_ctr_gen = NULL;
  /*  if (nvirt_all != NULL)
      *nvirt_all = 1;*/
    for (i=0; i<num_tot; i++){
      virt_dim[i] = 1;
      i_A = idx_arr[3*i+0];
      i_B = idx_arr[3*i+1];
      i_C = idx_arr[3*i+2];
      /* If this index belongs to exactly two tensors */
      if ((i_A != -1 && i_B != -1 && i_C == -1) ||
          (i_A != -1 && i_B == -1 && i_C != -1) ||
          (i_A == -1 && i_B != -1 && i_C != -1)) {
        ctr_2d_general * ctr_gen = new ctr_2d_general;
  #ifdef OFFLOAD
        ctr_gen->alloc_host_buf = false;
  #endif
        int is_built = 0;
        if (i_A == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C);
        }
        if (i_B == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A);
        }
        if (i_C == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B);
        }
        ctr_gen->sr_A = A->sr;
        ctr_gen->sr_B = B->sr;
        ctr_gen->sr_C = C->sr;
        if (is_built){
          if (is_top){
            hctr = ctr_gen;
            hctr->idx_lyr = 0;
            hctr->num_lyr = 1;
            is_top = 0;
          } else {
            *rec_ctr = ctr_gen;
          }
          if (bottom_ctr_gen == NULL)
            bottom_ctr_gen = ctr_gen;
          rec_ctr = &ctr_gen->rec_ctr;
        } else {
          ctr_gen->rec_ctr = NULL;
          delete ctr_gen;
        }
      } else {
        if (i_A != -1){
          map = &A->edge_map[i_A];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_B != -1){
          map = &B->edge_map[i_B];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_C != -1){
          map = &C->edge_map[i_C];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        }
      }
      if (sA && i_A != -1){
        nvirt = virt_dim[i]/str_A->strip_dim[i_A];
      } else if (sB && i_B != -1){
        nvirt = virt_dim[i]/str_B->strip_dim[i_B];
      } else if (sC && i_C != -1){
        nvirt = virt_dim[i]/str_C->strip_dim[i_C];
      }
      
      nvirt = nvirt * virt_dim[i];
    }
    if (nvirt_all != NULL)
      *nvirt_all = nvirt;

    ASSERT(blk_sz_A >= vrt_sz_A);
    ASSERT(blk_sz_B >= vrt_sz_B);
    ASSERT(blk_sz_C >= vrt_sz_C);
      
    int * new_sym_A, * new_sym_B, * new_sym_C;
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, A->sym, sizeof(int)*A->order);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, B->sym, sizeof(int)*B->order);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&new_sym_C);
    memcpy(new_sym_C, C->sym, sizeof(int)*C->order);

  #ifdef OFFLOAD
    if (ftsr.is_offloadable || is_inner > 0){
      if (bottom_ctr_gen != NULL)
        bottom_ctr_gen->alloc_host_buf = true;
      ctr_offload * ctroff = new ctr_offload;
      if (is_top){
        hctr = ctroff;
        hctr->idx_lyr = 0;
        hctr->num_lyr = 0;
        is_top = 0;
      } else {
        *rec_ctr = ctroff;
      }
      rec_ctr = &ctroff->rec_ctr;

      ctroff->size_A = blk_sz_A;
      ctroff->size_B = blk_sz_B;
      ctroff->size_C = blk_sz_C;
      ctroff->total_iter = total_iter;
      ctroff->upload_phase_A = upload_phase_A;
      ctroff->upload_phase_B = upload_phase_B;
      ctroff->download_phase_C = download_phase_C;
    }
  #endif

    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
  #ifdef USE_VIRT_25D
      ctr_virt_25d * ctrv = new ctr_virt_25d;
  #else
      ctr_virt * ctrv = new ctr_virt;
  #endif
      if (is_top) {
        hctr = ctrv;
        hctr->idx_lyr = 0;
        hctr->num_lyr = 1;
        is_top = 0;
      } else {
        *rec_ctr = ctrv;
      }
      rec_ctr = &ctrv->rec_ctr;
      ctrv->sr_A = A->sr;
      ctrv->sr_B = B->sr;
      ctrv->sr_C = C->sr;

      ctrv->num_dim   = num_tot;
      ctrv->virt_dim  = virt_dim;
      ctrv->order_A   = A->order;
      ctrv->blk_sz_A  = vrt_sz_A;
      ctrv->idx_map_A = idx_A;
      ctrv->order_B   = B->order;
      ctrv->blk_sz_B  = vrt_sz_B;
      ctrv->idx_map_B = idx_B;
      ctrv->order_C   = C->order;
      ctrv->blk_sz_C  = vrt_sz_C;
      ctrv->idx_map_C = idx_C;
    } else
      CTF_int::cdealloc(virt_dim);

    seq_tsr_ctr * ctrseq = new seq_tsr_ctr;
    ctrseq->sr_A = A->sr;
    ctrseq->sr_B = B->sr;
    ctrseq->sr_C = C->sr;
    if (is_top) {
      hctr = ctrseq;
      hctr->idx_lyr = 0;
      hctr->num_lyr = 1;
      is_top = 0;
    } else {
      *rec_ctr = ctrseq;
    }
    if (!is_inner){
      ctrseq->is_inner  = 0;
    } else if (is_inner == 1) {
      if (global_comm.rank == 0){
        DPRINTF(1,"Folded tensor n=%d m=%d k=%d\n", inner_params->n,
          inner_params->m, inner_params->k);
      }

      ctrseq->is_inner    = 1;
      ctrseq->inner_params  = *inner_params;
      ctrseq->inner_params.sz_C = vrt_sz_C;
      tensor * itsr;
      itsr = A->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = A->inner_ordering[i];
        for (k=0; k<A->order; k++){
          if (A->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && A->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
  /*        printf("inner_ordering[%d]=%d setting dim %d of A, to len %d from len %d\n",
                  i, A->inner_ordering[i], k, 1, virt_blk_len_A[k]);*/
          virt_blk_len_A[k] = 1;
          new_sym_A[k] = NS;
        }
      }
      itsr = B->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = B->inner_ordering[i];
        for (k=0; k<B->order; k++){
          if (B->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && B->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of B, to len %d from len %d\n",
                  i, B->inner_ordering[i], k, 1, virt_blk_len_B[k]);*/
          virt_blk_len_B[k] = 1;
          new_sym_B[k] = NS;
        }
      }
      itsr = C->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = C->inner_ordering[i];
        for (k=0; k<C->order; k++){
          if (C->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && C->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of C, to len %d from len %d\n",
                  i, C->inner_ordering[i], k, 1, virt_blk_len_C[k]);*/
          virt_blk_len_C[k] = 1;
          new_sym_C[k] = NS;
        }
      }
    }
    ctrseq->is_custom  = is_custom;
    ctrseq->alpha      = alpha;
    if (is_custom){
      ctrseq->func     = func;
    }
    ctrseq->order_A    = A->order;
    ctrseq->idx_map_A  = idx_A;
    ctrseq->edge_len_A = virt_blk_len_A;
    ctrseq->sym_A      = new_sym_A;
    ctrseq->order_B    = B->order;
    ctrseq->idx_map_B  = idx_B;
    ctrseq->edge_len_B = virt_blk_len_B;
    ctrseq->sym_B      = new_sym_B;
    ctrseq->order_C    = C->order;
    ctrseq->idx_map_C  = idx_C;
    ctrseq->edge_len_C = virt_blk_len_C;
    ctrseq->sym_C      = new_sym_C;

    hctr->beta = this->beta;
    C->sr->isequal(hctr->beta, C->sr->mulid());
    hctr->A    = A->data;
    hctr->B    = B->data;
    hctr->C    = C->data;
  /*  if (global_comm.rank == 0){
      int64_t n,m,k;
      dtype old_flops;
      dtype new_flops;
      ggg_sym_nmk(A->order, A->pad_edge_len, idx_A, A->sym,
      B->order, B->pad_edge_len, idx_B, B->sym,
      C->order, &n, &m, &k);
      old_flops = 2.0*(dtype)n*(dtype)m*(dtype)k;
      new_flops = A->calc_nvirt();
      new_flops *= B->calc_nvirt();
      new_flops *= C->calc_nvirt();
      new_flops *= global_comm.np;
      new_flops = sqrt(new_flops);
      new_flops *= global_comm.np;
      ggg_sym_nmk(A->order, virt_blk_len_A, idx_A, A->sym,
      B->order, virt_blk_len_B, idx_B, B->sym,
      C->order, &n, &m, &k);
      printf("Each subcontraction is a " PRId64 " by " PRId64 " by " PRId64 " DGEMM performing %E flops\n",n,m,k,
        2.0*(dtype)n*(dtype)m*(dtype)k);
      new_flops *= 2.0*(dtype)n*(dtype)m*(dtype)k;
      printf("Contraction performing %E flops rather than %E, a factor of %lf more flops due to padding\n",
        new_flops, old_flops, new_flops/old_flops);

    }*/

    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(blk_len_A);
    CTF_int::cdealloc(blk_len_B);
    CTF_int::cdealloc(blk_len_C);
    CTF_int::cdealloc(phys_mapped);
    TAU_FSTOP(construct_contraction);
    return hctr;
  }

  int contraction::contract(){
    int stat;
    ctr * ctrf;
    CommData global_comm = C->wrld->cdt;

    if (A->has_zero_edge_len || B->has_zero_edge_len
        || C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }
    //FIXME: create these tensors without home
    if (A == B || A == C){
      tensor * new_tsr = new tensor(A);
      contraction new_ctr = contraction(*this);
      new_ctr.A = new_tsr;
      stat = new_ctr.contract();
      delete new_tsr;
      return stat;
    }
    if (B == C){
      tensor * new_tsr = new tensor(B);
      contraction new_ctr = contraction(*this);
      new_ctr.B = new_tsr;
      stat = new_ctr.contract();
      delete new_tsr;
      return stat;
    }
  #if DEBUG >= 1 //|| VERBOSE >= 1)
    if (global_comm.rank == 0)
      printf("Contraction permutation:\n");
    print();
  #endif

    TAU_FSTART(contract);
  #if 0 //VERIFY
    int64_t nsA, nsB;
    int64_t nA, nB, nC, up_nC;
    dtype * sA, * sB, * ans_C;
    dtype * uA, * uB, * uC;
    dtype * up_C, * up_ans_C, * pup_C;
    int order_A, order_B, order_C, i, pass;
    int * edge_len_A, * edge_len_B, * edge_len_C;
    int * sym_A, * sym_B, * sym_C;
    int * sym_tmp;
    stat = allread_tsr(type->tid_A, &nsA, &sA);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_B, &nsB, &sB);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_C, &nC, &ans_C);
    assert(stat == SUCCESS);
  #endif
    /* Check if the current tensor mappings can be contracted on */
    /*fseq_tsr_ctr fftsr=ftsr;
    if (ftsr.func_ptr == NULL){
      fftsr.func_ptr = &sym_seq_ctr_ref;
  #ifdef OFFLOAD
      fftsr.is_offloadable = 0;
  #endif
    }*/
  #if REDIST
    //stat = map_tensors(type, fftsr, felm, alpha, beta, &ctrf);
    stat = map(&ctrf);
    if (stat == ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return ERROR;
    }
  #else
    if (check_mapping() == 0) {
      /* remap if necessary */
      stat = map(&ctrf);
      if (stat == ERROR) {
        printf("Failed to map tensors to physical grid\n");
        return ERROR;
      }
    } else {
      /* Construct the tensor algorithm we would like to use */
  #if DEBUG > 2
      if (global_comm.rank == 0)
        printf("Keeping mappings:\n");
      A->print_map(stdout);
      B->print_map(stdout);
      C->print_map(stdout);
  #endif
      ctrf = construct_ctr();
  #ifdef VERBOSE
      if (global_comm.rank == 0){
        int64_t memuse = ctrf->mem_rec();
        DPRINTF(1,"Contraction does not require redistribution, will use %E bytes per processor out of %E available memory and take an estimated of %E sec\n",
                (double)memuse,(double)proc_bytes_available(),ctrf->est_time_rec(1));
      }
  #endif
    }
  #endif
    ASSERT(check_mapping());
  #if FOLD_TSR
    if (!is_custom && can_fold()){
      iparam prm;
      TAU_FSTART(map_fold);
      prm = map_fold();
      TAU_FSTOP(map_fold);
      delete ctrf;
      ctrf = construct_ctr(1, &prm);
    } 
  #endif
  #if DEBUG >=2
    if (global_comm.rank == 0)
      ctrf->print();
  #endif
  #ifdef DEBUG
    double dtt = MPI_Wtime();
    if (global_comm.rank == 0){
      DPRINTF(1,"[%d] performing contraction\n",
          global_comm.rank);
      DPRINTF(1,"%E bytes of buffer space will be needed for this contraction\n",
        (double)ctrf->mem_rec());
      DPRINTF(1,"System memory = %E bytes total, %E bytes used, %E bytes available.\n",
        (double)proc_bytes_total(),
        (double)proc_bytes_used(),
        (double)proc_bytes_available());
    }
  #endif
  #if DEBUG >=2
    A->print_map();
    B->print_map();
    C->print_map();
  #endif
  //  stat = zero_out_padding(type->tid_A);
  //  stat = zero_out_padding(type->tid_B);
    TAU_FSTART(ctr_func);
    /* Invoke the contraction algorithm */
    A->topo->activate();
    ctrf->run();
    A->topo->deactivate();

    TAU_FSTOP(ctr_func);
  #ifndef SEQ
    if (C->is_cyclic)
      stat = C->zero_out_padding();
  #endif
    A->unfold();
    B->unfold();
    if (A->wrld->rank == 0){
      DPRINTF(1, "Contraction permutation completed in %lf sec.\n",MPI_Wtime()-dtt);
    }


  #if 0 //VERIFY
    stat = allread_tsr(type->tid_A, &nA, &uA);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_A, &order_A, &edge_len_A, &sym_A);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_B, &nB, &uB);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_B, &order_B, &edge_len_B, &sym_B);
    assert(stat == SUCCESS);

    if (nsA != nA) { printf("nsA = " PRId64 ", nA = " PRId64 "\n",nsA,nA); ABORT; }
    if (nsB != nB) { printf("nsB = " PRId64 ", nB = " PRId64 "\n",nsB,nB); ABORT; }
    for (i=0; (int64_t)i<nA; i++){
      if (fabs(uA[i] - sA[i]) > 1.E-6){
        printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
      }
    }
    for (i=0; (int64_t)i<nB; i++){
      if (fabs(uB[i] - sB[i]) > 1.E-6){
        printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
      }
    }

    stat = allread_tsr(type->tid_C, &nC, &uC);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_C, &order_C, &edge_len_C, &sym_C);
    assert(stat == SUCCESS);
    DEBUG_PRINTF("packed size of C is " PRId64 " (should be " PRId64 ")\n", nC,
      sy_packed_size(order_C, edge_len_C, sym_C));

    pup_C = (dtype*)CTF_int::alloc(nC*sizeof(dtype));

    cpy_sym_ctr(alpha,
          uA, order_A, edge_len_A, edge_len_A, sym_A, idx_A,
          uB, order_B, edge_len_B, edge_len_B, sym_B, idx_B,
          beta,
      ans_C, order_C, edge_len_C, edge_len_C, sym_C, idx_C);
    assert(stat == SUCCESS);

  #if ( DEBUG>=5)
    for (i=0; i<nC; i++){
  //    if (fabs(C[i]-ans_C[i]) > 1.E-6){
        printf("PACKED: C[%d] = %lf, ans_C[%d] = %lf\n",
         i, C[i], i, ans_C[i]);
  //     }
    }
  #endif

    punpack_tsr(uC, order_C, edge_len_C,
          sym_C, 1, &sym_tmp, &up_C);
    punpack_tsr(ans_C, order_C, edge_len_C,
          sym_C, 1, &sym_tmp, &up_ans_C);
    punpack_tsr(up_ans_C, order_C, edge_len_C,
          sym_C, 0, &sym_tmp, &pup_C);
    for (i=0; (int64_t)i<nC; i++){
      assert(fabs(pup_C[i] - ans_C[i]) < 1.E-6);
    }
    pass = 1;
    up_nC = 1;
    for (i=0; i<order_C; i++){ up_nC *= edge_len_C[i]; };

    for (i=0; i<(int)up_nC; i++){
      if (fabs((up_C[i]-up_ans_C[i])/up_ans_C[i]) > 1.E-6 &&
    fabs((up_C[i]-up_ans_C[i])) > 1.E-6){
        printf("C[%d] = %lf, ans_C[%d] = %lf\n",
         i, up_C[i], i, up_ans_C[i]);
        pass = 0;
      }
    }
    if (!pass) ABORT;

  #endif

    delete ctrf;

    TAU_FSTOP(contract);
    return SUCCESS;
  }


  int contraction::sym_contract(){
    int i;
    //int ** scl_idxs_C;
    //dtype * scl_alpha_C;
    int stat = SUCCESS;
    int * new_idx;
    int * map_A, * map_B, * map_C;
    tensor ** dstack_tsr_C;
    int ** dstack_map_C;
    int nst_C;
    std::vector<contraction> perm_types;
    std::vector<int> signs;
    char const * dbeta;
    ctr * ctrf;
    tensor * tnsr_A, * tnsr_B, * tnsr_C;
  
    this->check_consistency();
  
    CommData global_comm = A->wrld->cdt;
  
    A->unfold();
    B->unfold();
    C->unfold();
    if (A->has_zero_edge_len || B->has_zero_edge_len
        || C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }
    CTF_int::alloc_ptr(sizeof(int)*A->order,          (void**)&map_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order,          (void**)&map_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order,          (void**)&map_C);
    CTF_int::alloc_ptr(sizeof(int*)*C->order,         (void**)&dstack_map_C);
    CTF_int::alloc_ptr(sizeof(tensor*)*C->order,      (void**)&dstack_tsr_C);
    memcpy(map_A, idx_A, A->order*sizeof(int));
    memcpy(map_B, idx_B, B->order*sizeof(int));
    memcpy(map_C, idx_C, C->order*sizeof(int));

    tnsr_A = A;
    tnsr_B = B;
    tnsr_C = C;
    
    tensor * new_tsr;
    while (tnsr_A->extract_diag(map_A, 1, new_tsr, &new_idx) == SUCCESS){
      if (tnsr_A != A) delete tnsr_A;
      CTF_int::cdealloc(map_A);
      tnsr_A = new_tsr;
      map_A = new_idx;
    }
    while (tnsr_B->extract_diag(map_B, 1, new_tsr, &new_idx) == SUCCESS){
      if (tnsr_B != B) delete tnsr_B;
      CTF_int::cdealloc(map_B);
      tnsr_B = new_tsr;
      map_B = new_idx;
    }
    nst_C = 0;
    while (tnsr_C->extract_diag(map_C, 1, new_tsr, &new_idx) == SUCCESS){
      dstack_map_C[nst_C] = map_C;
      dstack_tsr_C[nst_C] = tnsr_C;
      nst_C++;
      tnsr_C = new_tsr;
      map_C = new_idx;
    }
    bivar_function * fptr;
    if (is_custom) fptr = func;
    else fptr = NULL;
    contraction new_ctr = contraction(tnsr_A, map_A, tnsr_B, map_B, alpha, tnsr_C, map_C, beta, fptr);

    tnsr_A->unfold();
    tnsr_B->unfold();
    tnsr_C->unfold();
    /*if (ntid_A == ntid_B || ntid_A == ntid_C){*/
    if (tnsr_A == tnsr_C){
      tensor * nnew_tsr = new tensor(tnsr_A);
      contraction nnew_ctr = contraction(new_ctr);
      nnew_ctr.A = nnew_tsr;
      stat = nnew_ctr.sym_contract();
      delete nnew_tsr;
    } else if (tnsr_B == tnsr_C){
      tensor * nnew_tsr = new tensor(tnsr_B);
      contraction nnew_ctr = contraction(new_ctr);
      nnew_ctr.B = nnew_tsr;
      stat = nnew_ctr.sym_contract();
      delete nnew_tsr;
    } else {

      int sign = align_symmetric_indices(tnsr_A->order,
                                         new_ctr.idx_A,
                                         tnsr_A->sym,
                                         tnsr_B->order,
                                         new_ctr.idx_B,
                                         tnsr_B->sym,
                                         tnsr_C->order,
                                         new_ctr.idx_C,
                                         tnsr_C->sym);

      /*
       * Apply a factor of n! for each set of n symmetric indices which are contracted over
       */
      int ocfact = overcounting_factor(tnsr_A->order,
                                       new_ctr.idx_A,
                                       tnsr_A->sym,
                                       tnsr_B->order,
                                       new_ctr.idx_B,
                                       tnsr_B->sym,
                                       tnsr_C->order,
                                       new_ctr.idx_C,
                                       tnsr_C->sym);
      char const * align_alpha = alpha;
      if (sign != 1){
        char * u_align_alpha = (char*)alloc(tnsr_C->sr->el_size);
        if (sign == -1){
          tnsr_C->sr->addinv(alpha, u_align_alpha);
//          alpha = new_alpha;
        }
        align_alpha = u_align_alpha;
        //FIXME free new_alpha
      }

      char * oc_align_alpha = (char*)alloc(tnsr_C->sr->el_size);
      tnsr_C->sr->copy(oc_align_alpha, align_alpha);
      if (ocfact != 1){
        if (ocfact != 1){
          tnsr_B->sr->copy(oc_align_alpha, tnsr_B->sr->addid());
          
          for (int i=0; i<ocfact; i++){
            tnsr_B->sr->add(oc_align_alpha, align_alpha, oc_align_alpha);
          }
//          alpha = new_alpha;
        }
      }
      //new_ctr.alpha = alpha;


      //std::cout << alpha << ' ' << alignfact << ' ' << ocfact << std::endl;

      if (new_ctr.unfold_broken_sym(NULL) != -1){
        if (global_comm.rank == 0)
          DPRINTF(1,"Contraction index is broken\n");

        contraction * unfold_ctr;
        new_ctr.unfold_broken_sym(&unfold_ctr);
  #if PERFORM_DESYM
        if (unfold_ctr->map(&ctrf, 0) == SUCCESS){
  #else
        int sy = 0;
        for (i=0; i<A->order; i++){
          if (A->sym[i] == SY) sy = 1;
        }
        for (i=0; i<B->order; i++){
          if (B->sym[i] == SY) sy = 1;
        }
        for (i=0; i<C->order; i++){
          if (C->sym[i] == SY) sy = 1;
        }
        if (sy && unfold_ctr->map(&ctrf, 0) == SUCCESS){
  #endif
          if (tnsr_A == tnsr_B){
            tnsr_A = new tensor(tnsr_B);
          }
          desymmetrize(tnsr_A, unfold_ctr->A, 0);
          desymmetrize(tnsr_B, unfold_ctr->B, 0);
          desymmetrize(tnsr_C, unfold_ctr->C, 1);
          if (global_comm.rank == 0)
            DPRINTF(1,"Performing index desymmetrization\n");
          unfold_ctr->alpha = align_alpha;
          stat = unfold_ctr->sym_contract();
          symmetrize(tnsr_C, unfold_ctr->C);
          if (tnsr_A != unfold_ctr->A){
            unfold_ctr->A->unfold();
            tnsr_A->pull_alias(unfold_ctr->A);
            delete unfold_ctr->A;
          }
          if (tnsr_B != unfold_ctr->B){
            unfold_ctr->B->unfold();
            tnsr_B->pull_alias(unfold_ctr->B);
            delete unfold_ctr->B;
          }
          if (tnsr_C != unfold_ctr->C){
            unfold_ctr->C->unfold();
            tnsr_C->pull_alias(unfold_ctr->C);
            delete unfold_ctr->C;
          }
        } else {
          get_sym_perms(new_ctr, perm_types, signs);
                        //&nscl_C, &scl_maps_C, &scl_alpha_C);
          dbeta = beta;
          char * new_alpha = (char*)alloc(tnsr_B->sr->el_size);
          for (i=0; i<(int)perm_types.size(); i++){
            if (signs[i] == 1)
              C->sr->copy(new_alpha, oc_align_alpha);
            else {
              ASSERT(signs[i]==-1);
              tnsr_C->sr->addinv(oc_align_alpha, new_alpha);
            }
            perm_types[i].alpha = new_alpha;
            perm_types[i].beta = dbeta;
            stat = perm_types[i].contract();
            dbeta = new_ctr.C->sr->mulid();
          }
          perm_types.clear();
          signs.clear();
        }
        delete unfold_ctr;
      } else {
        new_ctr.alpha = oc_align_alpha;
        stat = new_ctr.contract();
      }
      if (tnsr_A != A) delete tnsr_A;
      if (tnsr_B != B) delete tnsr_B;
      for (int i=nst_C-1; i>=0; i--){
        dstack_tsr_C[i]->extract_diag(dstack_map_C[i], 0, tnsr_C, &new_idx);
        delete tnsr_C;
        tnsr_C = dstack_tsr_C[i];
      }
      ASSERT(tnsr_C == C);
      CTF_int::cdealloc(oc_align_alpha);
    }

    CTF_int::cdealloc(map_A);
    CTF_int::cdealloc(map_B);
    CTF_int::cdealloc(map_C);
    CTF_int::cdealloc(dstack_map_C);
    CTF_int::cdealloc(dstack_tsr_C);

    return stat;
  }

  int contraction::home_contract(){
  #ifndef HOME_CONTRACT
    return sym_contract();
  #else
    int ret;
    int was_home_A, was_home_B, was_home_C;
    A->unfold();
    B->unfold();
    C->unfold();
    
    if (A->has_zero_edge_len || 
        B->has_zero_edge_len || 
        C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=new_idx_C[j];
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }

    CTF_int::contract_mst();

    //if (stype->tid_A == stype->tid_B || stype->tid_A == stype->tid_C){
    /*if (stype->tid_A == stype->tid_C){
      clone_tensor(stype->tid_A, 1, &new_tid);
      CTF_ctr_type_t new_type = *stype;
      new_type.tid_A = new_tid;
      ret = home_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return ret;
    } else if (stype->tid_B == stype->tid_C){
      clone_tensor(stype->tid_B, 1, &new_tid);
      CTF_ctr_type_t new_type = *stype;
      new_type.tid_B = new_tid;
      ret = home_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return ret;
    }*/ 

    //CTF_ctr_type_t ntype = *stype;
    contraction new_ctr = contraction(*this);;

    was_home_A = A->is_home;
    was_home_B = B->is_home;
    was_home_C = C->is_home;

    if (was_home_A){
//      clone_tensor(stype->tid_A, 0, &ntype.tid_A, 0);
      new_ctr.A = new tensor(A, 0, 0); //tensors[ntype.tid_A];
      new_ctr.A = new_ctr.A;
      new_ctr.A->data = A->data;
      new_ctr.A->home_buffer = A->home_buffer;
      new_ctr.A->is_home = 1;
      new_ctr.A->is_mapped = 1;
      new_ctr.A->topo = A->topo;
      copy_mapping(A->order, A->edge_map, new_ctr.A->edge_map);
      new_ctr.A->set_padding();
    }     
    if (was_home_B){
      if (A == B){ //stype->tid_A == stype->tid_B){
        new_ctr.B = new_ctr.A; //tensors[ntype.tid_B];
      } else {
        new_ctr.B = new tensor(B, 0, 0); //tensors[ntype.tid_A];
/*        clone_tensor(stype->tid_B, 0, &ntype.tid_B, 0);
        new_ctr.B = tensors[ntype.tid_B];*/
        new_ctr.B->data = B->data;
        new_ctr.B->home_buffer = B->home_buffer;
        new_ctr.B->is_home = 1;
        new_ctr.B->is_mapped = 1;
        new_ctr.B->topo = B->topo;
        copy_mapping(B->order, B->edge_map, new_ctr.B->edge_map);
        new_ctr.B->set_padding();
      }
    }
    if (was_home_C){
      if (C == A){ //stype->tid_C == stype->tid_A){
        new_ctr.C = new_ctr.A; //tensors[ntype.tid_C];
      } else if (C == B){ //stype->tid_C == stype->tid_B){
        new_ctr.C = new_ctr.B; //tensors[ntype.tid_C];
      } else {
        new_ctr.C = new tensor(C, 0, 0); //tensors[ntype.tid_C];
        /*clone_tensor(stype->tid_C, 0, &ntype.tid_C, 0);
        new_ctr.C = tensors[ntype.tid_C];*/
        new_ctr.C->data = C->data;
        new_ctr.C->home_buffer = C->home_buffer;
        new_ctr.C->is_home = 1;
        new_ctr.C->is_mapped = 1;
        new_ctr.C->topo = C->topo;
        copy_mapping(C->order, C->edge_map, new_ctr.C->edge_map);
        new_ctr.C->set_padding();
      }
    }

    ret = new_ctr.sym_contract();//&ntype, ftsr, felm, alpha, beta);
    if (ret!= SUCCESS) return ret;
    if (was_home_A) new_ctr.A->unfold();
    if (was_home_B && A != B) new_ctr.B->unfold();
    if (was_home_C) new_ctr.C->unfold();

    if (was_home_C && !new_ctr.C->is_home){
      if (C->wrld->rank == 0)
        DPRINTF(2,"Migrating tensor %s back to home\n", C->name);
      distribution dC = distribution(new_ctr.C);
/*      save_mapping(new_ctr.C,
                   &old_phase_C, &old_rank_C,
                   &old_virt_dim_C, &old_pe_lda_C,
                   &old_size_C,
                   &was_cyclic_C, &old_padding_C,
                   &old_edge_len_C, &topovec[new_ctr.C->itopo]);*/
      C->data = new_ctr.C->data;
      C->is_home = 0;
      TAU_FSTART(redistribute_for_ctr_home);
      C->redistribute(dC);
/*      remap_tensor(stype->tid_C, C, C->topo, old_size_C,
                   old_phase_C, old_rank_C, old_virt_dim_C,
                   old_pe_lda_C, was_cyclic_C,
                   old_padding_C, old_edge_len_C, global_comm);*/
      TAU_FSTOP(redistribute_for_ctr_home);
      memcpy(C->home_buffer, C->data, C->size*C->sr->el_size);
      CTF_int::cdealloc(C->data);
      C->data = C->home_buffer;
      C->is_home = 1;
      new_ctr.C->is_data_aliased = 1;
      delete new_ctr.C;
    } else if (was_home_C) {
  /*    C->itopo = new_ctr.C->itopo;
      copy_mapping(C->order, new_ctr.C->edge_map, C->edge_map);
      C->set_padding();*/
      ASSERT(new_ctr.C->data == C->data);
      new_ctr.C->is_data_aliased = 1;
      delete new_ctr.C;
    }
    if (new_ctr.A != new_ctr.C){ //ntype.tid_A != ntype.tid_C){
      if (was_home_A && !new_ctr.A->is_home){
        new_ctr.A->has_home = 0;
        delete new_ctr.A;
      } else if (was_home_A) {
        new_ctr.A->is_data_aliased = 1;
        delete new_ctr.A;
      }
    }
    if (new_ctr.B != new_ctr.A && new_ctr.B != new_ctr.C){
      if (was_home_B && A != B && !new_ctr.B->is_home){
        new_ctr.B->has_home = 0;
        delete new_ctr.B;
      } else if (was_home_B && A != B) {
        new_ctr.B->is_data_aliased = 1;
        delete new_ctr.B;
      }
    }
    return SUCCESS;
  #endif
  }



  int  ctr_2d_gen_build(int                        is_used,
                        CommData                   global_comm,
                        int                        i,
                        int *                      virt_dim,
                        int &                      cg_edge_len,
                        int &                      total_iter,
                        tensor *                   A,
                        int                        i_A,
                        CommData *&                cg_cdt_A,
                        int64_t &                  cg_ctr_lda_A,
                        int64_t &                  cg_ctr_sub_lda_A,
                        bool &                     cg_move_A,
                        int *                      blk_len_A,
                        int64_t &                  blk_sz_A,
                        int const *                virt_blk_len_A,
                        int &                      load_phase_A,
                        tensor *                   B,
                        int                        i_B,
                        CommData *&                cg_cdt_B,
                        int64_t &                  cg_ctr_lda_B,
                        int64_t &                  cg_ctr_sub_lda_B,
                        bool &                     cg_move_B,
                        int *                      blk_len_B,
                        int64_t &                  blk_sz_B,
                        int const *                virt_blk_len_B,
                        int &                      load_phase_B,
                        tensor *                   C,
                        int                        i_C,
                        CommData *&                cg_cdt_C,
                        int64_t &                  cg_ctr_lda_C,
                        int64_t &                  cg_ctr_sub_lda_C,
                        bool &                     cg_move_C,
                        int *                      blk_len_C,
                        int64_t &                  blk_sz_C,
                        int const *                virt_blk_len_C,
                        int &                      load_phase_C){
    mapping * map;
    int j;
    int nstep = 1;
    if (comp_dim_map(&C->edge_map[i_C], &B->edge_map[i_B])){
      map = &B->edge_map[i_B];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
      }
      return 0;
    } else {
      if (B->edge_map[i_B].type == VIRTUAL_MAP &&
        C->edge_map[i_C].type == VIRTUAL_MAP){
        virt_dim[i] = B->edge_map[i_B].np;
        return 0;
      } else {
        cg_edge_len = 1;
        if (B->edge_map[i_B].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, B->edge_map[i_B].calc_phase());
          cg_cdt_B = &B->topo->dim_comm[B->edge_map[i_B].cdt];
          /*if (is_used && cg_cdt_B.alive == 0)
            cg_cdt_B.activate(global_comm.cm);*/
          nstep = B->edge_map[i_B].calc_phase();
          cg_move_B = 1;
        } else
          cg_move_B = 0;
        if (C->edge_map[i_C].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, C->edge_map[i_C].calc_phase());
          cg_cdt_C = &C->topo->dim_comm[C->edge_map[i_C].cdt];
          /*if (is_used && cg_cdt_C.alive == 0)
            cg_cdt_C.activate(global_comm.cm);*/
          nstep = MAX(nstep, C->edge_map[i_C].calc_phase());
          cg_move_C = 1;
        } else
          cg_move_C = 0;
        cg_ctr_lda_A = 1;
        cg_ctr_sub_lda_A = 0;
        cg_move_A = 0;
  
  
        /* Adjust the block lengths, since this algorithm will cut
           the block into smaller ones of the min block length */
        /* Determine the LDA of this dimension, based on virtualization */
        cg_ctr_lda_B  = 1;
        if (B->edge_map[i_B].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_B= blk_sz_B*B->edge_map[i_B].np/cg_edge_len;
        else
          cg_ctr_sub_lda_B= blk_sz_B/cg_edge_len;
        for (j=i_B+1; j<B->order; j++) {
          cg_ctr_sub_lda_B = (cg_ctr_sub_lda_B *
                virt_blk_len_B[j]) / blk_len_B[j];
          cg_ctr_lda_B = (cg_ctr_lda_B*blk_len_B[j])
                /virt_blk_len_B[j];
        }
        cg_ctr_lda_C  = 1;
        if (C->edge_map[i_C].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_C= blk_sz_C*C->edge_map[i_C].np/cg_edge_len;
        else
          cg_ctr_sub_lda_C= blk_sz_C/cg_edge_len;
        for (j=i_C+1; j<C->order; j++) {
          cg_ctr_sub_lda_C = (cg_ctr_sub_lda_C *
                virt_blk_len_C[j]) / blk_len_C[j];
          cg_ctr_lda_C = (cg_ctr_lda_C*blk_len_C[j])
                /virt_blk_len_C[j];
        }
        if (B->edge_map[i_B].type != PHYSICAL_MAP){
          blk_sz_B  = blk_sz_B / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] / nstep;
        } else {
          blk_sz_B  = blk_sz_B * B->edge_map[i_B].np / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] * B->edge_map[i_B].np / nstep;
        }
        if (C->edge_map[i_C].type != PHYSICAL_MAP){
          blk_sz_C  = blk_sz_C / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] / nstep;
        } else {
          blk_sz_C  = blk_sz_C * C->edge_map[i_C].np / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] * C->edge_map[i_C].np / nstep;
        }
  
        if (B->edge_map[i_B].has_child){
          ASSERT(B->edge_map[i_B].child->type == VIRTUAL_MAP);
          virt_dim[i] = B->edge_map[i_B].np*B->edge_map[i_B].child->np/nstep;
        }
        if (C->edge_map[i_C].has_child) {
          ASSERT(C->edge_map[i_C].child->type == VIRTUAL_MAP);
          virt_dim[i] = C->edge_map[i_C].np*C->edge_map[i_C].child->np/nstep;
        }
        if (C->edge_map[i_C].type == VIRTUAL_MAP){
          virt_dim[i] = C->edge_map[i_C].np/nstep;
        }
        if (B->edge_map[i_B].type == VIRTUAL_MAP)
          virt_dim[i] = B->edge_map[i_B].np/nstep;
  #ifdef OFFLOAD
        total_iter *= nstep;
        if (cg_ctr_sub_lda_A == 0)
          load_phase_A *= nstep;
        else 
          load_phase_A  = 1;
        if (cg_ctr_sub_lda_B == 0)   
          load_phase_B *= nstep;
        else 
          load_phase_B  = 1;
        if (cg_ctr_sub_lda_C == 0) 
          load_phase_C *= nstep;
        else 
          load_phase_C  = 1;
  #endif
      }
    } 
    return 1;
  }

  void contraction::print(){
    int i,j,max,ex_A, ex_B,ex_C;
    max = A->order+B->order+C->order;
    CommData global_comm = A->wrld->cdt;
    MPI_Barrier(global_comm.cm);
    if (global_comm.rank == 0){
      printf("Contracting Tensor %s with %s into %s\n", A->name, B->name, C->name);
      if (alpha != NULL){
        printf("alpha is "); 
        A->sr->print(alpha);
        printf("\nbeta is "); 
        B->sr->print(beta);
        printf("\n");
      }

      printf("Contraction index table:\n");
      printf("     A     B     C\n");
      for (i=0; i<max; i++){
        ex_A=0;
        ex_B=0;
        ex_C=0;
        printf("%d:   ",i);
        for (j=0; j<A->order; j++){
          if (idx_A[j] == i){
            ex_A++;
            if (A->sym[j] != NS)
              printf("%d' ",j);
            else
              printf("%d  ",j);
          }
        }
        if (ex_A == 0)
          printf("      ");
        if (ex_A == 1)
          printf("   ");
        for (j=0; j<B->order; j++){
          if (idx_B[j] == i){
            ex_B=1;
            if (B->sym[j] != NS)
              printf("%d' ",j);
            else
              printf("%d  ",j);
          }
        }
        if (ex_B == 0)
          printf("      ");
        if (ex_B == 1)
          printf("   ");
        for (j=0; j<C->order; j++){
          if (idx_C[j] == i){
            ex_C=1;
            if (C->sym[j] != NS)
              printf("%d' ",j);
            else
              printf("%d ",j);
          }
        }
        printf("\n");
        if (ex_A + ex_B + ex_C == 0) break;
      }
    }
  }
}
