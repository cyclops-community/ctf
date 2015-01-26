#include "contraction.h"
#include "../redistribution/folding.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_ctr.h"
#include "ctr_tsr.h"
#include "ctr_comm.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/folding.h"
#include "../redistribution/redist.h"


namespace CTF_int {

  contraction::~contraction(){
    if (idx_A != NULL) free(idx_A);
    if (idx_B != NULL) free(idx_B);
    if (idx_C != NULL) free(idx_C);
  }

  contraction::contraction(contraction const & other){
    A     = other.A;
    idx_A = (int*)malloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B     = other.B;
    idx_B = (int*)malloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    C     = other.C;
    idx_C = (int*)malloc(sizeof(int)*other.C->order);
    memcpy(idx_C, other.idx_C, sizeof(int)*other.C->order);
    if (other.is_custom){
      func      = other.func;
      is_custom = 1;
    } else {
      alpha = other.alpha;
      beta  = other.beta;
    }
  }

  contraction::contraction(tensor *     A_,
                           int const *  idx_A_,
                           tensor *     B_,
                           int const *  idx_B_,
                           char const * alpha_,
                           tensor *     C_,
                           int const *  idx_C_,
                           char const * beta_){
    A         = A_;
    idx_A     = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    B         = B_;
    idx_B     = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    alpha     = alpha_;
    C         = C_;
    idx_C     = (int*)malloc(sizeof(int)*C->order);
    memcpy(idx_C, idx_C_, sizeof(int)*C->order);
    beta      = beta_;
    is_custom = 0;
  }
 
  contraction::contraction(tensor *       A_,
                           int const *    idx_A_,
                           tensor *       B_,
                           int const *    idx_B_,
                           tensor *       C_,
                           int const *    idx_C_,
                           bivar_function func_){
    A         = A_;
    idx_A     = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    B         = B_;
    idx_B     = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    C         = C_;
    idx_C     = (int*)malloc(sizeof(int)*C->order);
    memcpy(idx_C, idx_C_, sizeof(int)*C->order);
    func      = func_;
    is_custom = 1;
  }

  void contraction::execute(){
    int stat = home_contract();
    assert(stat == SUCCESS); 
  }
  
  double contraction::estimate_time(){
    assert(0); //FIXME
    return 0.0;
  }

  int contraction::is_equal(contracton const & os){
    if (this->A != os.A) return 0;
    if (this->B != os.B) return 0;
    if (this->C != os.C) return 0;
    
    for (i=0; i<A->ndim; i++){
      if (idx_A[i] != os.idx_A[i]) return 0;
    }
    for (i=0; i<B->ndim; i++){
      if (idx_B[i] != os.idx_B[i]) return 0;
    }
    for (i=0; i<C->ndim; i++){
      if (idx_C[i] != os.idx_C[i]) return 0;
    }
    return 1;
  }

  void contraction::calc_fold_nmk(
                      int const *    ordering_A,
                      int const *    ordering_B,
                      tensor const * A,
                      tensor const * B,
                      tensor const * C,
                      iparam *       inner_prm){
    int i, num_ctr, num_tot;
    int * idx_arr;
    int * edge_len_A, * edge_len_B;
    iparam prm;
      
    edge_len_A = A->edge_len;
    edge_len_B = B->edge_len;

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
        prm.m = prm.m * edge_len_A[ordering_A[i]];
      else 
        prm.k = prm.k * edge_len_A[ordering_A[i]];
    }
    for (i=0; i<B->order; i++){
      if (i >= num_ctr)
        prm.n = prm.n * edge_len_B[ordering_B[i]];
    }
    /* This gets set later */
    prm.sz_C = 0;
    CTF_free(idx_arr);
    *inner_prm = prm;  
  }

  void contraction::get_fold_indices(int *  num_fold,
                                     int ** fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, iC, inA, inB, inC, iiA, iiB, iiC;
    int * idx_arr, * idx;
    tensor<dtype> * A, * B, * C;
    A = tensors[type->tid_A];
    B = tensors[type->tid_B];
    C = tensors[type->tid_C];
    inv_idx(A->order, idx_A, A->edge_map,
      B->order, idx_B, B->edge_map,
      C->order, idx_C, C->edge_map,
      &num_tot, &idx_arr);
    CTF_alloc_ptr(num_tot*sizeof(int), (void**)&idx);

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
    CTF_free(idx_arr);

  }

  int contraction::can_fold(){
    int nfold, * fold_idx, i, j;
    tensor<dtype> * tsr;
    tsr = tensors[type->tid_A];
    for (i=0; i<tsr->order; i++){
      for (j=i+1; j<tsr->order; j++){
        if (idx_A[i] == idx_A[j]) return 0;
      }
    }
    tsr = tensors[type->tid_B];
    for (i=0; i<tsr->order; i++){
      for (j=i+1; j<tsr->order; j++){
        if (idx_B[i] == idx_B[j]) return 0;
      }
    }
    tsr = tensors[type->tid_C];
    for (i=0; i<tsr->order; i++){
      for (j=i+1; j<tsr->order; j++){
        if (idx_C[i] == idx_C[j]) return 0;
      }
    }
    get_fold_indices(type, &nfold, &fold_idx);
    CTF_free(fold_idx);
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
    tensor<dtype> * A, * B, * C;
    
    A = tensors[type->tid_A];
    B = tensors[type->tid_B];
    C = tensors[type->tid_C];
    CTF_alloc_ptr(sizeof(int)*A->order, (void**)&ordering_A);
    CTF_alloc_ptr(sizeof(int)*B->order, (void**)&ordering_B);
    CTF_alloc_ptr(sizeof(int)*C->order, (void**)&ordering_C);

    inv_idx(A->order, idx_A, A->edge_map,
            B->order, idx_B, B->edge_map,
            C->order, idx_C, C->edge_map,
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
    CTF_free(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
    *new_ordering_C = ordering_C;
  }
  iparam summation::map_fold(){
    int i, j, nfold, nf, all_fdim_A, all_fdim_B, all_fdim_C;
    int nvirt_A, nvirt_B, nvirt_C;
    int * fold_idx, * fidx_map_A, * fidx_map_B, * fidx_map_C;
    int * fnew_ord_A, * fnew_ord_B, * fnew_ord_C;
    int * all_flen_A, * all_flen_B, * all_flen_C;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    tensor<dtype> * ftsr_A, * ftsr_B, * ftsr_C;
    CTF_ctr_type_t fold_type;
    iparam iprm;

    get_fold_indices(type, &nfold, &fold_idx);
    if (nfold == 0) {
      CTF_free(fold_idx);
      return CTF_ERROR;
    }

    /* overestimate this space to not bother with it later */
    CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_A);
    CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_B);
    CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_C);

    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];

    fold_tsr(tsr_A, nfold, fold_idx, type->idx_map_A, 
             &all_fdim_A, &all_flen_A);
    fold_tsr(tsr_B, nfold, fold_idx, type->idx_map_B, 
             &all_fdim_B, &all_flen_B);
    fold_tsr(tsr_C, nfold, fold_idx, type->idx_map_C,
             &all_fdim_C, &all_flen_C);

    nf = 0;
    for (i=0; i<tsr_A->order; i++){
      for (j=0; j<nfold; j++){
        if (tsr_A->sym[i] == NS && type->idx_map_A[i] == fold_idx[j]){
          fidx_map_A[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<tsr_B->order; i++){
      for (j=0; j<nfold; j++){
        if (tsr_B->sym[i] == NS && type->idx_map_B[i] == fold_idx[j]){
          fidx_map_B[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<tsr_C->order; i++){
      for (j=0; j<nfold; j++){
        if (tsr_C->sym[i] == NS && type->idx_map_C[i] == fold_idx[j]){
          fidx_map_C[nf] = j;
          nf++;
        }
      }
    }

    ftsr_A = tensors[tsr_A->rec_tid];
    ftsr_B = tensors[tsr_B->rec_tid];
    ftsr_C = tensors[tsr_C->rec_tid];

    fold_type.tid_A = tsr_A->rec_tid;
    fold_type.tid_B = tsr_B->rec_tid;
    fold_type.tid_C = tsr_C->rec_tid;

    conv_idx(ftsr_A->order, fidx_map_A, &fold_type.idx_map_A,
             ftsr_B->order, fidx_map_B, &fold_type.idx_map_B,
             ftsr_C->order, fidx_map_C, &fold_type.idx_map_C);

  #if DEBUG>=2
    if (global_comm.rank == 0){
      printf("Folded contraction type:\n");
    }
    print_ctr(&fold_type,0.0,0.0);
  #endif
    
    //for type order 1 to 3 
    get_len_ordering(&fold_type, &fnew_ord_A, &fnew_ord_B, &fnew_ord_C); 
    
    //permute_target(ftsr_A->order, fnew_ord_A, cpy_tsr_A_inner_ordering);
    //permute_target(ftsr_B->order, fnew_ord_B, cpy_tsr_B_inner_ordering);

    //get nosym_transpose_estimate cost estimate an save best

    permute_target(ftsr_A->order, fnew_ord_A, tsr_A->inner_ordering);
    permute_target(ftsr_B->order, fnew_ord_B, tsr_B->inner_ordering);
    permute_target(ftsr_C->order, fnew_ord_C, tsr_C->inner_ordering);
    

    nvirt_A = calc_nvirt(tsr_A);
    for (i=0; i<nvirt_A; i++){
      nosym_transpose<dtype>(all_fdim_A, tsr_A->inner_ordering, all_flen_A, 
                             tsr_A->data + i*(tsr_A->size/nvirt_A), 1);
    }
    nvirt_B = calc_nvirt(tsr_B);
    for (i=0; i<nvirt_B; i++){
      nosym_transpose<dtype>(all_fdim_B, tsr_B->inner_ordering, all_flen_B, 
                             tsr_B->data + i*(tsr_B->size/nvirt_B), 1);
    }
    nvirt_C = calc_nvirt(tsr_C);
    for (i=0; i<nvirt_C; i++){
      nosym_transpose<dtype>(all_fdim_C, tsr_C->inner_ordering, all_flen_C, 
                             tsr_C->data + i*(tsr_C->size/nvirt_C), 1);
    }

    calc_fold_nmk<dtype>(&fold_type, fnew_ord_A, fnew_ord_B, 
                         ftsr_A, ftsr_B, ftsr_C, &iprm);

    //FIXME: try all possibilities
    iprm.tA = 'T';
    iprm.tB = 'N';

    CTF_free(fidx_map_A);
    CTF_free(fidx_map_B);
    CTF_free(fidx_map_C);
    CTF_free(fold_type.idx_map_A);
    CTF_free(fold_type.idx_map_B);
    CTF_free(fold_type.idx_map_C);
    CTF_free(fnew_ord_A);
    CTF_free(fnew_ord_B);
    CTF_free(fnew_ord_C);
    CTF_free(all_flen_A);
    CTF_free(all_flen_B);
    CTF_free(all_flen_C);
    CTF_free(fold_idx);

    *inner_prm = iprm;
    return CTF_SUCCESS;
  }

  int summation::unfold_broken_sym(contraction ** new_contraction){
    int i, num_tot, iA, iB, iC, iA2, iB2;
    int * idx_arr;
    
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    
    
    if (new_type == NULL){
      tsr_A = tensors[type->tid_A];
      tsr_B = tensors[type->tid_B];
      tsr_C = tensors[type->tid_C];
    } else {
      clone_tensor(type->tid_A, 0, &new_type->tid_A, 0);
      clone_tensor(type->tid_B, 0, &new_type->tid_B, 0);
      clone_tensor(type->tid_C, 0, &new_type->tid_C, 0);

      tsr_A = tensors[new_type->tid_A];
      tsr_B = tensors[new_type->tid_B];
      tsr_C = tensors[new_type->tid_C];
      
      CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)&new_type->idx_map_A);
      CTF_alloc_ptr(tsr_B->order*sizeof(int), (void**)&new_type->idx_map_B);
      CTF_alloc_ptr(tsr_C->order*sizeof(int), (void**)&new_type->idx_map_C);

      memcpy(new_type->idx_map_A, type->idx_map_A, tsr_A->order*sizeof(int));
      memcpy(new_type->idx_map_B, type->idx_map_B, tsr_B->order*sizeof(int));
      memcpy(new_type->idx_map_C, type->idx_map_C, tsr_C->order*sizeof(int));
    }

    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);

    for (i=0; i<tsr_A->order; i++){
      if (tsr_A->sym[i] != NS){
        iA = type->idx_map_A[i];
        if (idx_arr[3*iA+1] != -1){
          if (tsr_B->sym[idx_arr[3*iA+1]] == NS ||
              type->idx_map_A[i+1] != type->idx_map_B[idx_arr[3*iA+1]+1]){
            if (new_type != NULL)
              tsr_A->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*type->idx_map_A[i+1]+1] != -1){
            if (new_type != NULL)
              tsr_A->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i;
          }       
        }
        if (idx_arr[3*iA+2] != -1){
          if (tsr_C->sym[idx_arr[3*iA+2]] == NS ||
              type->idx_map_A[i+1] != type->idx_map_C[idx_arr[3*iA+2]+1]){
            if (new_type != NULL)
              tsr_A->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*type->idx_map_A[i+1]+2] != -1){
            if (new_type != NULL)
              tsr_A->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i;
          }       
        }
      }
    }

   
    for (i=0; i<tsr_B->order; i++){
      if (tsr_B->sym[i] != NS){
        iB = type->idx_map_B[i];
        if (idx_arr[3*iB+0] != -1){
          if (tsr_A->sym[idx_arr[3*iB+0]] == NS ||
              type->idx_map_B[i+1] != type->idx_map_A[idx_arr[3*iB+0]+1]){
            if (new_type != NULL)
              tsr_B->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*type->idx_map_B[i+1]+0] != -1){
            if (new_type != NULL)
              tsr_B->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+1;
          }       
        }
        if (idx_arr[3*iB+2] != -1){
          if (tsr_C->sym[idx_arr[3*iB+2]] == NS || 
              type->idx_map_B[i+1] != type->idx_map_C[idx_arr[3*iB+2]+1]){
            if (new_type != NULL)
              tsr_B->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*type->idx_map_B[i+1]+2] != -1){
            if (new_type != NULL)
              tsr_B->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+1;
          }       
        }
      }
    } 
    //if A=B, output symmetry may still be preserved, so long as all indices in A and B are proper
    bool is_preserv = true;
    if (type->tid_A != type->tid_B) is_preserv = false; 
    else {
      for (int j=0; j<tsr_A->order; j++){
        if (type->idx_map_A[j] != type->idx_map_B[j]){
          iA = type->idx_map_A[j];
          iB = type->idx_map_B[j];
          if (idx_arr[3*iA+2] == -1 || idx_arr[3*iB+2] == -1) is_preserv = false;
          else {
            for (int k=MIN(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k<MAX(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k++){
               if (tsr_C->sym[k] != SY) is_preserv = false;
            }
          }
        }
      }
    }
    if (!is_preserv){
      for (i=0; i<tsr_C->order; i++){
        if (tsr_C->sym[i] != NS){
          iC = type->idx_map_C[i];
          if (idx_arr[3*iC+1] != -1){
            if (tsr_B->sym[idx_arr[3*iC+1]] == NS ||
                type->idx_map_C[i+1] != type->idx_map_B[idx_arr[3*iC+1]+1]){
              if (new_type != NULL)
                tsr_C->sym[i] = NS;
              CTF_free(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*type->idx_map_C[i+1]+1] != -1){
            if (new_type != NULL)
              tsr_C->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+2;
          }       
          if (idx_arr[3*iC+0] != -1){
            if (tsr_A->sym[idx_arr[3*iC+0]] == NS ||
                type->idx_map_C[i+1] != type->idx_map_A[idx_arr[3*iC+0]+1]){
              if (new_type != NULL)
                tsr_C->sym[i] = NS;
              CTF_free(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*iC+0] == -1){
            if (idx_arr[3*type->idx_map_C[i+1]] != -1){
              if (new_type != NULL)
                tsr_C->sym[i] = NS;
              CTF_free(idx_arr); 
              return 3*i+2;
            }       
          }
        }
      }
    }
    for (i=0; i<tsr_A->order; i++){
      if (tsr_A->sym[i] == SY){
        iA = type->idx_map_A[i];
        iA2 = type->idx_map_A[i+1];
        if (idx_arr[3*iA+2] == -1 &&
            idx_arr[3*iA2+2] == -1){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i;
        }
      }
    }
    for (i=0; i<tsr_B->order; i++){
      if (tsr_B->sym[i] == SY){
        iB = type->idx_map_B[i];
        iB2 = type->idx_map_B[i+1];
        if (idx_arr[3*iB+2] == -1 &&
            idx_arr[3*iB2+2] == -1){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+1;
        }
      }
    }

    CTF_free(idx_arr);
    return -1;
  }

  void contraction::check_consistency(){
    int i, num_tot, len;
    int iA, iB, iC;
    int order_A, order_B, order_C;
    int * len_A, * len_B, * len_C;
    int * sym_A, * sym_B, * sym_C;
    int * idx_arr;
    
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;

    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];
      

    get_tsr_info(type->tid_A, &order_A, &len_A, &sym_A);
    get_tsr_info(type->tid_B, &order_B, &len_B, &sym_B);
    get_tsr_info(type->tid_C, &order_C, &len_C, &sym_C);
    
    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);

    for (i=0; i<num_tot; i++){
      len = -1;
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      if (iA != -1){
        len = len_A[iA];
      }
      if (len != -1 && iB != -1 && len != len_B[iB]){
        if (global_comm.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %d does not",
                  iA, type->tid_A);
          printf("match the %dth edge length of tensor %d.\n",
                  iB, type->tid_B);
        }
        ABORT;
      }
      if (len != -1 && iC != -1 && len != len_C[iC]){
        if (global_comm.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %d (%d) does not",
                  iA, type->tid_A, len);
          printf("match the %dth edge length of tensor %d (%d).\n",
                  iC, type->tid_C, len_C[iC]);
        }
        ABORT;
      }
      if (iB != -1){
        len = len_B[iB];
      }
      if (len != -1 && iC != -1 && len != len_C[iC]){
        if (global_comm.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %d does not",
                  iB, type->tid_B);
          printf("match the %dth edge length of tensor %d.\n",
                  iC, type->tid_C);
        }
        ABORT;
      }
    }
    CTF_free(len_A);
    CTF_free(len_B);
    CTF_free(len_C);
    CTF_free(sym_A);
    CTF_free(sym_B);
    CTF_free(sym_C);
    CTF_free(idx_arr);
    return CTF_SUCCESS;
  }

    
  int contraction::check_mapping(){

    int num_tot, i, ph_A, ph_B, iA, iB, iC, pass, order, topo_order;
    int * idx_arr;
    int * phys_mismatched, * phys_mapped;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    mapping * map;

    pass = 1;

    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];
      

    if (tsr_A->is_mapped == 0) pass = 0;
    if (tsr_B->is_mapped == 0) pass = 0;
    if (tsr_C->is_mapped == 0) pass = 0;
    ASSERT(pass==1);
    
    if (tsr_A->is_folded == 1) pass = 0;
    if (tsr_B->is_folded == 1) pass = 0;
    if (tsr_C->is_folded == 1) pass = 0;
    
    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    if (tsr_A->itopo != tsr_B->itopo) pass = 0;
    if (tsr_A->itopo != tsr_C->itopo) pass = 0;

    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    topo_order = topovec[tsr_A->itopo].order;
    CTF_alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mismatched);
    CTF_alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mapped);
    memset(phys_mismatched, 0, sizeof(int)*topo_order);
    memset(phys_mapped, 0, sizeof(int)*topo_order);


    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);
    
    if (!check_self_mapping(type->tid_A, type->idx_map_A))
      pass = 0;
    if (!check_self_mapping(type->tid_B, type->idx_map_B))
      pass = 0;
    if (!check_self_mapping(type->tid_C, type->idx_map_C))
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
  //      printf("tsr_A[%d].np = %d\n", iA, tsr_A->edge_map[iA].np);
        //printf("tsr_B[%d].np = %d\n", iB, tsr_B->edge_map[iB].np);
        //printf("tsr_C[%d].np = %d\n", iC, tsr_C->edge_map[iC].np);
        if (0 == comp_dim_map(&tsr_B->edge_map[iB], &tsr_A->edge_map[iA]) || 
            0 == comp_dim_map(&tsr_B->edge_map[iB], &tsr_C->edge_map[iC])){
          DPRINTF(3,"failed confirmation here %d %d %d\n",iA,iB,iC);
          pass = 0;
          break;
        } else {
          map = &tsr_A->edge_map[iA];
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
              tsr_A = tensors[type->tid_A];
              tsr_B = tensors[type->tid_B];
              tsr_C = tensors[type->tid_C];
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+2];
              break;
            case 1:
              tsr_A = tensors[type->tid_A];
              tsr_B = tensors[type->tid_C];
              tsr_C = tensors[type->tid_B];
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+2];
              iC = idx_arr[3*i+1];
              break;
            case 2:
              tsr_A = tensors[type->tid_C];
              tsr_B = tensors[type->tid_B];
              tsr_C = tensors[type->tid_A];
              iA = idx_arr[3*i+2];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+0];
              break;
          }
          if (iC == -1){
            if (iB == -1){
              if (iA != -1) {
                map = &tsr_A->edge_map[iA];
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
              map = &tsr_B->edge_map[iB];
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
              ph_A = calc_phase(&tsr_A->edge_map[iA]);
              ph_B = calc_phase(&tsr_B->edge_map[iB]);

              if (ph_A != ph_B){
                //if (global_comm.rank == 0) 
                  DPRINTF(3,"failed confirmation here iA=%d iB=%d\n",iA,iB);
                pass = 0;
                break;
              }
              /* If the mapping along this dimension is the same make sure
                 its mapped to a onto a unique free dimension */
              if (comp_dim_map(&tsr_B->edge_map[iB], &tsr_A->edge_map[iA])){
                map = &tsr_B->edge_map[iB];
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
                map = &tsr_A->edge_map[iA];
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
                      map = &tsr_B->edge_map[iB];
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


    CTF_free(idx_arr);
    CTF_free(phys_mismatched);
    CTF_free(phys_mapped);
    return pass;
  }

  int contraction::
      map_weigh_indices(int const *      idx_arr,
                        int const *      idx_weigh,
                        int              num_tot,
                        int              num_weigh,
                        topology const * topo){
    int tsr_order, iweigh, iA, iB, iC, i, j, k, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * weigh_map;

    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    TAU_FSTART(map_weigh_indices);

    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    tsr_C = tensors[tid_C];

    tsr_order = num_weigh;

    
    for (i=0; i<num_weigh; i++){
      iweigh = idx_weigh[i];
      iA = idx_arr[iweigh*3+0];
      iB = idx_arr[iweigh*3+1];
      iC = idx_arr[iweigh*3+2];

      if (tsr_A->edge_map[iA].type == PHYSICAL_MAP ||
          tsr_B->edge_map[iB].type == PHYSICAL_MAP ||
          tsr_C->edge_map[iC].type == PHYSICAL_MAP)
        return CTF_NEGATIVE; 
    }  
    CTF_alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_alloc_ptr(tsr_order*sizeof(mapping),            (void**)&weigh_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));
    extract_free_comms(topo, tsr_A->order, tsr_A->edge_map,
                             tsr_B->order, tsr_B->edge_map,
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

      
      weigh_map[i].np = lcm(weigh_map[i].np,tsr_A->edge_map[iA].np);
      weigh_map[i].np = lcm(weigh_map[i].np,tsr_B->edge_map[iB].np);
      weigh_map[i].np = lcm(weigh_map[i].np,tsr_C->edge_map[iC].np);

      tsr_edge_len[i] = tsr_A->edge_len[iA];

      for (j=i+1; j<num_weigh; j++){
        jX = idx_arr[idx_weigh[j]*3+0];

        for (k=MIN(iA,jX); k<MAX(iA,jX); k++){
          if (tsr_A->sym[k] == NS)
            break;
        }
        if (k==MAX(iA,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+1];

        for (k=MIN(iB,jX); k<MAX(iB,jX); k++){
          if (tsr_B->sym[k] == NS)
            break;
        }
        if (k==MAX(iB,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+2];

        for (k=MIN(iC,jX); k<MAX(iC,jX); k++){
          if (tsr_C->sym[k] == NS)
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

    if (stat == CTF_ERROR)
      return CTF_ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == CTF_SUCCESS){
      for (i=0; i<num_weigh; i++){
        iweigh = idx_weigh[i];
        iA = idx_arr[iweigh*3+0];
        iB = idx_arr[iweigh*3+1];
        iC = idx_arr[iweigh*3+2];

        copy_mapping(1, &weigh_map[i], &tsr_A->edge_map[iA]);
        copy_mapping(1, &weigh_map[i], &tsr_B->edge_map[iB]);
        copy_mapping(1, &weigh_map[i], &tsr_C->edge_map[iC]);
      }
    }
    CTF_free(restricted);
    CTF_free(tsr_edge_len);
    CTF_free(tsr_sym_table);
    for (i=0; i<num_weigh; i++){
      clear_mapping(weigh_map+i);
    }
    CTF_free(weigh_map);
    CTF_free(sub_phys_comm);
    CTF_free(comm_idx);

    TAU_FSTOP(map_weigh_indices);
    return stat;
  }

  int contraction::
      map_ctr_indices(int const *      idx_arr,
                      int const *      idx_ctr,
                      int              num_tot,
                      int              num_ctr,
                      topology const * topo){
    int tsr_order, ictr, iA, iB, i, j, jctr, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * ctr_map;

    tensor<dtype> * tsr_A, * tsr_B;
    TAU_FSTART(map_ctr_indices);

    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];

    tsr_order = num_ctr*2;

    CTF_alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_alloc_ptr(tsr_order*sizeof(mapping),            (void**)&ctr_map);

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

      copy_mapping(1, &tsr_A->edge_map[iA], &ctr_map[2*i+0]);
      copy_mapping(1, &tsr_B->edge_map[iB], &ctr_map[2*i+1]);
    }
  /*  for (i=0; i<tsr_order; i++){ 
      if (ctr_map[i].type == PHYSICAL_MAP) is_premapped = 1;
    }*/

    extract_free_comms(topo, tsr_A->order, tsr_A->edge_map,
                             tsr_B->order, tsr_B->edge_map,
                       num_sub_phys_dims, &sub_phys_comm, &comm_idx);
    

    /* Map a tensor of dimension 2*num_ctr, with symmetries among each pair.
     * Set the edge lengths and symmetries according to those in ctr dims of A and B.
     * This gives us a mapping for the contraction dimensions of tensors A and B. */
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

      tsr_edge_len[2*i+0] = tsr_A->edge_len[iA];
      tsr_edge_len[2*i+1] = tsr_A->edge_len[iA];

      tsr_sym_table[2*i*tsr_order+2*i+1] = 1;
      tsr_sym_table[(2*i+1)*tsr_order+2*i] = 1;

      /* Check if A has symmetry among the dimensions being contracted over.
       * Ignore symmetry with non-contraction dimensions.
       * FIXME: this algorithm can be more efficient but should not be a bottleneck */
      if (tsr_A->sym[iA] != NS){
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
      if (tsr_B->sym[iB] != NS){
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
    if (stat == CTF_ERROR)
      return CTF_ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == CTF_SUCCESS){
      for (i=0; i<num_ctr; i++){
        ictr = idx_ctr[i];
        iA = idx_arr[ictr*3+0];
        iB = idx_arr[ictr*3+1];

  /*      tsr_A->edge_map[iA] = ctr_map[2*i+0];
        tsr_B->edge_map[iB] = ctr_map[2*i+1];*/
        copy_mapping(1, &ctr_map[2*i+0], &tsr_A->edge_map[iA]);
        copy_mapping(1, &ctr_map[2*i+1], &tsr_B->edge_map[iB]);
      }
    }
    CTF_free(restricted);
    CTF_free(tsr_edge_len);
    CTF_free(tsr_sym_table);
    for (i=0; i<2*num_ctr; i++){
      clear_mapping(ctr_map+i);
    }
    CTF_free(ctr_map);
    CTF_free(sub_phys_comm);
    CTF_free(comm_idx);

    TAU_FSTOP(map_ctr_indices);
    return stat;
  }

  int contraction::
      map_no_ctr_indices(int const *              idx_arr,
                         int const *              idx_no_ctr,
                         int                      num_tot,
                         int                      num_no_ctr,
                         topology const *         topo){
    int stat, i, inoctr, iA, iB, iC;

    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    
    TAU_FSTART(map_noctr_indices);

    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    tsr_C = tensors[tid_C];

  /*  for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iC != -1 && iA != -1){
        copy_mapping(1, tsr_C->edge_map + iC, tsr_A->edge_map + iA); 
      } 
      if (iB != -1 && iA != -1){
        copy_mapping(1, tsr_C->edge_map + iB, tsr_A->edge_map + iA); 
      }
    }*/
    /* Map remainders of A and B to remainders of phys grid */
    stat = map_tensor_rem(topo->order, topo->dim_comm, tsr_A, 1);
    if (stat != CTF_SUCCESS){
      if (tsr_A->order != 0 || tsr_B->order != 0 || tsr_C->order != 0){
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
        copy_mapping(1, tsr_A->edge_map + iA, tsr_C->edge_map + iC); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, tsr_B->edge_map + iB, tsr_C->edge_map + iC); 
      } 
    }
    stat = map_tensor_rem(topo->order, topo->dim_comm, tsr_C, 0);
    if (stat != CTF_SUCCESS){
      TAU_FSTOP(map_noctr_indices);
      return stat;
    }
    for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iA != -1 && iC != -1){
        copy_mapping(1, tsr_C->edge_map + iC, tsr_A->edge_map + iA); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, tsr_C->edge_map + iC, tsr_B->edge_map + iB); 
      }
    }
    TAU_FSTOP(map_noctr_indices);

    return CTF_SUCCESS;
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
  int contraction::
      map_extra_indices(int const * idx_arr,
                        int const * idx_extra,
                        int         num_extra){
    int i, iA, iB, iC, iextra;

    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    
    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    tsr_C = tensors[tid_C];

    for (i=0; i<num_extra; i++){
      iextra = idx_extra[i];
      iA = idx_arr[3*iextra+0];
      iB = idx_arr[3*iextra+1];
      iC = idx_arr[3*iextra+2];

      if (iA != -1){
        //FIXME handle extra indices via reduction
        if (tsr_A->edge_map[iA].type == PHYSICAL_MAP)
          return CTF_NEGATIVE;
        if (tsr_A->edge_map[iA].type == NOT_MAPPED){
          tsr_A->edge_map[iA].type = VIRTUAL_MAP;
          tsr_A->edge_map[iA].np = 1;
          tsr_A->edge_map[iA].has_child = 0;
        }
      } else {
        if (iB != -1) {
          if (tsr_B->edge_map[iB].type == PHYSICAL_MAP)
            return CTF_NEGATIVE;
          if (tsr_B->edge_map[iB].type == NOT_MAPPED){
            tsr_B->edge_map[iB].type = VIRTUAL_MAP;
            tsr_B->edge_map[iB].np = 1;
            tsr_B->edge_map[iB].has_child = 0;
          }
        } else {
          ASSERT(iC != -1);
          if (tsr_C->edge_map[iC].type == PHYSICAL_MAP)
            return CTF_NEGATIVE;
          if (tsr_C->edge_map[iC].type == NOT_MAPPED){
            tsr_C->edge_map[iC].type = VIRTUAL_MAP;
            tsr_C->edge_map[iC].np = 1;
            tsr_C->edge_map[iC].has_child = 0;
          }
        }
      }
    }
    return CTF_SUCCESS;
  }


  int contraction::
      map_to_topology(int   itopo,
                      int   order,
                      int * idx_arr,
                      int * idx_ctr,
                      int * idx_extra,
                      int * idx_no_ctr,
                      int * idx_weigh){
    int tA, tB, tC, num_tot, num_ctr, num_no_ctr, num_weigh, num_extra, i, ret;
    int const * map_A, * map_B, * map_C;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    switch (order){
      case 0:
        tA = tid_A;
        tB = tid_B;
        tC = tid_C;
        map_A = idx_map_A;
        map_B = idx_map_B;
        map_C = idx_map_C;
        break;
      case 1:
        tA = tid_A;
        tB = tid_C;
        tC = tid_B;
        map_A = idx_map_A;
        map_B = idx_map_C;
        map_C = idx_map_B;
        break;
      case 2:
        tA = tid_B;
        tB = tid_A;
        tC = tid_C;
        map_A = idx_map_B;
        map_B = idx_map_A;
        map_C = idx_map_C;
        break;
      case 3:
        tA = tid_B;
        tB = tid_C;
        tC = tid_A;
        map_A = idx_map_B;
        map_B = idx_map_C;
        map_C = idx_map_A;
        break;
      case 4:
        tA = tid_C;
        tB = tid_A;
        tC = tid_B;
        map_A = idx_map_C;
        map_B = idx_map_A;
        map_C = idx_map_B;
        break;
      case 5:
        tA = tid_C;
        tB = tid_B;
        tC = tid_A;
        map_A = idx_map_C;
        map_B = idx_map_B;
        map_C = idx_map_A;
        break;
      default:
        return CTF_ERROR;
        break;
    }
    
    tsr_A = tensors[tA];
    tsr_B = tensors[tB];
    tsr_C = tensors[tC];

    inv_idx(tsr_A->order, map_A, tsr_A->edge_map,
            tsr_B->order, map_B, tsr_B->edge_map,
            tsr_C->order, map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);
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
    tsr_A->itopo = itopo;
    tsr_B->itopo = itopo;
    tsr_C->itopo = itopo;
    
    /* Map the weigh indices of A, B, and C*/
    ret = map_weigh_indices(idx_arr, idx_weigh, num_tot, num_weigh, 
                            tA, tB, tC, &topovec[itopo]);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }

    
    /* Map the contraction indices of A and B */
    ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, 
                              tA, tB, &topovec[itopo]);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }


  /*  ret = map_self_indices(tA, map_A);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }
    ret = map_self_indices(tB, map_B);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }
    ret = map_self_indices(tC, map_C);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }*/
    ret = map_extra_indices(idx_arr, idx_extra, num_extra,
                                tA, tB, tC);
    if (ret == CTF_NEGATIVE) {
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      CTF_free(idx_arr);
      return CTF_ERROR;
    }


    /* Map C or equivalently, the non-contraction indices of A and B */
    ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, 
                                  tA, tB, tC, &topovec[itopo]);
    if (ret == CTF_NEGATIVE){
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      return CTF_ERROR;
    }
    ret = map_symtsr(tsr_A->order, tsr_A->sym_table, tsr_A->edge_map);
    if (ret!=CTF_SUCCESS) return ret;
    ret = map_symtsr(tsr_B->order, tsr_B->sym_table, tsr_B->edge_map);
    if (ret!=CTF_SUCCESS) return ret;
    ret = map_symtsr(tsr_C->order, tsr_C->sym_table, tsr_C->edge_map);
    if (ret!=CTF_SUCCESS) return ret;

    /* Do it again to make sure everything is properly mapped. FIXME: loop */
    ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, 
                              tA, tB, &topovec[itopo]);
    if (ret == CTF_NEGATIVE){
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      return CTF_ERROR;
    }
    ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, 
                                  tA, tB, tC, &topovec[itopo]);
    if (ret == CTF_NEGATIVE){
      CTF_free(idx_arr);
      return CTF_NEGATIVE;
    }
    if (ret == CTF_ERROR) {
      return CTF_ERROR;
    }

    /*ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, 
                              tA, tB, &topovec[itopo]);*/
    /* Map C or equivalently, the non-contraction indices of A and B */
    /*ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, 
                                  tA, tB, tC, &topovec[itopo]);*/
    ret = map_symtsr(tsr_A->order, tsr_A->sym_table, tsr_A->edge_map);
    if (ret!=CTF_SUCCESS) return ret;
    ret = map_symtsr(tsr_B->order, tsr_B->sym_table, tsr_B->edge_map);
    if (ret!=CTF_SUCCESS) return ret;
    ret = map_symtsr(tsr_C->order, tsr_C->sym_table, tsr_C->edge_map);
    if (ret!=CTF_SUCCESS) return ret;
    
    CTF_free(idx_arr);

    return CTF_SUCCESS;

  }

  int contraction::try_topo_morph(){
    int itA, itB, itC, ret;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    tensor<dtype> * tsr_keep, * tsr_change_A, * tsr_change_B;
    
    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    tsr_C = tensors[tid_C];

    itA = tsr_A->itopo;
    itB = tsr_B->itopo;
    itC = tsr_C->itopo;

    if (itA == itB && itB == itC){
      return CTF_SUCCESS;
    }

    if (topovec[itA].order >= topovec[itB].order){
      if (topovec[itA].order >= topovec[itC].order){
        tsr_keep = tsr_A;
        tsr_change_A = tsr_B;
        tsr_change_B = tsr_C;
      } else {
        tsr_keep = tsr_C;
        tsr_change_A = tsr_A;
        tsr_change_B = tsr_B;
      } 
    } else {
      if (topovec[itB].order >= topovec[itC].order){
        tsr_keep = tsr_B;
        tsr_change_A = tsr_A;
        tsr_change_B = tsr_C;
      } else {
        tsr_keep = tsr_C;
        tsr_change_A = tsr_A;
        tsr_change_B = tsr_B;
      }
    }
    
    itA = tsr_change_A->itopo;
    itB = tsr_change_B->itopo;
    itC = tsr_keep->itopo;

    if (itA != itC){
      ret = can_morph(&topovec[itC], &topovec[itA]);
      if (!ret)
        return CTF_NEGATIVE;
    }
    if (itB != itC){
      ret = can_morph(&topovec[itC], &topovec[itB]);
      if (!ret)
        return CTF_NEGATIVE;
    }
    
    if (itA != itC){
      morph_topo(&topovec[itC], &topovec[itA], 
                 tsr_change_A->order, tsr_change_A->edge_map);
      tsr_change_A->itopo = itC;
    }
    if (itB != itC){
      morph_topo(&topovec[itC], &topovec[itB], 
                 tsr_change_B->order, tsr_change_B->edge_map);
      tsr_change_B->itopo = itC;
    }
    return CTF_SUCCESS;

  }

  int contraction::map(ctr ** ctrf){
    int num_tot, i, ret, j, need_remap, d;
    int need_remap_A, need_remap_B, need_remap_C;
    uint64_t memuse;//, bmemuse;
    double est_time, best_time;
    int btopo, gtopo;
    int old_nvirt_all;
    int was_cyclic_A, was_cyclic_B, was_cyclic_C, nvirt_all;
    int64_t old_size_A, old_size_B, old_size_C;
    int64_t nvirt;
    int * idx_arr, * idx_ctr, * idx_no_ctr, * idx_extra, * idx_weigh;
    int * old_phase_A, * old_rank_A, * old_virt_dim_A, * old_pe_lda_A;
    int * old_padding_A, * old_edge_len_A;
    int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B;
    int * old_padding_B, * old_edge_len_B;
    int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
    int * old_padding_C, * old_edge_len_C;
    mapping * old_map_A, * old_map_B, * old_map_C;
    int old_topo_A, old_topo_B, old_topo_C;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    ctr<dtype> * sctr;
    old_topo_A = -1;
    old_topo_B = -1;
    old_topo_C = -1;

    TAU_FSTART(map_tensors);

    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];

    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);

    CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_no_ctr);
    CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_extra);
    CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_weigh);
    CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_ctr);
  #if BEST_VOL
    CTF_alloc_ptr(sizeof(int)*tsr_A->order,     (void**)&virt_blk_len_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order,     (void**)&virt_blk_len_B);
    CTF_alloc_ptr(sizeof(int)*tsr_C->order,     (void**)&virt_blk_len_C);
  #endif
    old_map_A = NULL;
    old_map_B = NULL;
    old_map_C = NULL;

    CTF_alloc_ptr(sizeof(mapping)*tsr_A->order,         (void**)&old_map_A);
    CTF_alloc_ptr(sizeof(mapping)*tsr_B->order,         (void**)&old_map_B);
    CTF_alloc_ptr(sizeof(mapping)*tsr_C->order,         (void**)&old_map_C);
    
    for (i=0; i<tsr_A->order; i++){
      old_map_A[i].type         = VIRTUAL_MAP;
      old_map_A[i].has_child    = 0;
      old_map_A[i].np           = 1;
    }
    old_topo_A = -1;
    if (tsr_A->is_mapped){
      copy_mapping(tsr_A->order, tsr_A->edge_map, old_map_A);
      old_topo_A = tsr_A->itopo;
    } 
    
    for (i=0; i<tsr_B->order; i++){
      old_map_B[i].type         = VIRTUAL_MAP;
      old_map_B[i].has_child    = 0;
      old_map_B[i].np           = 1;
    }
    old_topo_B = -1;
    if (tsr_B->is_mapped){
      copy_mapping(tsr_B->order, tsr_B->edge_map, old_map_B);
      old_topo_B = tsr_B->itopo;
    }

    for (i=0; i<tsr_C->order; i++){
      old_map_C[i].type         = VIRTUAL_MAP;
      old_map_C[i].has_child    = 0;
      old_map_C[i].np           = 1;
    }
    old_topo_C = -1;
    if (tsr_C->is_mapped){
      copy_mapping(tsr_C->order, tsr_C->edge_map, old_map_C);
      old_topo_C = tsr_C->itopo;
    } 

    copy_mapping(tsr_B->order, tsr_B->edge_map, old_map_B);
    copy_mapping(tsr_C->order, tsr_C->edge_map, old_map_C);
    old_topo_B = tsr_B->itopo;
    old_topo_C = tsr_C->itopo;
    if (do_remap){
      ASSERT(tsr_A->is_mapped);
      ASSERT(tsr_B->is_mapped);
      ASSERT(tsr_C->is_mapped);
    #if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("Initial mappings:\n");
      print_map(stdout, type->tid_A);
      print_map(stdout, type->tid_B);
      print_map(stdout, type->tid_C);
    #endif
      unmap_inner(tsr_A);
      unmap_inner(tsr_B);
      unmap_inner(tsr_C);
      set_padding(tsr_A);
      set_padding(tsr_B);
      set_padding(tsr_C);
      /* Save the current mappings of A, B, C */
      save_mapping(tsr_A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A, 
                   &old_size_A, &was_cyclic_A, &old_padding_A, 
                   &old_edge_len_A, &topovec[tsr_A->itopo]);
      save_mapping(tsr_B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B, 
                   &old_size_B, &was_cyclic_B, &old_padding_B, 
                   &old_edge_len_B, &topovec[tsr_B->itopo]);
      save_mapping(tsr_C, &old_phase_C, &old_rank_C, &old_virt_dim_C, &old_pe_lda_C, 
                   &old_size_C, &was_cyclic_C, &old_padding_C, 
                   &old_edge_len_C, &topovec[tsr_C->itopo]);
    } else {
      CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&old_phase_A);
      for (j=0; j<tsr_A->order; j++){
        old_phase_A[j]   = calc_phase(tsr_A->edge_map+j);
      }
      CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&old_phase_B);
      for (j=0; j<tsr_B->order; j++){
        old_phase_B[j]   = calc_phase(tsr_B->edge_map+j);
      }
      CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&old_phase_C);
      for (j=0; j<tsr_C->order; j++){
        old_phase_C[j]   = calc_phase(tsr_C->edge_map+j);
      }
    }
    btopo = -1;
    best_time = DBL_MAX;
    //bmemuse = UINT64_MAX;

    for (j=0; j<6; j++){
      /* Attempt to map to all possible permutations of processor topology */
  #if DEBUG < 3 
      for (int t=global_comm.rank; t<(int)topovec.size()+3; t+=global_comm.np){
  #else
      for (int t=global_comm.rank*(topovec.size()+3); t<(int)topovec.size()+3; t++){
  #endif
        clear_mapping(tsr_A);
        clear_mapping(tsr_B);
        clear_mapping(tsr_C);
        set_padding(tsr_A);
        set_padding(tsr_B);
        set_padding(tsr_C);
      
        if (t < 3){
          switch (t){
            case 0:
            if (old_topo_A == -1) continue;
            i = old_topo_A;
            copy_mapping(tsr_A->order, old_map_A, tsr_A->edge_map);
            break;
          
            case 1:
            if (old_topo_B == -1) continue;
            i = old_topo_B;
            copy_mapping(tsr_B->order, old_map_B, tsr_B->edge_map);
            break;

            case 2:
            if (old_topo_C == -1) continue;
            i = old_topo_C;
            copy_mapping(tsr_C->order, old_map_C, tsr_C->edge_map);
            break;
          }
        } else i = t-3;
      

        ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                              type->idx_map_B, type->idx_map_C, i, j, 
                              idx_arr, idx_ctr, idx_extra, idx_no_ctr, idx_weigh);
        

        if (ret == CTF_ERROR) {
          TAU_FSTOP(map_tensors);
          return CTF_ERROR;
        }
        if (ret == CTF_NEGATIVE){
          //printf("map_to_topology returned negative\n");
          continue;
        }
    
        tsr_A->is_mapped = 1;
        tsr_B->is_mapped = 1;
        tsr_C->is_mapped = 1;
        tsr_A->itopo = i;
        tsr_B->itopo = i;
        tsr_C->itopo = i;
  #if DEBUG >= 3
        printf("\nTest mappings:\n");
        print_map(stdout, type->tid_A, 0);
        print_map(stdout, type->tid_B, 0);
        print_map(stdout, type->tid_C, 0);
  #endif
        
        if (check_contraction_mapping(type) == 0) continue;
        est_time = 0.0;
        
        nvirt_all = -1;
        old_nvirt_all = -2;
  #if 0
        while (nvirt_all < MIN_NVIRT){
          old_nvirt_all = nvirt_all;
          set_padding(tsr_A);
          set_padding(tsr_B);
          set_padding(tsr_C);
          sctr = construct_contraction(type, buffer, buffer_len, func_ptr, 
                                        alpha, beta, 0, NULL, &nvirt_all);
          /* If this cannot be stretched */
          if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
            clear_mapping(tsr_A);
            clear_mapping(tsr_B);
            clear_mapping(tsr_C);
            set_padding(tsr_A);
            set_padding(tsr_B);
            set_padding(tsr_C);

            ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                                  type->idx_map_B, type->idx_map_C, i, j, 
                                  idx_arr, idx_ctr, idx_extra, idx_no_ctr);
            tsr_A->is_mapped = 1;
            tsr_B->is_mapped = 1;
            tsr_C->is_mapped = 1;
            tsr_A->itopo = i;
            tsr_B->itopo = i;
            tsr_C->itopo = i;
            break;

          }
          if (nvirt_all < MIN_NVIRT){
            stretch_virt(tsr_A->order, 2, tsr_A->edge_map);
            stretch_virt(tsr_B->order, 2, tsr_B->edge_map);
            stretch_virt(tsr_C->order, 2, tsr_C->edge_map);
          }
        }
  #endif
        set_padding(tsr_A);
        set_padding(tsr_B);
        set_padding(tsr_C);
        sctr = construct_contraction(type, ftsr, felm, 
                                      alpha, beta, 0, NULL, &nvirt_all, 0);
       
        est_time = sctr->est_time_rec(sctr->num_lyr);
        //sctr->print();
  #if DEBUG >= 3
        printf("mapping passed contr est_time = %lf sec\n", est_time);
  #endif 
        ASSERT(est_time > 0.0);
        memuse = 0;
        need_remap_A = 0;
        need_remap_B = 0;
        need_remap_C = 0;
        if (i == old_topo_A){
          for (d=0; d<tsr_A->order; d++){
            if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
              need_remap_A = 1;
          }
        } else
          need_remap_A = 1;
        if (need_remap_A) {
          nvirt = (uint64_t)calc_nvirt(tsr_A);
          est_time += global_comm.estimate_alltoallv_time(sizeof(dtype)*tsr_A->size);
          if (can_block_reshuffle(tsr_A->order, old_phase_A, tsr_A->edge_map)){
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_A->size);
          } else {
            est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_A->size+global_comm.estimate_alltoall_time(nvirt);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_A->size;
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_A->size*2.5);
          }
        } else
          memuse = 0;
        if (i == old_topo_B){
          for (d=0; d<tsr_B->order; d++){
            if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
              need_remap_B = 1;
          }
        } else
          need_remap_B = 1;
        if (need_remap_B) {
          nvirt = (uint64_t)calc_nvirt(tsr_B);
          est_time += global_comm.estimate_alltoallv_time(sizeof(dtype)*tsr_B->size);
          if (can_block_reshuffle(tsr_B->order, old_phase_B, tsr_B->edge_map)){
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_B->size);
          } else {
            est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_B->size+global_comm.estimate_alltoall_time(nvirt);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_B->size;
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_B->size*2.5);
          }
        }
        if (i == old_topo_C){
          for (d=0; d<tsr_C->order; d++){
            if (!comp_dim_map(&tsr_C->edge_map[d],&old_map_C[d]))
              need_remap_C = 1;
          }
        } else
          need_remap_C = 1;
        if (need_remap_C) {
          nvirt = (uint64_t)calc_nvirt(tsr_C);
          est_time += global_comm.estimate_alltoallv_time(sizeof(dtype)*tsr_B->size);
          if (can_block_reshuffle(tsr_C->order, old_phase_C, tsr_C->edge_map)){
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_C->size);
          } else {
            est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_C->size+global_comm.estimate_alltoall_time(nvirt);
            if (nvirt > 1) 
              est_time += 5.*COST_MEMBW*sizeof(dtype)*tsr_C->size;
            memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_C->size*2.5);
          }
        }
        memuse = MAX((uint64_t)sctr->mem_rec(), memuse);
  #if DEBUG >= 3
        printf("total (with redistribution) est_time = %lf\n", est_time);
  #endif
        ASSERT(est_time > 0.0);

        if ((uint64_t)memuse >= proc_bytes_available()){
          DPRINTF(2,"Not enough memory available for topo %d with order %d\n", i, j);
          delete sctr;
          continue;
        } 

        /* be careful about overflow */
  /*      nvirt = (uint64_t)calc_nvirt(tsr_A);
        tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_B);
        if (tnvirt < nvirt) nvirt = UINT64_MAX;
        else {
          nvirt = tnvirt;
          tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_C);
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
    COMM_BARRIER(global_comm);
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
    DEBUG_PRINTF("bnvirt = " PRIu64 "\n", (uint64_t)bnvirt);
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
    ALLREDUCE(&best_time, &gbest_time, 1, MPI_DOUBLE, MPI_MIN, global_comm);
    if (best_time != gbest_time){
      btopo = INT_MAX;
    }
    int ttopo;
    ALLREDUCE(&btopo, &ttopo, 1, MPI_INT, MPI_MIN, global_comm);
    
    clear_mapping(tsr_A);
    clear_mapping(tsr_B);
    clear_mapping(tsr_C);
    set_padding(tsr_A);
    set_padding(tsr_B);
    set_padding(tsr_C);
    
    if (!do_remap || ttopo == INT_MAX || ttopo == -1){
      CTF_free((void*)idx_arr);
      CTF_free((void*)idx_no_ctr);
      CTF_free((void*)idx_ctr);
      CTF_free((void*)idx_extra);
      CTF_free((void*)idx_weigh);
      CTF_free(old_phase_A);
      CTF_free(old_phase_B);
      CTF_free(old_phase_C);
      for (i=0; i<tsr_A->order; i++)
        clear_mapping(old_map_A+i);
      for (i=0; i<tsr_B->order; i++)
        clear_mapping(old_map_B+i);
      for (i=0; i<tsr_C->order; i++)
        clear_mapping(old_map_C+i);
      CTF_free(old_map_A);
      CTF_free(old_map_B);
      CTF_free(old_map_C);

      TAU_FSTOP(map_tensors);
      if (ttopo == INT_MAX || ttopo == -1){
        printf("ERROR: Failed to map contraction!\n");
        //ABORT;
        return CTF_ERROR;
      }
      return CTF_SUCCESS;
    }
    if (ttopo < 18){
      switch (ttopo/6){
        case 0:
        gtopo = old_topo_A*6+(ttopo%6);
        copy_mapping(tsr_A->order, old_map_A, tsr_A->edge_map);
        break;
      
        case 1:
        gtopo = old_topo_B*6+(ttopo%6);
        copy_mapping(tsr_B->order, old_map_B, tsr_B->edge_map);
        break;

        case 2:
        gtopo = old_topo_C*6+(ttopo%6);
        copy_mapping(tsr_C->order, old_map_C, tsr_C->edge_map);
        break;
      }
    } else gtopo=ttopo-18;
   

    tsr_A->itopo = gtopo/6;
    tsr_B->itopo = gtopo/6;
    tsr_C->itopo = gtopo/6;
    
    ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                          type->idx_map_B, type->idx_map_C, gtopo/6, gtopo%6, 
                          idx_arr, idx_ctr, idx_extra, idx_no_ctr, idx_weigh);


    if (ret == CTF_NEGATIVE || ret == CTF_ERROR) {
      printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
      TAU_FSTOP(map_tensors);
      return CTF_ERROR;
    }
    tsr_A->is_mapped = 1;
    tsr_B->is_mapped = 1;
    tsr_C->is_mapped = 1;
  #if DEBUG > 2
    if (!check_contraction_mapping(type))
      printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
  //  else if (global_comm.rank == 0) printf("Mapping successful estimated execution time = %lf sec\n",best_time);
  #endif
    ASSERT(check_contraction_mapping(type));


    nvirt_all = -1;
    old_nvirt_all = -2;
    while (nvirt_all < MIN_NVIRT){
      old_nvirt_all = nvirt_all;
      set_padding(tsr_A);
      set_padding(tsr_B);
      set_padding(tsr_C);
      *ctrf = construct_contraction(type, ftsr, felm, 
                                    alpha, beta, 0, NULL, &nvirt_all, 0);
      delete *ctrf;
      /* If this cannot be stretched */
      if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
        clear_mapping(tsr_A);
        clear_mapping(tsr_B);
        clear_mapping(tsr_C);
        set_padding(tsr_A);
        set_padding(tsr_B);
        set_padding(tsr_C);
        tsr_A->itopo = gtopo/6;
        tsr_B->itopo = gtopo/6;
        tsr_C->itopo = gtopo/6;

        ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                              type->idx_map_B, type->idx_map_C, gtopo/6, gtopo%6, 
                              idx_arr, idx_ctr, idx_extra, idx_no_ctr, idx_weigh);
        tsr_A->is_mapped = 1;
        tsr_B->is_mapped = 1;
        tsr_C->is_mapped = 1;
        break;
      }
      if (nvirt_all < MIN_NVIRT){
        stretch_virt(tsr_A->order, 2, tsr_A->edge_map);
        stretch_virt(tsr_B->order, 2, tsr_B->edge_map);
        stretch_virt(tsr_C->order, 2, tsr_C->edge_map);
      }
    }
    set_padding(tsr_A);
    set_padding(tsr_B);
    set_padding(tsr_C);
    *ctrf = construct_contraction(type, ftsr, felm, 
                                  alpha, beta, 0, NULL, &nvirt_all, 1);
  #if DEBUG >= 2
    if (global_comm.rank == 0)
      printf("New mappings:\n");
    print_map(stdout, type->tid_A);
    print_map(stdout, type->tid_B);
    print_map(stdout, type->tid_C);
  #endif
   
        
    memuse = MAX((uint64_t)(*ctrf)->mem_rec(), (uint64_t)(tsr_A->size+tsr_B->size+tsr_C->size)*sizeof(dtype)*3);
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      VPRINTF(1,"Contraction will use %E bytes per processor out of %E available memory and take an estimated of %lf sec\n",
              (double)memuse,(double)proc_bytes_available(),gbest_time);
  #endif          

    if (tsr_A->is_cyclic == 0 &&
        tsr_B->is_cyclic == 0 &&
        tsr_C->is_cyclic == 0){
      tsr_A->is_cyclic = 0;
      tsr_B->is_cyclic = 0;
      tsr_C->is_cyclic = 0;
    } else {
      tsr_A->is_cyclic = 1;
      tsr_B->is_cyclic = 1;
      tsr_C->is_cyclic = 1;
    }
    TAU_FSTOP(map_tensors);
    /* redistribute tensor data */
    TAU_FSTART(redistribute_for_contraction);
    need_remap = 0;
    if (tsr_A->itopo == old_topo_A){
      for (d=0; d<tsr_A->order; d++){
        if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      remap_tensor(type->tid_A, tsr_A, &topovec[tsr_A->itopo], old_size_A, 
                   old_phase_A, old_rank_A, old_virt_dim_A, 
                   old_pe_lda_A, was_cyclic_A, 
                   old_padding_A, old_edge_len_A, global_comm);
    need_remap = 0;
    if (tsr_B->itopo == old_topo_B){
      for (d=0; d<tsr_B->order; d++){
        if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      remap_tensor(type->tid_B, tsr_B, &topovec[tsr_A->itopo], old_size_B, 
                   old_phase_B, old_rank_B, old_virt_dim_B, 
                   old_pe_lda_B, was_cyclic_B, 
                   old_padding_B, old_edge_len_B, global_comm);
    need_remap = 0;
    if (tsr_C->itopo == old_topo_C){
      for (d=0; d<tsr_C->order; d++){
        if (!comp_dim_map(&tsr_C->edge_map[d],&old_map_C[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      remap_tensor(type->tid_C, tsr_C, &topovec[tsr_A->itopo], old_size_C, 
                   old_phase_C, old_rank_C, old_virt_dim_C, 
                   old_pe_lda_C, was_cyclic_C, 
                   old_padding_C, old_edge_len_C, global_comm);
                   
    TAU_FSTOP(redistribute_for_contraction);
    
    (*ctrf)->A    = tsr_A->data;
    (*ctrf)->B    = tsr_B->data;
    (*ctrf)->C    = tsr_C->data;

    CTF_free( old_phase_A );          CTF_free( old_rank_A );
    CTF_free( old_virt_dim_A );       CTF_free( old_pe_lda_A );
    CTF_free( old_padding_A );        CTF_free( old_edge_len_A );
    CTF_free( old_phase_B );          CTF_free( old_rank_B );
    CTF_free( old_virt_dim_B );       CTF_free( old_pe_lda_B );
    CTF_free( old_padding_B );        CTF_free( old_edge_len_B );
    CTF_free( old_phase_C );          CTF_free( old_rank_C );
    CTF_free( old_virt_dim_C );       CTF_free( old_pe_lda_C );
    CTF_free( old_padding_C );        CTF_free( old_edge_len_C );
    
    for (i=0; i<tsr_A->order; i++)
      clear_mapping(old_map_A+i);
    for (i=0; i<tsr_B->order; i++)
      clear_mapping(old_map_B+i);
    for (i=0; i<tsr_C->order; i++)
      clear_mapping(old_map_C+i);
    CTF_free(old_map_A);
    CTF_free(old_map_B);
    CTF_free(old_map_C);

    CTF_free((void*)idx_arr);
    CTF_free((void*)idx_no_ctr);
    CTF_free((void*)idx_ctr);
    CTF_free((void*)idx_extra);
    CTF_free((void*)idx_weigh);
    


    return CTF_SUCCESS;
  }


  ctr * construct_ctr(int            is_inner,
                      iparam const * inner_params,
                      int *          nvirt_C,
                      int            is_used=1){
    int num_tot, i, i_A, i_B, i_C, is_top, j, nphys_dim,  k;
    int64_t nvirt;
    int64_t blk_sz_A, blk_sz_B, blk_sz_C;
    int64_t vrt_sz_A, vrt_sz_B, vrt_sz_C;
    int sA, sB, sC, need_rep;
    int * blk_len_A, * virt_blk_len_A, * blk_len_B;
    int * virt_blk_len_B, * blk_len_C, * virt_blk_len_C;
    int * idx_arr, * virt_dim, * phys_mapped;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    strp_tsr<dtype> * str_A, * str_B, * str_C;
    mapping * map;
    ctr<dtype> * hctr = NULL;
    ctr<dtype> ** rec_ctr = NULL;

    TAU_FSTART(construct_contraction);

    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];

    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
            &num_tot, &idx_arr);

    nphys_dim = topovec[tsr_A->itopo].order;

    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&virt_blk_len_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&virt_blk_len_B);
    CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&virt_blk_len_C);

    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&blk_len_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&blk_len_B);
    CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&blk_len_C);
    CTF_alloc_ptr(sizeof(int)*num_tot, (void**)&virt_dim);
    CTF_alloc_ptr(sizeof(int)*nphys_dim*3, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*3);


    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = tsr_A->size;
    blk_sz_B = tsr_B->size;
    blk_sz_C = tsr_C->size;
    calc_dim(tsr_A->order, blk_sz_A, tsr_A->edge_len, tsr_A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(tsr_B->order, blk_sz_B, tsr_B->edge_len, tsr_B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);
    calc_dim(tsr_C->order, blk_sz_C, tsr_C->edge_len, tsr_C->edge_map,
             &vrt_sz_C, virt_blk_len_C, blk_len_C);

    /* Strip out the relevant part of the tensor if we are contracting over diagonal */
    sA = strip_diag<dtype>( tsr_A->order, num_tot, type->idx_map_A, vrt_sz_A,
                            tsr_A->edge_map, &topovec[tsr_A->itopo],
                            blk_len_A, &blk_sz_A, &str_A);
    sB = strip_diag<dtype>( tsr_B->order, num_tot, type->idx_map_B, vrt_sz_B,
                            tsr_B->edge_map, &topovec[tsr_B->itopo],
                            blk_len_B, &blk_sz_B, &str_B);
    sC = strip_diag<dtype>( tsr_C->order, num_tot, type->idx_map_C, vrt_sz_C,
                            tsr_C->edge_map, &topovec[tsr_C->itopo],
                            blk_len_C, &blk_sz_C, &str_C);

    is_top = 1;
    if (sA || sB || sC){
      if (global_comm.rank == 0)
        DPRINTF(1,"Stripping tensor\n");
      strp_ctr<dtype> * sctr = new strp_ctr<dtype>;
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

    for (i=0; i<tsr_A->order; i++){
      map = &tsr_A->edge_map[i];
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
    for (i=0; i<tsr_B->order; i++){
      map = &tsr_B->edge_map[i];
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
    for (i=0; i<tsr_C->order; i++){
      map = &tsr_C->edge_map[i];
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

      ctr_replicate<dtype> * rctr = new ctr_replicate<dtype>;
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
      for (i=0; i<nphys_dim; i++){
        if (phys_mapped[3*i+0] == 0 &&
            phys_mapped[3*i+1] == 0 &&
            phys_mapped[3*i+2] == 0){
  /*        printf("ERROR: ALL-TENSOR REPLICATION NO LONGER DONE\n");
          ABORT;
          ASSERT(rctr->num_lyr == 1);
          hctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i].rank;
          hctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;
          rctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i].rank;
          rctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;*/
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
        CTF_alloc_ptr(sizeof(CommData)*rctr->ncdt_A, (void**)&rctr->cdt_A);
      if (rctr->ncdt_B > 0)
        CTF_alloc_ptr(sizeof(CommData)*rctr->ncdt_B, (void**)&rctr->cdt_B);
      if (rctr->ncdt_C > 0)
        CTF_alloc_ptr(sizeof(CommData)*rctr->ncdt_C, (void**)&rctr->cdt_C);
      rctr->ncdt_A = 0;
      rctr->ncdt_B = 0;
      rctr->ncdt_C = 0;
      for (i=0; i<nphys_dim; i++){
        if (!(phys_mapped[3*i+0] == 0 &&
              phys_mapped[3*i+1] == 0 &&
              phys_mapped[3*i+2] == 0)){
          if (phys_mapped[3*i+0] == 0){
            rctr->cdt_A[rctr->ncdt_A] = topovec[tsr_A->itopo].dim_comm[i];
            if (is_used && rctr->cdt_A[rctr->ncdt_A].alive == 0)
              SHELL_SPLIT(global_comm, rctr->cdt_A[rctr->ncdt_A]);
            rctr->ncdt_A++;
          }
          if (phys_mapped[3*i+1] == 0){
            rctr->cdt_B[rctr->ncdt_B] = topovec[tsr_B->itopo].dim_comm[i];
            if (is_used && rctr->cdt_B[rctr->ncdt_B].alive == 0)
              SHELL_SPLIT(global_comm, rctr->cdt_B[rctr->ncdt_B]);
            rctr->ncdt_B++;
          }
          if (phys_mapped[3*i+2] == 0){
            rctr->cdt_C[rctr->ncdt_C] = topovec[tsr_C->itopo].dim_comm[i];
            if (is_used && rctr->cdt_C[rctr->ncdt_C].alive == 0)
              SHELL_SPLIT(global_comm, rctr->cdt_C[rctr->ncdt_C]);
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

    ctr_2d_general<dtype> * bottom_ctr_gen = NULL;
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
        ctr_2d_general<dtype> * ctr_gen = new ctr_2d_general<dtype>;
        ctr_gen->buffer = NULL; //fix learn to use buffer space
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
                                      topovec,
                                      tsr_A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      tsr_B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      tsr_C,
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
                                      topovec,
                                      tsr_B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      tsr_C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      tsr_A,
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
                                      topovec,
                                      tsr_C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      tsr_A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      tsr_B,
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
          map = &tsr_A->edge_map[i_A];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_B != -1){
          map = &tsr_B->edge_map[i_B];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_C != -1){
          map = &tsr_C->edge_map[i_C];
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
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, tsr_A->sym, sizeof(int)*tsr_A->order);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, tsr_B->sym, sizeof(int)*tsr_B->order);
    CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&new_sym_C);
    memcpy(new_sym_C, tsr_C->sym, sizeof(int)*tsr_C->order);

  #ifdef OFFLOAD
    if (ftsr.is_offloadable || is_inner > 0){
      if (bottom_ctr_gen != NULL)
        bottom_ctr_gen->alloc_host_buf = true;
      ctr_offload<dtype> * ctroff = new ctr_offload<dtype>;
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
      ctr_virt_25d<dtype> * ctrv = new ctr_virt_25d<dtype>;
  #else
      ctr_virt<dtype> * ctrv = new ctr_virt<dtype>;
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

      ctrv->num_dim   = num_tot;
      ctrv->virt_dim  = virt_dim;
      ctrv->order_A  = tsr_A->order;
      ctrv->blk_sz_A  = vrt_sz_A;
      ctrv->idx_map_A = type->idx_map_A;
      ctrv->order_B  = tsr_B->order;
      ctrv->blk_sz_B  = vrt_sz_B;
      ctrv->idx_map_B = type->idx_map_B;
      ctrv->order_C  = tsr_C->order;
      ctrv->blk_sz_C  = vrt_sz_C;
      ctrv->idx_map_C = type->idx_map_C;
      ctrv->buffer  = NULL;
    } else
      CTF_free(virt_dim);

    seq_tsr_ctr<dtype> * ctrseq = new seq_tsr_ctr<dtype>;
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
      ctrseq->func_ptr  = ftsr;
    } else if (is_inner == 1) {
      ctrseq->is_inner    = 1;
      ctrseq->inner_params  = *inner_params;
      ctrseq->inner_params.sz_C = vrt_sz_C;
      tensor<dtype> * itsr;
      int * iphase;
      itsr = tensors[tsr_A->rec_tid];
      iphase = calc_phase<dtype>(itsr);
      for (i=0; i<tsr_A->order; i++){
        if (virt_blk_len_A[i]%iphase[i] > 0)
          virt_blk_len_A[i] = virt_blk_len_A[i]/iphase[i]+1;
        else
          virt_blk_len_A[i] = virt_blk_len_A[i]/iphase[i];

      }
      CTF_free(iphase);
      itsr = tensors[tsr_B->rec_tid];
      iphase = calc_phase<dtype>(itsr);
      for (i=0; i<tsr_B->order; i++){
        if (virt_blk_len_B[i]%iphase[i] > 0)
          virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i]+1;
        else
          virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i];
      }
      CTF_free(iphase);
      itsr = tensors[tsr_C->rec_tid];
      iphase = calc_phase<dtype>(itsr);
      for (i=0; i<tsr_C->order; i++){
        if (virt_blk_len_C[i]%iphase[i] > 0)
          virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i]+1;
        else
          virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i];
      }
      CTF_free(iphase);
    } else if (is_inner == 2) {
      if (global_comm.rank == 0){
        DPRINTF(1,"Folded tensor n=%d m=%d k=%d\n", inner_params->n,
          inner_params->m, inner_params->k);
      }

      ctrseq->is_inner    = 1;
      ctrseq->inner_params  = *inner_params;
      ctrseq->inner_params.sz_C = vrt_sz_C;
      tensor<dtype> * itsr;
      itsr = tensors[tsr_A->rec_tid];
      for (i=0; i<itsr->order; i++){
        j = tsr_A->inner_ordering[i];
        for (k=0; k<tsr_A->order; k++){
          if (tsr_A->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && tsr_A->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
  /*        printf("inner_ordering[%d]=%d setting dim %d of A, to len %d from len %d\n",
                  i, tsr_A->inner_ordering[i], k, 1, virt_blk_len_A[k]);*/
          virt_blk_len_A[k] = 1;
          new_sym_A[k] = NS;
        }
      }
      itsr = tensors[tsr_B->rec_tid];
      for (i=0; i<itsr->order; i++){
        j = tsr_B->inner_ordering[i];
        for (k=0; k<tsr_B->order; k++){
          if (tsr_B->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && tsr_B->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of B, to len %d from len %d\n",
                  i, tsr_B->inner_ordering[i], k, 1, virt_blk_len_B[k]);*/
          virt_blk_len_B[k] = 1;
          new_sym_B[k] = NS;
        }
      }
      itsr = tensors[tsr_C->rec_tid];
      for (i=0; i<itsr->order; i++){
        j = tsr_C->inner_ordering[i];
        for (k=0; k<tsr_C->order; k++){
          if (tsr_C->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && tsr_C->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of C, to len %d from len %d\n",
                  i, tsr_C->inner_ordering[i], k, 1, virt_blk_len_C[k]);*/
          virt_blk_len_C[k] = 1;
          new_sym_C[k] = NS;
        }
      }
    }
    ctrseq->alpha         = alpha;
    ctrseq->order_A        = tsr_A->order;
    ctrseq->idx_map_A     = type->idx_map_A;
    ctrseq->edge_len_A    = virt_blk_len_A;
    ctrseq->sym_A         = new_sym_A;
    ctrseq->order_B        = tsr_B->order;
    ctrseq->idx_map_B     = type->idx_map_B;
    ctrseq->edge_len_B    = virt_blk_len_B;
    ctrseq->sym_B         = new_sym_B;
    ctrseq->order_C        = tsr_C->order;
    ctrseq->idx_map_C     = type->idx_map_C;
    ctrseq->edge_len_C    = virt_blk_len_C;
    ctrseq->sym_C         = new_sym_C;
    ctrseq->custom_params = felm;
    ctrseq->is_custom     = (felm.func_ptr != NULL);

    hctr->A   = tsr_A->data;
    hctr->B   = tsr_B->data;
    hctr->C   = tsr_C->data;
    hctr->beta  = beta;
  /*  if (global_comm.rank == 0){
      int64_t n,m,k;
      dtype old_flops;
      dtype new_flops;
      ggg_sym_nmk(tsr_A->order, tsr_A->edge_len, type->idx_map_A, tsr_A->sym,
      tsr_B->order, tsr_B->edge_len, type->idx_map_B, tsr_B->sym,
      tsr_C->order, &n, &m, &k);
      old_flops = 2.0*(dtype)n*(dtype)m*(dtype)k;
      new_flops = calc_nvirt(tsr_A);
      new_flops *= calc_nvirt(tsr_B);
      new_flops *= calc_nvirt(tsr_C);
      new_flops *= global_comm.np;
      new_flops = sqrt(new_flops);
      new_flops *= global_comm.np;
      ggg_sym_nmk(tsr_A->order, virt_blk_len_A, type->idx_map_A, tsr_A->sym,
      tsr_B->order, virt_blk_len_B, type->idx_map_B, tsr_B->sym,
      tsr_C->order, &n, &m, &k);
      printf("Each subcontraction is a " PRId64 " by " PRId64 " by " PRId64 " DGEMM performing %E flops\n",n,m,k,
        2.0*(dtype)n*(dtype)m*(dtype)k);
      new_flops *= 2.0*(dtype)n*(dtype)m*(dtype)k;
      printf("Contraction performing %E flops rather than %E, a factor of %lf more flops due to padding\n",
        new_flops, old_flops, new_flops/old_flops);

    }*/

    CTF_free(idx_arr);
    CTF_free(blk_len_A);
    CTF_free(blk_len_B);
    CTF_free(blk_len_C);
    CTF_free(phys_mapped);
    TAU_FSTOP(construct_contraction);
    return hctr;
  }

  int contraction::contract(){
    int stat, new_tid;
    ctr<dtype> * ctrf;

    if (tensors[type->tid_A]->has_zero_edge_len || tensors[type->tid_B]->has_zero_edge_len
        || tensors[type->tid_C]->has_zero_edge_len){
      tensor<dtype> * tsr_C = tensors[type->tid_C];
      if (beta != 1.0 && !tsr_C->has_zero_edge_len){ 
        int * new_idx_map_C; 
        int num_diag = 0;
        new_idx_map_C = (int*)CTF_alloc(sizeof(int)*tsr_C->order);
        for (int i=0; i<tsr_C->order; i++){
          new_idx_map_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (type->idx_map_C[i] == type->idx_map_C[j]){
              new_idx_map_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        scale_tsr(beta, type->tid_C, new_idx_map_C, fs, felm); 
        CTF_free(new_idx_map_C);
      }
      return CTF_SUCCESS;
    }
    if (type->tid_A == type->tid_B || type->tid_A == type->tid_C){
      clone_tensor(type->tid_A, 1, &new_tid);
      CTF_ctr_type_t new_type = *type;
      new_type.tid_A = new_tid;
      stat = contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return stat;
    }
    if (type->tid_B == type->tid_C){
      clone_tensor(type->tid_B, 1, &new_tid);
      CTF_ctr_type_t new_type = *type;
      new_type.tid_B = new_tid;
      stat = contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return stat;
    }
  #if DEBUG >= 1 //|| VERBOSE >= 1)
    if (get_global_comm().rank == 0)
      printf("Contraction permutation:\n");
    print_ctr(type, alpha, beta);
  #endif

    TAU_FSTART(contract);
  #if VERIFY
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
    assert(stat == CTF_SUCCESS);

    stat = allread_tsr(type->tid_B, &nsB, &sB);
    assert(stat == CTF_SUCCESS);

    stat = allread_tsr(type->tid_C, &nC, &ans_C);
    assert(stat == CTF_SUCCESS);
  #endif
    /* Check if the current tensor mappings can be contracted on */
    fseq_tsr_ctr<dtype> fftsr=ftsr;
    if (ftsr.func_ptr == NULL){
      fftsr.func_ptr = &sym_seq_ctr_ref<dtype>;
  #ifdef OFFLOAD
      fftsr.is_offloadable = 0;
  #endif
    }
  #if REDIST
    stat = map_tensors(type, fftsr, felm, alpha, beta, &ctrf);
    if (stat == CTF_ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return CTF_ERROR;
    }
  #else
    if (check_contraction_mapping(type) == 0) {
      /* remap if necessary */
      stat = map_tensors(type, fftsr, felm, alpha, beta, &ctrf);
      if (stat == CTF_ERROR) {
        printf("Failed to map tensors to physical grid\n");
        return CTF_ERROR;
      }
    } else {
      /* Construct the tensor algorithm we would like to use */
  #if DEBUG >= 2
      if (get_global_comm().rank == 0)
        printf("Keeping mappings:\n");
      print_map(stdout, type->tid_A);
      print_map(stdout, type->tid_B);
      print_map(stdout, type->tid_C);
  #endif
      ctrf = construct_contraction(type, fftsr, felm, alpha, beta);
  #ifdef VERBOSE
      if (global_comm.rank == 0){
        uint64_t memuse = ctrf->mem_rec();
        DPRINTF(1,"Contraction does not require redistribution, will use %E bytes per processor out of %E available memory and take an estimated of %lf sec\n",
                (double)memuse,(double)proc_bytes_available(),ctrf->est_time_rec(1));
      }
  #endif
    }
  #endif
    ASSERT(check_contraction_mapping(type));
  #if FOLD_TSR
    if (felm.func_ptr == NULL && 
        ftsr.func_ptr == NULL && //sym_seq_ctr_ref<dtype> && 
        can_fold(type)){
      iparam prm;
      TAU_FSTART(map_fold);
      stat = map_fold(type, &prm);
      TAU_FSTOP(map_fold);
      if (stat == CTF_ERROR){
        return CTF_ERROR;
      }
      if (stat == CTF_SUCCESS){
        delete ctrf;
        ctrf = construct_contraction(type, fftsr, felm, alpha, beta, 2, &prm);
      }
    } 
  #endif
  #if DEBUG >=2
    if (get_global_comm().rank == 0)
      ctrf->print();
  #endif
  #if DEBUG >=1
    double dtt = MPI_Wtime();
    if (get_global_comm().rank == 0){
      DPRINTF(1,"[%d] performing contraction\n",
          get_global_comm().rank);
      DPRINTF(1,"%E bytes of buffer space will be needed for this contraction\n",
        (double)ctrf->mem_rec());
      DPRINTF(1,"System memory = %E bytes total, %E bytes used, %E bytes available.\n",
        (double)proc_bytes_total(),
        (double)proc_bytes_used(),
        (double)proc_bytes_available());
    }
  #endif
  /*  print_map(stdout, type->tid_A);
    print_map(stdout, type->tid_B);
    print_map(stdout, type->tid_C);*/
  //  stat = zero_out_padding(type->tid_A);
  //  stat = zero_out_padding(type->tid_B);
    TAU_FSTART(ctr_func);
    /* Invoke the contraction algorithm */
    ctrf->run();

    TAU_FSTOP(ctr_func);
  #ifndef SEQ
    if (tensors[type->tid_C]->is_cyclic)
      stat = zero_out_padding(type->tid_C);
  #endif
    if (get_global_comm().rank == 0){
      DPRINTF(1, "Contraction permutation completed in %lf sec.\n",MPI_Wtime()-dtt);
    }


  #if VERIFY
    stat = allread_tsr(type->tid_A, &nA, &uA);
    assert(stat == CTF_SUCCESS);
    stat = get_tsr_info(type->tid_A, &order_A, &edge_len_A, &sym_A);
    assert(stat == CTF_SUCCESS);

    stat = allread_tsr(type->tid_B, &nB, &uB);
    assert(stat == CTF_SUCCESS);
    stat = get_tsr_info(type->tid_B, &order_B, &edge_len_B, &sym_B);
    assert(stat == CTF_SUCCESS);

    if (nsA != nA) { printf("nsA = " PRId64 ", nA = " PRId64 "\n",nsA,nA); ABORT; }
    if (nsB != nB) { printf("nsB = " PRId64 ", nB = " PRId64 "\n",nsB,nB); ABORT; }
    for (i=0; (uint64_t)i<nA; i++){
      if (fabs(uA[i] - sA[i]) > 1.E-6){
        printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
      }
    }
    for (i=0; (uint64_t)i<nB; i++){
      if (fabs(uB[i] - sB[i]) > 1.E-6){
        printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
      }
    }

    stat = allread_tsr(type->tid_C, &nC, &uC);
    assert(stat == CTF_SUCCESS);
    stat = get_tsr_info(type->tid_C, &order_C, &edge_len_C, &sym_C);
    assert(stat == CTF_SUCCESS);
    DEBUG_PRINTF("packed size of C is " PRId64 " (should be " PRId64 ")\n", nC,
      sy_packed_size(order_C, edge_len_C, sym_C));

    pup_C = (dtype*)CTF_alloc(nC*sizeof(dtype));

    cpy_sym_ctr(alpha,
          uA, order_A, edge_len_A, edge_len_A, sym_A, type->idx_map_A,
          uB, order_B, edge_len_B, edge_len_B, sym_B, type->idx_map_B,
          beta,
      ans_C, order_C, edge_len_C, edge_len_C, sym_C, type->idx_map_C);
    assert(stat == CTF_SUCCESS);

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
    for (i=0; (uint64_t)i<nC; i++){
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
    return CTF_SUCCESS;


  }


  int contraction::sym_contract(){
    int i;
    //int ** scl_idx_maps_C;
    //dtype * scl_alpha_C;
    int stat, new_tid;
    int * new_idx_map;
    int * map_A, * map_B, * map_C, * dstack_tid_C;
    int ** dstack_map_C;
    int ntid_A, ntid_B, ntid_C, nst_C;
    CTF_ctr_type_t unfold_type, ntype = *stype;
    CTF_ctr_type_t * type = &ntype;
    std::vector<CTF_ctr_type_t> perm_types;
    std::vector<dtype> signs;
    dtype dbeta;
    ctr<dtype> * ctrf;
    check_contraction(stype);
    unmap_inner(tensors[stype->tid_A]);
    unmap_inner(tensors[stype->tid_B]);
    unmap_inner(tensors[stype->tid_C]);
    if (tensors[stype->tid_A]->has_zero_edge_len || tensors[stype->tid_B]->has_zero_edge_len
        || tensors[stype->tid_C]->has_zero_edge_len){
      tensor<dtype>* tsr_C = tensors[stype->tid_C];
      if (beta != 1.0 && !tsr_C->has_zero_edge_len){ 
        int * new_idx_map_C; 
        int num_diag = 0;
        new_idx_map_C = (int*)CTF_alloc(sizeof(int)*tsr_C->order);
        for (int i=0; i<tsr_C->order; i++){
          new_idx_map_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (stype->idx_map_C[i] == stype->idx_map_C[j]){
              new_idx_map_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        scale_tsr(beta, stype->tid_C, new_idx_map_C, fs, felm); 
        CTF_free(new_idx_map_C);
      }
      return CTF_SUCCESS;
    }
    ntid_A = type->tid_A;
    ntid_B = type->tid_B;
    ntid_C = type->tid_C;
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_A]->order,   (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->order,   (void**)&map_B);
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_C]->order,   (void**)&map_C);
    CTF_alloc_ptr(sizeof(int*)*tensors[ntid_C]->order,   (void**)&dstack_map_C);
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_C]->order,   (void**)&dstack_tid_C);
    memcpy(map_A, type->idx_map_A, tensors[ntid_A]->order*sizeof(int));
    memcpy(map_B, type->idx_map_B, tensors[ntid_B]->order*sizeof(int));
    memcpy(map_C, type->idx_map_C, tensors[ntid_C]->order*sizeof(int));
    while (extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      if (ntid_A != type->tid_A) del_tsr(ntid_A);
      CTF_free(map_A);
      ntid_A = new_tid;
      map_A = new_idx_map;
    }
    while (extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      if (ntid_B != type->tid_B) del_tsr(ntid_B);
      CTF_free(map_B);
      ntid_B = new_tid;
      map_B = new_idx_map;
    }
    nst_C = 0;
    while (extract_diag(ntid_C, map_C, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      dstack_map_C[nst_C] = map_C;
      dstack_tid_C[nst_C] = ntid_C;
      nst_C++;
      ntid_C = new_tid;
      map_C = new_idx_map;
    }
    type->tid_A = ntid_A;
    type->tid_B = ntid_B;
    type->tid_C = ntid_C;
    type->idx_map_A = map_A;
    type->idx_map_B = map_B;
    type->idx_map_C = map_C;

    unmap_inner(tensors[ntid_A]);
    unmap_inner(tensors[ntid_B]);
    unmap_inner(tensors[ntid_C]);
    /*if (ntid_A == ntid_B || ntid_A == ntid_C){*/
    if (ntid_A == ntid_C){
      clone_tensor(ntid_A, 1, &new_tid);
      CTF_ctr_type_t new_type = *type;
      new_type.tid_A = new_tid;
      stat = sym_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      ASSERT(stat == CTF_SUCCESS);
    } else if (ntid_B == ntid_C){
      clone_tensor(ntid_B, 1, &new_tid);
      CTF_ctr_type_t new_type = *type;
      new_type.tid_B = new_tid;
      stat = sym_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      ASSERT(stat == CTF_SUCCESS);
    } else {

      double alignfact = align_symmetric_indices(tensors[ntid_A]->order,
                                                map_A,
                                                tensors[ntid_A]->sym,
                                                tensors[ntid_B]->order,
                                                map_B,
                                                tensors[ntid_B]->sym,
                                                tensors[ntid_C]->order,
                                                map_C,
                                                tensors[ntid_C]->sym);

      /*
       * Apply a factor of n! for each set of n symmetric indices which are contracted over
       */
      double ocfact = overcounting_factor(tensors[ntid_A]->order,
                                         map_A,
                                         tensors[ntid_A]->sym,
                                         tensors[ntid_B]->order,
                                         map_B,
                                         tensors[ntid_B]->sym,
                                         tensors[ntid_C]->order,
                                         map_C,
                                         tensors[ntid_C]->sym);

      //std::cout << alpha << ' ' << alignfact << ' ' << ocfact << std::endl;

      if (unfold_broken_sym(type, NULL) != -1){
        if (global_comm.rank == 0)
          DPRINTF(1,"Contraction index is broken\n");

        unfold_broken_sym(type, &unfold_type);
  #if PERFORM_DESYM
        if (map_tensors(&unfold_type, 
                        ftsr, felm, alpha, beta, &ctrf, 0) == CTF_SUCCESS){
  #else
        int * sym, dim, sy;
        sy = 0;
        sym = get_sym(ntid_A);
        dim = get_dim(ntid_A);
        for (i=0; i<dim; i++){
          if (sym[i] == SY) sy = 1;
        }
        CTF_free(sym);
        sym = get_sym(ntid_B);
        dim = get_dim(ntid_B);
        for (i=0; i<dim; i++){
          if (sym[i] == SY) sy = 1;
        }
        CTF_free(sym);
        sym = get_sym(ntid_C);
        dim = get_dim(ntid_C);
        for (i=0; i<dim; i++){
          if (sym[i] == SY) sy = 1;
        }
        CTF_free(sym);
        if (sy && map_tensors(&unfold_type,
                              ftsr, felm, alpha, beta, &ctrf, 0) == CTF_SUCCESS){
  #endif
          if (ntid_A == ntid_B){
            clone_tensor(ntid_A, 1, &ntid_A);
          }
          desymmetrize(ntid_A, unfold_type.tid_A, 0);
          desymmetrize(ntid_B, unfold_type.tid_B, 0);
          desymmetrize(ntid_C, unfold_type.tid_C, 1);
          if (global_comm.rank == 0)
            DPRINTF(1,"Performing index desymmetrization\n");
          sym_contract(&unfold_type, ftsr, felm,
                       alpha*alignfact, beta);
          symmetrize(ntid_C, unfold_type.tid_C);
          if (ntid_A != unfold_type.tid_A){
            unmap_inner(tensors[unfold_type.tid_A]);
            dealias(ntid_A, unfold_type.tid_A);
            del_tsr(unfold_type.tid_A);
            CTF_free(unfold_type.idx_map_A);
          }
          if (ntid_B != unfold_type.tid_B){
            unmap_inner(tensors[unfold_type.tid_B]);
            dealias(ntid_B, unfold_type.tid_B);
            del_tsr(unfold_type.tid_B);
            CTF_free(unfold_type.idx_map_B);
          }
          if (ntid_C != unfold_type.tid_C){
            unmap_inner(tensors[unfold_type.tid_C]);
            dealias(ntid_C, unfold_type.tid_C);
            del_tsr(unfold_type.tid_C);
            CTF_free(unfold_type.idx_map_C);
          }
        } else {
          get_sym_perms(type, alpha*alignfact*ocfact, 
                        perm_types, signs);
                        //&nscl_C, &scl_maps_C, &scl_alpha_C);
          dbeta = beta;
          for (i=0; i<(int)perm_types.size(); i++){
            contract(&perm_types[i], ftsr, felm,
                      signs[i], dbeta);
            free_type(&perm_types[i]);
            dbeta = 1.0;
        }
        perm_types.clear();
        signs.clear();
        }
      } else {
        contract(type, ftsr, felm, alpha*alignfact*ocfact, beta);
      }
      if (ntid_A != type->tid_A) del_tsr(ntid_A);
      if (ntid_B != type->tid_B) del_tsr(ntid_B);
      for (i=nst_C-1; i>=0; i--){
        extract_diag(dstack_tid_C[i], dstack_map_C[i], 0, &ntid_C, &new_idx_map);
        del_tsr(ntid_C);
        ntid_C = dstack_tid_C[i];
      }
      ASSERT(ntid_C == type->tid_C);
    }

    CTF_free(map_A);
    CTF_free(map_B);
    CTF_free(map_C);
    CTF_free(dstack_map_C);
    CTF_free(dstack_tid_C);

    return CTF_SUCCESS;
  }

  int contraction::home_contract(){
  #ifndef HOME_CONTRACT
    return sym_contract(stype, ftsr, felm, alpha, beta);
  #else
    int ret;
    int was_home_A, was_home_B, was_home_C;
    int was_cyclic_C;
    int64_t old_size_C;
    int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
    int * old_padding_C, * old_edge_len_C;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    tensor<dtype> * ntsr_A, * ntsr_B, * ntsr_C;
    tsr_A = tensors[stype->tid_A];
    tsr_B = tensors[stype->tid_B];
    tsr_C = tensors[stype->tid_C];
    unmap_inner(tsr_A);
    unmap_inner(tsr_B);
    unmap_inner(tsr_C);
    
    if (tsr_A->has_zero_edge_len || 
        tsr_B->has_zero_edge_len || 
        tsr_C->has_zero_edge_len){
      if (beta != 1.0 && !tsr_C->has_zero_edge_len){ 
        int * new_idx_map_C; 
        int num_diag = 0;
        new_idx_map_C = (int*)CTF_alloc(sizeof(int)*tsr_C->order);
        for (int i=0; i<tsr_C->order; i++){
          new_idx_map_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (stype->idx_map_C[i] == stype->idx_map_C[j]){
              new_idx_map_C[i]=new_idx_map_C[j];
              num_diag++;
              break;
            }
          }
        }
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        scale_tsr(beta, stype->tid_C, new_idx_map_C, fs, felm); 
        CTF_free(new_idx_map_C);
      }
      return CTF_SUCCESS;
    }

    contract_mst();

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

    CTF_ctr_type_t ntype = *stype;

    was_home_A = tsr_A->is_home;
    was_home_B = tsr_B->is_home;
    was_home_C = tsr_C->is_home;

    if (was_home_A){
      clone_tensor(stype->tid_A, 0, &ntype.tid_A, 0);
      ntsr_A = tensors[ntype.tid_A];
      ntsr_A->data = tsr_A->data;
      ntsr_A->home_buffer = tsr_A->home_buffer;
      ntsr_A->is_home = 1;
      ntsr_A->is_mapped = 1;
      ntsr_A->itopo = tsr_A->itopo;
      copy_mapping(tsr_A->order, tsr_A->edge_map, ntsr_A->edge_map);
      set_padding(ntsr_A);
    }     
    if (was_home_B){
      if (stype->tid_A == stype->tid_B){
        ntype.tid_B = ntype.tid_A;
        ntsr_B = tensors[ntype.tid_B];
      } else {
        clone_tensor(stype->tid_B, 0, &ntype.tid_B, 0);
        ntsr_B = tensors[ntype.tid_B];
        ntsr_B->data = tsr_B->data;
        ntsr_B->home_buffer = tsr_B->home_buffer;
        ntsr_B->is_home = 1;
        ntsr_B->is_mapped = 1;
        ntsr_B->itopo = tsr_B->itopo;
        copy_mapping(tsr_B->order, tsr_B->edge_map, ntsr_B->edge_map);
        set_padding(ntsr_B);
      }
    }
    if (was_home_C){
      if (stype->tid_C == stype->tid_A){
        ntype.tid_C = ntype.tid_A;
        ntsr_C = tensors[ntype.tid_C];
      } else if (stype->tid_C == stype->tid_B){
        ntype.tid_C = ntype.tid_B;
        ntsr_C = tensors[ntype.tid_C];
      } else {
        clone_tensor(stype->tid_C, 0, &ntype.tid_C, 0);
        ntsr_C = tensors[ntype.tid_C];
        ntsr_C->data = tsr_C->data;
        ntsr_C->home_buffer = tsr_C->home_buffer;
        ntsr_C->is_home = 1;
        ntsr_C->is_mapped = 1;
        ntsr_C->itopo = tsr_C->itopo;
        copy_mapping(tsr_C->order, tsr_C->edge_map, ntsr_C->edge_map);
        set_padding(ntsr_C);
      }
    }

    ret = sym_contract(&ntype, ftsr, felm, alpha, beta);
    if (ret!= CTF_SUCCESS) return ret;
    if (was_home_A) unmap_inner(ntsr_A);
    if (was_home_B && stype->tid_A != stype->tid_B) unmap_inner(ntsr_B);
    if (was_home_C) unmap_inner(ntsr_C);

    if (was_home_C && !ntsr_C->is_home){
      if (global_comm.rank == 0)
        DPRINTF(2,"Migrating tensor %d back to home\n", stype->tid_C);
      save_mapping(ntsr_C,
                   &old_phase_C, &old_rank_C, 
                   &old_virt_dim_C, &old_pe_lda_C, 
                   &old_size_C,  
                   &was_cyclic_C, &old_padding_C, 
                   &old_edge_len_C, &topovec[ntsr_C->itopo]);
      tsr_C->data = ntsr_C->data;
      tsr_C->is_home = 0;
      TAU_FSTART(redistribute_for_ctr_home);
      remap_tensor(stype->tid_C, tsr_C, &topovec[tsr_C->itopo], old_size_C, 
                   old_phase_C, old_rank_C, old_virt_dim_C, 
                   old_pe_lda_C, was_cyclic_C, 
                   old_padding_C, old_edge_len_C, global_comm);
      TAU_FSTOP(redistribute_for_ctr_home);
      memcpy(tsr_C->home_buffer, tsr_C->data, tsr_C->size*sizeof(dtype));
      CTF_free(tsr_C->data);
      tsr_C->data = tsr_C->home_buffer;
      tsr_C->is_home = 1;
      ntsr_C->is_data_aliased = 1;
      del_tsr(ntype.tid_C);
      CTF_free(old_phase_C);
      CTF_free(old_rank_C);
      CTF_free(old_virt_dim_C);
      CTF_free(old_pe_lda_C);
      CTF_free(old_padding_C);
      CTF_free(old_edge_len_C);
    } else if (was_home_C) {
  /*    tsr_C->itopo = ntsr_C->itopo;
      copy_mapping(tsr_C->order, ntsr_C->edge_map, tsr_C->edge_map);
      set_padding(tsr_C);*/
      ASSERT(ntsr_C->data == tsr_C->data);
      ntsr_C->is_data_aliased = 1;
      del_tsr(ntype.tid_C);
    }
    if (ntype.tid_A != ntype.tid_C){
      if (was_home_A && !ntsr_A->is_home){
        ntsr_A->has_home = 0;
        del_tsr(ntype.tid_A);
      } else if (was_home_A) {
        ntsr_A->is_data_aliased = 1;
        del_tsr(ntype.tid_A);
      }
    }
    if (ntype.tid_B != ntype.tid_A &&
        ntype.tid_B != ntype.tid_C){
      if (was_home_B && stype->tid_A != stype->tid_B && !ntsr_B->is_home){
        ntsr_B->has_home = 0;
        del_tsr(ntype.tid_B);
      } else if (was_home_B && stype->tid_A != stype->tid_B) {
        ntsr_B->is_data_aliased = 1;
        del_tsr(ntype.tid_B);
      }
    }
    return CTF_SUCCESS;
  #endif
  }




  /**
   * \brief sets up a ctr_2d_general (2D SUMMA) level where A is not communicated
   *        function will be called with A/B/C permuted depending on desired alg
   *
   * \param[in] is_used whether this ctr will actually be run
   * \param[in] global_comm comm for this CTF instance
   * \param[in] i index in the total index map currently worked on
   * \param[in,out] virt_dim virtual processor grid lengths
   * \param[out] cg_edge_len edge lengths of ctr_2d_gen object to set
   * \param[in,out] total_iter the total number of ctr_2d_gen iterations
   * \param[in] topovec vector of topologies
   * \param[in] tsr_A A tensor
   * \param[in] i_A the index in A to which index i corresponds
   * \param[out] cg_cdt_A the communicator for A to be set for ctr_2d_gen
   * \param[out] cg_ctr_lda_A parameter of ctr_2d_gen corresponding to upper lda for lda_cpy
   * \param[out] cg_ctr_sub_lda_A parameter of ctr_2d_gen corresponding to lower lda for lda_cpy
   * \param[out] cg_move_A tells ctr_2d_gen whether A should be communicated
   * \param[in,out] blk_len_A lengths of local A piece after this ctr_2d_gen level
   * \param[in,out] blk_sz_A size of local A piece after this ctr_2d_gen level
   * \param[in] virt_blk_edge_len_A edge lengths of virtual blocks of A
   * \param[in] load_phase_A tells the offloader how often A buffer changes for ctr_2d_gen
   *
   * ... the other parameters are specified the same as for _A but this time for _B and _C
   */
  template<typename dtype>
  int  ctr_2d_gen_build(int                     is_used,
                        CommData              global_comm,
                        int                     i,
                        int                   * virt_dim,
                        int                   & cg_edge_len,
                        int                   & total_iter,
                        std::vector<topology> & topovec,
                        tensor<dtype>         * tsr_A,
                        int                     i_A,
                        CommData            & cg_cdt_A,
                        int64_t               & cg_ctr_lda_A,
                        int64_t               & cg_ctr_sub_lda_A,
                        bool                  & cg_move_A,
                        int                   * blk_len_A,
                        int64_t               & blk_sz_A,
                        int const             * virt_blk_len_A,
                        int                   & load_phase_A,
                        tensor<dtype>         * tsr_B,
                        int                     i_B,
                        CommData            & cg_cdt_B,
                        int64_t               & cg_ctr_lda_B,
                        int64_t               & cg_ctr_sub_lda_B,
                        bool                  & cg_move_B,
                        int                   * blk_len_B,
                        int64_t               & blk_sz_B,
                        int const             * virt_blk_len_B,
                        int                   & load_phase_B,
                        tensor<dtype>         * tsr_C,
                        int                     i_C,
                        CommData            & cg_cdt_C,
                        int64_t               & cg_ctr_lda_C,
                        int64_t               & cg_ctr_sub_lda_C,
                        bool                  & cg_move_C,
                        int                   * blk_len_C,
                        int64_t               & blk_sz_C,
                        int const             * virt_blk_len_C,
                        int                   & load_phase_C){
    mapping * map;
    int j;
    int nstep = 1;
    if (comp_dim_map(&tsr_C->edge_map[i_C], &tsr_B->edge_map[i_B])){
      map = &tsr_B->edge_map[i_B];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
      }
      return 0;
    } else {
      if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP &&
        tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
        virt_dim[i] = tsr_B->edge_map[i_B].np;
        return 0;
      } else {
        cg_edge_len = 1;
        if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, tsr_B->edge_map[i_B].np);
          cg_cdt_B = topovec[tsr_B->itopo].dim_comm[tsr_B->edge_map[i_B].cdt];
          if (is_used && cg_cdt_B.alive == 0)
            SHELL_SPLIT(global_comm, cg_cdt_B);
          nstep = tsr_B->edge_map[i_B].np;
          cg_move_B = 1;
        } else
          cg_move_B = 0;
        if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP){
          cg_edge_len = lcm(cg_edge_len, tsr_C->edge_map[i_C].np);
          cg_cdt_C = topovec[tsr_C->itopo].dim_comm[tsr_C->edge_map[i_C].cdt];
          if (is_used && cg_cdt_C.alive == 0)
            SHELL_SPLIT(global_comm, cg_cdt_C);
          nstep = MAX(nstep, tsr_C->edge_map[i_C].np);
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
        if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_B= blk_sz_B*tsr_B->edge_map[i_B].np/cg_edge_len;
        else
          cg_ctr_sub_lda_B= blk_sz_B/cg_edge_len;
        for (j=i_B+1; j<tsr_B->order; j++) {
          cg_ctr_sub_lda_B = (cg_ctr_sub_lda_B *
                virt_blk_len_B[j]) / blk_len_B[j];
          cg_ctr_lda_B = (cg_ctr_lda_B*blk_len_B[j])
                /virt_blk_len_B[j];
        }
        cg_ctr_lda_C  = 1;
        if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP)
          cg_ctr_sub_lda_C= blk_sz_C*tsr_C->edge_map[i_C].np/cg_edge_len;
        else
          cg_ctr_sub_lda_C= blk_sz_C/cg_edge_len;
        for (j=i_C+1; j<tsr_C->order; j++) {
          cg_ctr_sub_lda_C = (cg_ctr_sub_lda_C *
                virt_blk_len_C[j]) / blk_len_C[j];
          cg_ctr_lda_C = (cg_ctr_lda_C*blk_len_C[j])
                /virt_blk_len_C[j];
        }
        if (tsr_B->edge_map[i_B].type != PHYSICAL_MAP){
          blk_sz_B  = blk_sz_B / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] / nstep;
        } else {
          blk_sz_B  = blk_sz_B * tsr_B->edge_map[i_B].np / nstep;
          blk_len_B[i_B] = blk_len_B[i_B] * tsr_B->edge_map[i_B].np / nstep;
        }
        if (tsr_C->edge_map[i_C].type != PHYSICAL_MAP){
          blk_sz_C  = blk_sz_C / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] / nstep;
        } else {
          blk_sz_C  = blk_sz_C * tsr_C->edge_map[i_C].np / nstep;
          blk_len_C[i_C] = blk_len_C[i_C] * tsr_C->edge_map[i_C].np / nstep;
        }
  
        if (tsr_B->edge_map[i_B].has_child){
          ASSERT(tsr_B->edge_map[i_B].child->type == VIRTUAL_MAP);
          virt_dim[i] = tsr_B->edge_map[i_B].np*tsr_B->edge_map[i_B].child->np/nstep;
        }
        if (tsr_C->edge_map[i_C].has_child) {
          ASSERT(tsr_C->edge_map[i_C].child->type == VIRTUAL_MAP);
          virt_dim[i] = tsr_C->edge_map[i_C].np*tsr_C->edge_map[i_C].child->np/nstep;
        }
        if (tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
          virt_dim[i] = tsr_C->edge_map[i_C].np/nstep;
        }
        if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP)
          virt_dim[i] = tsr_B->edge_map[i_B].np/nstep;
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
  
  
  /**
   * \brief stretch virtualization by a factor
   * \param[in] order number of maps to stretch
   * \param[in] stretch_factor factor to strech by
   * \param[in] maps mappings along each dimension to stretch
   */
  inline 
  int stretch_virt(int const order,
       int const stretch_factor,
       mapping * maps){
    int i;
    mapping * map;
    for (i=0; i<order; i++){
      map = &maps[i];
      while (map->has_child) map = map->child;
      if (map->type == PHYSICAL_MAP){
        if (map->has_child){
          map->has_child    = 1;
          map->child    = (mapping*)CTF_alloc(sizeof(mapping));
          map->child->type  = VIRTUAL_MAP;
          map->child->np    = stretch_factor;
          map->child->has_child   = 0;
        }
      } else if (map->type == VIRTUAL_MAP){
        map->np = map->np * stretch_factor;
      } else {
        map->type = VIRTUAL_MAP;
        map->np   = stretch_factor;
      }
    }
    return CTF_SUCCESS;
  }


 
  /**
   * \brief extracts the set of physical dimensions still available for mapping
   * \param[in] topo topology
   * \param[in] order_A dimension of A
   * \param[in] edge_map_A mapping of A
   * \param[in] order_B dimension of B
   * \param[in] edge_map_B mapping of B
   * \param[out] num_sub_phys_dims number of free torus dimensions
   * \param[out] sub_phys_comm the torus dimensions
   * \param[out] comm_idx index of the free torus dimensions in the origin topology
   */
  void extract_free_comms(topology const *  topo,
                          int               order_A,
                          mapping const *   edge_map_A,
                          int               order_B,
                          mapping const *   edge_map_B,
                          int &             num_sub_phys_dims,
                          CommData *  *     psub_phys_comm,
                          int **            pcomm_idx){
    int i;
    int phys_mapped[topo->order];
    CommData *   sub_phys_comm;
    int * comm_idx;
    mapping const * map;
    memset(phys_mapped, 0, topo->order*sizeof(int));  
    
    num_sub_phys_dims = 0;

    for (i=0; i<order_A; i++){
      map = &edge_map_A[i];
      while (map->type == PHYSICAL_MAP){
        phys_mapped[map->cdt] = 1;
        if (map->has_child) map = map->child;
        else break;
      } 
    }
    for (i=0; i<order_B; i++){
      map = &edge_map_B[i];
      while (map->type == PHYSICAL_MAP){
        phys_mapped[map->cdt] = 1;
        if (map->has_child) map = map->child;
        else break;
      } 
    }

    num_sub_phys_dims = 0;
    for (i=0; i<topo->order; i++){
      if (phys_mapped[i] == 0){
        num_sub_phys_dims++;
      }
    }
    CTF_alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
    CTF_alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
    num_sub_phys_dims = 0;
    for (i=0; i<topo->order; i++){
      if (phys_mapped[i] == 0){
        sub_phys_comm[num_sub_phys_dims] = topo->dim_comm[i];
        comm_idx[num_sub_phys_dims] = i;
        num_sub_phys_dims++;
      }
    }
    *pcomm_idx = comm_idx;
    *psub_phys_comm = sub_phys_comm;

  }

  /**
   * \brief determines if two topologies are compatible with each other
   * \param topo_keep topology to keep (larger dimension)
   * \param topo_change topology to change (smaller dimension)
   * \return true if its possible to change
   */
  inline 
  int can_morph(topology const * topo_keep, topology const * topo_change){
    int i, j, lda;
    lda = 1;
    j = 0;
    for (i=0; i<topo_keep->order; i++){
      lda *= topo_keep->dim_comm[i].np;
      if (lda == topo_change->dim_comm[j].np){
        j++;
        lda = 1;
      } else if (lda > topo_change->dim_comm[j].np){
        return 0;
      }
    }
    return 1;
  }

  /**
   * \brief morphs a tensor topology into another
   * \param[in] new_topo topology to change to
   * \param[in] old_topo topology we are changing from
   * \param[in] order number of tensor dimensions
   * \param[in,out] edge_map mapping whose topology mapping we are changing
   */
  inline 
  void morph_topo(topology const *  new_topo, 
      topology const *  old_topo, 
      int const     order,
      mapping *     edge_map){
    int i,j,old_lda,new_np;
    mapping * old_map, * new_map, * new_rec_map;

    for (i=0; i<order; i++){
      if (edge_map[i].type == PHYSICAL_MAP){
        old_map = &edge_map[i];
        CTF_alloc_ptr(sizeof(mapping), (void**)&new_map);
        new_rec_map = new_map;
        for (;;){
          old_lda = old_topo->lda[old_map->cdt];
          new_np = 1;
          do {
            for (j=0; j<new_topo->order; j++){
              if (new_topo->lda[j] == old_lda) break;
            } 
            ASSERT(j!=new_topo->order);
            new_rec_map->type   = PHYSICAL_MAP;
            new_rec_map->cdt    = j;
            new_rec_map->np     = new_topo->dim_comm[j].np;
            new_np    *= new_rec_map->np;
            if (new_np<old_map->np) {
              old_lda = old_lda * new_rec_map->np;
              new_rec_map->has_child = 1;
              CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map = new_rec_map->child;
            }
          } while (new_np<old_map->np);

          if (old_map->has_child){
            if (old_map->child->type == VIRTUAL_MAP){
              new_rec_map->has_child = 1;
              CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map->child->type  = VIRTUAL_MAP;
              new_rec_map->child->np    = old_map->child->np;
              new_rec_map->child->has_child   = 0;
              break;
            } else {
              new_rec_map->has_child = 1;
              CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
              new_rec_map = new_rec_map->child;
              old_map = old_map->child;
              //continue
            }
          } else {
            new_rec_map->has_child = 0;
            break;
          }
        }
        clear_mapping(&edge_map[i]);      
        edge_map[i] = *new_map;
        CTF_free(new_map);
      }
    }
  }
}
