#include "contraction.h"

namespace CTF_int {

  void contraction::calc_fold_nmk(
                      int const *             ordering_A, 
                      int const *             ordering_B, 
                      tensor const *   tsr_A, 
                      tensor const *   tsr_B,
                      tensor const *   tsr_C,
                      iparam *                inner_prm) {
    int i, num_ctr, num_tot;
    int * idx_arr;
    int * edge_len_A, * edge_len_B;
    iparam prm;

      
    edge_len_A = tsr_A->edge_len;
    edge_len_B = tsr_B->edge_len;

    inv_idx(tsr_A->order, type->idx_map_A, NULL,
            tsr_B->order, type->idx_map_B, NULL,
            tsr_C->order, type->idx_map_C, NULL,
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
    for (i=0; i<tsr_A->order; i++){
      if (i >= num_ctr)
        prm.m = prm.m * edge_len_A[ordering_A[i]];
      else 
        prm.k = prm.k * edge_len_A[ordering_A[i]];
    }
    for (i=0; i<tsr_B->order; i++){
      if (i >= num_ctr)
        prm.n = prm.n * edge_len_B[ordering_B[i]];
    }
    /* This gets set later */
    prm.sz_C = 0;
    CTF_free(idx_arr);
    *inner_prm = prm;  
  }

  void contraction::get_fold_indices(int *                   num_fold,
                            int **                  fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, iC, inA, inB, inC, iiA, iiB, iiC;
    int * idx_arr, * idx;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];
    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
      tsr_B->order, type->idx_map_B, tsr_B->edge_map,
      tsr_C->order, type->idx_map_C, tsr_C->edge_map,
      &num_tot, &idx_arr);
    CTF_alloc_ptr(num_tot*sizeof(int), (void**)&idx);

    for (i=0; i<num_tot; i++){
      idx[i] = 1;
    }

    for (iA=0; iA<tsr_A->order; iA++){
      i = type->idx_map_A[iA];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      broken = 0;
      inA = iA;
      do {
        in = type->idx_map_A[inA];
        inB = idx_arr[3*in+1];
        inC = idx_arr[3*in+2];
        if (((inA>=0) + (inB>=0) + (inC>=0) != 2) ||
            ((inB == -1) ^ (iB == -1)) ||
            ((inC == -1) ^ (iC == -1)) ||
            (iB != -1 && inB - iB != in-i) ||
            (iC != -1 && inC - iC != in-i) ||
            (iB != -1 && tsr_A->sym[inA] != tsr_B->sym[inB]) ||
            (iC != -1 && tsr_A->sym[inA] != tsr_C->sym[inC])){
          broken = 1;
        }
        inA++;
      } while (tsr_A->sym[inA-1] != NS);
      if (broken){
        for (iiA=iA;iiA<inA;iiA++){
          idx[type->idx_map_A[iiA]] = 0;
        }
      }
    }
    
    for (iC=0; iC<tsr_C->order; iC++){
      i = type->idx_map_C[iC];
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      broken = 0;
      inC = iC;
      do {
        in = type->idx_map_C[inC];
        inA = idx_arr[3*in+0];
        inB = idx_arr[3*in+1];
        if (((inC>=0) + (inA>=0) + (inB>=0) != 2) ||
            ((inA == -1) ^ (iA == -1)) ||
            ((inB == -1) ^ (iB == -1)) ||
            (iA != -1 && inA - iA != in-i) ||
            (iB != -1 && inB - iB != in-i) ||
            (iA != -1 && tsr_C->sym[inC] != tsr_A->sym[inA]) ||
            (iB != -1 && tsr_C->sym[inC] != tsr_B->sym[inB])){
          broken = 1;
        }
        inC++;
      } while (tsr_C->sym[inC-1] != NS);
      if (broken){
        for (iiC=iC;iiC<inC;iiC++){
          idx[type->idx_map_C[iiC]] = 0;
        }
      }
    }
    
    for (iB=0; iB<tsr_B->order; iB++){
      i = type->idx_map_B[iB];
      iC = idx_arr[3*i+2];
      iA = idx_arr[3*i+0];
      broken = 0;
      inB = iB;
      do {
        in = type->idx_map_B[inB];
        inC = idx_arr[3*in+2];
        inA = idx_arr[3*in+0];
        if (((inB>=0) + (inC>=0) + (inA>=0) != 2) ||
            ((inC == -1) ^ (iC == -1)) ||
            ((inA == -1) ^ (iA == -1)) ||
            (iC != -1 && inC - iC != in-i) ||
            (iA != -1 && inA - iA != in-i) ||
            (iC != -1 && tsr_B->sym[inB] != tsr_C->sym[inC]) ||
            (iA != -1 && tsr_B->sym[inB] != tsr_A->sym[inA])){
              broken = 1;
            }
        inB++;
      } while (tsr_B->sym[inB-1] != NS);
      if (broken){
        for (iiB=iB;iiB<inB;iiB++){
          idx[type->idx_map_B[iiB]] = 0;
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
        if (type->idx_map_A[i] == type->idx_map_A[j]) return 0;
      }
    }
    tsr = tensors[type->tid_B];
    for (i=0; i<tsr->order; i++){
      for (j=i+1; j<tsr->order; j++){
        if (type->idx_map_B[i] == type->idx_map_B[j]) return 0;
      }
    }
    tsr = tensors[type->tid_C];
    for (i=0; i<tsr->order; i++){
      for (j=i+1; j<tsr->order; j++){
        if (type->idx_map_C[i] == type->idx_map_C[j]) return 0;
      }
    }
    get_fold_indices(type, &nfold, &fold_idx);
    CTF_free(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }


  void contraction::get_len_ordering(
            int **      new_ordering_A,
            int **      new_ordering_B,
            int **      new_ordering_C){
    int i, num_tot, num_ctr, idx_ctr, num_no_ctr_A;
    int idx_no_ctr_A, idx_no_ctr_B;
    int * ordering_A, * ordering_B, * ordering_C, * idx_arr;
    tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
    
    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&ordering_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&ordering_B);
    CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&ordering_C);

    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            tsr_C->order, type->idx_map_C, tsr_C->edge_map,
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

}
