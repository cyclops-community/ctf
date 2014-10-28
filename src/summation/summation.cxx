#include "summation.h"

namespace CTF_int {

  void sum_tsr::get_fold_indices(int *        num_fold,
                            int **         fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, inA, inB, iiA, iiB;
    int * idx_arr, * idx;
    tensor<dtype> * tsr_A, * tsr_B;
    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            &num_tot, &idx_arr);
    CTF_alloc_ptr(num_tot*sizeof(int), (void**)&idx);

    for (i=0; i<num_tot; i++){
      idx[i] = 1;
    }
    
    for (iA=0; iA<tsr_A->order; iA++){
      i = type->idx_map_A[iA];
      iB = idx_arr[2*i+1];
      broken = 0;
      inA = iA;
      do {
        in = type->idx_map_A[inA];
        inB = idx_arr[2*in+1];
        if (((inA>=0) + (inB>=0) != 2) ||
            (iB != -1 && inB - iB != in-i) ||
            (iB != -1 && tsr_A->sym[inA] != tsr_B->sym[inB])){
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

    for (iB=0; iB<tsr_B->order; iB++){
      i = type->idx_map_B[iB];
      iA = idx_arr[2*i+0];
      broken = 0;
      inB = iB;
      do {
        in = type->idx_map_B[inB];
        inA = idx_arr[2*in+0];
        if (((inB>=0) + (inA>=0) != 2) ||
            (iA != -1 && inA - iA != in-i) ||
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

  int summation::can_fold(){
    int i, j, nfold, * fold_idx;
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
    get_fold_indices(type, &nfold, &fold_idx);
    CTF_free(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }

  void summation::get_len_ordering(
                                        int **      new_ordering_A,
                                        int **      new_ordering_B){
    int i, num_tot;
    int * ordering_A, * ordering_B, * idx_arr;
    tensor<dtype> * tsr_A, * tsr_B;
    
    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&ordering_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&ordering_B);

    inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
            tsr_B->order, type->idx_map_B, tsr_B->edge_map,
            &num_tot, &idx_arr);
    for (i=0; i<num_tot; i++){
      ordering_A[i] = idx_arr[2*i];
      ordering_B[i] = idx_arr[2*i+1];
    }
    CTF_free(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
  }


}
