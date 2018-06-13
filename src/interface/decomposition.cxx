#include <stdlib.h>
#include <stdio.h>
#include "decomposition.h"

namespace CTF {

  template<typename dtype>
  void fold_unfold(Tensor<dtype>& X, Tensor<dtype>& Y){

    int64_t * inds_X; 
    double * vals_X;
    int64_t n_X;
    //if global index ordering is preserved between the two tensors, we can fold simply
    X.read_local(&n_X, &inds_X, &vals_X);
    Y.write(n_X, inds_X, vals_X);

  }


  template<typename dtype>
  Contract_Term HoSVD::operator[](char const * idx_map){
    char int_inds[core_tensor.order];
    for (int i=0; i<core_tensor.order; i++){
      int_inds = "["+i;
      //FIXME: how to make this robust?
    }
    char factor_inds[2];
    factor_inds[0] = idx_map[0];
    factor_inds[1] = int_inds[0];
    Contract_Term t(core_tensor[ind_inds],factor_matrices[0][factor_inds]);
    for (int i=1; i<core_tensor.order; i++){
      factor_inds[0] = idx_map[i];
      factor_inds[1] = int_inds[i];
      Contract_Term t(t,factor_matrices[0][factor_inds]);
    }
    return t;
  }
  

  /**
   * \calculate the rank[i] left singular columns of the i-mode unfoldings of a tensor
   * \param[in] ranks array of ints that denote number of leading columns of left singular matrix to store
   */
  template<typename dtype>
  std::vector< Matrix <dtype> > get_factor_matrices(Tensor<dtype>& T, int * ranks) {
	
    std::vector< Matrix <dtype> > factor_matrices (T.order);

    char arg[T.order];
    int transformed_lens[T.order];
    char transformed_arg[T.order];
    for (int i = 0; i < T.order; i++) {
      arg[i] = 'i' + i;
      transformed_arg[i] = 'i' + i;
      transformed_lens[i] = T.lens[i];
    }

    for (int i = 0; i < T.order; i++) {
      for (int j = i; j > 0; j--) { 
        transformed_lens[j] = T.lens[j-1];
      }

      transformed_lens[0] = T.lens[i];
      for (int j = 0; j < i; j++) {
        transformed_arg[j] = arg[j+1];
      }
      transformed_arg[i] = arg[0];

      int unfold_lens [2];
      unfold_lens[0] = T.lens[i];
      int ncol = 1;

      for (int j = 0; j < T.order; j++) {  
        if (j != i) 
          ncol *= T.lens[j];	
      }
      unfold_lens[1] = ncol;

      Tensor<dtype> transformed_T(T.order, transformed_lens, *(T.wrld));
      transformed_T[arg] = T[transformed_arg];

      Tensor<dtype> cur_unfold(2, unfold_lens,  *(T.wrld));
      fold_unfold(transformed_T, cur_unfold);

      Matrix<dtype> M(cur_unfold);
      Matrix<dtype> U;
      Matrix<dtype> VT;
      Vector<dtype> S;
      M.svd(U, S, VT, ranks[i]);
      factor_matrices[i] = U;
		
    }

    return factor_matrices;
  }
  template<typename dtype>
  Tensor<dtype> get_core_tensor(Tensor<dtype>& T, std::vector< Matrix <dtype> > factor_matrices, int * ranks) {

    std::vector< Tensor <> > core_tensors(T.order+1);
    core_tensors[0] = T;
    int lens[T.order];
    for (int i = 0; i < T.order; i++) {
      lens[i] = T.lens[i];
    } 
    for (int i = 1; i < T.order+1; i++) {
      lens[i-1] = ranks[i-1];
      Tensor<dtype> core(T.order, lens,  *(T.wrld));
      core_tensors[i] = core;   
    }

    //calculate core tensor
    char arg[T.order];
    char core_arg[T.order];
    for (int i = 0; i < T.order; i++) {
      arg[i] = 'i' + i;
      core_arg[i] = 'i' + i;
    }
    char matrix_arg[2];
    matrix_arg[0] = 'a';
    for (int i = 0; i < T.order; i++) {
      core_arg[i] = 'a';
      matrix_arg[1] = arg[i];
      Matrix<dtype> transpose(factor_matrices[i].ncol, factor_matrices[i].nrow, *(T.wrld)); 
      transpose["ij"] = factor_matrices[i]["ji"];
      core_tensors[i+1][core_arg] = transpose[matrix_arg] * core_tensors[i][arg];
      core_arg[i] = arg[i];
    }
    return core_tensors[T.order];
  }

  template<typename dtype>
  void HoSVD::HoSVD(Tensor<dtype>& T, int * ranks) {
    factor_matrices = get_factor_matrices(T, ranks);
	  core_tensor = get_core_tensor(T, factor_matrices, ranks); 
  }

  template<typename dtype>
  void HoSVD::HoSVD(int * lens, int * ranks) {
    //initialize to zero
    assert(0);
    /*factor_matrices = get_factor_matrices(T, ranks);
	  core_tensor = get_core_tensor(T, factor_matrices, ranks); */
  }

}
   


