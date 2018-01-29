#ifndef __DECOMPOSITION_H__
#define __DECOMPOSITION_H__
#include "tensor.h"
#include "matrix.h"
#include "vector.h"
namespace CTF {

  template<typename dtype>  
  class decomposition {
    public:
      void fold_unfold(Tensor<dtype>& X, Tensor<dtype>& Y);
       
       /**
       * \calculate the rank[i] left singular columns of the i-mode unfoldings of a tensor
       * \param[in] ranks array of ints that denote number of leading columns of left singular matrix to store
       */
      std::vector< Matrix<dtype> > get_factor_matrices(Tensor<dtype> T, int * ranks);

      Tensor<dtype> get_core_tensor(Tensor<dtype>& T, std::vector< Matrix <dtype> > factor_matrices, int ranks[]);

      /**
       * \calculate higher order singular value decomposition of a tensor
       * \param[out] core core tensor with dimensions corresponding to ranks
       * \param[out] factor_matrices rank-i left singular columns of i-mode unfolding of T
       * \param[in] ranks ranks(dimensions) of the core tensor and factor matrices
       */
      void hosvd(Tensor<dtype> T, Tensor<dtype>& core, std::vector< Matrix<dtype> > factor_matrices, int * ranks);
  };

}

#endif
