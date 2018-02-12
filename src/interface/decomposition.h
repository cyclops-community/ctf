#ifndef __DECOMPOSITION_H__
#define __DECOMPOSITION_H__
#include "tensor.h"
#include "matrix.h"
#include "vector.h"
namespace CTF {
 
  void fold_unfold(Tensor<dtype>& X, Tensor<dtype>& Y);

  template<typename dtype>  
  class Decomposition {
    public:
      /**
       * \brief associated an index map with the tensor decomposition for algebra
       * \param[in] idx_map index assignment for this tensor
       */
      virtual Contract_Term operator[](char const * idx_map) = 0;
  };

  template<typename dtype>  
  class HoSVD : public Decomposition {
    public:
      Tensor<dtype> core_tensor;
      std::vector< Matrix<dtype> > factor_matrices;

      /**
       * \calculate higher order singular value decomposition of a tensor
       * \param[in] ranks ranks(dimensions) of the core tensor and factor matrices
       */
      HoSVD(Tensor<dtype> T, int * ranks);

      /**
       * \calculate initialize a higher order singular value decomposition of a tensor to zero
       * \param[in] lens ranks(dimensions) of the factored tensor
       * \param[in] ranks ranks(dimensions) of the core tensor and factor matrices
       */
      HoSVD(int * lens, int * ranks);

      /**
       * \brief associated an index map with the tensor decomposition for algebra
       * \param[in] idx_map index assignment for this tensor
       */
      Contract_Term operator[](char const * idx_map);

  };

}

#endif
