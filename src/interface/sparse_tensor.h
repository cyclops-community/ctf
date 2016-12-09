#ifndef __SPARSE_TENSOR_H__
#define __SPARSE_TENSOR_H__

namespace CTF {
  /**
   * \defgroup CTF CTF Tensor
   * \addtogroup CTF
   * @{
   */
  /**
   * \brief a sparse subset of a tensor 
   */
  template<typename dtype=double>
  class Sparse_Tensor {
    public:
      /** \brief dense tensor whose subset this sparse tensor is of */
      Tensor<dtype> * parent;
      /** \brief indices of the sparse elements of this tensor */
      std::vector<int64_t > indices;
      /** \brief scaling factor by which to scale the tensor elements */
      dtype scale;

      /** 
        * \brief base constructor 
        */
      Sparse_Tensor();
      
      /**
       * \brief initialize a tensor which corresponds to a set of indices 
       * \param[in] indices a vector of global indices to tensor values
       * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
       */
      Sparse_Tensor(std::vector<int64_t >   indices,
                    Tensor<dtype> * parent);

      /**
       * \brief initialize a tensor which corresponds to a set of indices 
       * \param[in] n number of values this sparse tensor will have locally
       * \param[in] indices an array of global indices to tensor values
       * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
       */
      Sparse_Tensor(int64_t                 n,
                    int64_t       *         indices,
                    Tensor<dtype> * parent);

      /**
       * \brief set the sparse set of indices on the parent tensor to values
       *        forall(j) i = indices[j]; parent[i] = beta*parent[i] + alpha*values[j];
       * \param[in] alpha scaling factor on values array 
       * \param[in] values data, should be of same size as the number of indices (n)
       * \param[in] beta scaling factor to apply to previously existing data
       */
      void write(dtype   alpha, 
                 dtype * values,
                 dtype   beta); 

      // C++ overload special-cases of above method
      void operator=(std::vector<dtype> values); 
      void operator+=(std::vector<dtype> values); 
      void operator-=(std::vector<dtype> values); 
      void operator=(dtype * values); 
      void operator+=(dtype * values); 
      void operator-=(dtype * values); 

      /**
       * \brief read the sparse set of indices on the parent tensor to values
       *        forall(j) i = indices[j]; values[j] = alpha*parent[i] + beta*values[j];
       * \param[in] alpha scaling factor on parent array 
       * \param[in] values data, should be preallocated to the same size as the number of indices (n)
       * \param[in] beta scaling factor to apply to previously existing data in values
       */
      void read(dtype   alpha, 
                dtype * values,
                dtype   beta); 

      // C++ overload special-cases of above method
      operator std::vector<dtype>();
      operator dtype*();
  };
  /**
   * @}
   */
}

#include "sparse_tensor.cxx"
#endif
