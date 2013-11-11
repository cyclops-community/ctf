#include <algorithm>
#include <iomanip>
#include <ostream>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "../shared/util.h"
#include "../../include/ctf.hpp"

template<typename dtype>
tCTF_Sparse_Tensor<dtype>::tCTF_Sparse_Tensor(){
  parent = NULL;
  scale = 1.0;
}

/**
 * \brief initialize a tensor which corresponds to a set of indices 
 * \param[in] indices a vector of global indices to tensor values
 * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
 */
template<typename dtype>
tCTF_Sparse_Tensor<dtype>::tCTF_Sparse_Tensor(std::vector<int64_t> indices_,
                                              tCTF_Tensor<dtype> * parent_){
  parent        = parent_;
  indices       = indices_;
  scale         = 1.0;
}

/**
 * \brief initialize a tensor which corresponds to a set of indices 
 * \param[in] number of values this sparse tensor will have locally
 * \param[in] indices an array of global indices to tensor values
 * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
 */
template<typename dtype>
tCTF_Sparse_Tensor<dtype>::tCTF_Sparse_Tensor(int64_t              n,
                                              int64_t *            indices_,
                                              tCTF_Tensor<dtype> * parent_){
  parent        = parent_;
  indices       = std::vector<int64_t>(indices_,indices_+n);
  scale         = 1.0;
}

/**
 * \brief set the sparse set of indices on the parent tensor to values
 *        forall(j) i = indices[j]; parent[i] = beta*parent[i] + alpha*values[j];
 * \param[in] alpha scaling factor on values array 
 * \param[in] values data, should be of same size as the number of indices (n)
 * \param[in] beta scaling factor to apply to previously existing data
 */
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::write(dtype    alpha, 
                                      dtype *  values,
                                      dtype    beta){
  parent->write(indices.size(),alpha,beta,&indices[0],&values[0]);
}

// C++ overload special-cases of above method
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator=(std::vector<dtype> values){
  write(get_one<dtype>(), &values[0], get_zero<dtype>());
}
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator=(dtype* values){
  write(get_one<dtype>(), values, get_zero<dtype>());
}

template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator+=(std::vector<dtype> values){
  write(get_one<dtype>(), &values[0], get_one<dtype>());
}
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator+=(dtype* values){
  write(get_one<dtype>(), values, get_one<dtype>());
}

template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator-=(std::vector<dtype> values){
  write(-get_one<dtype>(), &values[0], get_one<dtype>());
}
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::operator-=(dtype* values){
  write(-get_one<dtype>(), values, get_one<dtype>());
}


/**
 * \brief read the sparse set of indices on the parent tensor to values
 *        forall(j) i = indices[j]; values[j] = alpha*parent[i] + beta*values[j];
 * \param[in] alpha scaling factor on parent array 
 * \param[in] values data, should be preallocated to the same size as the number of indices (n)
 * \param[in] beta scaling factor to apply to previously existing data in values
 */
template<typename dtype>
void tCTF_Sparse_Tensor<dtype>::read(dtype   alpha, 
                                     dtype * values,
                                     dtype   beta){
  parent->read(indices.size(),alpha,beta,&indices[0],values);
}
template<typename dtype>
tCTF_Sparse_Tensor<dtype>::operator std::vector<dtype>(){
  std::vector<dtype> values(indices.size());
  read(1.0, &values[0], 0.0);
  return values;
}

template<typename dtype>
tCTF_Sparse_Tensor<dtype>::operator dtype*(){
  dtype * values = (dtype*)malloc(sizeof(dtype)*indices.size());
  read(1.0, values, 0.0);
  return values;
}


template class tCTF_Sparse_Tensor<double>;
template class tCTF_Sparse_Tensor< std::complex<double> >;
