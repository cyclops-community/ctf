/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"

namespace CTF {

  template<typename dtype>
  Sparse_Tensor<dtype>::Sparse_Tensor(){
    parent = NULL;
  }

  template<typename dtype>
  Sparse_Tensor<dtype>::Sparse_Tensor(std::vector<int64_t>   indices_,
                                             Tensor<dtype> * parent_){
    parent  = parent_;
    indices = indices_;
    scale   = *(dtype*)parent_->sr->mulid();
  }

  template<typename dtype>
  Sparse_Tensor<dtype>::Sparse_Tensor(int64_t                n,
                                             int64_t *              indices_,
                                             Tensor<dtype> * parent_){
    parent  = parent_;
    indices = std::vector<int64_t>(indices_,indices_+n);
    scale   = *(dtype*)parent_->sr->mulid();
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::write(dtype   alpha,
                                          dtype * values,
                                          dtype   beta){
    parent->write(indices.size(),alpha,beta,&indices[0],&values[0]);
  }

  // C++ overload special-cases of above method
  template<typename dtype>
  void Sparse_Tensor<dtype>::operator=(std::vector<dtype> values){
    write(*(dtype const*)parent->sr->mulid(), &values[0], *(dtype const*)parent->sr->addid());
  }
  template<typename dtype>
  void Sparse_Tensor<dtype>::operator=(dtype* values){
    write(*(dtype const*)parent->sr->mulid(), values, *(dtype const*)parent->sr->addid());
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::operator+=(std::vector<dtype> values){
    write(*(dtype const*)parent->sr->mulid(), &values[0], *(dtype const*)parent->sr->mulid());
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::operator+=(dtype* values){
    write(*(dtype const*)parent->sr->mulid(), values, *(dtype const*)parent->sr->mulid());
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::operator-=(std::vector<dtype> values){
    write(-*(dtype const*)parent->sr->mulid(), &values[0], *(dtype const*)parent->sr->mulid());
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::operator-=(dtype* values){
    write(-*(dtype const*)parent->sr->mulid(), values, *(dtype const*)parent->sr->mulid());
  }

  template<typename dtype>
  void Sparse_Tensor<dtype>::read(dtype   alpha, 
                                         dtype * values,
                                         dtype   beta){
    parent->read(indices.size(),alpha,beta,&indices[0],values);
  }
  template<typename dtype>
  Sparse_Tensor<dtype>::operator std::vector<dtype>(){
    std::vector<dtype> values(indices.size());
    read(parent->sr->mulid(), &values[0], parent->sr->addid());
    return values;
  }

  template<typename dtype>
  Sparse_Tensor<dtype>::operator dtype*(){
    dtype * values = (dtype*)malloc(sizeof(dtype)*indices.size());
    read(parent->sr->mulid(), values, parent->sr->addid());
    return values;
  }
}
