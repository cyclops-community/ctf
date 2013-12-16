/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/

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
tCTF_Idx_Tensor<dtype> * get_full_intm(tCTF_Idx_Tensor<dtype>& A, 
                                     tCTF_Idx_Tensor<dtype>& B){
  int * len_C, * sym_C;
  char * idx_C;
  int ndim_C, i, j, idx;
  
  ndim_C = 0;
  for (i=0; i<A.parent->ndim; i++){
    ndim_C++;
    for (j=0; j<i; j++){
      if (A.idx_map[i] == A.idx_map[j]){
        ndim_C--;
        break;
      }
    }
  }
  for (j=0; j<B.parent->ndim; j++){
    ndim_C++;
    for (i=0; i<MAX(A.parent->ndim, B.parent->ndim); i++){
      if (i<j && B.idx_map[i] == B.idx_map[j]){
        ndim_C--;
        break;
      }
      if (i<A.parent->ndim && A.idx_map[i] == B.idx_map[j]){
        ndim_C--;
        break;
      }
    }
  }

  idx_C = (char*)CTF_alloc(sizeof(char)*ndim_C);
  sym_C = (int*)CTF_alloc(sizeof(int)*ndim_C);
  len_C = (int*)CTF_alloc(sizeof(int)*ndim_C);
  idx = 0;
  for (i=0; i<A.parent->ndim; i++){
    for (j=0; j<i && A.idx_map[i] != A.idx_map[j]; j++){}
    if (j!=i) break;
    idx_C[idx] = A.idx_map[i];
    len_C[idx] = A.parent->len[i];
    if (idx >= 1 && i >= 1 && idx_C[idx-1] == A.idx_map[i-1] && A.parent->sym[i-1] != NS){
      sym_C[idx-1] = A.parent->sym[i-1];
    }
    sym_C[idx] = NS;
    idx++;
  }
  int ndim_AC = idx;
  for (j=0; j<B.parent->ndim; j++){
    for (i=0; i<j && B.idx_map[i] != B.idx_map[j]; i++){}
    if (i!=j) break;
    for (i=0; i<ndim_AC && idx_C[i] != B.idx_map[j]; i++){}
    if (i!=ndim_AC){
      if (sym_C[i] != NS) {
        if (i==0){
          if (B.parent->sym[i] != sym_C[j]){
            sym_C[j] = NS;
          }
        } else if (j>0 && idx_C[i+1] == B.idx_map[j-1]){
          if (B.parent->sym[j-1] == NS) 
            sym_C[j] = NS;
        } else if (B.parent->sym[j] != sym_C[j]){
          sym_C[j] = NS;
        } else if (idx_C[i+1] != B.idx_map[j+1]){
          sym_C[j] = NS;
        }
      }
      break;
    }
    idx_C[idx] = B.idx_map[j];
    len_C[idx] = B.parent->len[j];
    if (idx >= 1 && j >= 1 && idx_C[idx-1] == B.idx_map[j-1] && B.parent->sym[j-1] != NS){
      sym_C[idx-1] = B.parent->sym[j-1];
    }
    sym_C[idx] = NS;
    idx++;
  }

  tCTF_Tensor<dtype> * tsr_C = new tCTF_Tensor<dtype>(ndim_C, len_C, sym_C, *(A.parent->world));
  tCTF_Idx_Tensor<dtype> * out = new tCTF_Idx_Tensor<dtype>(tsr_C, idx_C);
  out->is_intm = 1;
  CTF_free(sym_C);
  CTF_free(len_C);
  CTF_free(idx_C);
  return out;
}


//general tCTF_Term functions, see ../../include/ctf.hpp for doxygen comments
template<typename dtype>
tCTF_Term<dtype>::tCTF_Term(){
  scale = 1.0;
}

template<typename dtype>
tCTF_Term<dtype>::operator dtype() const {
  assert(where_am_i() != NULL);
  tCTF_Scalar<dtype> sc(*where_am_i());
  tCTF_Idx_Tensor<dtype> isc(&sc,""); 
  execute(isc);
//  delete isc;
  return sc.get_val();
}

//template<typename dtype>
//void tCTF_Term<dtype>::execute(tCTF_Idx_Tensor<dtype> output){
//  ABORT; //I don't see why this part of the code should ever be reached
////  output.scale *= scale;
//}
//
//template<typename dtype>
//tCTF_Idx_Tensor<dtype> tCTF_Term<dtype>::execute(){
//  ABORT; //I don't see why this part of the code should ever be reached
//  return tCTF_Idx_Tensor<dtype>();
//}


template<typename dtype>
tCTF_Contract_Term<dtype> tCTF_Term<dtype>::operator*(tCTF_Term<dtype> const & A) const {
  tCTF_Contract_Term<dtype> trm;
  trm.operands.push_back(this->clone());
  trm.operands.push_back(A.clone());
  return trm;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Term<dtype>::operator+(tCTF_Term<dtype> const & A) const {
  tCTF_Sum_Term<dtype> trm;
  trm.operands.push_back(this->clone());
  trm.operands.push_back(A.clone());
  return trm;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Term<dtype>::operator-(tCTF_Term<dtype> const & A) const {
  tCTF_Sum_Term<dtype> trm;
  trm.operands.push_back(this->clone());
  trm.operands.push_back(A.clone());
  trm.operands[1]->scale = -1.0 * A.scale;
  return trm;
}

template<typename dtype>
tCTF_Contract_Term<dtype> tCTF_Term<dtype>::operator*(dtype scl) const {
  tCTF_Contract_Term<dtype> trm;
  tCTF_Idx_Tensor<dtype> iscl(scl);
  trm.operands.push_back(this->clone());
  trm.operands.push_back(iscl.clone());
  return trm;
}

//functions spectific to tCTF_Sum_Term
template<typename dtype>
tCTF_Sum_Term<dtype>::~tCTF_Sum_Term(){
  for (int i=0; i<(int)operands.size(); i++){
    delete operands[i];
  }
  operands.clear();
}

template<typename dtype>
tCTF_Sum_Term<dtype>::tCTF_Sum_Term(
    tCTF_Sum_Term<dtype> const & other,
    std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap){
  this->scale = other.scale;
  for (int i=0; i<(int)other.operands.size(); i++){
    this->operands.push_back(other.operands[i]->clone(remap));
  }
}

template<typename dtype>
tCTF_Term<dtype> * tCTF_Sum_Term<dtype>::clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap) const{
  return new tCTF_Sum_Term<dtype>(*this, remap);
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Sum_Term<dtype>::operator+(tCTF_Term<dtype> const & A) const {
  tCTF_Sum_Term st(*this);
  st.operands.push_back(A.clone());
  return st;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Sum_Term<dtype>::operator-(tCTF_Term<dtype> const & A) const {
  tCTF_Sum_Term st(*this);
  st.operands.push_back(A.clone());
  st.operands.back()->scale = -1.0 * A.scale;
  return st;
}
template<typename dtype>
tCTF_Idx_Tensor<dtype> tCTF_Sum_Term<dtype>::estimate_cost(long_int & cost) const {
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 1){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->estimate_cost(cost);
    tCTF_Idx_Tensor<dtype> op_B = pop_B->estimate_cost(cost);
    tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
    cost += intm->parent->estimate_cost(*(op_A.parent), op_A.idx_map,
                                    intm->idx_map);
    cost += intm->parent->estimate_cost(*(op_B.parent), op_B.idx_map,
                                    intm->idx_map);
    tmp_ops.push_back(intm);
    delete pop_A;
    delete pop_B;
  }
  tCTF_Idx_Tensor<dtype> ans = tmp_ops[0]->estimate_cost(cost);
  delete tmp_ops[0];
  tmp_ops.clear();
  return ans;
}

template<typename dtype>
long_int tCTF_Sum_Term<dtype>::estimate_cost(tCTF_Idx_Tensor<dtype> output) const{
  std::vector< tCTF_Term<dtype>* > tmp_ops = operands;
  long_int cost = 0;
  for (int i=0; i<((int)tmp_ops.size())-1; i++){
    cost += tmp_ops[i]->estimate_cost(output);
  }
  tCTF_Idx_Tensor<dtype> itsr = tmp_ops.back()->estimate_cost(cost);
  cost += output.parent->estimate_cost( *(itsr.parent), itsr.idx_map,
                       output.idx_map); 
  return cost;
}


template<typename dtype>
tCTF_Idx_Tensor<dtype> tCTF_Sum_Term<dtype>::execute() const {
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 1){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->execute();
    tCTF_Idx_Tensor<dtype> op_B = pop_B->execute();
    tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
    intm->parent->sum(op_A.scale, *(op_A.parent), op_A.idx_map,
                     intm->scale,               intm->idx_map);
    intm->parent->sum(op_B.scale, *(op_B.parent), op_B.idx_map,
                     intm->scale,               intm->idx_map);
    tmp_ops.push_back(intm);
    delete pop_A;
    delete pop_B;
  }
  tmp_ops[0]->scale *= this->scale; 
  tCTF_Idx_Tensor<dtype> ans = tmp_ops[0]->execute();
  delete tmp_ops[0];
  tmp_ops.clear();
  return ans;
}

template<typename dtype>
void tCTF_Sum_Term<dtype>::execute(tCTF_Idx_Tensor<dtype> output) const{
  std::vector< tCTF_Term<dtype>* > tmp_ops = operands;
  for (int i=0; i<((int)tmp_ops.size())-1; i++){
    tmp_ops[i]->execute(output);
    output.scale = 1.0;
  }
  tCTF_Idx_Tensor<dtype> itsr = tmp_ops.back()->execute();
  output.parent->sum(itsr.scale, *(itsr.parent), itsr.idx_map,
                      output.scale, output.idx_map); 
}

template<typename dtype>
void tCTF_Sum_Term<dtype>::get_inputs(std::set<tCTF_Tensor<dtype>*>* inputs_set) const {
  for (int i=0; i<(int)operands.size(); i++){
    operands[i]->get_inputs(inputs_set);
  }
}

template<typename dtype>
tCTF_World<dtype> * tCTF_Sum_Term<dtype>::where_am_i() const {
  tCTF_World<dtype> * w = NULL;
  for (int i=0; i<(int)operands.size(); i++){
    if (operands[i]->where_am_i() != NULL) {
      w = operands[i]->where_am_i();
    }
  }
  return w;
}


//functions spectific to tCTF_Contract_Term
template<typename dtype>
tCTF_Contract_Term<dtype>::~tCTF_Contract_Term(){
  for (int i=0; i<(int)operands.size(); i++){
    delete operands[i];
  }
  operands.clear();
}

template<typename dtype>
tCTF_World<dtype> * tCTF_Contract_Term<dtype>::where_am_i() const {
  tCTF_World<dtype> * w = NULL;
  for (int i=0; i<(int)operands.size(); i++){
    if (operands[i]->where_am_i() != NULL) {
      w = operands[i]->where_am_i();
    }
  }
  return w;
}

template<typename dtype>
tCTF_Contract_Term<dtype>::tCTF_Contract_Term(
    tCTF_Contract_Term<dtype> const & other,
    std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap){
  this->scale = other.scale;
  for (int i=0; i<(int)other.operands.size(); i++){
    tCTF_Term<dtype> * t = other.operands[i]->clone(remap);
    operands.push_back(t);
  }
}

template<typename dtype>
tCTF_Term<dtype> * tCTF_Contract_Term<dtype>::clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap) const {
  return new tCTF_Contract_Term<dtype>(*this, remap);
}

template<typename dtype>
tCTF_Contract_Term<dtype> tCTF_Contract_Term<dtype>::operator*(tCTF_Term<dtype> const & A) const {
  tCTF_Contract_Term<dtype> ct(*this);
  ct.operands.push_back(A.clone());
  return ct;
}

template<typename dtype>
void tCTF_Contract_Term<dtype>::execute(tCTF_Idx_Tensor<dtype> output)const {
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 2){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->execute();
    tCTF_Idx_Tensor<dtype> op_B = pop_B->execute();
    if (op_A.parent == NULL) {
      op_B.scale *= op_A.scale;
      tmp_ops.push_back(op_B.clone());
    } else if (op_B.parent == NULL) {
      op_A.scale *= op_B.scale;
      tmp_ops.push_back(op_A.clone());
    } else {
      tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
      intm->parent->contract(this->scale*op_A.scale*op_B.scale, 
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                              intm->scale,         intm->idx_map);
      tmp_ops.push_back(intm);
    }
    delete pop_A;
    delete pop_B;
  } 
  {
    LIBT_ASSERT(tmp_ops.size() == 2);
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->execute();
    tCTF_Idx_Tensor<dtype> op_B = pop_B->execute();
    
    if (op_A.parent == NULL && op_B.parent == NULL){
      assert(0); //FIXME write scalar to whole tensor
    } else if (op_A.parent == NULL){
      output.parent->sum(this->scale*op_A.scale*op_B.scale, 
                                *(op_B.parent), op_B.idx_map,
                         output.scale,        output.idx_map);
    } else if (op_B.parent == NULL){
      output.parent->sum(this->scale*op_A.scale*op_B.scale, 
                                *(op_A.parent), op_A.idx_map,
                         output.scale,        output.idx_map);
    } else {
      output.parent->contract(this->scale*op_A.scale*op_B.scale, 
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                             output.scale,        output.idx_map);
    }
    delete pop_A;
    delete pop_B;
  } 
}

template<typename dtype>
tCTF_Idx_Tensor<dtype> tCTF_Contract_Term<dtype>::execute() const {
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 1){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->execute();
    tCTF_Idx_Tensor<dtype> op_B = pop_B->execute();
    if (op_A.parent == NULL) {
      op_B.scale *= op_A.scale;
      tmp_ops.push_back(op_B.clone());
    } else if (op_B.parent == NULL) {
      op_A.scale *= op_B.scale;
      tmp_ops.push_back(op_A.clone());
    } else {
      tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
      intm->parent->contract(this->scale*op_A.scale*op_B.scale, 
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                              intm->scale,         intm->idx_map);
      tmp_ops.push_back(intm);
    }
    delete pop_A;
    delete pop_B;
  } 
  return tmp_ops[0]->execute();
}

template<typename dtype>
long_int tCTF_Contract_Term<dtype>::estimate_cost(tCTF_Idx_Tensor<dtype> output)const {
  long_int cost = 0;
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 2){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->estimate_cost(cost);
    tCTF_Idx_Tensor<dtype> op_B = pop_B->estimate_cost(cost);
    if (op_A.parent == NULL) {
      tmp_ops.push_back(op_B.clone());
    } else if (op_B.parent == NULL) {
      op_A.scale *= op_B.scale;
      tmp_ops.push_back(op_A.clone());
    } else {
      tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
      cost += intm->parent->estimate_cost(
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                                       intm->idx_map);
      tmp_ops.push_back(intm);
    }
    delete pop_A;
    delete pop_B;
  } 
  {
    LIBT_ASSERT(tmp_ops.size() == 2);
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->estimate_cost(cost);
    tCTF_Idx_Tensor<dtype> op_B = pop_B->estimate_cost(cost);
    
    if (op_A.parent == NULL && op_B.parent == NULL){
      assert(0); //FIXME write scalar to whole tensor
    } else if (op_A.parent == NULL){
      cost += output.parent->estimate_cost(*(op_B.parent), op_B.idx_map,
                            output.idx_map);
    } else if (op_B.parent == NULL){
      cost += output.parent->estimate_cost(
                                *(op_A.parent), op_A.idx_map,
                                 output.idx_map);
    } else {
      cost += output.parent->estimate_cost(
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                                     output.idx_map);
    }
    delete pop_A;
    delete pop_B;
  } 
  return cost;
}

template<typename dtype>
tCTF_Idx_Tensor<dtype> tCTF_Contract_Term<dtype>::estimate_cost(long_int & cost) const {
  std::vector< tCTF_Term<dtype>* > tmp_ops;
  for (int i=0; i<(int)operands.size(); i++){
    tmp_ops.push_back(operands[i]->clone());
  }
  while (tmp_ops.size() > 1){
    tCTF_Term<dtype> * pop_A = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Term<dtype> * pop_B = tmp_ops.back();
    tmp_ops.pop_back();
    tCTF_Idx_Tensor<dtype> op_A = pop_A->estimate_cost(cost);
    tCTF_Idx_Tensor<dtype> op_B = pop_B->estimate_cost(cost);
    if (op_A.parent == NULL) {
      tmp_ops.push_back(op_B.clone());
    } else if (op_B.parent == NULL) {
      tmp_ops.push_back(op_A.clone());
    } else {
      tCTF_Idx_Tensor<dtype> * intm = get_full_intm(op_A, op_B);
      cost += intm->parent->estimate_cost(
                                    *(op_A.parent), op_A.idx_map,
                                    *(op_B.parent), op_B.idx_map,
                                      intm->idx_map);
      tmp_ops.push_back(intm);
    }
    delete pop_A;
    delete pop_B;
  } 
  return tmp_ops[0]->estimate_cost(cost);
}


template<typename dtype>
void tCTF_Contract_Term<dtype>::get_inputs(std::set<tCTF_Tensor<dtype>*>* inputs_set) const {
  for (int i=0; i<(int)operands.size(); i++){
    operands[i]->get_inputs(inputs_set);
  }
}

template class tCTF_Term<double>;
template class tCTF_Sum_Term<double>;
template class tCTF_Contract_Term<double>;
#ifdef CTF_COMPLEX
template class tCTF_Term< std::complex<double> >;
template class tCTF_Sum_Term< std::complex<double> >;
template class tCTF_Contract_Term< std::complex<double> >;
#endif
