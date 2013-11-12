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
tCTF_Idx_Tensor<dtype> get_full_intm(tCTF_Idx_Tensor<dtype>& A, 
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
      if (i<j && A.idx_map[i] == A.idx_map[j]){
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
  for (j=0; j<B.parent->ndim; j++){
    for (i=0; i<j && B.idx_map[i] != B.idx_map[j]; i++){}
    if (i!=j) break;
    for (i=0; i<idx && idx_C[i] != B.idx_map[j]; i++){}
    if (i!=j && sym_C[i] != NS) {
      if (i==0){
        if (B.parent->sym[i] != sym_C[j]){
          sym_C[j] = NS;
        }
      } else if (j>0 && idx_C[i+1] == B.idx_map[j-1]){
        if (B.parent->sym[j-1] == NS) 
          sym_C[j] = NS;
      } else if (B->parent.sym[j] != sym_C[j]){
        sym_C[j] = NS;
      } else if (idx_C[i+1] != B.idx_map[j+1]){
        sym_C[j] = NS;
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
  tCTF_Idx_Tensor<dtype> out(tsr_C, idx_C);
  out.is_intm = 1;
  CTF_free(sym_C);
  CTF_free(len_C);
  return out;
}


//general tCTF_Term functions, see ../../include/ctf.hpp for doxygen comments
template<typename dtype>
tCTF_Term::tCTF_Term(){
  scale = 1.0;
}

template<typename dtype>
void tCTF_Term::execute(tCTF_Idx_Tensor<dtype> output){
  ABORT; //I don't see why this part of the code should ever be reached
//  output.scale *= scale;
}

template<typename dtype>
tCTF_Idx_Tensor<dtype> CTF_Term::execute(){
  ABORT; //I don't see why this part of the code should ever be reached
  return tCTF_Idx_Tensor();
}


template<typename dtype>
tCTF_Contract_Term<dtype> tCTF_Term::operator*(tCTF_Term<dtype> const & A){
  tCTF_Contract_Term trm();
  trm.operands.push_back(*this)
  trm.operands.push_back(A);
  return trm;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Term::operator+(tCTF_Term<dtype> const & A){
  tCTF_Sum_Term trm();
  trm.operands.push_back(*this)
  trm.operands.push_back(A);
  return trm;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Term::operator-(tCTF_Term<dtype> const & A){
  tCTF_Sum_Term trm();
  trm.operands.push_back(*this)
  trm.operands.push_back(A);
  trm.operands[1].scale = -1.0 * A.scale;
  return trm;
}

//functions spectific to tCTF_Sum_Term
template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Sum_Term::operator+(tCTF_Term<dtype> const & A){
  operands.push_back(A);
  return *this;
}

template<typename dtype>
tCTF_Sum_Term<dtype> tCTF_Sum_Term::operator-(tCTF_Term<dtype> const & A){
  operands.push_back(A);
  operands.back().scale = -1.0 * A.scale;
  return *this;
}

template<typename dtype>
void tCTF_Sum_Term::execute(tCTF_Idx_Tensor<dtype> output){
  double oscale = output.scale;
  output.scale = 1.0;
  for (int i=0; i<operands.size()-1; i++){
    operands[i].execute(output);
  }
  CTF_Idx_Tensor itsr = operands.back().execute();
  output->parent.sum(operands.back().scale, *(operands.back().execute().parent), operands.back().idx_map,
                                    oscale, *(output->parent), output.idx_map); 
}


//functions spectific to tCTF_Contract_Term
template<typename dtype>
tCTF_Contract_Term<dtype> tCTF_Contract_Term::operator*(tCTF_Term<dtype> const & A){
  operands.push_back(A);
  return *this;
}

template<typename dtype>
void tCTF_Contract_Term::execute(tCTF_Idx_Tensor<dtype> output){
  while (operands.size() > 2){
    CTF_Idx_Tensor<dtype> op_A = operands.pop_back().execute();
    CTF_Idx_Tensor<dtype> op_B = operands.pop_back().execute();
    CTF_Idx_Tensor intm = get_full_intm(op_A, op_B);
    intm.parent->contract(scale*op_A.scale*op_B.scale, 
                                  *(op_A.parent), op_A.idx_map,
                                  *(op_B.parent), op_B.idx_map,
                            intm.scale,           itnm.idx_map);
    operands.push_back(intm);
    if (op_A.is_intm) { delete op_A.parent; } 
    if (op_B.is_intm) { delete op_B.parent; } 
  } 
  {
    LIBT_ASSERT(operands.size() == 2);
    CTF_Idx_Tensor<dtype> op_A = operands.pop_back().execute();
    CTF_Idx_Tensor<dtype> op_B = operands.pop_back().execute();
    
    output.parent->contract(scale*op_A.scale*op_B.scale, 
                                  *(op_A.parent), op_A.idx_map,
                                  *(op_B.parent), op_B.idx_map,
                           output.scale,        output.idx_map);
  } 
}

template class tCTF_Term<double>;
template class tCTF_Term< std::complex<double> >;
template class tCTF_Sum_Term<double>;
template class tCTF_Sum_Term< std::complex<double> >;
template class tCTF_Contract_Term<double>;
template class tCTF_Contract_Term< std::complex<double> >;
