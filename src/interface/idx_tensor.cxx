/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "schedule.h"

namespace CTF {

  //template<typename dtype>
  //Idx_Tensor<dtype> get_intermediate(Idx_Tensor<dtype>& A, 
  //                                        Idx_Tensor<dtype>& B){
  //  int * len_C, * sym_C;
  //  char * idx_C;
  //  int order_C, i, j, idx;
  //  
  //  order_C = 0;
  //  for (i=0; i<A.parent->order; i++){
  //    order_C++;
  //    for (j=0; j<B.parent->order; j++){
  //      if (A.idx_map[i] == B.idx_map[j]){
  //        order_C--;
  //        break;
  //      }
  //    }
  //  }
  //  for (j=0; j<B.parent->order; j++){
  //    order_C++;
  //    for (i=0; i<A.parent->order; i++){
  //      if (A.idx_map[i] == B.idx_map[j]){
  //        order_C--;
  //        break;
  //      }
  //    }
  //  }
  //
  //  idx_C = (char*)alloc(sizeof(char)*order_C);
  //  sym_C = (int*)alloc(sizeof(int)*order_C);
  //  len_C = (int*)alloc(sizeof(int)*order_C);
  //  idx = 0;
  //  for (i=0; i<A.parent->order; i++){
  //    for (j=0; j<B.parent->order; j++){
  //      if (A.idx_map[i] == B.idx_map[j]){
  //        break;
  //      }
  //    }
  //    if (j == B.parent->order){
  //      idx_C[idx] = A.idx_map[i];
  //      len_C[idx] = A.parent->len[i];
  //      if (idx >= 1 && i >= 1 && idx_C[idx-1] == A.idx_map[i-1] && A.parent->sym[i-1] != NS){
  //        sym_C[idx-1] = A.parent->sym[i-1];
  //      }
  //      sym_C[idx] = NS;
  //      idx++;
  //    }
  //  }
  //  for (j=0; j<B.parent->order; j++){
  //    for (i=0; i<A.parent->order; i++){
  //      if (A.idx_map[i] == B.idx_map[j]){
  //        break;
  //      }
  //    }
  //    if (i == A.parent->order){
  //      idx_C[idx] = B.idx_map[j];
  //      len_C[idx] = B.parent->len[j];
  //      if (idx >= 1 && j >= 1 && idx_C[idx-1] == B.idx_map[j-1] && B.parent->sym[j-1] != NS){
  //        sym_C[idx-1] = B.parent->sym[j-1];
  //      }
  //      sym_C[idx] = NS;
  //      idx++;
  //    }
  //  }
  //
  //  Tensor<dtype> * tsr_C = new Tensor<dtype>(order_C, len_C, sym_C, *(A.parent->world));
  //  Idx_Tensor<dtype> out(tsr_C, idx_C);
  //  out.is_intm = 1;
  //  free(sym_C);
  //  free(len_C);
  //  return out;
  //}

  template<typename dtype>
  Idx_Tensor<dtype>::Idx_Tensor(Tensor<dtype> *  parent_, 
                                          const char *          idx_map_, 
                                          int                   copy){
    if (copy){
      parent = new Tensor<dtype>(*parent,1);
      idx_map = (char*)alloc(parent->order*sizeof(char));
    } else {
      idx_map = (char*)alloc(parent_->order*sizeof(char));
      parent        = parent_;
    }
    memcpy(idx_map, idx_map_, parent->order*sizeof(char));
    is_intm       = 0;
    this->scale    = 1.0;
  }

  template<typename dtype>
  Idx_Tensor<dtype>::Idx_Tensor(
      Idx_Tensor<dtype> const &  other,
      int                       copy,
      std::map<Tensor<dtype>*, Tensor<dtype>*>* remap) {
    if (other.parent == NULL){
      parent        = NULL;
      idx_map       = NULL;
      is_intm       = 0;
    } else {
      parent = other.parent;
      if (remap != NULL) {
        typename std::map<Tensor<dtype>*, Tensor<dtype>*>::iterator it = remap->find(parent);
        assert(it != remap->end()); // assume a remapping will be complete
        parent = it->second;
      }

      if (copy || other.is_intm){
        parent = new Tensor<dtype>(*parent,1);
        is_intm = 1;
      } else {
        // leave parent as is - already correct
        is_intm = 0;
      }
      idx_map = (char*)alloc(other.parent->order*sizeof(char));
      memcpy(idx_map, other.idx_map, parent->order*sizeof(char));
    }
    this->scale    = other.scale;
  }

  template<typename dtype>
  Idx_Tensor<dtype>::Idx_Tensor(){
    parent        = NULL;
    idx_map       = NULL;
    is_intm       = 0;
    this->scale    = 1.0;
  }

  template<typename dtype>
  Idx_Tensor<dtype>::Idx_Tensor(dtype val){
    parent        = NULL;
    idx_map       = NULL;
    is_intm       = 0;
    this->scale   = val;
  }

  template<typename dtype>
  Idx_Tensor<dtype>::~Idx_Tensor(){
    if (is_intm) { 
      delete parent;
      is_intm = 0;
    }
    if (idx_map != NULL)  free(idx_map);
    idx_map = NULL;
  }

  template<typename dtype>
  Term<dtype> * Idx_Tensor<dtype>::clone(std::map<Tensor<dtype>*, Tensor<dtype>*>* remap) const {
    return new Idx_Tensor<dtype>(*this, 0, remap);
  }

  template<typename dtype>
  World * Idx_Tensor<dtype>::where_am_i() const {
    if (parent == NULL) return NULL;
    return parent->world;
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::operator=(Idx_Tensor<dtype> const & B){
    if (global_schedule != NULL) {
      std::cout << "op= tensor" << std::endl;
      assert(false);
    } else {
      this->scale = 0.0;
      B.execute(*this);
      this->scale = 1.0;
    }
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::operator=(Term<dtype> const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation<dtype>(TENSOR_OP_SET, new Idx_Tensor(*this), B.clone()));
    } else {
      this->scale = 0.0;
      B.execute(*this);
      this->scale = 1.0;
    }
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::operator+=(Term<dtype> const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation<dtype>(TENSOR_OP_SUM, new Idx_Tensor(*this), B.clone()));
    } else {
      //this->scale = 1.0;
      B.execute(*this);
      this->scale = 1.0;
    }
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::operator-=(Term<dtype> const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation<dtype>(TENSOR_OP_SUBTRACT, new Idx_Tensor(*this), B.clone()));
    } else {
      Term<dtype> * Bcpy = B.clone();
      Bcpy->scale *= -1.0;
      Bcpy->execute(*this);
      this->scale = 1.0;
      delete Bcpy;
    }
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::operator*=(Term<dtype> const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation<dtype>(TENSOR_OP_MULTIPLY, new Idx_Tensor(*this), B.clone()));
    } else {
      Contract_Term<dtype> ctrm = (*this)*B;
      *this = ctrm;
    }
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::execute(Idx_Tensor<dtype> output) const {
    if (parent == NULL){
      output.scale *= this->scale;
      Scalar<dtype> ts(this->scale, *(output.where_am_i()));
      output.parent->sum(1.0, ts, "",
                         output.scale, output.idx_map);
    } else {
      output.parent->sum(this->scale, *this->parent, idx_map,
                         output.scale, output.idx_map);
    } 
  }

  template<typename dtype>
  Idx_Tensor<dtype> Idx_Tensor<dtype>::execute() const {
    return *this;
  }

  template<typename dtype>
  int64_t Idx_Tensor<dtype>::estimate_cost(Idx_Tensor<dtype> output) const {
    int64_t cost = 0;
    if (parent == NULL){
      Scalar<dtype> ts(this->scale, *(output.where_am_i()));
      cost += output.parent->estimate_cost(ts, "",
                         output.idx_map);
    } else {
      cost += output.parent->estimate_cost(*this->parent, idx_map,
                          output.idx_map);
    } 
    return cost;
  }

  template<typename dtype>
  Idx_Tensor<dtype> Idx_Tensor<dtype>::estimate_cost(int64_t & cost) const {
    return *this;
  }

  template<typename dtype>
  void Idx_Tensor<dtype>::get_inputs(std::set<Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const {
    if (parent) {
      inputs_set->insert(parent);
    }
  }

  /*template<typename dtype>
  void Idx_Tensor<dtype>::operator=(dtype B){
    *this=(Scalar<dtype>(B,*(this->parent->world))[""]);
  }
  template<typename dtype>
  void Idx_Tensor<dtype>::operator+=(dtype B){
    *this+=(Scalar<dtype>(B,*(this->parent->world))[""]);
  }
  template<typename dtype>
  void Idx_Tensor<dtype>::operator-=(dtype B){
    *this-=(Scalar<dtype>(B,*(this->parent->world))[""]);
  }
  template<typename dtype>
  void Idx_Tensor<dtype>::operator*=(dtype B){
    *this*=(Scalar<dtype>(B,*(this->parent->world))[""]);
  }*/

  /*
  template<typename dtype>
  Idx_Tensor<dtype> Idx_Tensor<dtype>::operator+(Idx_Tensor<dtype> tsr){
    if (has_contract || has_sum){
      *NBR = (*NBR)-tsr;
      return *this;
    }
    NBR = &tsr;
    has_sum = 1;
    return *this;
  }

  template<typename dtype>
  Idx_Tensor<dtype> Idx_Tensor<dtype>::operator-(Idx_Tensor<dtype> tsr){
    if (has_contract || has_sum){
      *NBR = (*NBR)-tsr;
      return *this;
    }
    NBR = &tsr;
    has_sum = 1;
    if (tsr.has_scale) tsr.scale = -1.0*tsr.scale;
    else {
      tsr.has_scale = 1;
      tsr.scale = -1.0;
    }
    return *this;
  }

  template<typename dtype>
  Idx_Tensor<dtype> Idx_Tensor<dtype>::operator*(double  scl){
    if (has_contract){
      *NBR =(*NBR)*scl;
      return *this;
    }
    if (has_scale){
      scale *= scl;
    } else {
      has_scale = 1;
      scale = scl;
    }
    return *this;
  }*/

  /*
  template<typename dtype>
  void Idx_Tensor<dtype>::run(Idx_Tensor<dtype>* output, dtype  beta){
    dtype  alpha;
    if (has_scale) alpha = scale;
    else alpha = 1.0;
    if (has_contract){
      if (NBR->has_scale){
        alpha *= NBR->scale;
      }
      if (NBR->has_contract || NBR->has_sum){
        Idx_Tensor itsr = get_intermediate(*this,*NBR);
        itsr.has_sum = NBR->has_sum;
        itsr.has_contract = NBR->has_contract;
        itsr.NBR = NBR->NBR;
        printf("erm tsr has_Contract = %d, NBR = %p, NBR.has_scale = %d\n", itsr.has_contract, itsr.NBR,
        itsr.NBR->has_scale);
        
        itsr.parent->contract(alpha, *(this->parent), this->idx_map,
                                      *(NBR->parent),  NBR->idx_map,
                               0.0,                    itsr.idx_map);
        itsr.run(output, beta);
      } else {
        output->parent->contract(alpha, *(this->parent), this->idx_map,
                                        *(NBR->parent),  NBR->idx_map,
                                 beta,                   output->idx_map);
      }
    } else {
      if (has_sum){
        Tensor<dtype> tcpy(*(this->parent),1);
        Idx_Tensor itsr(&tcpy, idx_map);
        NBR->run(&itsr, alpha);
        output->parent->sum(1.0, tcpy, idx_map, beta, output->idx_map);
  //      delete itsr;
  //      delete tcpy;
      } else {
        output->parent->sum(alpha, *(this->parent), idx_map, beta, output->idx_map);
      }
    }  
  //  if (!is_perm)
  //    delete this;
  }*/
}
