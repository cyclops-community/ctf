/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "schedule.h"
#include "../summation/summation.h"

using namespace CTF_int;

namespace CTF {

  //template<typename dtype, bool is_ord>
  //Idx_Tensor get_intermediate(Idx_Tensor& A,
  //                                        Idx_Tensor& B){
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
  //  idx_C = (char*)CTF_int::alloc(sizeof(char)*order_C);
  //  sym_C = (int*)CTF_int::alloc(sizeof(int)*order_C);
  //  len_C = (int*)CTF_int::alloc(sizeof(int)*order_C);
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
  //  CTF_int::tensor * tsr_C = new CTF_int::tensor(order_C, len_C, sym_C, *(A.parent->world));
  //  Idx_Tensor out(tsr_C, idx_C);
  //  out.is_intm = 1;
  //  free(sym_C);
  //  free(len_C);
  //  return out;
  //}

  Idx_Tensor::Idx_Tensor(CTF_int::tensor * parent_,
                         const char *      idx_map_,
                         int               copy) : Term(parent_->sr) {
    if (parent_->order > -1)
      idx_map = (char*)CTF_int::alloc((parent_->order+1)*sizeof(char));
    else
      idx_map = (char*)CTF_int::alloc((strlen(idx_map_)+1)*sizeof(char));
    if (copy){
      parent = new CTF_int::tensor(parent,1);
    } else {
      parent        = parent_;
    }
    if (parent->order > -1){
      memcpy(idx_map, idx_map_, parent->order*sizeof(char));
      idx_map[parent->order] = '\0';
    } else
      memcpy(idx_map, idx_map_, strlen(idx_map_)+1);
    is_intm       = 0;
  }

  Idx_Tensor::Idx_Tensor(algstrct const * sr) : Term(sr) {
    idx_map = NULL;
    parent  = NULL;
    is_intm = 0;
  }

  Idx_Tensor::Idx_Tensor(algstrct const * sr, double scl) : Term(sr) {
    idx_map = NULL;
    parent  = NULL;
    is_intm = 0;
    sr->cast_double(scl, scale);
  }

  Idx_Tensor::Idx_Tensor(algstrct const * sr, int64_t scl) : Term(sr) {
    idx_map = NULL;
    parent  = NULL;
    is_intm = 0;
    sr->cast_int(scl, scale);
  }

  Idx_Tensor::Idx_Tensor(Idx_Tensor const & other,
                         int                copy,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    if (other.parent == NULL){
      parent  = NULL;
      idx_map = NULL;
      is_intm = 0;
    } else {
      parent = other.parent;
      if (remap != NULL) {
        typename std::map<CTF_int::tensor*, CTF_int::tensor*>::iterator it = remap->find(parent);
        assert(it != remap->end()); // assume a remapping will be complete
        parent = it->second;
      }

      if (copy || other.is_intm){
        parent = new CTF_int::tensor(other.parent,other.parent->is_mapped,other.parent->is_mapped);
        is_intm = 1;
      } else {
        // leave parent as is - already correct
        is_intm = 0;
      }
      idx_map = (char*)CTF_int::alloc(other.parent->order*sizeof(char));
      memcpy(idx_map, other.idx_map, parent->order*sizeof(char));
    }
    sr->safecopy(scale,other.scale);
  }

/*  Idx_Tensor::Idx_Tensor(){
    parent      = NULL;
    idx_map     = NULL;
    is_intm     = 0;
    sr->copy(scale,sr->mulid());
  }

  Idx_Tensor::Idx_Tensor(double val) : Term(NULL) {
    parent  = NULL;
    idx_map = NULL;
    is_intm = 0;
    scale   = (char*)alloc(sizeof(double));
    sr->copy(scale,(char const*)&val);
  }*/

  Idx_Tensor::~Idx_Tensor(){
    if (is_intm) { 
      delete parent;
      is_intm = 0;
    }
    if (parent != NULL)  cdealloc(idx_map);
    idx_map = NULL;
  }

  Term * Idx_Tensor::clone(std::map<CTF_int::tensor*, CTF_int::tensor*>* remap) const {
    return new Idx_Tensor(*this, 0, remap);
  }

  World * Idx_Tensor::where_am_i() const {
    if (parent == NULL) return NULL;
    return parent->wrld;
  }

  void Idx_Tensor::operator=(Idx_Tensor const & B){
    if (global_schedule != NULL) {
      std::cout << "op= tensor" << std::endl;
      assert(false);
    } else {
      if (sr->has_mul()){
        sr->safecopy(scale,sr->addid());
      } else {
        for (int i=0; i<this->parent->order; i++){
          for (int j=0; j<i; j++){
            if (this->idx_map[i] == this->idx_map[j]){
              printf("CTF ERROR: operations such as B[\"...i...i...\"] = A[\"...i...\"] are not supported when B is defined on an algebraic structure that does not have a multiplicative identity (and here we CTF wants to try to do B[\"...i...i...\"] = 0*B[\"...i...i...\"] + A[\"...i...\"]), a workaround is to supply a multiplicative operator that outputs the additive identity element when any element is multiplied by a multiplicative identity element\n");
              IASSERT(0);
              return;
            }
          }
        }
        this->parent->set_zero();
      }
      B.execute(*this);
      sr->safecopy(scale,sr->mulid());
    }
  }

  void Idx_Tensor::operator=(Term const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation(TENSOR_OP_SET, new Idx_Tensor(*this), B.clone()));
    } else {
      if (sr->has_mul()){
        sr->safecopy(scale,sr->addid());
      } else {
        for (int i=0; i<this->parent->order; i++){
          for (int j=0; j<i; j++){
            if (this->idx_map[i] == this->idx_map[j]){
              printf("CTF ERROR: operations such as B[\"...i...i...\"] = A[\"...i...\"] are not supported (only += is ok if an output index is repeated) when B is defined on an algebraic structure that does not have a multiplicative identity (and here we CTF wants to try to do B[\"...i...i...\"] = 0*B[\"...i...i...\"] + A[\"...i...\"]), a workaround is to supply a multiplicative operator that outputs the additive identity element when any element is multiplied by a multiplicative identity element\n");
              IASSERT(0);
              return;
            }
          }
        }
        this->parent->set_zero();
      }
      B.execute(*this);
      sr->safecopy(scale,sr->mulid());
    }
  }

  void Idx_Tensor::operator+=(Term const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation(TENSOR_OP_SUM, new Idx_Tensor(*this), B.clone()));
    } else {
      //sr->copy(scale,sr->mulid());
      B.execute(*this);
      sr->safecopy(scale,sr->mulid());
    }
  }
  
  void Idx_Tensor::operator=(double scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator+=(double scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator-=(double scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator*=(double scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,scl); }
  void Idx_Tensor::multeq(double scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,scl); }

  void Idx_Tensor::operator=(int64_t scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator+=(int64_t scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator-=(int64_t scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,scl); }
  void Idx_Tensor::operator*=(int64_t scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,scl); }

  void Idx_Tensor::operator=(int scl){ this->execute(this->get_uniq_inds()) = Idx_Tensor(sr,(int64_t)scl); }
  void Idx_Tensor::operator+=(int scl){ this->execute(this->get_uniq_inds()) += Idx_Tensor(sr,(int64_t)scl); }
  void Idx_Tensor::operator-=(int scl){ this->execute(this->get_uniq_inds()) -= Idx_Tensor(sr,(int64_t)scl); }
  void Idx_Tensor::operator*=(int scl){ this->execute(this->get_uniq_inds()) *= Idx_Tensor(sr,(int64_t)scl); }

  /*Idx_Tensor Idx_Tensor::operator-() const {

    Idx_Tensor trm(*this);
    sr->safeaddinv(trm.scale,trm.scale);
    return trm;
  }*/

  void Idx_Tensor::operator-=(Term const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation(TENSOR_OP_SUBTRACT, new Idx_Tensor(*this), B.clone()));
    } else {
      Term * Bcpy = B.clone();
      char * ainv = NULL;
      B.sr->safeaddinv(B.sr->mulid(),ainv);
      B.sr->safemul(Bcpy->scale,ainv,Bcpy->scale);
      Bcpy->execute(*this);
      sr->safecopy(scale,sr->mulid());
      if (ainv != NULL) free(ainv);
      delete Bcpy;
    }
  }

  void Idx_Tensor::operator*=(Term const & B){
    if (global_schedule != NULL) {
      global_schedule->add_operation(
          new TensorOperation(TENSOR_OP_MULTIPLY, new Idx_Tensor(*this), B.clone()));
    } else {
      Contract_Term ctrm = (*this)*B;
      *this = ctrm;
    }
  }




  void Idx_Tensor::execute(Idx_Tensor output) const {
    if (parent == NULL){
//      output.sr->safemul(output.scale, scale, output.scale);
      CTF_int::tensor ts(output.sr, 0, (int64_t*)NULL, NULL, output.where_am_i(), true, NULL, 0);
      char * data;
      int64_t sz;
      ts.get_raw_data(&data, &sz);
      if (ts.wrld->rank == 0) ts.sr->safecopy(data, scale);
      summation s(&ts, NULL, ts.sr->mulid(), 
                  output.parent, output.idx_map, output.scale);
      s.execute();
    } else {
      summation s(this->parent, idx_map, scale,
                  output.parent, output.idx_map, output.scale);
      s.execute();
//      output.parent->sum(scale, *this->parent, idx_map,
  //                       output.scale, output.idx_map);
    } 
  }

  Idx_Tensor Idx_Tensor::execute(std::vector<char> out_inds) const {
    return *this;
  }

  double Idx_Tensor::estimate_time(Idx_Tensor output) const {
    if (parent == NULL){
      CTF_int::tensor ts(output.sr, 0, (int64_t*)NULL, NULL, output.where_am_i(), true, NULL, 0);
      summation s(&ts, NULL, output.sr->mulid(), 
                  output.parent, output.idx_map, output.scale);
      return s.estimate_time();
    } else {
      summation s(this->parent, idx_map, scale,
                  output.parent, output.idx_map, output.scale);
      return s.estimate_time();
    } 
  }

  Idx_Tensor Idx_Tensor::estimate_time(double & cost, std::vector<char> out_inds) const {
    return *this;
  }

  std::vector<char> Idx_Tensor::get_uniq_inds() const {
    if (parent == NULL) return std::vector<char>();
    std::set<char> uniq_inds;
    for (int k=0; k<this->parent->order; k++){
      uniq_inds.insert(idx_map[k]);
    }
    return std::vector<char>(uniq_inds.begin(), uniq_inds.end());
  }

  void Idx_Tensor::get_inputs(std::set<Idx_Tensor*, tensor_name_less >* inputs_set) const {
    inputs_set->insert((Idx_Tensor*)this);
  }

  /*template<typename dtype, bool is_ord>
  void Idx_Tensor::operator=(dtype B){
    *this=(Scalar(B,*(this->parent->world))[""]);
  }
  void Idx_Tensor::operator+=(dtype B){
    *this+=(Scalar(B,*(this->parent->world))[""]);
  }
  void Idx_Tensor::operator-=(dtype B){
    *this-=(Scalar(B,*(this->parent->world))[""]);
  }
  void Idx_Tensor::operator*=(dtype B){
    *this*=(Scalar(B,*(this->parent->world))[""]);
  }*/

  /*
  Idx_Tensor Idx_Tensor::operator+(Idx_Tensor tsr){
    if (has_contract || has_sum){
      *NBR = (*NBR)-tsr;
      return *this;
    }
    NBR = &tsr;
    has_sum = 1;
    return *this;
  }

  Idx_Tensor Idx_Tensor::operator-(Idx_Tensor tsr){
    if (has_contract || has_sum){
      *NBR = (*NBR)-tsr;
      return *this;
    }
    NBR = &tsr;
    has_sum = 1;
    if (tsr.has_scale) tsr.scale = -sr->mulid*tsr.scale;
    else {
      tsr.has_scale = 1;
      tsr.scale = -sr->mulid();
    }
    return *this;
  }

  Idx_Tensor Idx_Tensor::operator*(double  scl){
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
  void Idx_Tensor::run(Idx_Tensor* output, dtype  beta){
    dtype  alpha;
    if (has_scale) alpha = scale;
    else alpha = sr->mulid();
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
                               sr->addid(),                    itsr.idx_map);
        itsr.run(output, beta);
      } else {
        output->parent->contract(alpha, *(this->parent), this->idx_map,
                                        *(NBR->parent),  NBR->idx_map,
                                 beta,                   output->idx_map);
      }
    } else {
      if (has_sum){
        CTF_int::tensor tcpy(*(this->parent),1);
        Idx_Tensor itsr(&tcpy, idx_map);
        NBR->run(&itsr, alpha);
        output->parent->sum(sr->mulid, tcpy, idx_map, beta, output->idx_map);
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
