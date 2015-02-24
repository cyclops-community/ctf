/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "idx_tensor.h"
#include "../tensor/algstrct.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"

using namespace CTF;

namespace CTF_int {
  Idx_Tensor * get_full_intm(Idx_Tensor& A, 
                             Idx_Tensor& B){
    int * len_C, * sym_C;
    char * idx_C;
    int order_C, i, j, idx;
    
    order_C = 0;
    for (i=0; i<A.parent->order; i++){
      order_C++;
      for (j=0; j<i; j++){
        if (A.idx_map[i] == A.idx_map[j]){
          order_C--;
          break;
        }
      }
    }
    for (j=0; j<B.parent->order; j++){
      order_C++;
      for (i=0; i<std::max(A.parent->order, B.parent->order); i++){
        if (i<j && B.idx_map[i] == B.idx_map[j]){
          order_C--;
          break;
        }
        if (i<A.parent->order && A.idx_map[i] == B.idx_map[j]){
          order_C--;
          break;
        }
      }
    }

    idx_C = (char*)malloc(sizeof(char)*order_C);
    sym_C = (int*)malloc(sizeof(int)*order_C);
    len_C = (int*)malloc(sizeof(int)*order_C);
    idx = 0;
    for (i=0; i<A.parent->order; i++){
      for (j=0; j<i && A.idx_map[i] != A.idx_map[j]; j++){}
      if (j!=i) break;
      idx_C[idx] = A.idx_map[i];
      len_C[idx] = A.parent->lens[i];
      if (idx >= 1 && i >= 1 && idx_C[idx-1] == A.idx_map[i-1] && A.parent->sym[i-1] != NS){
        sym_C[idx-1] = A.parent->sym[i-1];
      }
      sym_C[idx] = NS;
      idx++;
    }
    int order_AC = idx;
    for (j=0; j<B.parent->order; j++){
      for (i=0; i<j && B.idx_map[i] != B.idx_map[j]; i++){}
      if (i!=j) break;
      for (i=0; i<order_AC && idx_C[i] != B.idx_map[j]; i++){}
      if (i!=order_AC){
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
      len_C[idx] = B.parent->lens[j];
      if (idx >= 1 && j >= 1 && idx_C[idx-1] == B.idx_map[j-1] && B.parent->sym[j-1] != NS){
        sym_C[idx-1] = B.parent->sym[j-1];
      }
      sym_C[idx] = NS;
      idx++;
    }

    tensor * tsr_C = new tensor(A.parent->sr, order_C, len_C, sym_C, A.parent->wrld);
    Idx_Tensor * out = new Idx_Tensor(tsr_C, idx_C);
    out->is_intm = 1;
    free(sym_C);
    free(len_C);
    free(idx_C);
    return out;
  }


  //general Term functions, see ../../include/ctf.hpp for doxygen comments

  /*Term::operator dtype() const {
    assert(where_am_i() != NULL);
    Scalar sc(*where_am_i());
    Idx_Tensor isc(&sc,""); 
    execute(isc);
  //  delete isc;
    return sc.get_val();
  }*/

  //
  //void Term::execute(Idx_Tensor output){
  //  ABORT; //I don't see why this part of the code should ever be reached
  ////  output.scale *= scale;
  //}
  //
  //
  //Idx_Tensor Term::execute(){
  //  ABORT; //I don't see why this part of the code should ever be reached
  //  return Idx_Tensor();
  //}


  Term::Term(algstrct const * sr_){
    sr = sr_;
    scale = (char*)malloc(sr->el_size);
    sr->copy(scale,sr->mulid());
  }
  
  Term::~Term(){
    free(scale);
  }

  Contract_Term Term::operator*(Term const & A) const {
    Contract_Term trm(this->clone(),A.clone());
    return trm;
  }


  Sum_Term Term::operator+(Term const & A) const {
    Sum_Term trm(this->clone(),A.clone());
    return trm;
  }


  Sum_Term Term::operator-(Term const & A) const {
    Sum_Term trm(this->clone(),A.clone());
    sr->addinv(A.scale, trm.operands[1]->scale);
    return trm;
  }

  void Term::operator=(Term const & B){ execute() = B; }
  void Term::operator=(CTF::Idx_Tensor const & B){ execute() = B; }
  void Term::operator+=(Term const & B){ execute() += B; }
  void Term::operator-=(Term const & B){ execute() -= B; }
  void Term::operator*=(Term const & B){ execute() *= B; }

 


  Contract_Term Term::operator*(int64_t scl) const {
    Idx_Tensor iscl(sr);
    sr->cast_int(scl, iscl.scale);
    Contract_Term trm(this->clone(),iscl.clone());
    return trm;
  }

  Contract_Term Term::operator*(double scl) const {
    Idx_Tensor iscl(sr);
    sr->cast_double(scl, iscl.scale);
    Contract_Term trm(this->clone(),iscl.clone());
    return trm;
  }

  //functions spectific to Sum_Term

  Sum_Term::Sum_Term(Term * B, Term * A) : Term(A->sr) {
    operands.push_back(B);
    operands.push_back(A);
  }

  Sum_Term::~Sum_Term(){
    for (int i=0; i<(int)operands.size(); i++){
      delete operands[i];
    }
    operands.clear();
  }


  Sum_Term::Sum_Term(
      Sum_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->copy(this->scale, other.scale);
    for (int i=0; i<(int)other.operands.size(); i++){
      this->operands.push_back(other.operands[i]->clone(remap));
    }
  }


  Term * Sum_Term::clone(std::map<tensor*, tensor*>* remap) const{
    return new Sum_Term(*this, remap);
  }


  Sum_Term Sum_Term::operator+(Term const & A) const {
    Sum_Term st(*this);
    st.operands.push_back(A.clone());
    return st;
  }


  Sum_Term Sum_Term::operator-(Term const & A) const {
    Sum_Term st(*this);
    st.operands.push_back(A.clone());
    sr->addinv(A.scale, st.operands.back()->scale);
    return st;
  }

  Idx_Tensor Sum_Term::estimate_time(double & cost) const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost);
      Idx_Tensor op_B = pop_B->estimate_time(cost);
      Idx_Tensor * intm = get_full_intm(op_A, op_B);
      summation s1(op_A.parent, op_A.idx_map, op_A.scale, 
                   intm->parent, intm->idx_map, intm->scale);
      cost += s1.estimate_time();
      summation s2(op_B.parent, op_B.idx_map, op_B.scale, 
                   intm->parent, intm->idx_map, intm->scale);
      cost += s2.estimate_time();
      tmp_ops.push_back(intm);
      delete pop_A;
      delete pop_B;
    }
    Idx_Tensor ans = tmp_ops[0]->estimate_time(cost);
    delete tmp_ops[0];
    tmp_ops.clear();
    return ans;
  }


  double Sum_Term::estimate_time(Idx_Tensor output) const{
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    double cost = 0.0;
    for (int i=0; i<((int)tmp_ops.size())-1; i++){
      cost += tmp_ops[i]->estimate_time(output);
    }
    Idx_Tensor itsr = tmp_ops.back()->estimate_time(cost);
    summation s(itsr.parent, itsr.idx_map, itsr.scale, output.parent, output.idx_map, output.scale);
    cost += s.estimate_time();
    tmp_ops.clear();
    return cost;
  }

  Idx_Tensor Sum_Term::execute() const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute();
      Idx_Tensor op_B = pop_B->execute();
      Idx_Tensor * intm = get_full_intm(op_A, op_B);
      summation s1(op_A.parent, op_A.idx_map, op_A.scale, 
                   intm->parent, intm->idx_map, intm->scale);
      s1.execute();
      //a little sloopy but intm->scale should always be 1 here
      summation s2(op_B.parent, op_B.idx_map, op_B.scale, 
                   intm->parent, intm->idx_map, intm->scale);
      s2.execute();
      tmp_ops.push_back(intm);
      delete pop_A;
      delete pop_B;
    }
    sr->mul(tmp_ops[0]->scale, this->scale, tmp_ops[0]->scale); 
    Idx_Tensor ans = tmp_ops[0]->execute();
    delete tmp_ops[0];
    tmp_ops.clear();
    return ans;
  }


  void Sum_Term::execute(Idx_Tensor output) const{
    std::vector< Term* > tmp_ops = operands;
    for (int i=0; i<((int)tmp_ops.size())-1; i++){
      tmp_ops[i]->execute(output);
      sr->copy(output.scale, sr->mulid());
    }
    Idx_Tensor itsr = tmp_ops.back()->execute();
    summation s(itsr.parent, itsr.idx_map, itsr.scale, output.parent, output.idx_map, output.scale);
    s.execute();
  }


  void Sum_Term::get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const {
    for (int i=0; i<(int)operands.size(); i++){
      operands[i]->get_inputs(inputs_set);
    }
  }


  World * Sum_Term::where_am_i() const {
    World * w = NULL;
    for (int i=0; i<(int)operands.size(); i++){
      if (operands[i]->where_am_i() != NULL) {
        w = operands[i]->where_am_i();
      }
    }
    return w;
  }


  //functions spectific to Contract_Term

  Contract_Term::~Contract_Term(){
    for (int i=0; i<(int)operands.size(); i++){
      delete operands[i];
    }
    operands.clear();
  }


  World * Contract_Term::where_am_i() const {
    World * w = NULL;
    for (int i=0; i<(int)operands.size(); i++){
      if (operands[i]->where_am_i() != NULL) {
        w = operands[i]->where_am_i();
      }
    }
    return w;
  }


  Contract_Term::Contract_Term(Term * B, Term * A) : Term(A->sr) {
    operands.push_back(B);
    operands.push_back(A);
  }


  Contract_Term::Contract_Term(
      Contract_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->copy(this->scale, other.scale);
    for (int i=0; i<(int)other.operands.size(); i++){
      Term * t = other.operands[i]->clone(remap);
      operands.push_back(t);
    }
  }


  Term * Contract_Term::clone(std::map<tensor*, tensor*>* remap) const {
    return new Contract_Term(*this, remap);
  }


  Contract_Term Contract_Term::operator*(Term const & A) const {
    Contract_Term ct(*this);
    ct.operands.push_back(A.clone());
    return ct;
  }


  void Contract_Term::execute(Idx_Tensor output)const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    char tscale[sr->el_size];
    sr->copy(tscale, this->scale);
    while (tmp_ops.size() > 2){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute();
      Idx_Tensor op_B = pop_B->execute();
      if (op_A.parent == NULL) {
        sr->mul(op_A.scale, op_B.scale, op_B.scale);
        tmp_ops.push_back(op_B.clone());
      } else if (op_B.parent == NULL) {
        sr->mul(op_A.scale, op_B.scale, op_B.scale);
        tmp_ops.push_back(op_A.clone());
      } else {
        Idx_Tensor * intm = get_full_intm(op_A, op_B);
        sr->mul(tscale, op_A.scale, tscale);
        sr->mul(tscale, op_B.scale, tscale);
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, tscale,
                      intm->parent, intm->idx_map, intm->scale);
        c.execute(); 
        sr->copy(tscale, sr->mulid());
        tmp_ops.push_back(intm);
      }
      delete pop_A;
      delete pop_B;
    } 
    {
      ASSERT(tmp_ops.size() == 2);
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute();
      Idx_Tensor op_B = pop_B->execute();
      char tscale[sr->el_size];
      sr->copy(tscale, this->scale);
      sr->mul(tscale, op_A.scale, tscale);
      sr->mul(tscale, op_B.scale, tscale);

      if (op_A.parent == NULL && op_B.parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A.parent == NULL){
        summation s(op_B.parent, op_B.idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else if (op_B.parent == NULL){
        summation s(op_A.parent, op_A.idx_map, tscale,
                    output.parent, output.idx_map, output.scale);
        s.execute();
      } else {
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, tscale,
                      output.parent, output.idx_map, output.scale);
        c.execute();
      }
      delete pop_A;
      delete pop_B;
    } 
  }


  Idx_Tensor Contract_Term::execute() const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    char tscale[sr->el_size];
    sr->copy(tscale, this->scale);
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->execute();
      Idx_Tensor op_B = pop_B->execute();
      if (op_A.parent == NULL) {
        sr->mul(op_B.scale, op_A.scale, op_B.scale);
        tmp_ops.push_back(op_B.clone());
      } else if (op_B.parent == NULL) {
        sr->mul(op_B.scale, op_A.scale, op_B.scale);
        tmp_ops.push_back(op_A.clone());
      } else {
        Idx_Tensor * intm = get_full_intm(op_A, op_B);
        sr->mul(tscale, op_A.scale, tscale);
        sr->mul(tscale, op_B.scale, tscale);
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, tscale,
                      intm->parent, intm->idx_map, intm->scale);
        c.execute(); 
        sr->copy(tscale, sr->mulid());
        tmp_ops.push_back(intm);
      }
      delete pop_A;
      delete pop_B;
    } 
    Idx_Tensor rtsr = tmp_ops[0]->execute();
    delete tmp_ops[0];
    tmp_ops.clear();
    return rtsr;
  }


  double Contract_Term::estimate_time(Idx_Tensor output)const {
    double cost = 0.0;
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 2){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost);
      Idx_Tensor op_B = pop_B->estimate_time(cost);
      if (op_A.parent == NULL) {
        tmp_ops.push_back(op_B.clone());
      } else if (op_B.parent == NULL) {
        tmp_ops.push_back(op_A.clone());
      } else {
        Idx_Tensor * intm = get_full_intm(op_A, op_B);
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, this->scale, 
                      intm->parent, intm->idx_map, intm->scale);
        cost += c.estimate_time();
        tmp_ops.push_back(intm);
      }
      delete pop_A;
      delete pop_B;
    } 
    {
      ASSERT(tmp_ops.size() == 2);
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost);
      Idx_Tensor op_B = pop_B->estimate_time(cost);
      
      if (op_A.parent == NULL && op_B.parent == NULL){
        assert(0); //FIXME write scalar to whole tensor
      } else if (op_A.parent == NULL){
        summation s(op_B.parent, op_B.idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else if (op_B.parent == NULL){
        summation s(op_A.parent, op_A.idx_map, this->scale,
                    output.parent, output.idx_map, output.scale);
        cost += s.estimate_time();
      } else {
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, this->scale,
                      output.parent, output.idx_map, output.scale);
        cost += c.estimate_time();
      }
      delete pop_A;
      delete pop_B;
    } 
    return cost;
  }


  Idx_Tensor Contract_Term::estimate_time(double & cost) const {
    std::vector< Term* > tmp_ops;
    for (int i=0; i<(int)operands.size(); i++){
      tmp_ops.push_back(operands[i]->clone());
    }
    while (tmp_ops.size() > 1){
      Term * pop_A = tmp_ops.back();
      tmp_ops.pop_back();
      Term * pop_B = tmp_ops.back();
      tmp_ops.pop_back();
      Idx_Tensor op_A = pop_A->estimate_time(cost);
      Idx_Tensor op_B = pop_B->estimate_time(cost);
      if (op_A.parent == NULL) {
        tmp_ops.push_back(op_B.clone());
      } else if (op_B.parent == NULL) {
        tmp_ops.push_back(op_A.clone());
      } else {
        Idx_Tensor * intm = get_full_intm(op_A, op_B);
        contraction c(op_A.parent, op_A.idx_map,
                      op_B.parent, op_B.idx_map, this->scale, 
                      intm->parent, intm->idx_map, intm->scale);
        cost += c.estimate_time();

        tmp_ops.push_back(intm);
      }
      delete pop_A;
      delete pop_B;
    } 
    return tmp_ops[0]->estimate_time(cost);
  }



  void Contract_Term::get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const {
    for (int i=0; i<(int)operands.size(); i++){
      operands[i]->get_inputs(inputs_set);
    }
  }

}

