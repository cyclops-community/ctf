/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/
#include "fun_term.h"
#include "common.h"
#include "../tensor/algstrct.h"
#include "../scaling/scaling.h"
#include "functions.h"
#include "idx_tensor.h"

namespace CTF_int {
  Unifun_Term::Unifun_Term(Term *                  A_,
                           univar_function const * func_) : Term(A_->sr) {
    A = A_;
    func = func_;
  }

  Unifun_Term::~Unifun_Term(){
    delete A;
  }

  Term * Unifun_Term::clone(std::map<tensor*, tensor*>* remap) const{
    return new Unifun_Term(*this, remap);
  }
  
  Unifun_Term::Unifun_Term(
      Unifun_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->safecopy(this->scale, other.scale);
    func = other.func;
    A = other.A->clone(remap);
  }

  void Unifun_Term::execute(CTF::Idx_Tensor output) const {
    CTF::Idx_Tensor opA = A->execute();
    summation s(opA.parent, opA.idx_map, opA.scale, output.parent, output.idx_map, output.scale, func);
    s.execute();
  }
 
  CTF::Idx_Tensor Unifun_Term::execute() const {
    printf("Univar Unifunction applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
 
  CTF::Idx_Tensor Unifun_Term::estimate_time(double & cost) const {
    printf("Univar Unifunction applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
  double Unifun_Term::estimate_time(CTF::Idx_Tensor output) const{
    double cost = 0.0;
    CTF::Idx_Tensor opA = A->estimate_time(cost);
    summation s(opA.parent, opA.idx_map, opA.scale, output.parent, output.idx_map, output.scale, func);
    cost += s.estimate_time();
    return cost;
  }


  void Unifun_Term::get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const {
    A->get_inputs(inputs_set);
  }

  CTF::World * Unifun_Term::where_am_i() const {
    return A->where_am_i();
  }

  Bifun_Term::Bifun_Term(Term *                  A_,
                         Term *                  B_,
                         bivar_function const * func_) : Term(A_->sr) {
    A = A_;
    B = B_;
    func = func_;
  }

  Bifun_Term::~Bifun_Term(){
    delete A;
    delete B;
  }

  Term * Bifun_Term::clone(std::map<tensor*, tensor*>* remap) const{
    return new Bifun_Term(*this, remap);
  }
  
  Bifun_Term::Bifun_Term(
      Bifun_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->safecopy(this->scale, other.scale);
    func = other.func;
    A = other.A->clone(remap);
    B = other.B->clone(remap);
  }

  void Bifun_Term::execute(CTF::Idx_Tensor output) const {
    CTF::Idx_Tensor opA = A->execute();
    CTF::Idx_Tensor opB = B->execute();
/*    char * scl;
    scl = NULL;
    if (opA.scale != NULL || opB.scale != NULL) 
      opA.sr->safemul(opA.scale, opB.scale, scl);*/
    if (!opA.sr->isequal(opA.scale, opA.sr->mulid()) ||
        !opB.sr->isequal(opB.scale, opB.sr->mulid()) /*||
        !output.sr->isequal(output.scale, output.sr->mulid())*/){
      if (opA.parent->wrld->rank == 0)
        printf("CTF ERROR: cannot scale tensors when using bilinear function or transform, aborting.\n");
      ASSERT(0);
      assert(0);
    }
    contraction c(opA.parent, opA.idx_map, opB.parent, opB.idx_map, output.sr->mulid(), output.parent, output.idx_map, output.scale, func);
    //contraction c(opA.parent, opA.idx_map, opB.parent, opB.idx_map, NULL, output.parent, output.idx_map, output.scale, func);
    c.execute();
//    if (scl != NULL) cdealloc(scl);
  }
 
  CTF::Idx_Tensor Bifun_Term::execute() const {
    printf("Bivar Bifunction applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
 
  CTF::Idx_Tensor Bifun_Term::estimate_time(double & cost) const {
    printf("Bivar Bifunction applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
  double Bifun_Term::estimate_time(CTF::Idx_Tensor output) const{
    double cost = 0.0;
    CTF::Idx_Tensor opA = A->estimate_time(cost);
    CTF::Idx_Tensor opB = B->estimate_time(cost);
    contraction c(opA.parent, opA.idx_map, opB.parent, opB.idx_map, opB.scale, output.parent, output.idx_map, output.scale, func);
    cost += c.estimate_time();
    return cost;
  }


  void Bifun_Term::get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const {
    A->get_inputs(inputs_set);
    B->get_inputs(inputs_set);
  }

  CTF::World * Bifun_Term::where_am_i() const {
    if (A->where_am_i() != NULL)
      return A->where_am_i();
    else
      return B->where_am_i();
  }


}
