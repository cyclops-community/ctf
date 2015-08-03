/*Copyright (c) 2013, Edgar Solomonik, all rights reserved.*/
#include "fun_term.h"
#include "common.h"
#include "../tensor/algstrct.h"
#include "../scaling/scaling.h"
#include "functions.h"
#include "idx_tensor.h"

namespace CTF_int {
  Fun_Term::Fun_Term(Term *                  A_,
                     univar_function const * func_) : Term(A_->sr) {
    A = A_;
    func = func_;
  }

  Fun_Term::~Fun_Term(){
    delete A;
  }

  Term * Fun_Term::clone(std::map<tensor*, tensor*>* remap) const{
    return new Fun_Term(*this, remap);
  }
  
  Fun_Term::Fun_Term(
      Fun_Term const & other,
      std::map<tensor*, tensor*>* remap) : Term(other.sr) {
    sr->safecopy(this->scale, other.scale);
    func = other.func;
    A = other.A->clone(remap);
  }

  void Fun_Term::execute(CTF::Idx_Tensor output) const {
    CTF::Idx_Tensor opA = A->execute();
    summation s(opA.parent, opA.idx_map, opA.scale, output.parent, output.idx_map, output.scale, func);
    s.execute();
  }
 
  CTF::Idx_Tensor Fun_Term::execute() const {
    printf("Univar Function applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
 
  CTF::Idx_Tensor Fun_Term::estimate_time(double & cost) const {
    printf("Univar Function applications cannot currently be a part of a longer algebraic expression\n");
    assert(0); 
    return CTF::Idx_Tensor(NULL);
  }
  double Fun_Term::estimate_time(CTF::Idx_Tensor output) const{
    double cost = 0.0;
    CTF::Idx_Tensor opA = A->estimate_time(cost);
    summation s(opA.parent, opA.idx_map, opA.scale, output.parent, output.idx_map, output.scale, func);
    cost += s.estimate_time();
    return cost;
  }


  void Fun_Term::get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const {
    A->get_inputs(inputs_set);
  }

  CTF::World * Fun_Term::where_am_i() const {
    return A->where_am_i();
  }

}
