#ifndef __CUST_TERM_H__
#define __CUST_TERM_H__

#include "term.h"

namespace CTF_int {
  class univar_function;
}

namespace CTF_int {
  class Fun_Term : public Term{
    public:
      Term * A;
      univar_function * func;

      Fun_Term(Term *            A,
               univar_function * func);

      Fun_Term(Fun_Term const & other,
               std::map<tensor*, tensor*>* remap=NULL);

      ~Fun_Term();

      Term * clone(std::map<tensor*, tensor*>* remap = NULL) const;

      void execute(CTF::Idx_Tensor output) const;

      CTF::Idx_Tensor execute() const;

      CTF::Idx_Tensor estimate_time(double  & cost) const;

      double  estimate_time(CTF::Idx_Tensor output) const;

      void get_inputs(std::set<tensor*, tensor_tid_less >* inputs_set) const;

      CTF::World * where_am_i() const;
  };
}

#endif
