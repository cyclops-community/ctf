#ifndef __CUST_TERM_H__
#define __CUST_TERM_H__

#include "term.h"

namespace CTF_int {
  class univar_function;
  class bivar_function;
}

namespace CTF_int {
  class Unifun_Term : public Term{
    public:
      Term * A;
      univar_function const * func;

      Unifun_Term(Term *                  A,
                  univar_function const * func);

      Unifun_Term(Unifun_Term const & other,
                  std::map<tensor*, tensor*>* remap=NULL);

      ~Unifun_Term();

      Term * clone(std::map<tensor*, tensor*>* remap = NULL) const;

      void execute(CTF::Idx_Tensor output) const;

      CTF::Idx_Tensor execute(std::vector<char> out_inds) const;

      CTF::Idx_Tensor estimate_time(double  & cost, std::vector<char> out_inds) const;

      double estimate_time(CTF::Idx_Tensor output) const;

      std::vector<char> get_uniq_inds() const;

      void get_inputs(std::set<CTF::Idx_Tensor*, tensor_name_less >* inputs_set) const;

      CTF::World * where_am_i() const;
  };

  class Bifun_Term : public Term {
    public:
      Term * A;
      Term * B;
      bivar_function const * func;

      Bifun_Term(Term *                 A,
                 Term *                 B,
                 bivar_function const * func);

      Bifun_Term(Bifun_Term const & other,
                 std::map<tensor*, tensor*>* remap=NULL);

      ~Bifun_Term();

      Term * clone(std::map<tensor*, tensor*>* remap = NULL) const;

      void execute(CTF::Idx_Tensor output) const;

      CTF::Idx_Tensor execute(std::vector<char> out_inds) const;

      CTF::Idx_Tensor estimate_time(double  & cost, std::vector<char> out_inds) const;

      double estimate_time(CTF::Idx_Tensor output) const;

      std::vector<char> get_uniq_inds() const;

      void get_inputs(std::set<CTF::Idx_Tensor*, tensor_name_less >* inputs_set) const;

      CTF::World * where_am_i() const;
  };

}

#endif
