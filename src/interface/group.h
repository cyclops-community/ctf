#ifndef __GROUP_H__
#define __GROUP_H__

#include "../tensor/algstrct.h"

namespace CTF {
  
  template <typename dtype>
  dtype default_addinv(dtype a){
    return -a;
  }

 
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_abs(dtype a){
    dtype b = default_addinv<dtype>(a);
    return a>=b ? a : b;
  }
  
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_abs(dtype a){
    printf("CTF ERROR: cannot compute abs unless the set is ordered");
    assert(0);
    return a;
  }

  template <typename dtype, dtype (*abs)(dtype)>
  void char_abs(char const * a,
                char * b){
    ((dtype*)b)[0]=abs(((dtype const*)a)[0]);
  }

  /**
   * \brief Group is a Monoid with operator '-' defined
   *   special case (parent) of a ring
   */
  template <typename dtype=double, bool is_ord=true> 
  class Group : public Monoid<dtype, is_ord> {
    public:
      Group(Group const & other) : Monoid<dtype, is_ord>(other) { }

      virtual CTF_int::algstrct * clone() const {
        return new Group<dtype, is_ord>(*this);
      }

      Group() : Monoid<dtype, is_ord>() { 
        this->abs = &char_abs< dtype, default_abs<dtype, is_ord> >;
      } 

      Group(dtype taddid_,
            dtype (*fadd_)(dtype a, dtype b),
            MPI_Op addmop_)
              : Monoid<dtype, is_ord>(taddid_, fadd_, addmop_) { 
        abs = &char_abs< dtype, default_abs<dtype, is_ord> >;
      }

      void addinv(char const * a, char * b) const {
        ((dtype*)b)[0] = -((dtype*)a)[0];
      }
  };

}
#include "semiring.h"
#endif
