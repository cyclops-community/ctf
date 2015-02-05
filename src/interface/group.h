#ifndef __GROUP_H__
#define __GROUP_H__

#include "../tensor/algstrct.h"

namespace CTF {
  
  template <typename dtype>
  dtype default_addinv(dtype a){
    return -a;
  }

  template <typename dtype, dtype (*fmax)(dtype a, dtype b), dtype (*faddinv)(dtype a, dtype b)>
  void fabs(char const * a, char * b) {
    dtype inva = faddinv(((dtype*)a)[0]);
    ((dtype*)b)[0] = fmax(a,inva);
  }

  /**
   * Group class defined by a datatype and an addition function
   *   addition must have an identity and be associative, does not need to be commutative
   *   define a Semiring/Ring instead if a multiplication
   */
  template <typename dtype=double, bool is_ord=true> 
  class Group : public Monoid<dtype, is_ord> {
    public:
      dtype (*faddinv)(dtype a);
      
      Group(dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
            dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Monoid(fmin_, fmax_) {
        faddinv = &default_addinv<dtype>;
      } 

      Group(dtype taddid_,
            dtype (*fadd_)(dtype a, dtype b),
            dtype (*faddinv_)(dtype a, dtype b),
            dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
            dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Monoid(taddid_, fadd_, fmin_, fmax_) {
        faddinv = faddinv_;
        abs = fabs<dtype, fmax_, faddinv_>;
      }
 
      Group(dtype taddid_,
            dtype (*fadd_)(dtype a, dtype b),
            dtype (*faddinv_)(dtype a, dtype b),
            void (*fxpy_)(int, dtype const *, dtype *),
            MPI_Op addmop_,
            dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
            dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Monoid(taddid_, fadd_, fxpy_, addmop_, fmin_, fmax_) {
        faddinv = faddinv_;
        abs = fabs<dtype, fmax_, faddinv_>;
      }

      void addinv(char const * a, char * b) const {
        ((dtype*)b)[0] = faddinv(((dtype*)a)[0]);
      }
  }

}
#include "algstrct.h"
#endif
