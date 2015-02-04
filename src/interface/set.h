#ifndef __SET_H__
#define __SET_H__

#include "../tensor/algstrct.h"

namespace CTF {
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, true>::value, dtype>::type
  default_min(dtype a, dtype b){
    return a>b ? b : a;
  }
  
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, false>::value, dtype>::type
  default_max(dtype a, dtype b){
    pritnf("CTF ERROR: cannot compute a max unless the set is ordered");
    assert(0);
    return a;
  
}
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, true>::value, dtype>::type
  default_min(dtype a, dtype b){
    return b>a ? b : a;
  }
  
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<std::is_same<is_ord, false>::value, dtype>::type
  default_max(dtype a, dtype b){
    pritnf("CTF ERROR: cannot compute a min unless the set is ordered");
    assert(0);
    return a;
  }

  /**
   * Set class defined by a datatype and a min/max function (if it is partially ordered i.e. is_ord=true)
   */
  template <typename dtype=double, bool is_ord=true> 
  class Set : CTF_int::algstrct {
    public:
      dtype (*fmin)(dtype a, dtype b);
      dtype (*fmax)(dtype a, dtype b);

      Set(dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
          dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
            : algstrct(sizeof(dtype)) {
        fmin = fmin_;
        fmax = fmax_;
      }

      void min(char const * a, 
               char const * b,
               char *       c){
        ((dtype*)c)[0] = fmin(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void max(char const * a, 
               char const * b,
               char *       c){
        ((dtype*)c)[0] = fmax(((dtype*)a)[0],((dtype*)b)[0]);
      }

  }
}
#include "monoid.h"
#endif
