#include "common.h"

namespace CTF {

  struct int1
  {
    int i[1];
    int1(int a)
    {
      i[0] = a;
    }
    operator const int*() const
    {
      return i;
    }
  };

  template<typename dtype, bool is_ord>
  Vector<dtype, is_ord>::Vector(int                       len_,
                                World &                   world_,
                                Set<dtype,is_ord> const & sr_,
                                char const *              name_,
                                int                       profile_)
   : Tensor<dtype,is_ord>(1, int1(len_), int1(NS), sr_, world_, name_, profile_) {
    len = len_;
  }

  template<typename dtype, bool is_ord>
  Vector<dtype, is_ord>::Vector(int                       len_,
                                World &                   world_,
                                char const *              name_,
                                int                       profile_,
                                Set<dtype,is_ord> const & sr_)
   : Tensor<dtype,is_ord>(1, int1(len_), int1(NS), sr_, world_, name_, profile_) {
    len = len_;
  }



}
