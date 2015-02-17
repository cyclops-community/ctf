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

  template<typename dtype>
  Vector<dtype>::Vector(int                len_,
                        World &            world_,
                        Set<dtype> const * sr_,
                        char const *       name_,
                        int                profile_)
   : Tensor<dtype>(1, int1(len_), int1(NS), sr_, world_, name_, profile_) {
    len = len_;
  }

}
