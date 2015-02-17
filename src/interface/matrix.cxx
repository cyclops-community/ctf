/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"

namespace CTF {

  struct int2
  {
    int i[2];
    int2(int a, int b)
    {
      i[0] = a;
      i[1] = b;
    }
    operator const int*() const
    {
      return i;
    }
  };

  template<typename dtype>
  Matrix<dtype>::Matrix(int                nrow_,
                        int                ncol_,
                        int                sym_,
                        World &            world_,
                        Set<dtype> const & sr_,
                        char const *       name_,
                        int                profile_)
    : Tensor<dtype>(2, int2(nrow_, ncol_), int2(sym_, NS), 
                        world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    sym = sym_;
  }


}
