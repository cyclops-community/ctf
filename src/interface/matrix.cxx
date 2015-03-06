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

  template<typename dtype, bool is_ord>
  Matrix<dtype, is_ord>::Matrix(int                       nrow_,
                                int                       ncol_,
                                int                       sym_,
                                World &                   world_,
                                char const *              name_,
                                int                       profile_,
                                Set<dtype,is_ord> const & sr_)
    : Tensor<dtype,is_ord>(2, int2(nrow_, ncol_), int2(sym_, NS), 
                           world_, sr_, name_, profile_) {
    nrow = nrow_;
    ncol = ncol_;
    sym = sym_;
  }

  template<typename dtype, bool is_ord>
  Matrix<dtype, is_ord>::Matrix(int                       nrow_,
                                int                       ncol_,
                                int                       sym_,
                                World &                   world_,
                                Set<dtype,is_ord> const & sr_)
    : Tensor<dtype,is_ord>(2, int2(nrow_, ncol_), int2(sym_, NS), 
                           world_, Ring<dtype,is_ord>(), NULL, 0) {
    nrow = nrow_;
    ncol = ncol_;
    sym = sym_;
  }
 
  template<typename dtype, bool is_ord>
  Matrix<dtype,is_ord> & Matrix<dtype,is_ord>::operator=(const Matrix<dtype,is_ord> & A){
    CTF_int::tensor::free_self();
    CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile);
    return *this;
  }

}
