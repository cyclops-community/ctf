#include "common.h"

namespace CTF_int {

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
}
namespace CTF {

  template<typename dtype, bool is_ord>
  Vector<dtype, is_ord>::Vector(int                       len_,
                                World &                   world_,
                                CTF_int::algstrct const & sr_)
   : Tensor<dtype,is_ord>(1, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_, NULL, 0) {
    len = len_;
  }

  template<typename dtype, bool is_ord>
  Vector<dtype, is_ord>::Vector(int                       len_,
                                World &                   world_,
                                char const *              name_,
                                int                       profile_,
                                CTF_int::algstrct const & sr_)
   : Tensor<dtype,is_ord>(1, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_, name_, profile_) {
    len = len_;
  }

 
  template<typename dtype, bool is_ord>
  Vector<dtype,is_ord> & Vector<dtype,is_ord>::operator=(const Vector<dtype,is_ord> & A){
    CTF_int::tensor::free_self();
    CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile);
    return *this;
  }


}
