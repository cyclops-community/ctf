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

  struct char1
  {
    char i[1];
    char1(char a)
    {
      i[0] = a;
    }
    operator const char*() const
    {
      return i;
    }
  };

}
namespace CTF {
  template<typename dtype>
  Vector<dtype>::Vector() : Tensor<dtype>() { }

  template<typename dtype>
  Vector<dtype>::Vector(Vector<dtype> const & A)
    : Tensor<dtype>(A) {
    len = A.len;
  }

  template<typename dtype>
  Vector<dtype>::Vector(Tensor<dtype> const & A)
    : Tensor<dtype>(A) {
    IASSERT(A.order == 1);
    len = A.lens[0];
  }

  template<typename dtype>
  Vector<dtype>::Vector(int                       len_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_, NULL, 0) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int                       len_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_, name_, profile_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int                       len_,
                        int                       atr_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, atr_>0, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_, name_, profile_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int                       len_,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, atr_>0, CTF_int::int1(len_), CTF_int::int1(NS), world_, sr_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int                       len_,
                        char                      idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        int                       atr,
                        World &                   wrld,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        int                       profile) 
    : Tensor<dtype>(1, (atr_&4)>0, CTF_int::int1(len_), CTF_int::int1(NS), 
                           world_, CTF_int::char1(idx), prl, blk, name_, profile_, sr_) {
    len = len_;
  }



  //template<typename dtype>
  //Vector<dtype> & Vector<dtype>::operator=(const Vector<dtype> & A){
  //  CTF_int::tensor::free_self();
  //  CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile, A.is_sparse);
  //  return *this;
  //}



}
