#include "common.h"

namespace CTF_int {

  struct int64_t1
  {
    int64_t i[1];
    int64_t1(int64_t a)
    {
      i[0] = a;
    }
    operator const int64_t*() const
    {
      return i;
    }
  };

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
  Vector<dtype>::Vector(int64_t                   len_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, CTF_int::int64_t1(len_), CTF_int::int1(NS), world_, sr_, NULL, 0) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int64_t                   len_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, CTF_int::int64_t1(len_), CTF_int::int1(NS), world_, sr_, name_, profile_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int64_t                   len_,
                        int                       atr_,
                        World &                   world_,
                        char const *              name_,
                        int                       profile_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, atr_>0, CTF_int::int64_t1(len_), CTF_int::int1(NS), world_, sr_, name_, profile_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int64_t                   len_,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_)
   : Tensor<dtype>(1, atr_>0, CTF_int::int64_t1(len_), CTF_int::int1(NS), world_, sr_) {
    len = len_;
  }

  template<typename dtype>
  Vector<dtype>::Vector(int64_t                   len_,
                        char                      idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        int                       atr_,
                        World &                   world_,
                        CTF_int::algstrct const & sr_,
                        char const *              name_,
                        int                       profile_) 
    : Tensor<dtype>(1, (atr_&4)>0, CTF_int::int64_t1(len_), CTF_int::int1(NS), 
                           world_, CTF_int::char1(idx), prl, blk, name_, profile_, sr_) {
    len = len_;
  }



  //template<typename dtype>
  //Vector<dtype> & Vector<dtype>::operator=(const Vector<dtype> & A){
  //  CTF_int::tensor::free_self();
  //  CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile, A.is_sparse);
  //  return *this;
  //}


  template <typename dtype>
  Vector<dtype> arange(dtype start,
                       int64_t n,
                       dtype step,
                       World & world,
                       CTF_int::algstrct const & sr){
    Vector<dtype> v(n, world, sr);
    int64_t * inds;
    dtype * vals;
    int64_t m;
    v.get_local_data(&m, &inds, &vals);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<m; i++){
      vals[i] = start + ((dtype)inds[i])*step;
    }
    v.write(m, inds, vals);
    return v;
  }

}
