#ifndef __MONOID_H__
#define __MONOID_H__

namespace CTF {
  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }
  
  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  void default_mxpy(void * X,
                    void * Y,
                    int    n){
    fxpy(n, (dtype const*)X, (dtype *)Y);
  }


  template <typename dtype, void (*fadd)(dtype, dtype)>
  void default_afxpy(int           n,
                     dtype const * X,
                     dtype *       Y){
    for (int i=0; i<n; i++){
      Y[i] = fadd(X[i],Y[i]);
    }
  }

  template <typename dtype>
  void default_fxpy(int           n,
                    dtype const * X,
                    dtype *       Y){
    for (int i=0; i<n; i++){
      Y[i] = X[i] + Y[i];
    }
  }

/*  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  void default_mopfun(void * X,
                      void * Y,
                      int    n){
    fxpy(n, (dtype const *)X, (dtype *)Y);
  }*/

  template <typename dtype>
  MPI_Datatype get_default_mdtype(){
    MPI_Datatype newtype;
    MPI_Type_contiguous(sizeof(dtype), MPI_CHAR, &newtype);
    return newtype;
  }
  template <>
  MPI_Datatype get_default_mdtype<char>(){ return MPI_CHAR; }
  template <>
  MPI_Datatype get_default_mdtype<bool>(){ return MPI_C_BOOL; }
  template <>
  MPI_Datatype get_default_mdtype<int>(){ return MPI_INT; }
  template <>
  MPI_Datatype get_default_mdtype<int64_t>(){ return MPI_INT64_T; }
  template <>
  MPI_Datatype get_default_mdtype<unsigned int>(){ return MPI_UNSIGNED; }
  template <>
  MPI_Datatype get_default_mdtype<uint64_t>(){ return MPI_UINT64_T; }
  template <>
  MPI_Datatype get_default_mdtype<float>(){ return MPI_FLOAT; }
  template <>
  MPI_Datatype get_default_mdtype<double>(){ return MPI_DOUBLE; }
  template <>
  MPI_Datatype get_default_mdtype<long double>(){ return MPI_LONG_DOUBLE; }
  template <>
  MPI_Datatype get_default_mdtype< std::complex<float> >(){ return MPI_COMPLEX; }
  template <>
  MPI_Datatype get_default_mdtype< std::complex<double> >(){ return MPI_DOUBLE_COMPLEX; }

  template <typename dtype>
  MPI_Op get_default_maddop(){
    //FIXME: assumes + operator commutes
    MPI_Op newop;
    MPI_Op_create(&default_afxpy<dtype,default_add<dtype>>, 1, &newop);
    return newop;
  }

  //c++ sucks...
  template <> MPI_Op get_default_maddop<char>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<bool>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<int>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<int64_t>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<unsigned int>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<uint64_t>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<float>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<double>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop<long double>(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop< std::complex<float> >(){ return MPI_SUM; }
  template <> MPI_Op get_default_maddop< std::complex<double> >(){ return MPI_SUM; }
  
  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  MPI_Op get_maddop(){
    //FIXME: assumes + operator commutes
    MPI_Op newop;
    MPI_Op_create(&default_mxpy<dtype, fxpy>, 1, &newop);
    return newop;
  }

  /**
   * Monoid class defined by a datatype (Set) and an addition function
   *   addition must have an identity and be associative, does not need to be commutative
   *   define a Group if there is an additive inverse and Semiring/Ring instead if there is 
   *   a multiplication
   */
  template <typename dtype=double, bool is_ord=true> 
  class Monoid : public Set<dtype, is_ord> {
    public:
      dtype taddid;
      dtype (*fadd)(dtype a, dtype b);
      void (*fxpy)(int, dtype const *, dtype *);
      MPI_Datatype tmdtype;
      MPI_Op       taddmop;
      
      Monoid(dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
             dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Set<dtype, is_ord>(fmin_, fmax_) {
        taddid  = (dtype)0;
        fadd    = &default_add<dtype>;
        fxpy    = &default_fxpy<dtype>;
        taddmop = get_default_maddop<dtype>();
        tmdtype = get_default_mdtype<dtype>();
      } 
 
      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
             dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Set<dtype, is_ord>(fmin_, fmax_) {
        taddid  = taddid_;
        fadd    = fadd_;
        fxpy    = &default_afxpy<dtype,fadd_>;
        taddmop = get_maddop<dtype,&default_afxpy<dtype,fadd_>>();
        tmdtype = get_default_mdtype<dtype>();
      }
 
      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             void (*fxpy_)(int, dtype const *, dtype *),
             MPI_Op addmop_,
             dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
             dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Set<dtype, is_ord>(fmin_, fmax_) {
        taddid  = taddid_;
        fadd    = fadd_;
        fxpy    = fxpy_;
        taddmop = addmop_;
        tmdtype = get_default_mdtype<dtype>();
      }

      void add(char const * a, 
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = fadd(((dtype*)a)[0],((dtype*)b)[0]);
      }
 
      char const * addid() const {
        return (char const *)&taddid;
      }

      MPI_Op addmop() const {
        return taddmop;        
      }
      
      MPI_Datatype mdtype() const {
        return tmdtype;        
      }

      void axpy(int          n,
                char const * alpha,
                char const * X,
                int          incX,
                char       * Y,
                int          incY) const {
        // FIXME: need arbitrary incX and incY? some assert on alpha?
        ASSERT(incX == 1);
        ASSERT(incY == 1);
        fxpy(n, X, Y);
      }

  };
}

#include "group.h"
#endif

