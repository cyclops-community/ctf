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

  template <typename dtype>
  void default_fxpy(int           n,
                    dtype const * X,
                    dtype *       Y){
    for (int i=0; i<n; i++){
      Y[i] = X[i] + Y[i];
    }
  }
  template <typename dtype>
  MPI_Datatype get_default_mdtype(){
    MPI_Datatype newtype;
    MPI_Type_contiguous(sizeof(dtype), MPI_CHAR, &newtype);
    return newtype;
  }
  template <>
  inline MPI_Datatype get_default_mdtype<char>(){ return MPI_CHAR; }
  template <>
  inline MPI_Datatype get_default_mdtype<bool>(){ return MPI_C_BOOL; }
  template <>
  inline MPI_Datatype get_default_mdtype<int>(){ return MPI_INT; }
  template <>
  inline MPI_Datatype get_default_mdtype<int64_t>(){ return MPI_INT64_T; }
  template <>
  inline MPI_Datatype get_default_mdtype<unsigned int>(){ return MPI_UNSIGNED; }
  template <>
  inline MPI_Datatype get_default_mdtype<uint64_t>(){ return MPI_UINT64_T; }
  template <>
  inline MPI_Datatype get_default_mdtype<float>(){ return MPI_FLOAT; }
  template <>
  inline MPI_Datatype get_default_mdtype<double>(){ return MPI_DOUBLE; }
  template <>
  inline MPI_Datatype get_default_mdtype<long double>(){ return MPI_LONG_DOUBLE; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<float> >(){ return MPI_COMPLEX; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<double> >(){ return MPI_DOUBLE_COMPLEX; }

  template <typename dtype>
  MPI_Op get_default_maddop(){
    //FIXME: assumes + operator commutes
    MPI_Op newop;
    MPI_Op_create(&default_mxpy<dtype,default_fxpy<dtype>>, 1, &newop);
    return newop;
  }

  //c++ sucks...
  template <> inline MPI_Op get_default_maddop<char>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<bool>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<int>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<int64_t>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<unsigned int>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<uint64_t>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<float>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<double>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop<long double>(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop< std::complex<float> >(){ return MPI_SUM; }
  template <> inline MPI_Op get_default_maddop< std::complex<double> >(){ return MPI_SUM; }
  
  template <typename dtype>
  MPI_Op get_maddop(void (*fxpy)(int, dtype const *, dtype *)){
    //FIXME: assumes + operator commutes
    MPI_Op newop;
    MPI_Op_create(&default_mxpy<dtype, fxpy>, 1, &newop);
    return newop;
  }

  /**
   * \brief A Monoid is a Set equipped with a binary addition operator '+' or a custom function
   *   addition must have an identity and be associative, does not need to be commutative
   *   special case (parent) of a semiring, group, and ring
   */
  template <typename dtype=double, bool is_ord=true> 
  class Monoid : public Set<dtype, is_ord> {
    public:
      dtype taddid;
      dtype (*fadd)(dtype a, dtype b);
      MPI_Datatype tmdtype;
      MPI_Op       taddmop;

      Monoid(Monoid const & other) : Set<dtype, is_ord>(other) {
        this->taddid  = other.taddid;
        this->fadd    = other.fadd;
        this->tmdtype = other.tmdtype;
        this->taddmop = other.taddmop;
      }
      
      virtual CTF_int::algstrct * clone() const {
        return new Monoid<dtype, is_ord>(*this);
      }
      Monoid() : Set<dtype, is_ord>() {
        taddid  = (dtype)0;
        fadd    = &default_add<dtype>;
        taddmop = get_default_maddop<dtype>();
        tmdtype = get_default_mdtype<dtype>();
      } 

      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             MPI_Op addmop_)
              : Set<dtype, is_ord>() {
        taddid  = taddid_;
        fadd    = fadd_;
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
        ASSERT(alpha == NULL);
        for (int64_t i=0; i<n; i++){
          add(X+sizeof(dtype)*i*incX,Y+sizeof(dtype)*i*incY,Y+sizeof(dtype)*i*incY);
        }
      }

  };
}

#include "group.h"
#endif

