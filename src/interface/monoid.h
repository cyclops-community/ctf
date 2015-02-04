#ifndef __MONOID_H__
#define __MONOID_H__

namespace CTF {
  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }

  template <typename dtype>
  void default_xpy(void * X,
                   void * Y,
                   int    n){
    for (int i=0; i<n; i++){
      ((dtype*)Y)[i] = ((dtype const *)X)[i] + ((dtype*)Y)[i];
    }
  }

  template <typename dtype, void (*fadd)(dtype, dtype)>
  void get_fxpy(int           n
                dtype const * X,
                dtype *       Y){
    for (int i=0; i<n; i++){
      Y[i] = fadd(X[i],Y[i]);
    }
  }

  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  void default_mopfun(void * X,
                      void * Y,
                      int    n){
    fxpy(n, (dtype const *)X, (dtype *)Y);
  }

  template <typename dtype>
  MPI_Datatype get_default_mdtype(){
    switch (dtype) {
      case char:
        return MPI_CHAR;
        break;
      case bool:
        return MPI_BOOL;
        break;
      case int:
        eturn MPI_INT;
        break;
      case int64_t:
        return MPI_INT64_T;
        break;
      case unsigned int:
        return MPI_UNSIGHED;
        break;
      case uint64_t:
        return MPI_UINT64_T;
        break;
      case float:
        return MPI_FLOAT;
        break;
      case double:
        return MPI_DOUBLE;
        break;
      case long double:
        return MPI_LONG_DOUBLE;
        break;
      case std::complex<float>:
        return MPI_COMPLEX;
        break;
      case std::complex<double>:
        return MPI_DOUBLE_COMPLEX;
        break;
      default:
        MPI_Datatype newtype;
        MPI_Type_contiguous(sizeof(dtype), MPI_CHAR, &newtype);
        return newtype;
        break;
    }
  }

  template <typename dtype>
  MPI_Datatype get_default_addop(){
    switch (dtype) {
      case char:
      case bool:
      case int:
      case int64_t:
      case unsigned int:
      case uint64_t:
      case float:
      case double:
      case long double:
      case std::complex<float>:
      case std::complex<double>:
        return MPI_SUM;
        break;
      default:
        //FIXME: assumes + operator commutes
        MPI_Op newop;
        MPI_Op_create(&default_xpy<dtype>, 1, &newop);
        return newop;
        break;
    }
  }

  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  MPI_Datatype get_maddop(){
      //FIXME: assumes + operator commutes
      MPI_Op newop;
      MPI_Op_create(&default_fxpy<dtype, fxpy>, 1, &newop);
      return newop;
    }
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
              : Set(fmin_, fmax_) {
        taddid  = (dtype)0;
        fadd    = &default_add<dtype>;
        fxpy    = &default_xpy<dtype>;
        taddmop = get_default_mop<dtype>();
        tmdtype = get_default_mdtype<dtype>();
      } 
 
      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
             dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Set(fmin_, fmax_) {
        taddid  = taddid_;
        fadd    = fadd_;
        fxpy    = get_fxpy<dtype,fadd>;
        taddmop = get_maddop<dtype,fxpy>();
        tmdtype = get_default_mdtype();
      }
 
      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             void (*fxpy_)(int, dtype const *, dtype *),
             MPI_Op addmop_,
             dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
             dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>)
              : Set(fmin_, fmax_) {
        taddid  = taddid_;
        fadd    = fadd_;
        fxpy    = fxpy_;
        taddmop = taddmop_;
        tmdtype = get_default_mdtype();
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

  }
}

#include "group.h"
#endif

