#ifndef __MONOID_H__
#define __MONOID_H__

namespace CTF_int {
  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }
  
  template <typename dtype, void (*fxpy)(int, dtype const *, dtype *)>
  void default_mxpy(void *         X,
                    void *         Y,
                    int *          n,
                    MPI_Datatype * d){
    fxpy(*n, (dtype const*)X, (dtype *)Y);
  }

  template <typename dtype>
  void default_fxpy(int            n,
                    dtype const *  X,
                    dtype *        Y){
    for (int i=0; i<n; i++){
      Y[i] = X[i] + Y[i];
    }
  }


  template <typename dtype>
  MPI_Op get_default_maddop(){
    //FIXME: assumes + operator commutes
    MPI_Op newop;
//    default_mxpy<dtype,default_fxpy<dtype>>(NULL, NULL, 0);
    MPI_Op_create(&default_mxpy< dtype, default_fxpy<dtype> >, 1, &newop);
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
}

namespace CTF {
  /**
   * \addtogroup algstrct 
   * @{
   **/
  /**
   * \brief A Monoid is a Set equipped with a binary addition operator '+' or a custom function
   *   addition must have an identity and be associative, does not need to be commutative
   *   special case (parent) of a semiring, group, and ring
   */
  template <typename dtype=double, bool is_ord=CTF_int::get_default_is_ord<dtype>()> 
  class Monoid : public Set<dtype, is_ord> {
    public:
      dtype taddid;
      dtype (*fadd)(dtype a, dtype b);
      MPI_Op       taddmop;

      Monoid(Monoid const & other) : Set<dtype, is_ord>(other), taddid(other.taddid), fadd(other.fadd), taddmop(other.taddmop) {
      }
      
      virtual CTF_int::algstrct * clone() const {
        return new Monoid<dtype, is_ord>(*this);
      }
      Monoid() : Set<dtype, is_ord>(), taddid(0) {
        fadd    = &CTF_int::default_add<dtype>;
        taddmop = CTF_int::get_default_maddop<dtype>();
      } 

      Monoid(dtype taddid_) : Set<dtype, is_ord>(), taddid(taddid_) {
        fadd    = &CTF_int::default_add<dtype>;
        taddmop = CTF_int::get_default_maddop<dtype>();
      } 



      Monoid(dtype taddid_,
             dtype (*fadd_)(dtype a, dtype b),
             MPI_Op addmop_)
              : Set<dtype, is_ord>(), taddid(taddid_) {
        fadd    = fadd_;
        taddmop = addmop_;
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

      void init(int64_t n, char * arr) const {
        this->set(arr, (char const *)&taddid, n);
      }

      void axpy(int          n,
                char const * alpha,
                char const * X,
                int          incX,
                char       * Y,
                int          incY) const {
        //assert(alpha == NULL);
        for (int64_t i=0; i<n; i++){
          add(X+sizeof(dtype)*i*incX,Y+sizeof(dtype)*i*incY,Y+sizeof(dtype)*i*incY);
        }
      }

      /** \brief adds CSR matrices A (stored in cA) and B (stored in cB) to create matric C (pointer to all_data returned), C data allocated internally */
      char * csr_add(char * cA, char * cB, bool is_ccsr) const {
        return CTF_int::algstrct::csr_add(cA, cB, is_ccsr);
      }


  };
  template <>
  char * Monoid<double,1>::csr_add(char *, char *, bool) const;
  
  /**
   * @}
   */
}

#include "group.h"
#endif

