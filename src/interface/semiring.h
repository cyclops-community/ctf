#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "../tensor/untyped_semiring.h"

namespace CTF {

  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }

  template <typename dtype>
  dtype default_mul(dtype a, dtype b){
    return a*b;
  }

  template <typename dtype=double> 
  class Semiring : public CTF_int::semiring {
    public:
      dtype addid;
      dtype mulid;
      MPI_Op addmop;
      dtype (*fadd)(dtype a, dtype b);
      dtype (*fmul)(dtype a, dtype b);
      void (*gemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);
      void (*axpy)(int,dtype,dtype const*,int,dtype*,int);
      void (*scal)(int,dtype,dtype*,int);
    public:
    /** 
     * \brief default constructor valid for only certain types:
     *         bool, int, unsigned int, int64_t, uint64_t,
     *         float, double, std::complex<float>, std::complex<double>
     */
    Semiring(){ 
      printf("CTF ERROR: identity must be specified for custom tensor types, use of default constructor not allowed, aborting.\n");
      assert(0);
    }
    /**
     * \brief constructor for semiring equipped with * and +
     * \param[in] addid_ additive identity
     * \param[in] mulid_ multiplicative identity
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] fadd_ binary addition function
     * \param[in] fmul_ binary multiplication function
     * \param[in] gemm_ block matrix multiplication function
     * \param[in] axpy_ vector sum function
     * \param[in] scal_ vector scale function
     */
    Semiring(dtype  addid_,
             dtype  mulid_,
             MPI_Op addmop_=MPI_SUM,
             dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
             dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
             void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&CTF_int::default_gemm<dtype>,
             void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&CTF_int::default_axpy<dtype>,
             void (*scal_)(int,dtype,dtype*,int)=&CTF_int::default_scal<dtype>){
      addid = addid_;
      mulid = mulid_;
      addmop = addmop_;
      fadd = fadd_;
      fmul = fmul_;
      gemm = gemm_;
      axpy = axpy_;
      scal = scal_;
    }

    /**
     * \brief constructor for semiring equipped with + only
     * \param[in] addid_ additive identity
     */
    Semiring(dtype  addid_) {
      addid = addid_;
      addmop = MPI_SUM;
      fadd = &default_add<dtype>;
      fmul = &default_mul<dtype>;
      gemm = &CTF_int::default_gemm<dtype>;
      axpy = &CTF_int::default_axpy<dtype>;
      scal = &CTF_int::default_scal<dtype>;
    }

    /**
     * \brief constructor for semiring equipped with + only
     * \param[in] addid_ additive identity
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] fadd_ binary addition function
     */
    Semiring(dtype  addid_,
             MPI_Op addmop_,
             dtype (*fadd_)(dtype a, dtype b)){
      addid = addid_;
      addmop = addmop_;
      fadd = fadd_;
      fmul = &default_mul<dtype>;
      gemm = &CTF_int::default_gemm<dtype>;
      axpy = &CTF_int::default_axpy<dtype>;
      scal = &CTF_int::default_scal<dtype>;
    }

  };

  // The following requires C++11 unfortunately...
  template<>
  Semiring<bool>::Semiring() : Semiring(false, true) {};
  template<>
  Semiring<int>::Semiring() : Semiring(0, 1) {};
  template<>
  Semiring<unsigned int>::Semiring() : Semiring(0, 1) {};
  template<>
  Semiring<int64_t>::Semiring() : Semiring(0, 1) {};
  template<>
  Semiring<uint64_t>::Semiring() : Semiring(0, 1) {};
  template<>
  Semiring<float>::Semiring() : Semiring(0.0, 1.0) {};
  template<>
  Semiring<double>::Semiring() : Semiring(0.0, 1.0) {};
  template<>
  Semiring< std::complex<float> >::Semiring() 
    : Semiring(std::complex<float>(0.0,0.0), 
               std::complex<float>(1.0,0.0)) {};
  template<>
  Semiring< std::complex<double> >::Semiring() 
    : Semiring(std::complex<double>(0.0,0.0), 
               std::complex<double>(1.0,0.0)) {};

}


#endif
