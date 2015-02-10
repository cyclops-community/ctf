#ifndef __RING_H__
#define __RING_H__

namespace CTF {

  /**
   * Semiring class defined by a datatype and addition and multiplicaton functions
   *   addition must have an identity, inverse, and be associative, does not need to be commutative
   *   multiplications must have an identity and be distributive
   */
  template <typename dtype=double, bool is_ord=true>
  class Ring : public Semiring<dtype, is_ord> {
    public:
    /** 
     * \brief default constructor valid for only certain types:
     *         bool, int, unsigned int, int64_t, uint64_t,
     *         float, double, std::complex<float>, std::complex<double>
     */
    Ring() : Semiring<dtype, is_ord>() { }

    /**
     * \brief constructor for algstrct equipped with * and +
     * \param[in] addid_ additive identity
     * \param[in] fadd_ binary addition function
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] mulid_ multiplicative identity
     * \param[in] fmul_ binary multiplication function
     * \param[in] gemm_ block matrix multiplication function
     * \param[in] axpy_ vector sum function
     * \param[in] scal_ vector scale function
     */
    Ring(dtype        addid_,
         dtype (*fadd_)(dtype a, dtype b),
         MPI_Op       addmop_,
         dtype        mulid_,
         dtype (*fmul_)(dtype a, dtype b),
         void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=NULL,
         void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=NULL,
         void (*scal_)(int,dtype,dtype*,int)=NULL)
          : Semiring<dtype,is_ord>(addid_, fadd_, mulid_, addmop_, fmul_, gemm_, axpy_, scal_) {}


    /**
     * \brief constructor for algstrct equipped with * and +
     * \param[in] addid_ additive identity
     * \param[in] mulid_ multiplicative identity
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] fadd_ binary addition function
     * \param[in] fmul_ binary multiplication function
     * \param[in] gemm_ block matrix multiplication function
     * \param[in] axpy_ vector sum function
     * \param[in] scal_ vector scale function
     */
/*    Ring(dtype  addid_,
         dtype  mulid_,
         dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
         dtype (*faddinv_)(dtype a)=&default_addinv<dtype>,
         dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
         dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
         dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>,
         void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
         void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
         void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>) 
          : Semiring<dtype,is_ord>(addid_, mulid_, fadd_, fmul_, 
                                   fmin_, fmax_, gemm_, axpy_, scal_) , 
            Group<dtype,is_ord>(addid_, fadd_, faddinv_, fxpy_from_faxpy<dtype,axpy_,addid_>,fmin_,fmax_) { }*/


      void addinv(char const * a, char * b) const {
        ((dtype*)b)[0] = -((dtype*)a)[0];
      }


    /**
     * \brief constructor for algstrct equipped with + only
     * \param[in] addid_ additive identity
     */
    /*Semiring(dtype addid_) : Semiring<dtype>(addid_) {
      faddinv = &default_addinv<dtype>;
      is_ring = true;
    }*/
  };
  // The following requires C++11 unfortunately...
  template<>
  Ring<bool>::Ring() : Ring() {};
  template<>
  Ring<int>::Ring() : Ring() {};
  template<>
  Ring<unsigned int>::Ring() : Ring() {};
  template<>
  Ring<int64_t>::Ring() : Ring() {};
  template<>
  Ring<uint64_t>::Ring() : Ring() {};
  template<>
  Ring<float>::Ring() : Ring() {};
  template<>
  Ring<double>::Ring() : Ring() {};
  template<>
  Ring< std::complex<float> >::Ring() : Ring() {};
  template<>
  Ring< std::complex<double> >::Ring() : Ring() {};


}

#endif
