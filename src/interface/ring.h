#ifndef __RING_H__
#define __RING_H__

namespace CTF {

  template <typename dtype>
  dtype default_addinv(dtype a){
    return -a;
  }

  /**
   * Semiring class defined by a datatype and addition and multiplicaton functions
   *   addition must have an identity, inverse, and be associative, does not need to be commutative
   *   multiplications must have an identity and be distributive
   */
  template <typename dtype=double>
  class Ring : public Semiring<dtype> {
    dtype (*faddinv)(dtype a);

    /** 
     * \brief default constructor valid for only certain types:
     *         bool, int, unsigned int, int64_t, uint64_t,
     *         float, double, std::complex<float>, std::complex<double>
     */
    Ring(){ 
      printf("CTF ERROR: at least the additive identity must be specified for rings of custom tensor types, use of default constructor not allowed, aborting.\n");
      assert(0);
    }

    /**
     * \brief constructor for semiring equipped with * and +
     * \param[in] addid_ additive identity
     * \param[in] mulid_ multiplicative identity
     * \param[in] mdtype MPI Datatype to use in reductions
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] fadd_ binary addition function
     * \param[in] fmul_ binary multiplication function
     * \param[in] gemm_ block matrix multiplication function
     * \param[in] axpy_ vector sum function
     * \param[in] scal_ vector scale function
     */
    Ring(dtype        addid_,
         dtype        mulid_,
         MPI_Datatype mdtype_,
         MPI_Op       addmop_,
         dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
         dtype (*faddinv_)(dtype a)=&default_addinv<dtype>,
         dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
         dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype>,
         dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype>,
         void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
         void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
         void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>) 
          : Semiring<dtype>(addid_, mulid_, mdtype_, addmop_, fadd_, fmul_, 
                            fmin_, fmax_, gemm_, axpy_, scal_){
      faddinv = faddinv_;
      is_ring = true;
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
    Ring(dtype  addid_,
         dtype  mulid_,
         dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
         dtype (*faddinv_)(dtype a)=&default_addinv<dtype>,
         dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
         dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype>,
         dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype>,
         void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
         void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
         void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>) 
          : Semiring<dtype>(addid_, mulid_, fadd_, fmul_, 
                            fmin_, fmax_, gemm_, axpy_, scal_) {
      faddinv = faddinv_;
      is_ring = true;
    }


    /**
     * \brief constructor for semiring equipped with + only
     * \param[in] addid_ additive identity
     */
    Semiring(dtype addid_) : Semiring<dtype>(addid_) {
      faddinv = &default_addinv<dtype>;
      is_ring = true;
    }
  }
  // The following requires C++11 unfortunately...
  template<>
  Ring<bool>::Ring() : Ring(false, true) {};
  template<>
  Ring<int>::Ring() : Ring(0, 1) {};
  template<>
  Ring<unsigned int>::Ring() : Ring(0, 1) {};
  template<>
  Ring<int64_t>::Ring() : Ring(0, 1) {};
  template<>
  Ring<uint64_t>::Ring() : Ring(0, 1) {};
  template<>
  Ring<float>::Ring() : Ring(0.0, 1.0) {};
  template<>
  Ring<double>::Ring() : Ring(0.0, 1.0) {};
  template<>
  Ring< std::complex<float> >::Ring() 
    : Ring(std::complex<float>(0.0,0.0), 
           std::complex<float>(1.0,0.0)) {};
  template<>
  Ring< std::complex<double> >::Ring() 
    : Ring(std::complex<double>(0.0,0.0), 
           std::complex<double>(1.0,0.0)) {};


}

#endif
