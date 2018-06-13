#ifndef __RING_H__
#define __RING_H__

#include "../tensor/algstrct.h"

namespace CTF {

  /**
   * \addtogroup algstrct 
   * @{
   */
  /**
   * \brief Ring class defined by a datatype and addition and multiplicaton functions
   *   addition must have an identity, inverse, and be associative, does not need to be commutative
   *   multiplications must have an identity and be distributive
   */
  template <typename dtype=double, bool is_ord=CTF_int::get_default_is_ord<dtype>()>
  class Ring : public Semiring<dtype, is_ord> {
    public:
      Ring(Ring const & other) : Semiring<dtype, is_ord>(other) { 
        this->abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
      }
      /** 
       * \brief default constructor valid for only certain types:
       *         bool, int, unsigned int, int64_t, uint64_t,
       *         float, double, std::complex<float>, std::complex<double>
       */
      Ring() : Semiring<dtype, is_ord>() { 
        this->abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
      }

      virtual CTF_int::algstrct * clone() const {
        return new Ring<dtype, is_ord>(*this);
      }

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
            : Semiring<dtype,is_ord>(addid_, fadd_, mulid_, addmop_, fmul_, gemm_, axpy_, scal_) {
          this->abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
        }

      //treat NULL as mulid
      void safeaddinv(char const * a, char *& b) const {
        if (b==NULL) b = (char*)malloc(this->el_size);
        if (a == NULL){
          
          ((dtype*)b)[0] = -this->tmulid;
        } else {
          ((dtype*)b)[0] = -((dtype*)a)[0];
        }
      }

        void addinv(char const * a, char * b) const {
          ((dtype*)b)[0] = -((dtype*)a)[0];
        }

  };
  /**
   * @}
   */

}

#endif
