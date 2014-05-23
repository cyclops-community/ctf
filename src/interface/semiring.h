#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "../ctr_seq/int_semiring.h"

template <typename dtype>
dtype default_add(dtype & a, dtype & b){
  return a+b;
}

template <typename dtype>
dtype default_mul(dtype & a, dtype & b){
  return a*b;
}

template <typename dtype=double> 
class Semiring : public Int_Semiring {
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
     * \brief default constructor, instantiates to (*,+) and {s,d,z}gemm if possible
     */
    Semiring();
    
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
             void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
             void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
             void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>){
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
     * \param[in] addmop_ MPI_Op operation for addition
     * \param[in] fadd_ binary addition function
     */
    Semiring(dtype  addid_,
             dtype  addmop_,
             dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>){
      addid = addid_;
      addmop = addmop_;
      fadd = fadd_;
      fmul = &default_mul<dtype>;
      gemm = &default_gemm<dtype>;
      axpy = &default_axpy<dtype>;
      scal = &default_scal<dtype>;
    }

};


#endif
