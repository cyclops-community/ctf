#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "../ctr_seq/int_semiring.h"

// it seems to not be possible to initialize template argument function pointers
// to NULL, so defining this dummy_gemm function instead
template<typename dtype>
void default_gemm(dtype         tA,
                  dtype         tB,
                  int           m,
                  int           n,
                  int           k,
                  dtype         alpha,
                  dtype const * A,
                  dtype const * B,
                  dtype         beta,
                  dtype *       C){
  int i,j,l;
  int istride_A, lstride_A, jstride_B, lstride_B;
  if (tA == 'N' || tA == 'n'){
    istride_A=1; 
    lstride_A=m; 
  } else {
    istride_A=k; 
    lstride_A=1; 
  }
  if (tB == 'N' || tB == 'n'){
    jstride_B=k; 
    lstride_B=1; 
  } else {
    jstride_B=1; 
    lstride_B=m; 
  }
  for (j=0; j<n; j++){
    for (i=0; i<m; i++){
      C[j*m+i] *= beta;
      for (l=0; l<k; l++){
        C[j*m+i] += A[istride_A*i+lstride_A*l]*B[lstride_B*l+jstride_B*j];
      }
    }
  }
}

template<>
void default_gemm<float>
          (float          tA,
           float          tB,
           int            m,
           int            n,
           int            k,
           float          alpha,
           float  const * A,
           float  const * B,
           float          beta,
           float  *       C){
  sgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template<>
void default_gemm<double>
          (double         tA,
           double         tB,
           int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           double const * B,
           double         beta,
           double *       C){
  dgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

template<>
void default_gemm< std::complex<double> >
          (std::complex<double>         tA,
           std::complex<double>         tB,
           int                          m,
           int                          n,
           int                          k,
           std::complex<double>         alpha,
           std::complex<double> const * A,
           std::complex<double> const * B,
           std::complex<double>         beta,
           std::complex<double> *       C){
  zgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
}

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
             void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype> >){
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
      fmul = &default_fmul<dtype>;
      gemm = &default_gemm<dtype>;
      axpy = &default_axpy<dtype>;
      scal = &default_scal<dtype>;
    }

};


#endif
