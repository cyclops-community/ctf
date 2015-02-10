#ifndef __SEMIRING_H__
#define __SEMIRING_H__

namespace CTF {
  template <typename dtype>
  dtype default_mul(dtype a, dtype b){
    return a*b;
  }

  template <typename dtype>
  void default_axpy(int           n,
                    dtype         alpha,
                    dtype const * X,
                    int           incX,
                    dtype *       Y,
                    int           incY){
    for (int i=0; i<n; i++){
      Y[incY*i] += alpha*X[incX*i];
    }
  }

  template <>
  void default_axpy<float>
                   (int           n,
                    float         alpha,
                    float const * X,
                    int           incX,
                    float *       Y,
                    int           incY){
    cblas_saxpy(n,alpha,X,incX,Y,incY);
  }

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY){
    cblas_daxpy(n,alpha,X,incX,Y,incY);
  }

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY){
    cblas_caxpy(n,&alpha,X,incX,Y,incY);
  }

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY){
    cblas_zaxpy(n,&alpha,X,incX,Y,incY);
  }


  template <typename dtype, void (*faxpy)(int,dtype,dtype const*,int,dtype*,int), dtype mulid>
  void fxpy_from_faxpy(int           n,
                       dtype const * X,
                       dtype *       Y){ 
    faxpy(n, mulid, X, 1, Y, 1);
  }


  template <typename dtype>
  void default_scal(int           n,
                    dtype         alpha,
                    dtype *       X,
                    int           incX){
    for (int i=0; i<n; i++){
      X[incX*i] *= alpha;
    }
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX){
    cblas_sscal(n,alpha,X,incX);
  }

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX){
    cblas_dscal(n,alpha,X,incX);
  }

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
    cblas_cscal(n,&alpha,X,incX);
  }

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
    cblas_zscal(n,&alpha,X,incX);
  }

  template<typename dtype>
  void default_gemm(char          tA,
                    char          tB,
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
            (char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             float          alpha,
             float  const * A,
             float  const * B,
             float          beta,
             float  *       C){
    CTF_int::sgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  void default_gemm<double>
            (char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             double         alpha,
             double const * A,
             double const * B,
             double         beta,
             double *       C){
    CTF_int::dgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  void default_gemm< std::complex<float> >
            (char                        tA,
             char                        tB,
             int                         m,
             int                         n,
             int                         k,
             std::complex<float>         alpha,
             std::complex<float> const * A,
             std::complex<float> const * B,
             std::complex<float>         beta,
             std::complex<float> *       C){
    CTF_int::cgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  void default_gemm< std::complex<double> >
            (char                         tA,
             char                         tB,
             int                          m,
             int                          n,
             int                          k,
             std::complex<double>         alpha,
             std::complex<double> const * A,
             std::complex<double> const * B,
             std::complex<double>         beta,
             std::complex<double> *       C){
    CTF_int::zgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template <typename dtype, dtype (*func)(dtype const a, dtype const b)>
  void detypedfunc(char const * a,
                   char const * b,
                   char *       c){
    dtype ans = (*func)(((dtype const *)a)[0], ((dtype const *)b)[0]);
    ((dtype *)c)[0] = ans;
  }

  template <typename dtype, 
            void (*gemm)(char,char,int,int,int,dtype,
                         dtype const*,dtype const*,dtype,dtype*)>
  void detypedgemm(char         tA,
                   char         tB,
                   int          m,
                   int          n,
                   int          k,
                   char const * alpha,
                   char const * A,
                   char const * B,
                   char const * beta,
                   char *       C){
    (*gemm)(tA,tB,m,n,k,
            ((dtype const *)alpha)[0],
             (dtype const *)A,
             (dtype const *)B,
            ((dtype const *)beta)[0],
             (dtype       *)C);
  }


  /**
   * Semiring class defined by a datatype and addition and multiplicaton functions
   *   addition must have an identity and be associative, does not need to be commutative
   *   multiplications must have an identity as well as be distributive and associative
   *   define a Ring instead if an additive inverse is also available
   */
  template <typename dtype=double, bool is_ord=true> 
  class Semiring : public Monoid<dtype, is_ord> {
    public:
      dtype tmulid;
      void (*fscal)(int,dtype,dtype*,int);
      void (*faxpy)(int,dtype,dtype const*,int,dtype*,int);
      dtype (*fmul)(dtype a, dtype b);
      void (*gemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);

      /**
       * \brief constructor for algstrct equipped with * and +
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
      Semiring(dtype        addid_,
               dtype        mulid_,
               MPI_Datatype mdtype_,
               MPI_Op       addmop_,
               dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
               dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
               dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
               dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>,
               void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
               void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
               void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>) 
                : Monoid<dtype, is_ord>(addid_, fadd_, fxpy_from_faxpy<dtype,axpy_,mulid_>, addmop_, fmin_, fmax_) {
        tmulid = mulid_;
        fmul   = fmul_;
        gemm   = gemm_;
        faxpy  = axpy_;
        fscal  = scal_;
      }

  
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
      Semiring(dtype  addid_,
               dtype  mulid_,
               dtype (*fadd_)(dtype a, dtype b)=&default_add<dtype>,
               dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
               dtype (*fmin_)(dtype a, dtype b)=&default_min<dtype,is_ord>,
               dtype (*fmax_)(dtype a, dtype b)=&default_max<dtype,is_ord>,
               void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
               void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
               void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>) 
                : Monoid<dtype, is_ord>(addid_, fadd_, fxpy_from_faxpy<dtype,axpy_,mulid_>, fmin_, fmax_) {
        tmulid = mulid_;
        fmul   = fmul_;
        gemm   = gemm_;
        faxpy  = axpy_;
        fscal  = scal_;
      }
  
      /**
       * \brief constructor for algstrct equipped with + only
       * \param[in] addid_ additive identity
       */
      Semiring(dtype addid_) : Monoid<dtype,is_ord>() {
        fmul  = &default_mul<dtype>;
        gemm  = &default_gemm<dtype>;
        faxpy = &default_axpy<dtype>;
        fscal = &default_scal<dtype>;
      }
  
      /**
       * \brief constructor for algstrct equipped with + only ---- now Monoid
       * \param[in] addid_ additive identity
       * \param[in] addmop_ MPI_Op operation for addition
       * \param[in] fadd_ binary addition function
       */
      /*Semiring(dtype  addid_,
               MPI_Op addmop_,
               dtype (*fadd_)(dtype a, dtype b)){
        addid   = addid_;
        addmop  = addmop_;
        fadd    = fadd_;
        faddinv = &default_addinv<dtype>;
        fmul    = &default_mul<dtype>; //FIXME: what if I want just an Abelian group with no mul operator?
        gemm    = &default_gemm<dtype>;
        axpy    = &default_axpy<dtype>;
        scal    = &default_scal<dtype>;
      }*/
  };
}
#include "ring.h"
#endif
