#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "../tensor/untyped_semiring.h"

namespace CTF {
  /**
   * \brief index-value pair used for tensor data input
   */
  template<typename dtype=double>
  class Pair  {
    public:
      /** \brief key, global index [i1,i2,...] specified as i1+len[0]*i2+... */
      int64_t k;

      /** \brief tensor value associated with index */
      dtype v;

      /**
       * \brief constructor builds pair
       * \param[in] k_ key
       * \param[in] v_ value
       */
      Pair(int64_t k_, dtype v_){
        this->k = k_; 
        v = v_;
      }
  };

  template<typename dtype>
  inline bool comp_pair(Pair<dtype> i,
                        Pair<dtype> j) {
    return (i.k<j.k);
  }

  template <typename dtype>
  dtype default_add(dtype a, dtype b){
    return a+b;
  }

  template <typename dtype>
  dtype default_addinv(dtype a){
    return -a;
  }

  template <typename dtype>
  dtype default_mul(dtype a, dtype b){
    return a*b;
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


  template <typename dtype=double> 
  class Semiring : public CTF_int::semiring {
    public:
      dtype addid;
      dtype mulid;
      MPI_Op addmop;
      dtype (*fadd)(dtype a, dtype b);
      dtype (*faddinv)(dtype a);
      dtype (*fmul)(dtype a, dtype b);
      void (*gemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);
      void (*axpy)(int,dtype,dtype const*,int,dtype*,int);
      void (*scal)(int,dtype,dtype*,int);
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
             dtype (*faddinv_)(dtype a)=&default_addinv<dtype>,
             dtype (*fmul_)(dtype a, dtype b)=&default_mul<dtype>,
             void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=&default_gemm<dtype>,
             void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=&default_axpy<dtype>,
             void (*scal_)(int,dtype,dtype*,int)=&default_scal<dtype>){
      addid = addid_;
      mulid = mulid_;
      addmop = addmop_;
      fadd = fadd_;
      faddinv = faddinv_;
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
      faddinv = &default_addinv<dtype>;
      fmul = &default_mul<dtype>;
      gemm = &default_gemm<dtype>;
      axpy = &default_axpy<dtype>;
      scal = &default_scal<dtype>;
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
      faddinv = &default_addinv<dtype>;
      fmul = &default_mul<dtype>;
      gemm = &default_gemm<dtype>;
      axpy = &default_axpy<dtype>;
      scal = &default_scal<dtype>;
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
