#include "set.h"
#include "../shared/blas_symbs.h"


namespace CTF_int {
  template <>
  void default_axpy<float>
                   (int           n,
                    float         alpha,
                    float const * X,
                    int           incX,
                    float *       Y,
                    int           incY){
    CTF_BLAS::SAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY){
    CTF_BLAS::DAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY){
    CTF_BLAS::CAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY){
    CTF_BLAS::ZAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX){
    CTF_BLAS::SSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX){
    CTF_BLAS::DSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
    CTF_BLAS::CSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
    CTF_BLAS::ZSCAL(&n,&alpha,X,&incX);
  }

#if USE_SP_MKL
  template <>
  void default_coomm< float >
          (int           m,
           int           n,
           int           k,
           float         alpha,
           float const * A,
           int const *   rows_A,
           int const *   cols_A,
           int           nnz_A,
           float const * B,
           float         beta,
           float *       C){
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_SCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (float*)A, rows_A, cols_A, &nnz_A,
               (float*)B, &k, &beta,
               (float*)C, &m);

  }

  template <>
  void default_coomm< double >
          (int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           int const *    rows_A,
           int const *    cols_A,
           int            nnz_A,
           double const * B,
           double         beta,
           double *       C){
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_DCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (double*)A, rows_A, cols_A, &nnz_A,
               (double*)B, &k, &beta,
               (double*)C, &m);

  }


  template <>
  void default_coomm< std::complex<float> >
          (int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           int const *                 rows_A,
           int const *                 cols_A,
           int                         nnz_A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C){
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_CCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<float>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<float>*)B, &k, &beta,
               (std::complex<float>*)C, &m);

  }

  template <>
  void default_coomm< std::complex<double> >
     (int                          m,
      int                          n,
      int                          k,
      std::complex<double>         alpha,
      std::complex<double> const * A,
      int const *                  rows_A,
      int const *                  cols_A,
      int                          nnz_A,
      std::complex<double> const * B,
      std::complex<double>         beta,
      std::complex<double> *       C){
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<double>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<double>*)B, &k, &beta,
               (std::complex<double>*)C, &m);

  }


 
#endif
}
