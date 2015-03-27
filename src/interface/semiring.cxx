#include "set.h"
#include "../shared/blas_symbs.h"


namespace CTF {
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
}
