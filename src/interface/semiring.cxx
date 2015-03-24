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
    SAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY){
    DAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY){
    CAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY){
    ZAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX){
    SSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX){
    DSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
    CSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
    ZSCAL(&n,&alpha,X,&incX);
  }
}
