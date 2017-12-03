#ifndef __BLAS_SYMBS__
#define __BLAS_SYMBS__

#include <complex>
#if FTN_UNDERSCORE
#define DDOT ddot_
#define SGEMM sgemm_
#define DGEMM dgemm_
#define CGEMM cgemm_
#define ZGEMM zgemm_
#define SAXPY saxpy_
#define DAXPY daxpy_
#define CAXPY caxpy_
#define ZAXPY zaxpy_
#define SSCAL sscal_
#define DSCAL dscal_
#define CSCAL cscal_
#define ZSCAL zscal_
#define SCOPY scopy_
#define DCOPY dcopy_
#define ZCOPY zcopy_
#else
#define DDOT ddot
#define SGEMM sgemm
#define DGEMM dgemm
#define CGEMM cgemm
#define ZGEMM zgemm
#define SAXPY saxpy
#define DAXPY daxpy
#define CAXPY caxpy
#define ZAXPY zaxpy
#define SSCAL sscal
#define DSCAL dscal
#define CSCAL cscal
#define ZSCAL zscal
#define SCOPY scopy
#define DCOPY dcopy
#define ZCOPY zcopy
#endif


namespace CTF_BLAS {
  extern "C"
  double DDOT(int * n,         const double * dX,      
              int * incX,      const double * dY,      
              int * incY);


  extern "C"
  void SGEMM(const char *,
             const char *,
             const int *,
             const int *,
             const int *,
             const float *,
             const float *,
             const int *,
             const float *,
             const int *,
             const float *,
             float *,
             const int *);


  extern "C"
  void DGEMM(const char *,
             const char *,
             const int *,
             const int *,
             const int *,
             const double *,
             const double *,
             const int *,
             const double *,
             const int *,
             const double *,
             double *,
             const int *);


  extern "C"
  void CGEMM(const char *,
             const char *,
             const int *,
             const int *,
             const int *,
             const std::complex<float> *,
             const std::complex<float> *,
             const int *,
             const std::complex<float> *,
             const int *,
             const std::complex<float> *,
             std::complex<float> *,
             const int *);

  extern "C"
  void ZGEMM(const char *,
             const char *,
             const int *,
             const int *,
             const int *,
             const std::complex<double> *,
             const std::complex<double> *,
             const int *,
             const std::complex<double> *,
             const int *,
             const std::complex<double> *,
             std::complex<double> *,
             const int *);


  extern "C"
  void SAXPY(const int *   n,
             float *       dA,
             const float * dX,
             const int *   incX,
             float *       dY,
             const int *   incY);


  extern "C"
  void DAXPY(const int *    n,
             double *       dA,
             const double * dX,
             const int *    incX,
             double *       dY,
             const int *    incY);

  extern "C"
  void CAXPY(const int *                 n,
             std::complex<float> *       dA,
             const std::complex<float> * dX,
             const int *                 incX,
             std::complex<float> *       dY,
             const int *                 incY);

  extern "C"
  void ZAXPY(const int *                  n,
             std::complex<double> *       dA,
             const std::complex<double> * dX,
             const int *                  incX,
             std::complex<double> *       dY,
             const int *                  incY);

  extern "C"
  void SCOPY(const int *   n,
             const float * dX,
             const int *   incX,
             float *       dY,
             const int *   incY);


  extern "C"
  void DCOPY(const int *    n,
             const double * dX,
             const int *    incX,
             double *       dY,
             const int *    incY);


  extern "C"
  void ZCOPY(const int *                  n,
             const std::complex<double> * dX,
             const int *                  incX,
             std::complex<double> *       dY,
             const int *                  incY);

  extern "C"
  void SSCAL(const int * n,
             float *     dA,
             float *     dX,
             const int * incX);

  extern "C"
  void DSCAL(const int * n,
             double *    dA,
             double *    dX,
             const int * incX);

  extern "C"
  void CSCAL(const int *           n,
             std::complex<float> * dA,
             std::complex<float> * dX,
             const int *           incX);

  extern "C"
  void ZSCAL(const int *            n,
             std::complex<double> * dA,
             std::complex<double> * dX,
             const int *            incX);

}
#endif
