#ifndef __BLAS_SYMBS__
#define __BLAS_SYMBS__

#include <complex>
#if FTN_UNDERSCORE
#define DDOT ddot_
#define SGEMM sgemm_
#define DGEMM dgemm_
#define CGEMM cgemm_
#define ZGEMM zgemm_
#define SGEMM_BATCH sgemm_batch_
#define DGEMM_BATCH dgemm_batch_
#define CGEMM_BATCH cgemm_batch_
#define ZGEMM_BATCH zgemm_batch_
#define SAXPY saxpy_
#define DAXPY daxpy_
#define CAXPY caxpy_
#define ZAXPY zaxpy_
#define SSYR   ssyr_
#define DSYR   dsyr_
#define CSYR   csyr_
#define ZSYR   zsyr_
#define SPOSV   sposv_
#define DPOSV   dposv_
#define CPOSV   cposv_
#define ZPOSV   zposv_
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
#define SGEMM_BATCH sgemm_batch
#define DGEMM_BATCH dgemm_batch
#define CGEMM_BATCH cgemm_batch
#define ZGEMM_BATCH zgemm_batch
#define SAXPY saxpy
#define DAXPY daxpy
#define CAXPY caxpy
#define ZAXPY zaxpy
#define SSYR   ssyr
#define DSYR   dsyr
#define CSYR   csyr
#define ZSYR   zsyr
#define SPOSV   sposv
#define DPOSV   dposv
#define CPOSV   cposv
#define ZPOSV   zposv
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


  template <typename dtype>
  void gemm(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            const dtype *,
            const int *,
            const dtype *,
            const int *,
            const dtype *,
            dtype *,
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
  void SSYR(const char *       UPLO ,
            const int *        N , 
            const float *     ALPHA, 
            const float *     X , 
            const int *        INCX , 
            float *           A , 
            const int *        LDA );


  extern "C"
  void DSYR(const char *       UPLO ,
            const int *        N , 
            const double *     ALPHA, 
            const double *     X , 
            const int *        INCX , 
            double *           A , 
            const int *        LDA );


  extern "C"
  void CSYR(const char *       UPLO ,
            const int *        N , 
            const std::complex<float> *     ALPHA, 
            const std::complex<float> *     X , 
            const int *        INCX , 
            std::complex<float> *           A , 
            const int *        LDA );


  extern "C"
  void ZSYR(const char *       UPLO ,
            const int *        N , 
            const std::complex<double> *     ALPHA, 
            const std::complex<double> *     X , 
            const int *        INCX , 
            std::complex<double> *           A , 
            const int *        LDA );

  template <typename dtype>
  void syr(const char *       UPLO ,
            const int *        N , 
            const dtype *     ALPHA, 
            const dtype *     X , 
            const int *        INCX , 
            dtype *           A , 
            const int *        LDA );



 
  extern "C"
  void SPOSV(char const *       UPLO ,
            const int *         N, 
            const int *         NRHS,
            float *            A, 
            const int *         LDA, 
            float *            B, 
            const int *         LDB, 
            int *               INFO);
 

  extern "C"
  void DPOSV(char const *       UPLO ,
            const int *         N, 
            const int *         NRHS,
            double *            A, 
            const int *         LDA, 
            double *            B, 
            const int *         LDB, 
            int *               INFO);
 
  extern "C"
  void CPOSV(char const *       UPLO ,
            const int *         N, 
            const int *         NRHS,
            std::complex<float> *            A, 
            const int *         LDA, 
            std::complex<float> *            B, 
            const int *         LDB, 
            int *               INFO);
 
  extern "C"
  void ZPOSV(char const *       UPLO ,
            const int *         N, 
            const int *         NRHS,
            std::complex<double> *            A, 
            const int *         LDA, 
            std::complex<double> *            B, 
            const int *         LDB, 
            int *               INFO);
 
  template <typename dtype>
  void posv(char const *        UPLO ,
            const int *         N, 
            const int *         NRHS,
            dtype *            A, 
            const int *         LDA, 
            dtype *            B, 
            const int *         LDB, 
            int *               INFO);
           
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


#ifdef USE_BATCH_GEMM
  extern "C"
  void SGEMM_BATCH(
            const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const float *,
            float **,
            const int *,
            float **,
            const int *,
            const float *,
            float **,
            const int *,
            const int *,
            const int *);

  extern "C"
  void DGEMM_BATCH(
            const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const double *,
            double **,
            const int *,
            double **,
            const int *,
            const double *,
            double **,
            const int *,
            const int *,
            const int *);

  extern "C"
  void CGEMM_BATCH(
            const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const std::complex<float> *,
            std::complex<float> **,
            const int *,
            std::complex<float> **,
            const int *,
            const std::complex<float> *,
            std::complex<float> **,
            const int *,
            const int *,
            const int *);

  extern "C"
  void ZGEMM_BATCH(
            const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const std::complex<double> *,
            std::complex<double> **,
            const int *,
            std::complex<double> **,
            const int *,
            const std::complex<double> *,
            std::complex<double> **,
            const int *,
            const int *,
            const int *);

  template <typename dtype>
  void gemm_batch(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            dtype **,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            const int *,
            const int *);
#endif //USE_BATCH_GEMM

}
#endif
