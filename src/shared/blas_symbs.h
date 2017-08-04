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
#define MKL_SCOOMM mkl_scoomm_
#define MKL_DCOOMM mkl_dcoomm_
#define MKL_CCOOMM mkl_ccoomm_
#define MKL_ZCOOMM mkl_zcoomm_
#define MKL_SCSRCOO mkl_scsrcoo_
#define MKL_DCSRCOO mkl_dcsrcoo_
#define MKL_CCSRCOO mkl_ccsrcoo_
#define MKL_ZCSRCOO mkl_zcsrcoo_
#define MKL_SCSRMM mkl_scsrmm_
#define MKL_DCSRMM mkl_dcsrmm_
#define MKL_CCSRMM mkl_ccsrmm_
#define MKL_ZCSRMM mkl_zcsrmm_
#define MKL_SCSRADD mkl_scsradd_
#define MKL_DCSRADD mkl_dcsradd_
#define MKL_CCSRADD mkl_ccsradd_
#define MKL_ZCSRADD mkl_zcsradd_
#define MKL_SCSRMULTD mkl_scsrmultd_
#define MKL_DCSRMULTD mkl_dcsrmultd_
#define MKL_CCSRMULTD mkl_ccsrmultd_
#define MKL_ZCSRMULTD mkl_zcsrmultd_
#define MKL_SCSRMULTCSR mkl_scsrmultcsr_
#define MKL_DCSRMULTCSR mkl_dcsrmultcsr_
#define MKL_CCSRMULTCSR mkl_ccsrmultcsr_
#define MKL_ZCSRMULTCSR mkl_zcsrmultcsr_
#define BLACS_GRIDINFO blacs_gridinfo_
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
#define MKL_SCOOMM mkl_scoomm
#define MKL_DCOOMM mkl_dcoomm
#define MKL_CCOOMM mkl_ccoomm
#define MKL_ZCOOMM mkl_zcoomm
#define MKL_SCSRCOO mkl_scsrcoo
#define MKL_DCSRCOO mkl_dcsrcoo
#define MKL_CCSRCOO mkl_ccsrcoo
#define MKL_ZCSRCOO mkl_zcsrcoo
#define MKL_SCSRMM mkl_scsrmm
#define MKL_DCSRMM mkl_dcsrmm
#define MKL_CCSRMM mkl_ccsrmm
#define MKL_ZCSRMM mkl_zcsrmm
#define MKL_SCSRADD mkl_scsradd
#define MKL_DCSRADD mkl_dcsradd
#define MKL_CCSRADD mkl_ccsradd
#define MKL_ZCSRADD mkl_zcsradd
#define MKL_SCSRMULTD mkl_scsrmultd
#define MKL_DCSRMULTD mkl_dcsrmultd
#define MKL_CCSRMULTD mkl_ccsrmultd
#define MKL_ZCSRMULTD mkl_zcsrmultd
#define MKL_SCSRMULTCSR mkl_scsrmultcsr
#define MKL_DCSRMULTCSR mkl_dcsrmultcsr
#define MKL_CCSRMULTCSR mkl_ccsrmultcsr
#define MKL_ZCSRMULTCSR mkl_zcsrmultcsr
#define BLACS_GRIDINFO blacs_gridinfo
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

#if USE_SP_MKL
  extern "C"
  void MKL_SCOOMM(char *        transa,
                  int *         m,
                  int *         n,
                  int *         k,
                  float *       alpha,
                  char *        matdescra,
                  float const * val,
                  int const *   rowind,
                  int const *   colind,
                  int *         nnz,
                  float const * b,
                  int *         ldb,
                  float *       beta,
                  float *       c,
                  int *         ldc);


  extern "C"
  void MKL_DCOOMM(char *         transa,
                  int *          m,
                  int *          n,
                  int *          k,
                  double *       alpha,
                  char *         matdescra,
                  double const * val,
                  int const *    rowind,
                  int const *    colind,
                  int *          nnz,
                  double const * b,
                  int *          ldb,
                  double *       beta,
                  double *       c,
                  int *          ldc);

  extern "C"
  void MKL_CCOOMM(char *                      transa,
                  int *                       m,
                  int *                       n,
                  int *                       k,
                  std::complex<float> *       alpha,
                  char *                      matdescra,
                  std::complex<float> const * val,
                  int const *                 rowind,
                  int const *                 colind,
                  int *                       nnz,
                  std::complex<float> const * b,
                  int *                       ldb,
                  std::complex<float> *       beta,
                  std::complex<float> *       c,
                  int *                       ldc);


  extern "C"
  void MKL_ZCOOMM(char *                       transa,
                  int *                        m,
                  int *                        n,
                  int *                        k,
                  std::complex<double> *       alpha,
                  char *                       matdescra,
                  std::complex<double> const * val,
                  int const *                  rowind,
                  int const *                  colind,
                  int *                        nnz,
                  std::complex<double> const * b,
                  int *                        ldb,
                  std::complex<double> *       beta,
                  std::complex<double> *       c,
                  int *                        ldc);

  extern "C"
  void MKL_SCSRCOO(int const * job,
                   int *       n,
                   float *     acsr,
                   int const * ja,
                   int const * ia,
                   int *       nnz,
                   float *     acoo,
                   int const * rowind,
                   int const * colind,
                   int *       info);

  extern "C"
  void MKL_DCSRCOO(int const * job,
                   int *       n,
                   double *    acsr,
                   int const * ja,
                   int const * ia,
                   int *       nnz,
                   double *    acoo,
                   int const * rowind,
                   int const * colind,
                   int *       info);

  extern "C"
  void MKL_CCSRCOO(int const *           job,
                   int *                 n,
                   std::complex<float> * acsr,
                   int const *           ja,
                   int const *           ia,
                   int *                 nnz,
                   std::complex<float> * acoo,
                   int const *           rowind,
                   int const *           colind,
                   int *                 info);

  extern "C"
  void MKL_ZCSRCOO(int const *            job,
                   int *                  n,
                   std::complex<double> * acsr,
                   int const *            ja,
                   int const *            ia,
                   int *                  nnz,
                   std::complex<double> * acoo,
                   int const *            rowind,
                   int const *            colind,
                   int *                  info);


  extern "C"
  void MKL_SCSRMM(const char *transa , const int *m , const int *n , const int *k , const float *alpha , const char *matdescra , const float *val , const int *indx , const int *pntrb , const int *pntre , const float *b , const int *ldb , const float *beta , float *c , const int *ldc );

  extern "C"
  void MKL_DCSRMM(const char *transa , const int *m , const int *n , const int *k , const double *alpha , const char *matdescra , const double *val , const int *indx , const int *pntrb , const int *pntre , const double *b , const int *ldb , const double *beta , double *c , const int *ldc );


  extern "C"
  void MKL_CCSRMM(const char *transa , const int *m , const int *n , const int *k , const std::complex<float> *alpha , const char *matdescra , const std::complex<float> *val , const int *indx , const int *pntrb , const int *pntre , const std::complex<float> *b , const int *ldb , const std::complex<float> *beta , std::complex<float> *c , const int *ldc );

  extern "C"
  void MKL_ZCSRMM(const char *transa , const int *m , const int *n , const int *k , const std::complex<double> *alpha , const char *matdescra , const std::complex<double> *val , const int *indx , const int *pntrb , const int *pntre , const std::complex<double> *b , const int *ldb , const std::complex<double> *beta , std::complex<double> *c , const int *ldc );

  extern "C"
  void MKL_SCSRMULTD(const char *transa , const int *m , const int *n , const int *k, const float *a , const int *ja , const int *ia , const float *b , const int *jb , const int *ib , float *c , const int *ldc );

  extern "C"
  void MKL_DCSRMULTD(const char *transa , const int *m , const int *n , const int *k, const double *a , const int *ja , const int *ia , const double *b , const int *jb , const int *ib , double *c , const int *ldc );


  extern "C"
  void MKL_CCSRMULTD(const char *transa , const int *m , const int *n , const int *k, const std::complex<float> *a , const int *ja , const int *ia , const std::complex<float> *b , const int *jb , const int *ib , std::complex<float> *c , const int *ldc );

  extern "C"
  void MKL_ZCSRMULTD(const char *transa , const int *m , const int *n , const int *k, const std::complex<double> *a , const int *ja , const int *ia , const std::complex<double> *b , const int *jb , const int *ib , std::complex<double> *c , const int *ldc );

  extern "C"
  void MKL_SCSRMULTCSR(const char *transa , const int *req, const int *sort, const int *m, const int *n , const int *k, const float *a , const int *ja , const int *ia , const float *b , const int *jb , const int *ib , float *c , int *jc, int *ic, int *nnz_max, int *info);
  
  extern "C"
  void MKL_DCSRMULTCSR(const char *transa , const int* req, const int *sort, const int *m , const int *n , const int *k, const double *a , const int *ja , const int *ia , const double *b , const int *jb , const int *ib , double *c , int *jc, int *ic, int *nnz_max, int *info);

  extern "C"
  void MKL_CCSRMULTCSR(const char *transa , const int *req, const int *sort, const int *m, const int *n , const int *k, const std::complex<float> *a , const int *ja , const int *ia , const std::complex<float> *b , const int *jb , const int *ib , std::complex<float> *c , int *jc, int *ic, int *nnz_max, int *info);
  
  extern "C"
  void MKL_ZCSRMULTCSR(const char *transa , const int* req, const int *sort, const int *m , const int *n , const int *k, const std::complex<double> *a , const int *ja , const int *ia , const std::complex<double> *b , const int *jb , const int *ib , std::complex<double> *c , int *jc, int *ic, int *nnz_max, int *info);


  extern "C"
  void MKL_SCSRADD(char const * transa, int const *req, int const * job, int const * sort, int const * n, int const * k, float const * a, int const * ja, int const * ia, float const * beta, float const * b, int const * jb, int const * ib, float * c, int * jc, int * ic, int const * nnzmax, int const * ierr);

  extern "C"
  void MKL_DCSRADD(char const * transa, int const * job, int const * sort, int const * n, int const * k, double const * a, int const * ja, int const * ia, double const * beta, double const * b, int const * jb, int const * ib, double * c, int * jc, int * ic, int const * nnzmax, int const * ierr);

  extern "C"
  void MKL_CCSRADD(char const * transa, int const * job, int const * sort, int const * n, int const * k, std::complex<float> const * a, int const * ja, int const * ia, std::complex<float> const * beta, std::complex<float> const * b, int const * jb, int const * ib, std::complex<float> * c, int * jc, int * ic, int const * nnzmax, int const * ierr);

  extern "C"
  void MKL_ZCSRADD(char const * transa, int const * job, int const * sort, int const * n, int const * k, std::complex<double> const * a, int const * ja, int const * ia, std::complex<double> const * beta, std::complex<double> const * b, int const * jb, int const * ib, std::complex<double> * c, int * jc, int * ic, int const * nnzmax, int const * ierr);
#endif


  extern "C"
  void BLACS_GRIDINFO(int * icontxt, int * nprow, int * npcol, int * iprow, int * ipcol);

  extern "C"
  void BLACS_GRIDINIT(int * icontxt, char * order, int * nprow, int * npcol);
}
#endif
