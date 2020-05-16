#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__
#include <stdlib.h>
#include <complex>
#include <assert.h>

namespace CTF_LAPACK{
  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, double cond, int * rank, double * work, int lwork, int * iwork, int * info);

  void cdgeqrf(int M, int N, double * A, int LDA, double * TAU2, double * WORK, int LWORK, int * INFO);

  void cdormqr(char SIDE, char TRANS, int M, int N, int K, double const * A, int LDA, double const * TAU2, double * C, int LDC, double * WORK, int LWORK, int * INFO);
}

namespace CTF_SCALAPACK {
  template <typename dtype>
  int get_int_fromreal(dtype r){
    assert(0);
    return -1;
  }

  template <>
  inline int get_int_fromreal<float>(float r){
    return (int)r;
  }
  template <>
  inline int get_int_fromreal<double>(double r){
    return (int)r;
  }
  template <>
  inline int get_int_fromreal<std::complex<float>>(std::complex<float> r){
    return (int)r.real();
  }
  template <>
  inline int get_int_fromreal<std::complex<double>>(std::complex<double> r){
    return (int)r.real();
  }


  template <typename dtype>
  void pgesvd(char    JOBU,
              char    JOBVT,
              int     M,
              int     N,
              dtype * A,
              int     IA,
              int     JA,
              int *   DESCA,
              dtype * S,
              dtype * U,
              int     IU,
              int     JU,
              int *   DESCU,
              dtype * VT,
              int     IVT,
              int     JVT,
              int *   DESCVT,
              dtype * WORK,
              int     LWORK,
              int *   info) {
    assert(0);
  }

  template <typename dtype>
  void pgeqrf(int         M,
              int         N,
              dtype *     A,
              int         IA,
              int         JA,
              int const * DESCA,
              dtype *     TAU2,
              dtype *     WORK,
              int         LWORK,
              int *       INFO){
    assert(0);
  }

  template <>
  void pgeqrf<float>(int         M,
                     int         N,
                     float *     A,
                     int         IA,
                     int         JA,
                     int const * DESCA,
                     float *     TAU2,
                     float *     WORK,
                     int         LWORK,
                     int *       INFO);
  template <>
  void pgeqrf<double>(int         M,
                      int         N,
                      double *    A,
                      int         IA,
                      int         JA,
                      int const * DESCA,
                      double *    TAU2,
                      double *    WORK,
                      int         LWORK,
                      int *       INFO);
  template <>
  void pgeqrf< std::complex<float> >(int                   M,
                                     int                   N,
                                     std::complex<float> * A,
                                     int                   IA,
                                     int                   JA,
                                     int const *           DESCA,
                                     std::complex<float> * TAU2,
                                     std::complex<float> * WORK,
                                     int                   LWORK,
                                     int *                 INFO);
  template <>
  void pgeqrf< std::complex<double> >(int                    M,
                                      int                    N,
                                      std::complex<double> * A,
                                      int                    IA,
                                      int                    JA,
                                      int const *            DESCA,
                                      std::complex<double> * TAU2,
                                      std::complex<double> * WORK,
                                      int                    LWORK,
                                      int *                  INFO);
  template <>
  void pgesvd<float>(char    JOBU,
                     char    JOBVT,
                     int     M,
                     int     N,
                     float * A,
                     int     IA,
                     int     JA,
                     int *   DESCA,
                     float * S,
                     float * U,
                     int     IU,
                     int     JU,
                     int *   DESCU,
                     float * VT,
                     int     IVT,
                     int     JVT,
                     int *   DESCVT,
                     float * WORK,
                     int     LWORK,
                     int *   info);

  template <>
  void pgesvd<float>(char    JOBU,
                     char    JOBVT,
                     int     M,
                     int     N,
                     float * A,
                     int     IA,
                     int     JA,
                     int *   DESCA,
                     float * S,
                     float * U,
                     int     IU,
                     int     JU,
                     int *   DESCU,
                     float * VT,
                     int     IVT,
                     int     JVT,
                     int *   DESCVT,
                     float * WORK,
                     int     LWORK,
                     int *   info);
  template <>
  void pgesvd<double>(char     JOBU,
                      char     JOBVT,
                      int      M,
                      int      N,
                      double * A,
                      int      IA,
                      int      JA,
                      int *    DESCA,
                      double * S,
                      double * U,
                      int      IU,
                      int      JU,
                      int *    DESCU,
                      double * VT,
                      int      IVT,
                      int      JVT,
                      int *    DESCVT,
                      double * WORK,
                      int      LWORK,
                      int *    info);

  template <>
  void pgesvd< std::complex<float> >(char                  JOBU,
                                     char                  JOBVT,
                                     int                   M,
                                     int                   N,
                                     std::complex<float> * A,
                                     int                   IA,
                                     int                   JA,
                                     int *                 DESCA,
                                     std::complex<float> * cS,
                                     std::complex<float> * U,
                                     int                   IU,
                                     int                   JU,
                                     int *                 DESCU,
                                     std::complex<float> * VT,
                                     int                   IVT,
                                     int                   JVT,
                                     int *                 DESCVT,
                                     std::complex<float> * WORK,
                                     int                   LWORK,
                                     int *                 info);


  template <>
  void pgesvd< std::complex<double> >(char                   JOBU,
                                      char                   JOBVT,
                                      int                    M,
                                      int                    N,
                                      std::complex<double> * A,
                                      int                    IA,
                                      int                    JA,
                                      int *                  DESCA,
                                      std::complex<double> * cS,
                                      std::complex<double> * U,
                                      int                    IU,
                                      int                    JU,
                                      int *                  DESCU,
                                      std::complex<double> * VT,
                                      int                    IVT,
                                      int                    JVT,
                                      int *                  DESCVT,
                                      std::complex<double> * WORK,
                                      int                    LWORK,
                                      int *                  info);

  template <typename dtype>
  void psyevx(char    JOBZ,
              char    RANGE,
              char    UPLO,
              int     N,
              dtype * A,
              int     IA,
              int     JA,
              int *   DESCA,
              dtype   VL,
              dtype   VU,
              int     IL,
              int     IU,
              dtype   ABSTOL,
              int *   M,
              int *   NZ,
              dtype * W,
              dtype   ORFAC,
              dtype * Z,
              int     IZ,
              int     JZ,
              int *   DESCZ,
              dtype * WORK,
              int     LWORK,
              int *   IWORK,
              int     LIWORK,
              int *   IFAIL,
              int *   ICLUSTR,
              dtype * GAP,
              int *   INFO){
    assert(0);
  }

  template <>
  void psyevx<float>(char    JOBZ,
                     char    RANGE,
                     char    UPLO,
                     int     N,
                     float * A,
                     int     IA,
                     int     JA,
                     int *   DESCA,
                     float   VL,
                     float   VU,
                     int     IL,
                     int     IU,
                     float   ABSTOL,
                     int *   M,
                     int *   NZ,
                     float * W,
                     float   ORFAC,
                     float * Z,
                     int     IZ,
                     int     JZ,
                     int *   DESCZ,
                     float * WORK,
                     int     LWORK,
                     int *   IWORK,
                     int     LIWORK,
                     int *   IFAIL,
                     int *   ICLUSTR,
                     float * GAP,
                     int *   INFO);

  template <>
  void psyevx<double>(char     JOBZ,
                      char     RANGE,
                      char     UPLO,
                      int      N,
                      double * A,
                      int      IA,
                      int      JA,
                      int *    DESCA,
                      double   VL,
                      double   VU,
                      int      IL,
                      int      IU,
                      double   ABSTOL,
                      int *    M,
                      int *    NZ,
                      double * W,
                      double   ORFAC,
                      double * Z,
                      int      IZ,
                      int      JZ,
                      int *    DESCZ,
                      double * WORK,
                      int      LWORK,
                      int *    IWORK,
                      int      LIWORK,
                      int *    IFAIL,
                      int *    ICLUSTR,
                      double * GAP,
                      int *    INFO);

  template <typename dtype>
  void pheevx(char                  JOBZ,
              char                  RANGE,
              char                  UPLO,
              int                   N,
              std::complex<dtype> * A,
              int                   IA,
              int                   JA,
              int *                 DESCA,
              dtype                 VL,
              dtype                 VU,
              int                   IL,
              int                   IU,
              dtype                 ABSTOL,
              int *                 M,
              int *                 NZ,
              dtype *               W,
              dtype                 ORFAC,
              std::complex<dtype> * Z,
              int                   IZ,
              int                   JZ,
              int *                 DESCZ,
              std::complex<dtype> * WORK,
              int                   LWORK,
              dtype *               RWORK,
              int                   LRWORK,
              int *                 IWORK,
              int                   LIWORK,
              int *                 IFAIL,
              int *                 ICLUSTR,
              dtype *               GAP,
              int *                 INFO){
    assert(0);
  }

  template <>
  void pheevx<float>(char                  JOBZ,
                     char                  RANGE,
                     char                  UPLO,
                     int                   N,
                     std::complex<float> * A,
                     int                   IA,
                     int                   JA,
                     int *                 DESCA,
                     float                 VL,
                     float                 VU,
                     int                   IL,
                     int                   IU,
                     float                 ABSTOL,
                     int *                 M,
                     int *                 NZ,
                     float *               W,
                     float                 ORFAC,
                     std::complex<float> * Z,
                     int                   IZ,
                     int                   JZ,
                     int *                 DESCZ,
                     std::complex<float> * WORK,
                     int                   LWORK,
                     float *               RWORK,
                     int                   LRWORK,
                     int *                 IWORK,
                     int                   LIWORK,
                     int *                 IFAIL,
                     int *                 ICLUSTR,
                     float *               GAP,
                     int *                 INFO);

  template <>
  void pheevx<double>(char                   JOBZ,
                      char                   RANGE,
                      char                   UPLO,
                      int                    N,
                      std::complex<double> * A,
                      int                    IA,
                      int                    JA,
                      int *                  DESCA,
                      double                 VL,
                      double                 VU,
                      int                    IL,
                      int                    IU,
                      double                 ABSTOL,
                      int *                  M,
                      int *                  NZ,
                      double *               W,
                      double                 ORFAC,
                      std::complex<double> * Z,
                      int                    IZ,
                      int                    JZ,
                      int *                  DESCZ,
                      std::complex<double> * WORK,
                      int                    LWORK,
                      double *               RWORK,
                      int                    LRWORK,
                      int *                  IWORK,
                      int                    LIWORK,
                      int *                  IFAIL,
                      int *                  ICLUSTR,
                      double *               GAP,
                      int *                  INFO);

  template <typename dtype>
  void pgeigh(char    UPLO,
              int64_t n,
              int64_t npr,
              int64_t npc,
              dtype * A,
              int *   DESCA,
              dtype * W,
              dtype * Z,
              int *   DESCZ){
    assert(0);
  }

  template <>
  void pgeigh<float>(char    UPLO,
                     int64_t n,
                     int64_t npr,
                     int64_t npc,
                     float * A,
                     int *   DESCA,
                     float * W,
                     float * Z,
                     int *   DESCZ);

  template <>
  void pgeigh<double>(char    UPLO,
                      int64_t n,
                      int64_t npr,
                      int64_t npc,
                      double * A,
                      int *   DESCA,
                      double * W,
                      double * Z,
                      int *   DESCZ);

  template <>
  void pgeigh<std::complex<float>>(char    UPLO,
                                   int64_t n,
                                   int64_t npr,
                                   int64_t npc,
                                   std::complex<float> * A,
                                   int *   DESCA,
                                   std::complex<float> * W,
                                   std::complex<float> * Z,
                                   int *   DESCZ);



  template <>
  void pgeigh<std::complex<double>>(char    UPLO,
                                    int64_t n,
                                    int64_t npr,
                                    int64_t npc,
                                    std::complex<double> * A,
                                    int *   DESCA,
                                    std::complex<double> * W,
                                    std::complex<double> * Z,
                                    int *   DESCZ);

  template <typename dtype>
  void porgqr(int         M,
              int         N,
              int         K,
              dtype *     A,
              int         IA,
              int         JA,
              int const * DESCA,
              dtype *     TAU2,
              dtype *     WORK,
              int         LWORK,
              int *       INFO){
    assert(0); // PORGQR not defined for this type
  }

  template <>
  void porgqr<float>(int         M,
                     int         N,
                     int         K,
                     float *     A,
                     int         IA,
                     int         JA,
                     int const * DESCA,
                     float *     TAU2,
                     float *     WORK,
                     int         LWORK,
                     int *       INFO);

  template <>
  void porgqr<double>(int         M,
                      int         N,
                      int         K,
                      double *    A,
                      int         IA,
                      int         JA,
                      int const * DESCA,
                      double *    TAU2,
                      double *    WORK,
                      int         LWORK,
                      int *       INFO);

  template <>
  void porgqr< std::complex<float> >(int                    M,
                                     int                    N,
                                     int                    K,
                                     std::complex<float>  * A,
                                     int                    IA,
                                     int                    JA,
                                     int const *            DESCA,
                                     std::complex<float>  * TAU2,
                                     std::complex<float>  * WORK,
                                     int                    LWORK,
                                     int *                  INFO);


  template <>
  void porgqr< std::complex<double> >(int                     M,
                                      int                     N,
                                      int                     K,
                                      std::complex<double>  * A,
                                      int                     IA,
                                      int                     JA,
                                      int const *             DESCA,
                                      std::complex<double>  * TAU2,
                                      std::complex<double>  * WORK,
                                      int                     LWORK,
                                      int *                   INFO);

  template <typename dtype>
  void ppotrf(char    uplo,
              int     n,
              dtype * A,
              int     ia,
              int     ja,
              int *   desca,
              int *   info){
    assert(0); // PPOTRF not defined for this type
  }

  template <>
  void ppotrf<float>(char    uplo,
                     int     n,
                     float * A,
                     int     ia,
                     int     ja,
                     int *   desca,
                     int *   info);

  template <>
  void ppotrf<double>(char     uplo,
                      int      n,
                      double * A,
                      int      ia,
                      int      ja,
                      int *    desca,
                      int *    info);

  template <>
  void ppotrf<std::complex<float>>(char                  uplo,
                                   int                   n,
                                   std::complex<float> * A,
                                   int                   ia,
                                   int                   ja,
                                   int *                 desca,
                                   int *                 info);

  template <>
  void ppotrf<std::complex<double>>(char                   uplo,
                                    int                    n,
                                    std::complex<double> * A,
                                    int                    ia,
                                    int                    ja,
                                    int *                  desca,
                                    int *                  info);


  template <typename dtype>
  void ptrsm(char SIDE, char UPLO, char TRANS, char DIAG,
             int M, int N, dtype ALPHA,
             dtype * A, int IA, int JA, int * DESCA,
             dtype * B, int IB, int JB, int * DESCB){
    assert(0); // PTRSM not defined for this type
  }

  template <>
  void ptrsm<float>(char SIDE, char UPLO, char TRANS, char DIAG,
                    int M, int N, float ALPHA,
                    float * A, int IA, int JA, int * DESCA,
                    float * B, int IB, int JB, int * DESCB);

  template <>
  void ptrsm<double>(char SIDE, char UPLO, char TRANS, char DIAG,
                     int M, int N, double ALPHA,
                     double * A, int IA, int JA, int * DESCA,
                     double * B, int IB, int JB, int * DESCB);

  template <>
  void ptrsm<std::complex<float>>(char SIDE, char UPLO, char TRANS, char DIAG,
                    int M, int N, std::complex<float> ALPHA,
                    std::complex<float> * A, int IA, int JA, int * DESCA,
                    std::complex<float> * B, int IB, int JB, int * DESCB);

  template <>
  void ptrsm<std::complex<double>>(char SIDE, char UPLO, char TRANS, char DIAG,
                     int M, int N, std::complex<double> ALPHA,
                     std::complex<double> * A, int IA, int JA, int * DESCA,
                     std::complex<double> * B, int IB, int JB, int * DESCB);


  template <typename dtype>
  void pposv(char UPLO, int N, int NRHS,
             dtype * A, int IA, int JA, int * DESCA,
             dtype * B, int IB, int JB, int * DESCB, int * info){
    assert(0); // PTRSM not defined for this type
  }

  template <>
  void pposv<float>(char UPLO, int N, int NRHS,
             float * A, int IA, int JA, int * DESCA,
             float * B, int IB, int JB, int * DESCB, int * info);
  template <>
  void pposv<double>(char UPLO, int N, int NRHS,
             double * A, int IA, int JA, int * DESCA,
             double * B, int IB, int JB, int * DESCB, int * info);
  template <>
  void pposv<std::complex<float>>(char UPLO, int N, int NRHS,
             std::complex<float> * A, int IA, int JA, int * DESCA,
             std::complex<float> * B, int IB, int JB, int * DESCB, int * info);
  template <>
  void pposv<std::complex<double>>(char UPLO, int N, int NRHS,
             std::complex<double> * A, int IA, int JA, int * DESCA,
             std::complex<double> * B, int IB, int JB, int * DESCB, int * info);


  void cdescinit( int *  desc,
                  int    m,
                  int    n,
                  int    mb,
                  int    nb,
                  int    irsrc,
                  int    icsrc,
                  int    ictxt,
                  int    LLD,
                  int *  info);

  void cblacs_pinfo(int * mypnum, int * nprocs);

  void cblacs_get(int contxt, int what, int * val);

  void cblacs_gridinit(int * contxt, char * row, int nprow, int npcol);

  void cblacs_gridinfo(int contxt, int * nprow, int * npcol, int * myprow, int * mypcol);

  void cblacs_gridmap(int * contxt, int * usermap, int ldup, int nprow0, int npcol0);

  void cblacs_barrier(int contxt, char * scope);

  void cblacs_gridexit(int contxt);
}
#endif

