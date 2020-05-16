#include <stdlib.h>
#include <complex>
#include <assert.h>
#include "lapack_symbs.h"
#include "../interface/common.h"

#if FTN_UNDERSCORE
#define DGELSD dgelsd_
#define DGEQRF dgeqrf_
#define DORMQR dormqr_
#define PDGESVD pdgesvd_
#define PSGESVD psgesvd_
#define PCGESVD pcgesvd_
#define PZGESVD pzgesvd_
#define PDSYEVX pdsyevx_
#define PSSYEVX pssyevx_
#define PCHEEVX pcheevx_
#define PZHEEVX pzheevx_
#define PSGEQRF psgeqrf_
#define PDGEQRF pdgeqrf_
#define PCGEQRF pcgeqrf_
#define PZGEQRF pzgeqrf_
#define PSORGQR psorgqr_
#define PDORGQR pdorgqr_
#define PCUNGQR pcungqr_
#define PZUNGQR pzungqr_
#define PSPOTRF pspotrf_
#define PDPOTRF pdpotrf_
#define PCPOTRF pcpotrf_
#define PZPOTRF pzpotrf_
#define PSTRSM pstrsm_
#define PDTRSM pdtrsm_
#define PCTRSM pctrsm_
#define PZTRSM pztrsm_
#define PSPOSV psposv_
#define PDPOSV pdposv_
#define PCPOSV pcposv_
#define PZPOSV pzposv_
#define DESCINIT descinit_
#define BLACS_GRIDINFO blacs_gridinfo_
#define BLACS_GRIDINIT blacs_gridinit_
#else
#define DGELSD dgelsd
#define DGEQRF dgeqrf
#define DORMQR dormqr
#define PDGESVD pdgesvd
#define PSGESVD psgesvd
#define PCGESVD pcgesvd
#define PZGESVD pzgesvd
#define PDSYEVX pdsyevx
#define PSSYEVX pssyevx
#define PCHEEVX pcheevx
#define PZHEEVX pzheevx
#define PSGEQRF psgeqrf
#define PDGEQRF pdgeqrf
#define PCGEQRF pcgeqrf
#define PZGEQRF pzgeqrf
#define PSORGQR psorgqr
#define PDORGQR pdorgqr
#define PCUNGQR pcungqr
#define PZUNGQR pzungqr
#define PSPOTRF pspotrf
#define PDPOTRF pdpotrf
#define PCPOTRF pcpotrf
#define PZPOTRF pzpotrf
#define PSTRSM pstrsm
#define PDTRSM pdtrsm
#define PCTRSM pctrsm
#define PZTRSM pztrsm
#define PSPOSV psposv
#define PDPOSV pdposv
#define PCPOSV pcposv
#define PZPOSV pzposv
#define DESCINIT descinit
#define BLACS_GRIDINFO blacs_gridinfo
#define BLACS_GRIDINIT blacs_gridinit
#endif

namespace CTF_LAPACK{
#ifdef USE_LAPACK
  extern "C"
  void DGELSD(int * m, int * n, int * k, double const * A, int * lda_A, double * B, int * lda_B, double * S, double * cond, int * rank, double * work, int * lwork, int * iwork, int * info);

  extern "C"
  void DGEQRF(int const *  M, int const *  N, double * A, int const *  LDA, double * TAU2, double * WORK, int const *  LWORK, int  * INFO);

  extern "C"
  void DORMQR(char const * SIDE, char const * TRANS, int const *  M, int const *  N, int const *  K, double const * A, int const *  LDA, double const * TAU2, double * C, int const *  LDC, double * WORK, int const *  LWORK, int  * INFO);
#endif

  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, double cond, int * rank, double * work, int lwork, int * iwork, int * info){
#ifdef USE_LAPACK
    DGELSD(&m, &n, &k, A, &lda_A, B, &lda_B, S, &cond, rank, work, &lwork, iwork, info);
#else
    assert(0);
#endif
  }

  void cdgeqrf(int M, int N, double * A, int LDA, double * TAU2, double * WORK, int LWORK, int * INFO){
#ifdef USE_LAPACK
    DGEQRF(&M, &N, A, &LDA, TAU2, WORK, &LWORK, INFO);
#else
    assert(0);
#endif
  }

  void cdormqr(char SIDE, char TRANS, int M, int N, int K, double const * A, int LDA, double const * TAU2, double * C, int LDC, double * WORK, int LWORK, int * INFO){
#ifdef USE_LAPACK
    DORMQR(&SIDE, &TRANS, &M, &N, &K, A, &LDA, TAU2, C, &LDC, WORK, &LWORK, INFO);
#else
    assert(0);
#endif
  }
}

namespace CTF_SCALAPACK{
#ifdef USE_SCALAPACK
  extern "C"
  void BLACS_GRIDINFO(int * icontxt, int * nprow, int * npcol, int * iprow, int * ipcol);

  extern "C"
  void BLACS_GRIDINIT(int * icontxt, char * order, int * nprow, int * npcol);

  extern "C"
  void PDGESVD( char *,
                char *,
                int *,
                int *,
                double *,
                int *,
                int *,
                int *,
                double *,
                double *,
                int *,
                int *,
                int *,
                double *,
                int *,
                int *,
                int *,
                double *,
                int *,
                int *);

  extern "C"
  void PSGESVD( char *,
                char *,
                int *,
                int *,
                float *,
                int *,
                int *,
                int *,
                float *,
                float *,
                int *,
                int *,
                int *,
                float *,
                int *,
                int *,
                int *,
                float *,
                int *,
                int *);

  extern "C"
  void PCGESVD( char *,
                char *,
                int *,
                int *,
                std::complex<float> *,
                int *,
                int *,
                int *,
                float *,
                std::complex<float> *,
                int *,
                int *,
                int *,
                std::complex<float> *,
                int *,
                int *,
                int *,
                float *,
                int *,
                float *,
                int *);

  extern "C"
  void PZGESVD( char *,
                char *,
                int *,
                int *,
                std::complex<double> *,
                int *,
                int *,
                int *,
                double *,
                std::complex<double> *,
                int *,
                int *,
                int *,
                std::complex<double> *,
                int *,
                int *,
                int *,
                double *,
                int *,
                double *,
                int *);

  extern "C"
  void PSSYEVX(char *  JOBZ,
               char *  RANGE,
               char *  UPLO,
               int *   N,
               float * A,
               int *   IA,
               int *   JA,
               int *   DESCA,
               float * VL,
               float * VU,
               int *   IL,
               int *   IU,
               float * ABSTOL,
               int *   M,
               int *   NZ,
               float * W,
               float * ORFAC,
               float * Z,
               int *   IZ,
               int *   JZ,
               int *   DESCZ,
               float * WORK,
               int *   LWORK,
               int *   IWORK,
               int *   LIWORK,
               int *   IFAIL,
               int *   ICLUSTR,
               float * GAP,
               int *   INFO);

  extern "C"
  void PDSYEVX(char *   JOBZ,
               char *   RANGE,
               char *   UPLO,
               int *    N,
               double * A,
               int *    IA,
               int *    JA,
               int *    DESCA,
               double * VL,
               double * VU,
               int *    IL,
               int *    IU,
               double * ABSTOL,
               int *    M,
               int *    NZ,
               double * W,
               double * ORFAC,
               double * Z,
               int *    IZ,
               int *    JZ,
               int *    DESCZ,
               double * WORK,
               int *    LWORK,
               int *    IWORK,
               int *    LIWORK,
               int *    IFAIL,
               int *    ICLUSTR,
               double * GAP,
               int *    INFO);

  extern "C"
  void PCHEEVX(char *                JOBZ,
               char *                RANGE,
               char *                UPLO,
               int *                 N,
               std::complex<float> * A,
               int *                 IA,
               int *                 JA,
               int *                 DESCA,
               float *               VL,
               float *               VU,
               int *                 IL,
               int *                 IU,
               float *               ABSTOL,
               int *                 M,
               int *                 NZ,
               float *               W,
               float *               ORFAC,
               std::complex<float> * Z,
               int *                 IZ,
               int *                 JZ,
               int *                 DESCZ,
               std::complex<float> * WORK,
               int *                 LWORK,
               float *               RWORK,
               int *                 LRWORK,
               int *                 IWORK,
               int *                 LIWORK,
               int *                 IFAIL,
               int *                 ICLUSTR,
               float *               GAP,
               int *                 INFO);

  extern "C"
  void PZHEEVX(char *                 JOBZ,
               char *                 RANGE,
               char *                 UPLO,
               int *                  N,
               std::complex<double> * A,
               int *                  IA,
               int *                  JA,
               int *                  DESCA,
               double *               VL,
               double *               VU,
               int *                  IL,
               int *                  IU,
               double *               ABSTOL,
               int *                  M,
               int *                  NZ,
               double *               W,
               double *               ORFAC,
               std::complex<double> * Z,
               int *                  IZ,
               int *                  JZ,
               int *                  DESCZ,
               std::complex<double> * WORK,
               int *                  LWORK,
               double *               RWORK,
               int *                  LRWORK,
               int *                  IWORK,
               int *                  LIWORK,
               int *                  IFAIL,
               int *                  ICLUSTR,
               double *               GAP,
               int *                  INFO);



  extern "C"
  void PSGEQRF(int *,
               int *,
               float *,
               int *,
               int *,
               int const *,
               float *,
               float *,
               int *,
               int *);

  extern "C"
  void PDGEQRF(int *,
               int *,
               double *,
               int *,
               int *,
               int const *,
               double *,
               double *,
               int *,
               int *);


  extern "C"
  void PCGEQRF(int *,
               int *,
               std::complex<float> *,
               int *,
               int *,
               int const *,
               std::complex<float> *,
               std::complex<float> *,
               int *,
               int *);

  extern "C"
  void PZGEQRF(int *,
               int *,
               std::complex<double> *,
               int *,
               int *,
               int const *,
               std::complex<double> *,
               std::complex<double> *,
               int *,
               int *);


  extern "C"
  void PSORGQR(int *,
                int *,
                int *,
                float *,
                int *,
                int *,
                int const *,
                float *,
                float *,
                int *,
                int *);



  extern "C"
  void PDORGQR(int *,
                int *,
                int *,
                double *,
                int *,
                int *,
                int const *,
                double *,
                double *,
                int *,
                int *);


  extern "C"
  void PCUNGQR(int *,
                int *,
                int *,
                std::complex<float> *,
                int *,
                int *,
                int const *,
                std::complex<float> *,
                std::complex<float> *,
                int *,
                int *);



  extern "C"
  void PZUNGQR(int *,
                int *,
                int *,
                std::complex<double> *,
                int *,
                int *,
                int const *,
                std::complex<double> *,
                std::complex<double> *,
                int *,
                int *);

  extern "C"
  void PSPOTRF(char *  uplo,
               int *   n,
               float * A,
               int *   ia,
               int *   ja,
               int *   desca,
               int *   info);

  extern "C"
  void PDPOTRF(char *   uplo,
               int *    n,
               double * A,
               int *    ia,
               int *    ja,
               int *    desca,
               int *    info);

  extern "C"
  void PCPOTRF(char *                uplo,
               int *                 n,
               std::complex<float> * A,
               int *                 ia,
               int *                 ja,
               int *                 desca,
               int *                 info);

  extern "C"
  void PZPOTRF(char *                 uplo,
               int *                  n,
               std::complex<double> * A,
               int *                  ia,
               int *                  ja,
               int *                  desca,
               int *                  info);

  extern "C"
  void PSTRSM(char * SIDE, char * UPLO, char * TRANS, char * DIAG,
              int * M, int * N, float * ALPHA,
              float * A, int * IA, int * JA, int * DESCA,
              float * B, int * IB, int * JB, int * DESCB);

  extern "C"
  void PDTRSM(char * SIDE, char * UPLO, char * TRANS, char * DIAG,
              int * M, int * N, double * ALPHA,
              double * A, int * IA, int * JA, int * DESCA,
              double * B, int * IB, int * JB, int * DESCB);

  extern "C"
  void PCTRSM(char * SIDE, char * UPLO, char * TRANS, char * DIAG,
              int * M, int * N, std::complex<float> * ALPHA,
              std::complex<float> * A, int * IA, int * JA, int * DESCA,
              std::complex<float> * B, int * IB, int * JB, int * DESCB);

  extern "C"
  void PZTRSM(char * SIDE, char * UPLO, char * TRANS, char * DIAG,
              int * M, int * N, std::complex<double> * ALPHA,
              std::complex<double> * A, int * IA, int * JA, int * DESCA,
              std::complex<double> * B, int * IB, int * JB, int * DESCB);

  extern "C"
  void PSPOSV(char * UPLO, int * N, int * NRHS,
             float * A, int * IA, int * JA, int * DESCA,
             float * B, int * IB, int * JB, int * DESCB, int * info);

  extern "C"
  void PDPOSV(char * UPLO, int * N, int * NRHS,
             double * A, int * IA, int * JA, int * DESCA,
             double * B, int * IB, int * JB, int * DESCB, int * info);


  extern "C"
  void PCPOSV(char * UPLO, int * N, int * NRHS,
             std::complex<float> * A, int * IA, int * JA, int * DESCA,
             std::complex<float> * B, int * IB, int * JB, int * DESCB, int * info);


  extern "C"
  void PZPOSV(char * UPLO, int * N, int * NRHS,
             std::complex<double> * A, int * IA, int * JA, int * DESCA,
             std::complex<double> * B, int * IB, int * JB, int * DESCB, int * info);


  extern "C"
  void DESCINIT(int *, int *,

                int *, int *,

                int *, int *,

                int *, int *,

                int *, int *);

    extern "C"
    void Cblacs_pinfo(int*, int*);
    extern "C"
    void Cblacs_get(int, int, int*);
    extern "C"
    void Cblacs_gridinit(int*, char*, int, int);
    extern "C"
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    extern "C"
    void Cblacs_gridmap(int*, int*, int, int, int);
    extern "C"
    void Cblacs_barrier(int , char*);
    extern "C"
    void Cblacs_gridexit(int);
#endif


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
                     int *   info) {
#ifdef USE_SCALAPACK
    PSGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
#else
    assert(0);
#endif
  }

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
                      int *    info) {
#ifdef USE_SCALAPACK
    PDGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
#else
    assert(0);
#endif
  }


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
                                     int *                 info) {
#ifdef USE_SCALAPACK
    float * S = (float*)cS;
    float * rwork;
    rwork = new float[4*std::max(M,N)+1];
    PCGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, (float*)WORK, &LWORK, rwork, info);
    delete [] rwork;
    if (LWORK != -1){
      for (int i=std::min(M,N)-1; i>=0; i--){
        cS[i].real(S[i]);
        cS[i].imag(0.0);
      }
    }
#else
    assert(0);
#endif
  }


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
                                      int *                  info){
#ifdef USE_SCALAPACK
    double * S = (double*)cS;
    double * rwork;
    rwork = new double[4*std::max(M,N)+1];
    PZGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, (double*)WORK, &LWORK, rwork, info);
    delete [] rwork;
    if (LWORK != -1){
      for (int i=std::min(M,N)-1; i>=0; i--){
        cS[i].real(S[i]);
        cS[i].imag(0.0);
      }
    }
#else
    assert(0);
#endif
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
                     int *   INFO){
#ifdef USE_SCALAPACK
    PSSYEVX(&JOBZ,&RANGE,&UPLO,&N,A,&IA,&JA,DESCA,&VL,&VU,&IL,&IU,&ABSTOL,M,NZ,W,&ORFAC,Z,&IZ,&JZ,DESCZ,WORK,&LWORK,IWORK,&LIWORK,IFAIL,ICLUSTR,GAP,INFO);
#else
    assert(0);
#endif
  }

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
                      int *    INFO){
#ifdef USE_SCALAPACK
    PDSYEVX(&JOBZ,&RANGE,&UPLO,&N,A,&IA,&JA,DESCA,&VL,&VU,&IL,&IU,&ABSTOL,M,NZ,W,&ORFAC,Z,&IZ,&JZ,DESCZ,WORK,&LWORK,IWORK,&LIWORK,IFAIL,ICLUSTR,GAP,INFO);
#else
    assert(0);
#endif
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
                     int *                 INFO){
#ifdef USE_SCALAPACK
    PCHEEVX(&JOBZ,&RANGE,&UPLO,&N,A,&IA,&JA,DESCA,&VL,&VU,&IL,&IU,&ABSTOL,M,NZ,W,&ORFAC,Z,&IZ,&JZ,DESCZ,WORK,&LWORK,RWORK,&LWORK,IWORK,&LIWORK,IFAIL,ICLUSTR,GAP,INFO);
#else
    assert(0);
#endif
  }

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
                      int *                  INFO){
#ifdef USE_SCALAPACK
    PZHEEVX(&JOBZ,&RANGE,&UPLO,&N,A,&IA,&JA,DESCA,&VL,&VU,&IL,&IU,&ABSTOL,M,NZ,W,&ORFAC,Z,&IZ,&JZ,DESCZ,WORK,&LWORK,RWORK,&LWORK,IWORK,&LIWORK,IFAIL,ICLUSTR,GAP,INFO);
#else
    assert(0);
#endif
  }

  template <typename dtype>
  void peigs(char    UPLO,
             int64_t n,
             int64_t npr,
             int64_t npc,
             dtype * A,
             int *   desca,
             dtype * d,
             dtype * u,
             int *   descu){
    int M, NZ, info;
    int lwork;
    dtype dlwork;
    int ilwork;
    //create copy variables due to weird ScaLAPACK bug, which zeros out n and descu pointer for pdsyevx
    //int64_t n_cpy = n;
    //char UPLO_cpy = UPLO;
    //int * descu_cpy = descu;
    //int * desca_cpy = desca;
    //printf("%c %c, %ld %ld, %p %p, %p %p",UPLO, UPLO_cpy, n, n_cpy, descu, descu_cpy, desca, desca_cpy);
    CTF_SCALAPACK::psyevx<dtype>('V', 'A', 'U', n, NULL, 1, 1, desca, (dtype)0, (dtype)0, 0, 0, (dtype)0, &M, &NZ, NULL, (dtype)0, NULL, 1, 1, descu, &dlwork, -1, &ilwork, -1, NULL, NULL, NULL, &info);
    //printf("%c %c, %ld %ld, %p %p, %p %p",UPLO, UPLO_cpy, n, n_cpy, descu, descu_cpy, desca, desca_cpy);
    //UPLO = UPLO_cpy;
    //n = n_cpy;
    //descu = descu_cpy;
    //desca = desca_cpy;
    lwork = get_int_fromreal<dtype>(dlwork);
    dtype * work = (dtype*)CTF_int::alloc(sizeof(dtype)*((int64_t)lwork));
    int * iwork = (int*)CTF_int::alloc(sizeof(int)*ilwork);
    int * IFAIL = (int*)CTF_int::alloc(sizeof(int)*n);
    int * ICLUSTR = (int*)CTF_int::alloc(sizeof(int)*2*npr*npc);
    dtype * GAP = (dtype*)CTF_int::alloc(sizeof(dtype)*npr*npc);
    CTF_SCALAPACK::psyevx<dtype>('V', 'A', UPLO, n, A, 1, 1, desca, (dtype)0., (dtype)0., 0, 0, (dtype)0., &M, &NZ, d, (dtype)0., u, 1, 1, descu, work, lwork, iwork, ilwork, IFAIL, ICLUSTR, GAP, &info);
    CTF_int::cdealloc(iwork);
    CTF_int::cdealloc(work);
    CTF_int::cdealloc(IFAIL);
    CTF_int::cdealloc(ICLUSTR);
    CTF_int::cdealloc(GAP);
    if (info != 0){
      printf("CTF ERROR: pysevx returned error code %d\n",info);
    }
  }

  template <typename dtype>
  void peigh(char    UPLO,
             int64_t n,
             int64_t npr,
             int64_t npc,
             std::complex<dtype> * A,
             int *   desca,
             std::complex<dtype> * d,
             std::complex<dtype> * u,
             int *   descu){
    int M, NZ, info;
    int lwork;
    dtype dlwork;
    int lcwork;
    std::complex<dtype> dlcwork;
    int ilwork;
    //create copy variables due to weird ScaLAPACK bug, which zeros out n and descu pointer for pdsyevx
    //int64_t n_cpy = n;
    //char UPLO_cpy = UPLO;
    //int * descu_cpy = descu;
    //int * desca_cpy = desca;
    //printf("%c %c, %ld %ld, %p %p, %p %p",UPLO, UPLO_cpy, n, n_cpy, descu, descu_cpy, desca, desca_cpy);
    CTF_SCALAPACK::pheevx<dtype>('V', 'A', 'U', n, NULL, 1, 1, desca, (dtype)0, (dtype)0, 0, 0, (dtype)0, &M, &NZ, NULL, (dtype)0, NULL, 1, 1, descu, &dlcwork, -1, &dlwork, -1, &ilwork, -1, NULL, NULL, NULL, &info);
    //printf("%c %c, %ld %ld, %p %p, %p %p",UPLO, UPLO_cpy, n, n_cpy, descu, descu_cpy, desca, desca_cpy);
    //UPLO = UPLO_cpy;
    //n = n_cpy;
    //descu = descu_cpy;
    //desca = desca_cpy;
    lwork = get_int_fromreal<dtype>(dlwork);
    lcwork = get_int_fromreal<std::complex<dtype>>(dlcwork);
    dtype * work = (dtype*)CTF_int::alloc(sizeof(dtype)*((int64_t)lwork));
    std::complex<dtype> * cwork = (std::complex<dtype>*)CTF_int::alloc(sizeof(std::complex<dtype>)*((int64_t)lcwork));
    int * iwork = (int*)CTF_int::alloc(sizeof(int)*ilwork);
    int * IFAIL = (int*)CTF_int::alloc(sizeof(int)*n);
    int * ICLUSTR = (int*)CTF_int::alloc(sizeof(int)*2*npr*npc);
    dtype * GAP = (dtype*)CTF_int::alloc(sizeof(dtype)*npr*npc);
    dtype * rd = (dtype*)CTF_int::alloc(sizeof(dtype)*n);
    CTF_SCALAPACK::pheevx<dtype>('V', 'A', UPLO, n, A, 1, 1, desca, (dtype)0., (dtype)0., 0, 0, (dtype)0., &M, &NZ, rd, (dtype)0., u, 1, 1, descu, cwork, lcwork, work, lwork, iwork, ilwork, IFAIL, ICLUSTR, GAP, &info);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<n; i++){
      d[i].real(rd[i]);
      d[i].imag(0.);
    }
    CTF_int::cdealloc(rd);
    CTF_int::cdealloc(cwork);
    CTF_int::cdealloc(iwork);
    CTF_int::cdealloc(work);
    CTF_int::cdealloc(IFAIL);
    CTF_int::cdealloc(ICLUSTR);
    CTF_int::cdealloc(GAP);
    if (info != 0){
      printf("CTF ERROR: pysevx returned error code %d\n",info);
    }
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
                     int *   DESCZ){
    peigs<float>(UPLO,n,npr,npc,A,DESCA,W,Z,DESCZ);
  }

  template <>
  void pgeigh<double>(char    UPLO,
                      int64_t n,
                      int64_t npr,
                      int64_t npc,
                      double * A,
                      int *   DESCA,
                      double * W,
                      double * Z,
                      int *   DESCZ){
    peigs<double>(UPLO,n,npr,npc,A,DESCA,W,Z,DESCZ);
  }


  template <>
  void pgeigh<std::complex<float>>(char    UPLO,
                                   int64_t n,
                                   int64_t npr,
                                   int64_t npc,
                                   std::complex<float> * A,
                                   int *   DESCA,
                                   std::complex<float> * W,
                                   std::complex<float> * Z,
                                   int *   DESCZ){
    peigh<float>(UPLO,n,npr,npc,A,DESCA,W,Z,DESCZ);
  }


  template <>
  void pgeigh<std::complex<double>>(char    UPLO,
                                    int64_t n,
                                    int64_t npr,
                                    int64_t npc,
                                    std::complex<double> * A,
                                    int *   DESCA,
                                    std::complex<double> * W,
                                    std::complex<double> * Z,
                                    int *   DESCZ){
    peigh<double>(UPLO,n,npr,npc,A,DESCA,W,Z,DESCZ);
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
                     int *       INFO){
#ifdef USE_SCALAPACK
    PSGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }

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
                      int *       INFO){
#ifdef USE_SCALAPACK
    PDGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }

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
                                     int *                 INFO){
#ifdef USE_SCALAPACK
    PCGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }


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
                                      int *                  INFO){
#ifdef USE_SCALAPACK
    PZGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
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
                     int *       INFO){
#ifdef USE_SCALAPACK
    PSORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }

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
                      int *       INFO){
#ifdef USE_SCALAPACK
    PDORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }

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
                                     int *                  INFO){
#ifdef USE_SCALAPACK
    PCUNGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }


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
                                      int *                   INFO){
#ifdef USE_SCALAPACK
    PZUNGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
#else
    assert(0);
#endif
  }

  template <>
  void ppotrf<float>(char    uplo,
                     int     n,
                     float * A,
                     int     ia,
                     int     ja,
                     int *   desca,
                     int *   info){
#ifdef USE_SCALAPACK
    PSPOTRF(&uplo, &n, A, &ia, &ja, desca, info);
#else
    assert(0);
#endif
  }

  template <>
  void ppotrf<double>(char     uplo,
                      int      n,
                      double * A,
                      int      ia,
                      int      ja,
                      int *    desca,
                      int *    info){
#ifdef USE_SCALAPACK
    PDPOTRF(&uplo, &n, A, &ia, &ja, desca, info);
#else
    assert(0);
#endif
  }

  template <>
  void ppotrf<std::complex<float>>(char                  uplo,
                                   int                   n,
                                   std::complex<float> * A,
                                   int                   ia,
                                   int                   ja,
                                   int *                 desca,
                                   int *                 info){
#ifdef USE_SCALAPACK
    PCPOTRF(&uplo, &n, A, &ia, &ja, desca, info);
#else
    assert(0);
#endif
  }

  template <>
  void ppotrf<std::complex<double>>(char                   uplo,
                                    int                    n,
                                    std::complex<double> * A,
                                    int                    ia,
                                    int                    ja,
                                    int *                  desca,
                                    int *                  info){
#ifdef USE_SCALAPACK
    PZPOTRF(&uplo, &n, A, &ia, &ja, desca, info);
#else
    assert(0);
#endif
  }


  template <>
  void ptrsm<float>(char SIDE, char UPLO, char TRANS, char DIAG,
                    int M, int N, float ALPHA,
                    float * A, int IA, int JA, int * DESCA,
                    float * B, int IB, int JB, int * DESCB){
#ifdef USE_SCALAPACK
    PSTRSM(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB);
#else
    assert(0);
#endif
  }



  template <>
  void ptrsm<double>(char SIDE, char UPLO, char TRANS, char DIAG,
                     int M, int N, double ALPHA,
                     double * A, int IA, int JA, int * DESCA,
                     double * B, int IB, int JB, int * DESCB){
#ifdef USE_SCALAPACK
    PDTRSM(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB);
#else
    assert(0);
#endif
  }


  template <>
  void ptrsm<std::complex<float>>(char SIDE, char UPLO, char TRANS, char DIAG,
                    int M, int N, std::complex<float> ALPHA,
                    std::complex<float> * A, int IA, int JA, int * DESCA,
                    std::complex<float> * B, int IB, int JB, int * DESCB){
#ifdef USE_SCALAPACK
    PCTRSM(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB);
#else
    assert(0);
#endif
  }


  template <>
  void ptrsm<std::complex<double>>(char SIDE, char UPLO, char TRANS, char DIAG,
                     int M, int N, std::complex<double> ALPHA,
                     std::complex<double> * A, int IA, int JA, int * DESCA,
                     std::complex<double> * B, int IB, int JB, int * DESCB){
#ifdef USE_SCALAPACK
    PZTRSM(&SIDE, &UPLO, &TRANS, &DIAG, &M, &N, &ALPHA, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB);
#else
    assert(0);
#endif
  }


  template <>
  void pposv<float>(char UPLO, int N, int NRHS,
             float * A, int IA, int JA, int * DESCA,
             float * B, int IB, int JB, int * DESCB, int * info){
#ifdef USE_SCALAPACK
    PSPOSV(&UPLO, &N, &NRHS, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, info);
#else
    assert(0);
#endif
  }
  template <>
  void pposv<double>(char UPLO, int N, int NRHS,
             double * A, int IA, int JA, int * DESCA,
             double * B, int IB, int JB, int * DESCB, int * info){
#ifdef USE_SCALAPACK
    PDPOSV(&UPLO, &N, &NRHS, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, info);
#else
    assert(0);
#endif

  }
  template <>
  void pposv<std::complex<float>>(char UPLO, int N, int NRHS,
             std::complex<float> * A, int IA, int JA, int * DESCA,
             std::complex<float> * B, int IB, int JB, int * DESCB, int * info){
#ifdef USE_SCALAPACK
    PCPOSV(&UPLO, &N, &NRHS, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, info);
#else
    assert(0);
#endif
  }
  template <>
  void pposv<std::complex<double>>(char UPLO, int N, int NRHS,
             std::complex<double> * A, int IA, int JA, int * DESCA,
             std::complex<double> * B, int IB, int JB, int * DESCB, int * info){
#ifdef USE_SCALAPACK
    PZPOSV(&UPLO, &N, &NRHS, A, &IA, &JA, DESCA, B, &IB, &JB, DESCB, info);
#else
    assert(0);
#endif
  }



  void cdescinit(int *  desc,
                 int    m,
                 int    n,
                 int    mb,
                 int    nb,
                 int    irsrc,
                 int    icsrc,
                 int    ictxt,
                 int    LLD,
                 int *  info){
#ifdef USE_SCALAPACK
    DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,&ictxt, &LLD, info);
#else
    assert(0);
#endif
  }

  void cblacs_pinfo(int * mypnum, int * nprocs){
#ifdef USE_SCALAPACK
    Cblacs_pinfo(mypnum, nprocs);
#else
    assert(0);
#endif
  }

  void cblacs_get(int contxt, int what, int * val){
#ifdef USE_SCALAPACK
    Cblacs_get(contxt, what, val);
#else
    assert(0);
#endif
  }

  void cblacs_gridinit(int * contxt, char * row, int nprow, int npcol){
#ifdef USE_SCALAPACK
    Cblacs_gridinit(contxt, row, nprow, npcol);
#else
    assert(0);
#endif
  }

  void cblacs_gridinfo(int contxt, int * nprow, int * npcol, int * myprow, int * mypcol){
#ifdef USE_SCALAPACK
    Cblacs_gridinfo(contxt, nprow, npcol, myprow, mypcol);
#else
    assert(0);
#endif
  }

  void cblacs_gridmap(int * contxt, int * usermap, int ldup, int nprow0, int npcol0){
#ifdef USE_SCALAPACK
    Cblacs_gridmap(contxt, usermap, ldup, nprow0, npcol0);
#else
    assert(0);
#endif
  }

  void cblacs_barrier(int contxt, char * scope){
#ifdef USE_SCALAPACK
    Cblacs_barrier(contxt, scope);
#else
    assert(0);
#endif
  }

  void cblacs_gridexit(int contxt){
#ifdef USE_SCALAPACK
    Cblacs_gridexit(contxt);
#else
    assert(0);
#endif
  }

}

