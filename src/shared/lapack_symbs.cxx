#include <stdlib.h>
#include <complex>
#include <assert.h>
#include "lapack_symbs.h"

#if FTN_UNDERSCORE
#define DGELSD dgelsd_
#define DGEQRF dgeqrf_
#define DORMQR dormqr_
#define PDGESVD pdgesvd_
#define PSGESVD psgesvd_
#define PCGESVD pcgesvd_
#define PZGESVD pzgesvd_
#define PSGEQRF psgeqrf_
#define PDGEQRF pdgeqrf_
#define PCGEQRF pcgeqrf_
#define PZGEQRF pzgeqrf_
#define PSORGQR psorgqr_
#define PDORGQR pdorgqr_
#define PCUNGQR pcungqr_
#define PZUNGQR pzungqr_
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
#define PSGEQRF psgeqrf
#define PDGEQRF pdgeqrf
#define PCGEQRF pcgeqrf
#define PZGEQRF pzgeqrf
#define PSORGQR psorgqr
#define PDORGQR pdorgqr
#define PCUNGQR pcungqr
#define PZUNGQR pzungqr
#define DESCINIT descinit
#define BLACS_GRIDINFO blacs_gridinfo
#define BLACS_GRIDINIT blacs_gridinit
#endif

namespace CTF_LAPACK{
#ifdef USE_LAPACK
  extern "C"
  void DGELSD(int * m, int * n, int * k, double const * A, int * lda_A, double * B, int * lda_B, double * S, int * cond, int * rank, double * work, int * lwork, int * iwork, int * info);

  extern "C"
  void DGEQRF(int const *  M, int const *  N, double * A, int const *  LDA, double * TAU2, double * WORK, int const *  LWORK, int  * INFO);

  extern "C"
  void DORMQR(char const * SIDE, char const * TRANS, int const *  M, int const *  N, int const *  K, double const * A, int const *  LDA, double const * TAU2, double * C, int const *  LDC, double * WORK, int const *  LWORK, int  * INFO);
#endif

  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, int cond, int rank, double * work, int lwork, int * iwork, int * info){
#ifdef USE_LAPACK
    DGELSD(&m, &n, &k, A, &lda_A, B, &lda_B, S, &cond, &rank, work, &lwork, iwork, info);
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
