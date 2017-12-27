#include <stdlib.h>
#include <iostream>
#include <complex>
#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__

#if FTN_UNDERSCORE
#define DGELSD dgelsd_
#define DGEQRF dgeqrf_
#define DORMQR dormqr_
#define PDGESVD pdgesvd_
#define PSGESVD psgesvd_
#define PCGESVD pcgesvd_
#define PZGESVD pzgesvd_
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
}
#ifdef USE_SCALAPACK
#define EXTERN_OR_INLINE extern "C"
#define SCAL_END ;
#else
#define EXTERN_OR_INLINE inline 
#define SCAL_END {}
#endif
namespace CTF_SCALAPACK{
  EXTERN_OR_INLINE
  void BLACS_GRIDINFO(int * icontxt, int * nprow, int * npcol, int * iprow, int * ipcol) SCAL_END

  EXTERN_OR_INLINE
  void BLACS_GRIDINIT(int * icontxt, char * order, int * nprow, int * npcol) SCAL_END

  EXTERN_OR_INLINE 
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
                int *) SCAL_END

  EXTERN_OR_INLINE 
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
                int *) SCAL_END

  EXTERN_OR_INLINE 
  void PCGESVD( char *,
                char *,
                int *,
                int *, 
                std::complex<float> *,
                int *,
                int *,
                int *,
                std::complex<float> *,
                std::complex<float> *,
                int *,
                int *,
                int *,
                std::complex<float> *,
                int *,
                int *,
                int *,
                std::complex<float> *,
                int *,
                int *) SCAL_END

  EXTERN_OR_INLINE 
  void PZGESVD( char *,
                char *,
                int *,
                int *, 
                std::complex<double> *,
                int *,
                int *,
                int *,
                std::complex<double> *,
                std::complex<double> *,
                int *,
                int *,
                int *,
                std::complex<double> *,
                int *,
                int *,
                int *,
                std::complex<double> *,
                int *,
                int *) SCAL_END

  EXTERN_OR_INLINE
  void DESCINIT(int *, int *,

                int *, int *,

                int *, int *,

                int *, int *,

                int *, int *) SCAL_END

  EXTERN_OR_INLINE 
    void Cblacs_pinfo(int*, int*) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_get(int, int, int*) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_gridinit(int*, char*, int, int) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_gridinfo(int, int*, int*, int*, int*) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_gridmap(int*, int*, int, int, int) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_barrier(int , char*) SCAL_END
  EXTERN_OR_INLINE 
    void Cblacs_gridexit(int) SCAL_END
  
  inline
  void cpdgesvd(  char JOBU,
                  char JOBVT,
                  int M,
                  int N,
                  double * A,
                  int IA,
                  int JA,
                  int * DESCA,
                  double * S,
                  double * U,
                  int IU,
                  int JU,
                  int * DESCU,
                  double * VT,
                  int IVT,
                  int JVT,
                  int * DESCVT,
                  double * WORK,
                  int LWORK,
                  int * info) {
    PDGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }

  inline
  void cpsgesvd(  char JOBU,
                  char JOBVT,
                  int M,
                  int N,
                  float * A,
                  int IA,
                  int JA,
                  int * DESCA,
                  float * S,
                  float * U,
                  int IU,
                  int JU,
                  int * DESCU,
                  float * VT,
                  int IVT,
                  int JVT,
                  int * DESCVT,
                  float * WORK,
                  int LWORK,
                  int * info) {
    PSGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }

  inline
  void cpcgesvd(  char JOBU,
                  char JOBVT,
                  int M,
                  int N,
                  std::complex<float> * A,
                  int IA,
                  int JA,
                  int * DESCA,
                  std::complex<float> * S,
                  std::complex<float> * U,
                  int IU,
                  int JU,
                  int * DESCU,
                  std::complex<float> * VT,
                  int IVT,
                  int JVT,
                  int * DESCVT,
                  std::complex<float> * WORK,
                  int LWORK,
                  int * info) {
    PCGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }

  inline
  void cpzgesvd(  char JOBU,
                  char JOBVT,
                  int M,
                  int N,
                  std::complex<double> * A,
                  int IA,
                  int JA,
                  int * DESCA,
                  std::complex<double> * S,
                  std::complex<double> * U,
                  int IU,
                  int JU,
                  int * DESCU,
                  std::complex<double> * VT,
                  int IVT,
                  int JVT,
                  int * DESCVT,
                  std::complex<double> * WORK,
                  int LWORK,
                  int * info) {
    PZGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }



  inline
  void cdescinit( int * desc, 
                  int m,	    
                  int n,
                  int mb,
                  int nb,
                  int irsrc,
                  int icsrc,
                  int ictxt,
                  int LLD,
                  int * info) {
    DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,&ictxt, &LLD, info);
  }
#undef EXTERN_OR_INLINE 
#undef SCAL_END

}
#endif
