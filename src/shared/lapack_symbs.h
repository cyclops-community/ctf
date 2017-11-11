#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__

#if FTN_UNDERSCORE
#define DGELSD dgelsd_
#define DGEQRF dgeqrf_
#define DORMQR dormqr_
#define PDGESVD pdgesvd_
#define DESCINIT descinit_
#else
#define DGELSD dgelsd
#define DGEQRF dgeqrf
#define DORMQR dormqr
#define PDGESVD pdgesvd
#define DESCINIT descinit
#endif

#define USE_LAPACK
#ifdef USE_LAPACK
namespace CTF_LAPACK{
  extern "C"
  void DGELSD(int * m, int * n, int * k, double const * A, int * lda_A, double * B, int * lda_B, double * S, int * cond, int * rank, double * work, int * lwork, int * iwork, int * info);


  extern "C"
  void DGEQRF(int const *  M, int const *  N, double * A, int const *  LDA, double * TAU2, double * WORK, int const *  LWORK, int  * INFO);


  extern "C"
  void DORMQR(char const * SIDE, char const * TRANS, int const *  M, int const *  N, int const *  K, double const * A, int const *  LDA, double const * TAU2, double * C, int const *  LDC, double * WORK, int const *  LWORK, int  * INFO);

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
  void DESCINIT(int *, int *,

                int *, int *,

                int *, int *,

                int *, int *,

                int *, int *);


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
    CTF_LAPACK::PDGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
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
    CTF_LAPACK::DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,&ictxt, &LLD, info);
  }

}
#endif

#endif
