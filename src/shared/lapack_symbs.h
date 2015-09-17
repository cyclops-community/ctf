#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__

#if FTN_UNDERSCORE
#define DDOT ddot_
#define DGELSD dgelsd_
#define DGEQRF dgeqrf_
#define DORMQR dormqr_
#else
#define DDOT ddot
#define DGELSD dgelsd
#define DGEQRF dgeqrf
#define DORMQR dormqr
#endif

namespace CTF_LAPACK{
  extern "C"
  double DDOT(int * n,         const double * dX,      
              int * incX,      const double * dY,      
              int * incY);

  extern "C"
  void DGELSD(int * m, int * n, int * k, double const * A, int * lda_A, double * B, int * lda_B, double * S, int * cond, int * rank, double * work, int * lwork, int * iwork, int * info);


  extern "C"
  void DGEQRF(int const *  M, int const *  N, double * A, int const *  LDA, double * TAU2, double * WORK, int const *  LWORK, int  * INFO);


  extern "C"
  void DORMQR(char const * SIDE, char const * TRANS, int const *  M, int const *  N, int const *  K, double const * A, int const *  LDA, double const * TAU2, double * C, int const *  LDC, double * WORK, int const *  LWORK, int  * INFO);


}


#endif
