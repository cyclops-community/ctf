#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__

#if FTN_UNDERSCORE
#define DDOT ddot_
#define DGELSD dgelsd_
#else
#define DDOT ddot
#define DGELSD dgelsd
#endif

namespace CTF_LAPACK{
  extern "C"
  double DDOT(int * n,         const double * dX,      
              int * incX,      const double * dY,      
              int * incY);

  extern "C"
  void DGELSD(int * m, int * n, int * k, double const * A, int * lda_A, double * B, int * lda_B, double * S, int * cond, int * rank, double * work, int * lwork, int * iwork, int * info);
}


#endif
