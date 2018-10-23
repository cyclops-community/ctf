#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__
#include <assert.h>

namespace CTF_LAPACK{
  void cdgelsd(int m, int n, int k, double const * A, int lda_A, double * B, int lda_B, double * S, int cond, int rank, double * work, int lwork, int * iwork, int * info);

  void cdgeqrf(int M, int N, double * A, int LDA, double * TAU2, double * WORK, int LWORK, int * INFO);

  void cdormqr(char SIDE, char TRANS, int M, int N, int K, double const * A, int LDA, double const * TAU2, double * C, int LDC, double * WORK, int LWORK, int * INFO);
}

namespace CTF_SCALAPACK {
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
  inline void pgeqrf(int         M,
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
