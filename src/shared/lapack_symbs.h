#ifndef __LAPACK_SYMBS__
#define __LAPACK_SYMBS__
#include <stdlib.h>
#include <complex>
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

