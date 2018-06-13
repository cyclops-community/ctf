#include <stdlib.h>
#include <complex>
#include <assert.h>
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
                int *) SCAL_END

  
  EXTERN_OR_INLINE 
  void PSGEQRF(int *,
               int *,
               float *,
               int *,
               int *,
               int const *,
               float *,
               float *,
               int *,
               int *) SCAL_END
  
  EXTERN_OR_INLINE 
  void PDGEQRF(int *,
               int *,
               double *,
               int *,
               int *,
               int const *,
               double *,
               double *,
               int *,
               int *) SCAL_END
   

  EXTERN_OR_INLINE 
  void PCGEQRF(int *,
               int *,
               std::complex<float> *,
               int *,
               int *,
               int const *,
               std::complex<float> *,
               std::complex<float> *,
               int *,
               int *) SCAL_END
  
  EXTERN_OR_INLINE 
  void PZGEQRF(int *,
               int *,
               std::complex<double> *,
               int *,
               int *,
               int const *,
               std::complex<double> *,
               std::complex<double> *,
               int *,
               int *) SCAL_END
   

  EXTERN_OR_INLINE 
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
                int *) SCAL_END

  

  EXTERN_OR_INLINE 
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
                int *) SCAL_END


  EXTERN_OR_INLINE 
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
                int *) SCAL_END

  

  EXTERN_OR_INLINE 
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

  template <>
  inline void pgesvd<float>(char    JOBU,
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
    PSGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }

  template <>
  inline void pgesvd<double>(char     JOBU,
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
    PDGESVD(&JOBU, &JOBVT, &M, &N, A, &IA, &JA, DESCA, S, U, &IU, &JU, DESCU, VT, &IVT, &JVT,  DESCVT, WORK, &LWORK, info);
  }


  template <>
  inline void pgesvd< std::complex<float> >(char                  JOBU,
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
  }


  template <>
  inline void pgesvd< std::complex<double> >(char                   JOBU,
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

  template <> 
  inline void pgeqrf<float>(int         M,
                            int         N,
                            float *     A,
                            int         IA,
                            int         JA,
                            int const * DESCA,
                            float *     TAU2,
                            float *     WORK,
                            int         LWORK,
                            int *       INFO){
    PSGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }

  template <> 
  inline void pgeqrf<double>(int         M,
                             int         N,
                             double *    A,
                             int         IA,
                             int         JA,
                             int const * DESCA,
                             double *    TAU2,
                             double *    WORK,
                             int         LWORK,
                             int *       INFO){
    PDGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }

  template <> 
  inline void pgeqrf< std::complex<float> >(int                   M,
                                            int                   N,
                                            std::complex<float> * A,
                                            int                   IA,
                                            int                   JA,
                                            int const *           DESCA,
                                            std::complex<float> * TAU2,
                                            std::complex<float> * WORK,
                                            int                   LWORK,
                                            int *                 INFO){
    PCGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }


  template <> 
  inline void pgeqrf< std::complex<double> >(int                    M,
                                             int                    N,
                                             std::complex<double> * A,
                                             int                    IA,
                                             int                    JA,
                                             int const *            DESCA,
                                             std::complex<double> * TAU2,
                                             std::complex<double> * WORK,
                                             int                    LWORK,
                                             int *                  INFO){
    PZGEQRF(&M,&N,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }

  template <typename dtype>
  inline void porgqr(int         M,
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
  inline void porgqr<float>(int         M,
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
    PSORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }
  
  template <>
  inline void porgqr<double>(int         M,
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
    PDORGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }
  
  template <>
  inline void porgqr< std::complex<float> >(int                    M,
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
    PCUNGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }
  
  
  template <>
  inline void porgqr< std::complex<double> >(int                     M,
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
    PZUNGQR(&M,&N,&K,A,&IA,&JA,DESCA,TAU2,WORK,&LWORK,INFO);
  }


  inline
  void cdescinit( int *  desc,
                  int    m,
                  int    n,
                  int    mb,
                  int    nb,
                  int    irsrc,
                  int    icsrc,
                  int    ictxt,
                  int    LLD,
                  int *  info){
    DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,&ictxt, &LLD, info);
  }
#undef EXTERN_OR_INLINE 
#undef SCAL_END

}
#endif
