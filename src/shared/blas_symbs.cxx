#include "blas_symbs.h"
#include "util.h"
namespace CTF_BLAS {
  template <typename dtype>
  void gemm(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            const dtype *,
            const int *,
            const dtype *,
            const int *,
            const dtype *,
            dtype *,
            const int *){
    printf("CTF ERROR GEMM not available for this type.\n");
    ASSERT(0);
    assert(0);
  }
#define INST_GEMM(dtype,s)                     \
  template <>                                  \
  void gemm<dtype>(const char * a,             \
            const char * b,                    \
            const int * c,                     \
            const int * d,                     \
            const int * e,                     \
            const dtype * f,                   \
            const dtype * g,                   \
            const int * h,                     \
            const dtype * i,                   \
            const int * j,                     \
            const dtype * k,                   \
            dtype * l,                         \
            const int * m){                    \
    s ## GEMM(a,b,c,d,e,f,g,h,i,j,k,l,m); \
  }
  INST_GEMM(float,S)
  INST_GEMM(double,D)
  INST_GEMM(std::complex<float>,C)
  INST_GEMM(std::complex<double>,Z)
#undef INST_GEMM

  template <typename dtype>
  void syr(const char *       UPLO ,
            const int *        N , 
            const dtype *     ALPHA, 
            const dtype *     X , 
            const int *        INCX , 
            dtype *           A , 
            const int *        LDA ){
     printf("CTF ERROR POSV not available for this type.\n");
    ASSERT(0);
    assert(0);
  }

#define INST_SYR(dtype,s)                     \
  template <>                                  \
  void syr<dtype>(const char * a, \
            const int *    b, \
            const dtype *        c, \
            const dtype *        d, \
            const int *    e, \
            dtype *        f, \
            const int *    g){ \
    s ## SYR(a,b,c,d,e,f,g); \
  }
  INST_SYR(float,S)
  INST_SYR(double,D)
  INST_SYR(std::complex<float>,C)
  INST_SYR(std::complex<double>,Z)
#undef INST_GEMM


  template <typename dtype>
  void posv(char const *        UPLO ,
            const int *         N, 
            const int *         NRHS,
            dtype *            A, 
            const int *         LDA, 
            dtype *            B, 
            const int *         LDB, 
            int *               INFO){
     printf("CTF ERROR POSV not available for this type.\n");
    ASSERT(0);
    assert(0);
  }



#define INST_POSV(dtype,s)                     \
  template <>                                  \
  void posv<dtype>(char const * a, \
            const int *    b, \
            const int *    c, \
            dtype *        d, \
            const int *    e, \
            dtype *        f, \
            const int *    g, \
            int *          h){ \
    s ## POSV(a,b,c,d,e,f,g,h); \
  }
  INST_POSV(float,S)
  INST_POSV(double,D)
  INST_POSV(std::complex<float>,C)
  INST_POSV(std::complex<double>,Z)
#undef INST_GEMM



#ifdef USE_BATCH_GEMM
  template <typename dtype>
  void gemm_batch(const char *,
            const char *,
            const int *,
            const int *,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            dtype **,
            const int *,
            const dtype *,
            dtype **,
            const int *,
            const int *,
            const int *){
    printf("CTF ERROR gemm_batch not available for this type.\n");
    ASSERT(0);
    assert(0);
  }

#define INST_GEMM_BATCH(dtype,s)                         \
  template <>                                            \
  void gemm_batch<dtype>(const char * a,                 \
            const char * b,                              \
            const int * c,                               \
            const int * d,                               \
            const int * e,                               \
            const dtype * f,                             \
            dtype ** g,                                  \
            const int * h,                               \
            dtype ** i,                                  \
            const int * j,                               \
            const dtype * k,                             \
            dtype ** l,                                  \
            const int * m,                               \
            const int * n,                               \
            const int * o){                              \
    s ## GEMM_BATCH(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o); \
  }
  INST_GEMM_BATCH(float,S)
  INST_GEMM_BATCH(double,D)
  INST_GEMM_BATCH(std::complex<float>,C)
  INST_GEMM_BATCH(std::complex<double>,Z)
#endif
}
#undef INST_GEMM_BATCH
