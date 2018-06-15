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
