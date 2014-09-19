#ifndef __COMMON_H__
#define __COMMON_H__
#include <string.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include <complex>
#include <unistd.h>
#include <iostream>


int CTF_alloc_ptr(int64_t len, void ** const ptr);
int CTF_mst_alloc_ptr(int64_t len, void ** const ptr);
void * CTF_alloc(int64_t len);
void * CTF_mst_alloc(int64_t len);
int CTF_free(void * ptr, int const tid);
int CTF_free(void * ptr);



namespace CTF {

  /**
   * \brief reduction types for tensor data
   */
  enum OP { OP_SUM, OP_SUMABS,
            OP_NORM1, OP_NORM2, OP_NORM_INFTY,
            OP_MAX, OP_MIN, OP_MAXABS, OP_MINABS};

  enum { SUCCESS, ERROR, NEGATIVE };

  int conv_idx(int          ndim,
               char const * cidx,
               int **       iidx);

  int conv_idx(int          ndim_A,
               char const * cidx_A,
               int **       iidx_A,
               int          ndim_B,
               char const * cidx_B,
               int **       iidx_B);

  int conv_idx(int          ndim_A,
               char const * cidx_A,
               int **       iidx_A,
               int          ndim_B,
               char const * cidx_B,
               int **       iidx_B,
               int          ndim_C,
               char const * cidx_C,
               int **       iidx_C);


}

namespace CTF_int {

  void csgemm(char transa,      char transb,
              int m,            int n,
              int k,            float a,
              float const * A,  int lda,
              float const * B,  int ldb,
              float b,          float * C,
                                int ldc);

  void cdgemm(char transa,      char transb,
              int m,            int n,
              int k,            double a,
              double const * A, int lda,
              double const * B, int ldb,
              double b,         double * C,
                                int ldc);

  void ccgemm(char transa,                    char transb,
              int m,                          int n,
              int k,                          std::complex<float> a,
              const std::complex<float> * A,  int lda,
              const std::complex<float> * B,  int ldb,
              std::complex<float> b,          std::complex<float> * C,
                                              int ldc);


  void czgemm(char transa,                    char transb,
              int m,                          int n,
              int k,                          std::complex<double> a,
              const std::complex<double> * A, int lda,
              const std::complex<double> * B, int ldb,
              std::complex<double> b,         std::complex<double> * C,
                                              int ldc);

  void csaxpy(int n,              float  dA,
              const float  * dX,  int incX,
              float  * dY,        int incY);

  void cdaxpy(int n,              double dA,
              const double * dX,  int incX,
              double * dY,        int incY);

  void ccaxpy(int n,                            std::complex<float> dA,
              const std::complex<float> * dX,   int incX,
              std::complex<float> * dY,         int incY);

  void czaxpy(int n,                            std::complex<double> dA,
              const std::complex<double> * dX,  int incX,
              std::complex<double> * dY,        int incY);

  void csscal(int n, float dA, float * dX, int incX);

  void cdscal(int n, double dA, double * dX, int incX);

  void ccscal(int n, std::complex<float> dA, std::complex<float> * dX, int incX);

  void czscal(int n, std::complex<double> dA, std::complex<double> * dX, int incX);

  void conv_idx(int          ndim,
                int const *  lens,
                int64_t      idx,
                int **       idx_arr);

  void conv_idx(int          ndim,
                int const *  lens,
                int64_t      idx,
                int *        idx_arr);

  void conv_idx(int          ndim,
                int const *  lens,
                int const *  idx_arr,
                int64_t *    idx);


  class CommData {
    public:
    MPI_Comm cm;
    int np;
    int rank;
    int color;
    int alive;
   
    double estimate_bcast_time(int64_t msg_sz);
 
    double estimate_allred_time(int64_t msg_sz);
   
    double estimate_alltoall_time(int64_t chunk_sz);
   
    double estimate_alltoallv_time(int64_t tot_sz);
  };
}

  #endif
