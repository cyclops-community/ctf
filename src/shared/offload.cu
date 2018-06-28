/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#include <complex>
#include <assert.h>
#include <stdio.h>
#include "int_timer.h"
#include <stdint.h>

#include "offload.h"
#include "../tensor/algstrct.h"
#include "../interface/timer.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "device_launch_parameters.h"
#endif

namespace CTF_int{
  #ifndef PROFILE
  #define TAU_PROFILE(NAME,ARG,USER)
  #define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)
  #define TAU_PROFILER_CREATE(ARG1, ARG2, ARG3, ARG4)
  #define TAU_PROFILE_STOP(ARG)
  #define TAU_PROFILE_START(ARG)
  #define TAU_PROFILE_SET_NODE(ARG)
  #define TAU_PROFILE_SET_CONTEXT(ARG)
  #define TAU_FSTART(ARG)
  #define TAU_FSTOP(ARG)
  #endif

  #define ABORT                                   \
    do{                                           \
     assert(0); } while (0)

#ifdef USE_CUDA
  int initialized = 0;
  cublasHandle_t cuhandle;

  void offload_init(){
    if (!initialized){
      int ndev=0;
      cudaError_t err = cudaGetDeviceCount(&ndev);
      assert(err == cudaSuccess);
      assert(ndev > 0);
      cublasStatus_t status = cublasCreate(&cuhandle);
      assert(status == CUBLAS_STATUS_SUCCESS);
    }
    initialized = 1;
  }

  void offload_exit(){
    if (initialized){
      cublasStatus_t status = cublasDestroy(cuhandle);
      assert(status == CUBLAS_STATUS_SUCCESS);
      initialized = 0;
    }
  }

  offload_tsr::offload_tsr(algstrct const * sr_, int64_t size_) : offload_arr(size_*sr_->el_size) {
    sr = sr_;
    size = size_;
  }

  /*offload_tsr::~offload_tsr(){
  }*/

  LinModel<2> upload_mdl(upload_mdl_init,"upload_mdl");
  LinModel<2> download_mdl(download_mdl_init,"download_mdl");

  double estimate_download_time(int64_t size){
    double ps[] = {1.0, (double)size};
    return download_mdl.est_time(ps);
  }

  double estimate_upload_time(int64_t size){
    double ps[] = {1.0, (double)size};
    return upload_mdl.est_time(ps);
  }


  template <typename dtype>
  __global__ void gset_zero(dtype *arr, int64_t size, dtype val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i=idx; i<size; i+= gridDim.x*blockDim.x) {
      arr[i]=val;
    }
  }

  void offload_tsr::set_zero(){
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / (size);
    TAU_FSTART(set_zero);
    switch (sr->el_size){
      case 4:
        gset_zero<<<blockSize, numBlocks>>>((float*)dev_spr, size, ((float*)sr->addid())[0]);
        break;
      case 8:
        gset_zero<<<blockSize, numBlocks>>>((double*)dev_spr, size, ((double*)sr->addid())[0]);
        break;
      case 16:
        gset_zero<<<blockSize, numBlocks>>>((std::complex<double>*)dev_spr, size, ((std::complex<double>*)sr->addid())[0]);
        break;
      default:
        assert(0);
        break;
    }
    TAU_FSTOP(set_zero);
  }


  offload_arr::offload_arr(int64_t nbytes_){
    nbytes = nbytes_;
    TAU_FSTART(offload_malloc);
    cudaError_t err = cudaMalloc((void**)&dev_spr, nbytes);
    TAU_FSTOP(offload_malloc);
    assert(err == cudaSuccess);
  }

  offload_arr::~offload_arr(){
    TAU_FSTART(offload_free);
    cudaError_t err = cudaFree(dev_spr);
    TAU_FSTOP(offload_free);
    assert(err == cudaSuccess);
  }


  void offload_arr::download(char * host_spr){
     // not-quite-sure
    assert(initialized);
    TAU_FSTART(cuda_download);
    double st_time = MPI_Wtime();
    cudaError_t err = cudaMemcpy(host_spr, dev_spr, nbytes,
                                 cudaMemcpyDeviceToHost);
    double exe_time = MPI_Wtime()-st_time;
    double tps[] = {exe_time, 1.0, (double)nbytes};
    download_mdl.observe(tps);
    TAU_FSTOP(cuda_download);
    assert(err == cudaSuccess);
  }

  void offload_arr::upload(char const * host_spr){
     // not-quite-sure
    TAU_FSTART(cuda_upload);
    double st_time = MPI_Wtime();
    cudaError_t err = cudaMemcpy(dev_spr, host_spr, nbytes,
                                 cudaMemcpyHostToDevice);

    double exe_time = MPI_Wtime()-st_time;
    double tps[] = {exe_time, 1.0, (double)nbytes};
    upload_mdl.observe(tps);
    TAU_FSTOP(cuda_upload);
    assert(err == cudaSuccess);
  }



  void host_pinned_alloc(void ** ptr, int64_t size){
    TAU_FSTART(host_pinned_malloc);
    cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocMapped);
    TAU_FSTOP(host_pinned_malloc);
    assert(err == cudaSuccess);
  }

  void host_pinned_free(void * ptr){
    TAU_FSTART(host_pinned_free);
    cudaError_t err = cudaFreeHost(ptr);
    TAU_FSTOP(host_pinned_free);
    assert(err == cudaSuccess);
  }
#endif

  template
  void offload_gemm(char          tA,
                    char          tB,
                    int           m,
                    int           n,
                    int           k,
                    double        alpha,
                    offload_tsr & A,
                    int           lda_A,
                    offload_tsr & B,
                    int           lda_B,
                    double        beta,
                    offload_tsr & C,
                    int           lda_C);

  template <>
  void offload_gemm<float>(char           tA,
                            char           tB,
                            int            m,
                            int            n,
                            int            k,
                            float         alpha,
                            float const * dev_A,
                            int            lda_A,
                            float const * dev_B,
                            int            lda_B,
                            float         beta,
                            float *       dev_C,
                            int            lda_C){
  #ifdef USE_CUDA
    assert(initialized);
  
    cublasOperation_t cuA;  
    switch (tA){
      case 'n':
      case 'N':
        cuA = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuA = CUBLAS_OP_T;
        break;
    }  
  
    cublasOperation_t cuB;
    switch (tB){
      case 'n':
      case 'N':
        cuB = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuB = CUBLAS_OP_T;
        break;
    }  
  
    cublasStatus_t status = 
      cublasDgemm(cuhandle, cuA, cuB, m, n, k, &alpha, 
                  dev_A, lda_A, 
                  dev_B, lda_B, &beta, 
                  dev_C, lda_C);
  #ifdef PROFILE
    cudaDeviceSynchronize();
  #endif
    
    assert(status == CUBLAS_STATUS_SUCCESS);
  #endif  
  }
  
  
  template <>
  void offload_gemm< std::complex<float> >(
                           char                         tA,
                           char                         tB,
                           int                          m,
                           int                          n,
                           int                          k,
                           std::complex<float>         alpha,
                           std::complex<float> const * dev_A,
                           int                          lda_A,
                           std::complex<float> const * dev_B,
                           int                          lda_B,
                           std::complex<float>         beta,
                           std::complex<float> *       dev_C,
                           int                          lda_C){
  #ifdef USE_CUDA
    assert(initialized);
    
    cublasOperation_t cuA;  
    switch (tA){
      case 'n':
      case 'N':
        cuA = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuA = CUBLAS_OP_T;
        break;
      case 'c':
      case 'C':
        cuA = CUBLAS_OP_C;
        break;
    }  
  
    cublasOperation_t cuB;
    switch (tB){
      case 'n':
      case 'N':
        cuB = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuB = CUBLAS_OP_T;
        break;
      case 'c':
      case 'C':
        cuB = CUBLAS_OP_C;
        break;
    }  
  
    TAU_FSTART(cublas_zgemm);
    cublasStatus_t status = 
      cublasZgemm(cuhandle, cuA, cuB, m, n, k, 
                  reinterpret_cast<cuDoubleComplex*>(&alpha), 
                  reinterpret_cast<const cuDoubleComplex*>(dev_A), lda_A, 
                  reinterpret_cast<const cuDoubleComplex*>(dev_B), lda_B, 
                  reinterpret_cast<cuDoubleComplex*>(&beta), 
                  reinterpret_cast<cuDoubleComplex*>(dev_C), lda_C);
  #ifdef PROFILE
    cudaDeviceSynchronize();
  #endif
    TAU_FSTOP(cublas_zgemm);
    
    assert(status == CUBLAS_STATUS_SUCCESS);
    assert(status == CUBLAS_STATUS_SUCCESS);
  #endif
  }

  template <>
  void offload_gemm<double>(char           tA,
                            char           tB,
                            int            m,
                            int            n,
                            int            k,
                            double         alpha,
                            double const * dev_A,
                            int            lda_A,
                            double const * dev_B,
                            int            lda_B,
                            double         beta,
                            double *       dev_C,
                            int            lda_C){
  #ifdef USE_CUDA
    assert(initialized);

    cublasOperation_t cuA;
    switch (tA){
      case 'n':
      case 'N':
        cuA = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuA = CUBLAS_OP_T;
        break;
    }

    cublasOperation_t cuB;
    switch (tB){
      case 'n':
      case 'N':
        cuB = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuB = CUBLAS_OP_T;
        break;
    }

    cublasStatus_t status =
      cublasDgemm(cuhandle, cuA, cuB, m, n, k, &alpha,
                  dev_A, lda_A,
                  dev_B, lda_B, &beta,
                  dev_C, lda_C);
  #ifdef PROFILE
    cudaDeviceSynchronize();
  #endif

    assert(status == CUBLAS_STATUS_SUCCESS);
  #endif
  }


  template <>
  void offload_gemm< std::complex<double> >(
                           char                         tA,
                           char                         tB,
                           int                          m,
                           int                          n,
                           int                          k,
                           std::complex<double>         alpha,
                           std::complex<double> const * dev_A,
                           int                          lda_A,
                           std::complex<double> const * dev_B,
                           int                          lda_B,
                           std::complex<double>         beta,
                           std::complex<double> *       dev_C,
                           int                          lda_C){
  #ifdef USE_CUDA
    assert(initialized);

    cublasOperation_t cuA;
    switch (tA){
      case 'n':
      case 'N':
        cuA = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuA = CUBLAS_OP_T;
        break;
      case 'c':
      case 'C':
        cuA = CUBLAS_OP_C;
        break;
    }

    cublasOperation_t cuB;
    switch (tB){
      case 'n':
      case 'N':
        cuB = CUBLAS_OP_N;
        break;
      case 't':
      case 'T':
        cuB = CUBLAS_OP_T;
        break;
      case 'c':
      case 'C':
        cuB = CUBLAS_OP_C;
        break;
    }

    TAU_FSTART(cublas_zgemm);
    cublasStatus_t status =
      cublasZgemm(cuhandle, cuA, cuB, m, n, k,
                  reinterpret_cast<cuDoubleComplex*>(&alpha),
                  reinterpret_cast<const cuDoubleComplex*>(dev_A), lda_A,
                  reinterpret_cast<const cuDoubleComplex*>(dev_B), lda_B,
                  reinterpret_cast<cuDoubleComplex*>(&beta),
                  reinterpret_cast<cuDoubleComplex*>(dev_C), lda_C);
  #ifdef PROFILE
    cudaDeviceSynchronize();
  #endif
    TAU_FSTOP(cublas_zgemm);

    assert(status == CUBLAS_STATUS_SUCCESS);
    assert(status == CUBLAS_STATUS_SUCCESS);
  #endif
  }

  template <typename dtype>
  void offload_gemm(char           tA,
                    char           tB,
                    int            m,
                    int            n,
                    int            k,
                    dtype          alpha,
                    offload_tsr &  A,
                    int            lda_A,
                    offload_tsr &  B,
                    int            lda_B,
                    dtype          beta,
                    offload_tsr &  C,
                    int            lda_C){
    TAU_FSTART(cuda_gemm);
    offload_gemm(tA, tB, m, n, k, alpha, (dtype*)A.dev_spr, lda_A, (dtype*)B.dev_spr, lda_B, beta, (dtype*)C.dev_spr, lda_C);
    TAU_FSTOP(cuda_gemm);
  }
}
