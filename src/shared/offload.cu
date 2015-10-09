/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>
#include <assert.h>
#include <stdio.h>
//#include "../shared/util.h"
#include "device_launch_parameters.h"
#include "int_timer.h"
#include <stdint.h>

#include "offload.h"
#include "../tensor/algstrct.h"

namespace CTF_int{
  volatile static int64_t int64_t_max = INT64_MAX;
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
  
  int initialized = 0;
  cublasHandle_t cuhandle;
  
  void offload_init(){
    if (!initialized){
      int ndev=0;
      cudaGetDeviceCount(&ndev);
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
  
  offload_ptr::offload_ptr(algstrct const * sr_, int64_t size_){
    sr = sr_;
    size = size_;
    cudaError_t err = cudaMalloc((void**)&dev_ptr, size*sr->el_size);
    printf("allocated dev %p size %ld\n", dev_ptr,size*sr->el_size);
    assert(err == cudaSuccess);
  }
  
  offload_ptr::~offload_ptr(){
    cudaError_t err = cudaFree(dev_ptr);
    assert(err == cudaSuccess);
  }
  
  void offload_ptr::download(char * host_ptr){
    assert(initialized);
    TAU_FSTART(cuda_download);
    cudaError_t err = cudaMemcpy(host_ptr, dev_ptr, size*sr->el_size,
                                 cudaMemcpyDeviceToHost);
    TAU_FSTOP(cuda_download);
    printf("err = %d size = %ld dev ptr = %p host ptr = %p\n",err,size*sr->el_size,dev_ptr,host_ptr);
    assert(err == cudaSuccess);
  }
  
  void offload_ptr::upload(char const * host_ptr){
    TAU_FSTART(cuda_upload);
    cudaError_t err = cudaMemcpy(dev_ptr, host_ptr, size*sr->el_size,
                                 cudaMemcpyHostToDevice);
    TAU_FSTOP(cuda_upload);
    assert(err == cudaSuccess);
  }
  
  
  template <typename dtype>
  __global__ void gset_zero(dtype *arr, int64_t size, dtype val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
  
    for (int i=idx; i<size; i+= gridDim.x*blockDim.x) {
      arr[i]=val;
    }
  }
  
  void offload_ptr::set_zero(){
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / (size);
    switch (sr->el_size){
      case 4:
        gset_zero<<<blockSize, numBlocks>>>((float*)dev_ptr, size, ((float*)sr->addid())[0]);
        break;
      case 8:
        gset_zero<<<blockSize, numBlocks>>>((double*)dev_ptr, size, ((double*)sr->addid())[0]);
        break;
      case 16:
        gset_zero<<<blockSize, numBlocks>>>((std::complex<double>*)dev_ptr, size, ((std::complex<double>*)sr->addid())[0]);
        break;
      default:
        assert(0);
        break;
    }
  }
  
  void host_pinned_alloc(void ** ptr, int64_t size){
    cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocMapped);
    assert(err == cudaSuccess);
    printf("made pinned alloc ptr = %p size = %ld\n", *ptr, size);
  }
  
  void host_pinned_free(void * ptr){
    cudaError_t err = cudaFreeHost(ptr);
    assert(err == cudaSuccess);
  }
  
  template <typename dtype>
  void offload_gemm(char           tA,
                    char           tB,
                    int            m,
                    int            n,
                    int            k,
                    dtype          alpha,
                    offload_ptr &  A,
                    int            lda_A,
                    offload_ptr &  B,
                    int            lda_B,
                    dtype          beta,
                    offload_ptr &  C,
                    int            lda_C){
    TAU_FSTART(cuda_gemm);
    offload_gemm(tA, tB, m, n, k, alpha, (dtype*)A.dev_ptr, lda_A, (dtype*)B.dev_ptr, lda_B, beta, (dtype*)C.dev_ptr, lda_C);
    TAU_FSTOP(cuda_gemm);
  }
  template 
  void offload_gemm(char          tA,
                    char          tB,
                    int           m,
                    int           n,
                    int           k,
                    double        alpha,
                    offload_ptr & A,
                    int           lda_A,
                    offload_ptr & B,
                    int           lda_B,
                    double        beta,
                    offload_ptr & C,
                    int           lda_C);
  template 
  void offload_gemm(char                 tA,
                    char                 tB,
                    int                  m,
                    int                  n,
                    int                  k,
                    std::complex<double> alpha,
                    offload_ptr &        A,
                    int                  lda_A,
                    offload_ptr &        B,
                    int                  lda_B,
                    std::complex<double> beta,
                    offload_ptr &        C,
                    int                  lda_C);
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
  
    //printf("offloading dgemm\n");
    cublasStatus_t status = 
      cublasDgemm(cuhandle, cuA, cuB, m, n, k, &alpha, 
                  dev_A, lda_A, 
                  dev_B, lda_B, &beta, 
                  dev_C, lda_C);
  //#ifdef PROFILE
    cudaDeviceSynchronize();
  //#endif
    
    assert(status == CUBLAS_STATUS_SUCCESS);
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
  //#ifdef PROFILE
    cudaDeviceSynchronize();
  //#endif
    TAU_FSTOP(cublas_zgemm);
    
    assert(status == CUBLAS_STATUS_SUCCESS);
    assert(status == CUBLAS_STATUS_SUCCESS);
  }
  
}
#endif
