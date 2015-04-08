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

typedef int64_t int64_t;
volatile static int64_t int64_t_max = INT64_MAX;
#include "offload.h"

#ifndef ASSERT
#if ENABLE_ASSERT
#define ASSERT(...)                \
do { if (!(__VA_ARGS__)) handler(); assert(__VA_ARGS__); } while (0)
#else
#define ASSERT(...) do {} while(0 && (__VA_ARGS__))
#endif
#endif

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
    ASSERT(ndev > 0);
    cublasStatus_t status = cublasCreate(&cuhandle);
    ASSERT(status == CUBLAS_STATUS_SUCCESS);
  }
  initialized = 1;
}

void offload_exit(){
  if (initialized){
    cublasStatus_t status = cublasDestroy(cuhandle);
    ASSERT(status == CUBLAS_STATUS_SUCCESS);
    initialized = 0;
  }
}

template <typename dtype>
offload_ptr<dtype>::offload_ptr(int el_size_, int64_t size_){
  el_size = el_size_;
  size = size_;
  cudaError_t err = cudaMalloc((void**)&dev_ptr, size_*el_size);
  ASSERT(err == cudaSuccess);
}

offload_ptr::~offload_ptr(){
  cudaError_t err = cudaFree(dev_ptr);
  ASSERT(err == cudaSuccess);
}

void offload_ptr::download(dtype * host_ptr){
  TAU_FSTART(cuda_download);
  cudaError_t err = cudaMemcpy(host_ptr, dev_ptr, size*el_size,
                               cudaMemcpyDeviceToHost);
  TAU_FSTOP(cuda_download);
  ASSERT(err == cudaSuccess);
}

void offload_ptr<dtype>::upload(char const * host_ptr){
  TAU_FSTART(cuda_upload);
  cudaError_t err = cudaMemcpy(dev_ptr, host_ptr, size*el_size,
                               cudaMemcpyHostToDevice);
  TAU_FSTOP(cuda_upload);
  ASSERT(err == cudaSuccess);
}

/*
template <typename dtype>
__global__ void gset_zero(dtype *arr, int64_t size, dtype val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i=idx; i<size; i+= gridDim.x*blockDim.x) {
    arr[i]=val;
  }
}

void offload_ptr<dtype>::set_zero(){
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / (size);
  gset_zero<<<blockSize, numBlocks>>>(dev_ptr, size, get_zero<dtype>());
}
*/

void host_pinned_alloc(void ** ptr, int64_t size){
  cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocMapped);
  ASSERT(err == cudaSuccess);
}

void host_pinned_free(void * ptr){
  cudaError_t err = cudaFreeHost(ptr);
  ASSERT(err == cudaSuccess);
}

/*template <typename dtype>
void offload_gemm(char                  tA,
                  char                  tB,
                  int                   m,
                  int                   n,
                  int                   k,
                  dtype                 alpha,
                  offload_ptr<dtype> &  A,
                  int                   lda_A,
                  offload_ptr<dtype> &  B,
                  int                   lda_B,
                  dtype                 beta,
                  offload_ptr<dtype> &  C,
                  int                   lda_C){
  TAU_FSTART(cuda_gemm);
  offload_gemm(tA, tB, m, n, k, alpha, A.dev_ptr, lda_A, B.dev_ptr, lda_B, beta, C.dev_ptr, lda_C);
  TAU_FSTOP(cuda_gemm);
}
template 
void offload_gemm(char                  tA,
                  char                  tB,
                  int                   m,
                  int                   n,
                  int                   k,
                  double                alpha,
                  offload_ptr<double> & A,
                  int                   lda_A,
                  offload_ptr<double> & B,
                  int                   lda_B,
                  double                beta,
                  offload_ptr<double> & C,
                  int                   lda_C);
template 
void offload_gemm(char                                  tA,
                  char                                  tB,
                  int                                   m,
                  int                                   n,
                  int                                   k,
                  std::complex<double>                  alpha,
                  offload_ptr< std::complex<double> > & A,
                  int                                   lda_A,
                  offload_ptr< std::complex<double> > & B,
                  int                                   lda_B,
                  std::complex<double>                  beta,
                  offload_ptr< std::complex<double> > & C,
                  int                                   lda_C);
template <>
void offload_gemm<double>(char                  tA,
                          char                  tB,
                          int                   m,
                          int                   n,
                          int                   k,
                          double                alpha,
                          double const        * dev_A,
                          int                   lda_A,
                          double const        * dev_B,
                          int                   lda_B,
                          double                beta,
                          double              * dev_C,
                          int                   lda_C){
  ASSERT(initialized);

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
#ifdef PROFILE
  cudaDeviceSynchronize();
#endif
  
  ASSERT(status == CUBLAS_STATUS_SUCCESS);
}
*/

/*
template <>
void offload_gemm< std::complex<double> >(
                         char                                  tA,
                         char                                  tB,
                         int                                   m,
                         int                                   n,
                         int                                   k,
                         std::complex<double>                  alpha,
                         std::complex<double> const          * dev_A,
                         int                                   lda_A,
                         std::complex<double> const          * dev_B,
                         int                                   lda_B,
                         std::complex<double>                  beta,
                         std::complex<double>                * dev_C,
                         int                                   lda_C){
  ASSERT(initialized);
  
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
  
  ASSERT(status == CUBLAS_STATUS_SUCCESS);
  assert(status == CUBLAS_STATUS_SUCCESS);
}
*/
#endif
