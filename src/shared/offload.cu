/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <complex>
#include <assert.h>
#include <stdio.h>
//#include "../shared/util.h"
#include "device_launch_parameters.h"

template<typename dtype>
dtype get_zero(){
  assert(0);
}
template<> inline
double get_zero<double>() { return 0.0; }

template<> inline
std::complex<double> get_zero< std::complex<double> >() { return std::complex<double>(0.0,0.0); }


typedef int64_t long_int;
#ifndef LIBT_ASSERT
#ifdef DEBUG
#define LIBT_ASSERT(...)                \
do { assert(__VA_ARGS__); } while (0)
#else
#define LIBT_ASSERT(...) do {} while(0 && (__VA_ARGS__))
#endif
#endif

#include "offload.h"

int initialized = 0;
cublasHandle_t cuhandle;

void offload_init(){
  if (!initialized){
    int ndev=0;
    cudaGetDeviceCount(&ndev);
    LIBT_ASSERT(ndev > 0);
    cublasStatus_t status = cublasCreate(&cuhandle);
    LIBT_ASSERT(status == CUBLAS_STATUS_SUCCESS);
  }
  initialized = 1;
}

void offload_exit(){
  if (initialized){
    cublasStatus_t status = cublasDestroy(cuhandle);
    LIBT_ASSERT(status == CUBLAS_STATUS_SUCCESS);
    initialized = 0;
  }
}


/**
 * \brief allocates offload device pointer
 * \param[in] size number of elements to create for buffer
 */
template <typename dtype>
offload_ptr<dtype>::offload_ptr(long_int size_){
  size = size_;
  cudaError_t err = cudaMalloc((void**)&dev_ptr, size_*sizeof(dtype));
  LIBT_ASSERT(err == cudaSuccess);
}

/**
 * \brief deallocates offload device pointer
 */
template <typename dtype>
offload_ptr<dtype>::~offload_ptr(){
  cudaError_t err = cudaFree(dev_ptr);
  LIBT_ASSERT(err == cudaSuccess);
}

/**
 * \brief downloads all data from device pointer to host pointer
 * \param[in,out] host_ptr preallocated host buffer to download to
 */
template <typename dtype>
void offload_ptr<dtype>::download(dtype * host_ptr){
  cudaError_t err = cudaMemcpy(host_ptr, dev_ptr, size*sizeof(dtype),
                               cudaMemcpyDeviceToHost);
  LIBT_ASSERT(err == cudaSuccess);
}
/**
 * \brief uploads all data to device pointer from host pointer
 * \param[in] host_ptr preallocated host buffer to upload from
 */
template <typename dtype>
void offload_ptr<dtype>::upload(dtype const * host_ptr){
  cudaError_t err = cudaMemcpy(dev_ptr, host_ptr, size*sizeof(dtype),
                               cudaMemcpyHostToDevice);
  LIBT_ASSERT(err == cudaSuccess);
}


template <typename dtype>
__global__ void gset_zero(dtype *arr, int64_t size, dtype val) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i=idx; i<size; i+= gridDim.x*blockDim.x) {
    arr[i]=val;
  }
}

/**
 * \brief set array to 0
 */
template <typename dtype>
void offload_ptr<dtype>::set_zero(){
  int blockSize = 256;
  int numBlocks = (size + blockSize - 1) / (size);
  gset_zero<<<blockSize, numBlocks>>>(dev_ptr, size, get_zero<dtype>());
}


void host_pinned_alloc(void ** ptr, long_int size){
  cudaError_t err = cudaHostAlloc(ptr, size, cudaHostAllocMapped);
  LIBT_ASSERT(err == cudaSuccess);
}

void host_pinned_free(void * ptr){
  cudaError_t err = cudaFreeHost(ptr);
  LIBT_ASSERT(err == cudaSuccess);
}

/**
 * \brief performs an offloaded gemm using device pointer of objects
 *        specialized instantization to double
 */
template <typename dtype>
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
  offload_gemm(tA, tB, m, n, k, alpha, A.dev_ptr, lda_A, B.dev_ptr, lda_B, beta, C.dev_ptr, lda_C);
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

/**
 * \brief performs an offloaded gemm using device pointer of objects
 *        specialized instantization to double
 */
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
  LIBT_ASSERT(initialized);

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
  
  LIBT_ASSERT(status == CUBLAS_STATUS_SUCCESS);
}

/**
 * \brief performs an offloaded gemm using device pointer of objects
 *        specialized instantization to complex<double>
 */
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
  LIBT_ASSERT(initialized);
  
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

  //printf("offloading zgemm\n");
  cublasStatus_t status = 
    cublasZgemm(cuhandle, cuA, cuB, m, n, k, 
                reinterpret_cast<cuDoubleComplex*>(&alpha), 
                reinterpret_cast<const cuDoubleComplex*>(dev_A), lda_A, 
                reinterpret_cast<const cuDoubleComplex*>(dev_B), lda_B, 
                reinterpret_cast<cuDoubleComplex*>(&beta), 
                reinterpret_cast<cuDoubleComplex*>(dev_C), lda_C);
  
  LIBT_ASSERT(status == CUBLAS_STATUS_SUCCESS);
}

template class offload_ptr<double>;
template class offload_ptr< std::complex<double> >;
#endif
