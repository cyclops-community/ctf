/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../shared/util.h"
#include "offload.h"

int initialized = 0;
cublasHandle_t cuhandle;

void offload_init(){
  if (!initialized){
/*    int dev = findCudaDevice();
    LIBT_ASSERT(dev != -1):*/
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
                          offload_ptr<double> & A,
                          int                   lda_A,
                          offload_ptr<double> & B,
                          int                   lda_B,
                          double                beta,
                          offload_ptr<double> & C,
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

  cublasStatus_t status = 
    cublasDgemm(cuhandle, cuA, cuB, m, n, k, &alpha, 
                A.dev_ptr, lda_A, 
                B.dev_ptr, lda_B, &beta, 
                C.dev_ptr, lda_C);
  
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
                         offload_ptr< std::complex<double> > & A,
                         int                                   lda_A,
                         offload_ptr< std::complex<double> > & B,
                         int                                   lda_B,
                         std::complex<double>                  beta,
                         offload_ptr< std::complex<double> > & C,
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

  cublasStatus_t status = 
    cublasZgemm(cuhandle, cuA, cuB, m, n, k, 
                reinterpret_cast<cuDoubleComplex*>(&alpha), 
                reinterpret_cast<cuDoubleComplex*>(A.dev_ptr), lda_A, 
                reinterpret_cast<cuDoubleComplex*>(B.dev_ptr), lda_B, 
                reinterpret_cast<cuDoubleComplex*>(&beta), 
                reinterpret_cast<cuDoubleComplex*>(C.dev_ptr), lda_C);
  
  LIBT_ASSERT(status == CUBLAS_STATUS_SUCCESS);
}

template class offload_ptr<double>;
template class offload_ptr< std::complex<double> >;


#endif
