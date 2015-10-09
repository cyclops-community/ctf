/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifndef __OFFLOAD_H__
#define __OFFLOAD_H__

#include "../interface/common.h"
#include "../tensor/algstrct.h"

namespace CTF_int{

  /** \brief initialize offloading, e.g. create cublas */
  void offload_init();
  /** \brief exit offloading, e.g. destroy cublas */
  void offload_exit();
  
  class offload_ptr {
    public:
      /** \brief device pointer */
      char * dev_ptr;
      /** \brief algebraic structure */
      algstrct const * sr;
      /** \brief number of elements */
      int64_t size;
  
      
      /**
       * \brief constructor allocates device buffer
       * \param[in] sr algebraic structure
       * \param[in] size number of elements
       */
      offload_ptr(algstrct const * sr, int64_t size);
  
      /**
       * \brief destructor allocates device buffer
       */
      ~offload_ptr();
  
      /**
       * \brief read data from device to host pointer
       * \param[in,out] host_ptr (should be preallocated)
       */
      void download(char * host_ptr);
  
      /**
       * \brief write data from host to device
       * \param[in] host_ptr
       */
      void upload(char const * host_ptr);
   
      /**
       * \brief set device data to zero
       */
      void set_zero();
  };
  
  /**
   * \brief allocate a pinned host buffer
   * \param[out] ptr pointer to define
   * \param[in] size amount of buffer space to allocate
   */
  void host_pinned_alloc(void ** ptr, int64_t size);
  
  /**
   * \brief free a pinned host buffer
   * \param[in] ptr pointer to free
   */
  void host_pinned_free(void * ptr);
  
  template <typename dtype>
  void offload_gemm(char          tA,
                    char          tB,
                    int           m,
                    int           n,
                    int           k,
                    dtype         alpha,
                    offload_ptr & A,
                    int           lda_A,
                    offload_ptr & B,
                    int           lda_B,
                    dtype         beta,
                    offload_ptr & C,
                    int           lda_C);
  
  template <typename dtype>
  void offload_gemm(char                 tA,
                    char                 tB,
                    int                  m,
                    int                  n,
                    int                  k,
                    dtype                alpha,
                    dtype const        * dev_A,
                    int                  lda_A,
                    dtype const        * dev_B,
                    int                  lda_B,
                    dtype                beta,
                    dtype              * dev_C,
                    int                  lda_C);
}  
#endif

