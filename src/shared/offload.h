/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifndef __OFFLOAD_H__
#define __OFFLOAD_H__

//#include "../interface/common.h"

namespace CTF_int{
  class algstrct;
  
  /** \brief initialize offloading, e.g. create cublas */
  void offload_init();
  /** \brief exit offloading, e.g. destroy cublas */
  void offload_exit();

  /** \brief estimate time it takes to upload */
  double estimate_download_time(int64_t size);

  /** \brief estimate time it takes to download */
  double estimate_upload_time(int64_t size);
 
  /** \brief offloaded array/buffer */
  class offload_arr {
    public:
      /** \brief device pointer */
      char * dev_spr;
      /** \brief number of bytes */
      int64_t nbytes;
      
      /**
       * \brief constructor allocates device buffer
       * \param[in] nbytes number of elements
       */
      offload_arr(int64_t nbytes);
  
      /**
       * \brief destructor allocates device buffer
       */
      ~offload_arr();
  
      /**
       * \brief read data from device to host pointer
       * \param[in,out] host_spr (should be preallocated)
       */
      void download(char * host_spr);
  
      /**
       * \brief write data from host to device
       * \param[in] host_spr
       */
      void upload(char const * host_spr);
  };

  /** \brief offloaded and serialized tensor data */
  class offload_tsr : public offload_arr {
    public:
      /** \brief algebraic structure */
      algstrct const * sr;
      /** \brief number of elements */
      int64_t size;
      
      /**
       * \brief constructor allocates device buffer
       * \param[in] sr algebraic structure
       * \param[in] size number of elements
       */
      offload_tsr(algstrct const * sr, int64_t size);

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
                    offload_tsr & A,
                    int           lda_A,
                    offload_tsr & B,
                    int           lda_B,
                    dtype         beta,
                    offload_tsr & C,
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

