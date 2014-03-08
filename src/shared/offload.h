/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/

#ifndef __OFFLOAD_H__
#define __OFFLOAD_H__


void offload_init();
void offload_exit();

template <typename dtype>
class offload_ptr {
  public:
    dtype * dev_ptr;
    long_int size;

    offload_ptr(long_int size_);
    ~offload_ptr();

    void download(dtype * host_ptr);
    void upload(dtype const * host_ptr);
    void set_zero();
};

void host_pinned_alloc(void ** ptr, long_int size);
void host_pinned_free(void * ptr);

template <typename dtype>
void offload_gemm( char                 tA,
                   char                 tB,
                   int                  m,
                   int                  n,
                   int                  k,
                   dtype                alpha,
                   offload_ptr<dtype> & A,
                   int                  lda_A,
                   offload_ptr<dtype> & B,
                   int                  lda_B,
                   dtype                beta,
                   offload_ptr<dtype> & C,
                   int                  lda_C);

template <typename dtype>
void offload_gemm( char                 tA,
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

#endif

