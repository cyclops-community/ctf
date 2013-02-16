/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include "../shared/util.h"
#include "ctr_comm.h"
#include "ctr_tsr.h"

#ifdef USE_OMP
#include <omp.h>
#endif
#ifndef VIRT_NTD
#define VIRT_NTD        1
#endif

/**
 * \brief deallocates ctr_virt object
 */
template<typename dtype>
ctr_virt<dtype>::~ctr_virt() {
  free(virt_dim);
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_virt<dtype>::ctr_virt(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_virt<dtype> * o   = (ctr_virt<dtype>*)other;
  rec_ctr       = o->rec_ctr->clone();
  num_dim       = o->num_dim;
  virt_dim      = (int*)malloc(sizeof(int)*num_dim);
  memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

  ndim_A        = o->ndim_A;
  blk_sz_A      = o->blk_sz_A;
  idx_map_A     = o->idx_map_A;

  ndim_B        = o->ndim_B;
  blk_sz_B      = o->blk_sz_B;
  idx_map_B     = o->idx_map_B;

  ndim_C        = o->ndim_C;
  blk_sz_C      = o->blk_sz_C;
  idx_map_C     = o->idx_map_C;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_virt<dtype>::clone() {
  return new ctr_virt<dtype>(this);
}

/**
 * \brief prints ctr object
 */
template<typename dtype>
void ctr_virt<dtype>::print() {
  int i;
  printf("ctr_virt:\n");
  printf("blk_sz_A = %lld, blk_sz_B = %lld, blk_sz_C = %lld\n",
          blk_sz_A, blk_sz_B, blk_sz_C);
  for (i=0; i<num_dim; i++){
    printf("virt_dim[%d] = %d\n", i, virt_dim[i]);
  }
}




/**
 * \brief returns the number of bytes of buffer space
   we need
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_virt<dtype>::mem_fp(){
  return (6+VIRT_NTD)*num_dim*sizeof(int);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_virt<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief iterates over the dense virtualization block grid and contracts
 */
template<typename dtype>
void ctr_virt<dtype>::run(){
  TAU_FSTART(ctr_virt);
  int * idx_arr, * tidx_arr, * lda_A, * lda_B, * lda_C, * beta_arr;
  int * ilda_A, * ilda_B, * ilda_C;
  long_int i, off_A, off_B, off_C;
  int nb_A, nb_B, nb_C, alloced, ret; 

  if (this->buffer != NULL){    
    alloced = 0;
    idx_arr = (int*)this->buffer;
  } else {
    alloced = 1;
    ret = posix_memalign((void**)&idx_arr,
                         ALIGN_BYTES,
                         mem_fp());
    LIBT_ASSERT(ret==0);
  }

  
  lda_A = idx_arr + VIRT_NTD*num_dim;
  lda_B = lda_A + num_dim;
  lda_C = lda_B + num_dim;
  ilda_A = lda_C + num_dim;
  ilda_B = ilda_A + num_dim;
  ilda_C = ilda_B + num_dim;

#define SET_LDA_X(__X)                                                  \
do {                                                                    \
  nb_##__X = 1;                                                         \
  for (i=0; i<ndim_##__X; i++){                                         \
    lda_##__X[i] = nb_##__X;                                            \
    nb_##__X = nb_##__X*virt_dim[idx_map_##__X[i]];                     \
  }                                                                     \
  memset(ilda_##__X, 0, num_dim*sizeof(int));                           \
  for (i=0; i<ndim_##__X; i++){                                         \
    ilda_##__X[idx_map_##__X[i]] += lda_##__X[i];                       \
  }                                                                     \
} while (0)
  SET_LDA_X(A);
  SET_LDA_X(B);
  SET_LDA_X(C);
#undef SET_LDA_X
 
  /* dynammically determined size */ 
  beta_arr = (int*)malloc(sizeof(int)*nb_C);
  memset(beta_arr, 0, nb_C*sizeof(int));
#if (VIRT_NTD>1)
#pragma omp parallel private(off_A,off_B,off_C,tidx_arr,i) 
#endif
  {
    int tid, ntd, start_off, end_off;
#if (VIRT_NTD>1)
    tid = omp_get_thread_num();
    ntd = MIN(VIRT_NTD, omp_get_num_threads());
#else
    tid = 0;
    ntd = 1;
#endif
#if (VIRT_NTD>1)
    DPRINTF(2,"%d/%d %d %d\n",tid,ntd,VIRT_NTD,omp_get_num_threads());
#endif
    if (tid < ntd){
      tidx_arr = idx_arr + tid*num_dim;
      memset(tidx_arr, 0, num_dim*sizeof(int));

      start_off = (nb_C/ntd)*tid;
      if (tid < nb_C%ntd){
        start_off += tid;
        end_off = start_off + nb_C/ntd + 1;
      } else {
        start_off += nb_C%ntd;
        end_off = start_off + nb_C/ntd;
      }

      ctr<dtype> * tid_rec_ctr;
      if (tid > 0)
        tid_rec_ctr = rec_ctr->clone();
      else
        tid_rec_ctr = rec_ctr;
      
      tid_rec_ctr->num_lyr = this->num_lyr;
      tid_rec_ctr->idx_lyr = this->idx_lyr;

      off_A = 0, off_B = 0, off_C = 0;
      for (;;){
        if (off_C >= start_off && off_C < end_off) {
          tid_rec_ctr->A        = this->A + off_A*blk_sz_A;
          tid_rec_ctr->B        = this->B + off_B*blk_sz_B;
          tid_rec_ctr->C        = this->C + off_C*blk_sz_C;
          tid_rec_ctr->beta     = beta_arr[off_C]>0 ? 1.0 : this->beta;
          beta_arr[off_C]       = 1;
          tid_rec_ctr->run();
        }

        for (i=0; i<num_dim; i++){
          off_A -= ilda_A[i]*tidx_arr[i];
          off_B -= ilda_B[i]*tidx_arr[i];
          off_C -= ilda_C[i]*tidx_arr[i];
          tidx_arr[i]++;
          if (tidx_arr[i] >= virt_dim[i])
            tidx_arr[i] = 0;
          off_A += ilda_A[i]*tidx_arr[i];
          off_B += ilda_B[i]*tidx_arr[i];
          off_C += ilda_C[i]*tidx_arr[i];
          if (tidx_arr[i] != 0) break;
        }
        if (i==num_dim) break;
      }
      if (tid > 0){
        delete tid_rec_ctr;
      }
    }
  }
  if (alloced){
    free(idx_arr);
    this->buffer = NULL;
  }
  free(beta_arr);
  TAU_FSTOP(ctr_virt);
}




