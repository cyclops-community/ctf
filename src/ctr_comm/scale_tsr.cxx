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
#include "scale_tsr.h"

/**
 * \brief copies generic scl object
 */
template<typename dtype>
scl<dtype>::scl(scl * other){
  A = other->A;
  alpha = other->alpha;
  buffer = NULL;
}

/**
 * \brief deallocates scl_virt object
 */
template<typename dtype>
scl_virt<dtype>::~scl_virt() {
  free(virt_dim);
  delete rec_scl;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
scl_virt<dtype>::scl_virt(scl<dtype> * other) : scl<dtype>(other) {
  scl_virt<dtype> * o   = (scl_virt<dtype>*)other;
  rec_scl       = o->rec_scl->clone();
  num_dim       = o->num_dim;
  virt_dim      = (int*)malloc(sizeof(int)*num_dim);
  memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

  ndim_A        = o->ndim_A;
  blk_sz_A      = o->blk_sz_A;
  idx_map_A     = o->idx_map_A;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
scl<dtype> * scl_virt<dtype>::clone() {
  return new scl_virt(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need
 * \return bytes needed
 */
template<typename dtype>
long_int scl_virt<dtype>::mem_fp(){
  return 3*num_dim*sizeof(int);
}

/**
 * \brief iterates over the dense virtualization block grid and contracts
 */
template<typename dtype>
void scl_virt<dtype>::run(){
  int * idx_arr, * lda_A;
  int * ilda_A;
  int i, off_A, nb_A, alloced, ret; 
  TAU_FSTART(scl_virt);

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
  
  lda_A = idx_arr + num_dim;
  ilda_A = lda_A + num_dim;
  

#define SET_LDA_X(__X)                                                  \
do {                                                                    \
  nb_##__X = 1;                                                         \
  for (i=0; i<ndim_##__X; i++){                                 \
    lda_##__X[i] = nb_##__X;                                            \
    nb_##__X = nb_##__X*virt_dim[idx_map_##__X[i]];     \
  }                                                                     \
  memset(ilda_##__X, 0, num_dim*sizeof(int));                   \
  for (i=0; i<ndim_##__X; i++){                                 \
    ilda_##__X[idx_map_##__X[i]] += lda_##__X[i];                       \
  }                                                                     \
} while (0)
  SET_LDA_X(A);
#undef SET_LDA_X
 
  memset(idx_arr, 0, num_dim*sizeof(int));
  rec_scl->alpha = this->alpha;
  off_A = 0;
  for (;;){
    rec_scl->A = this->A + off_A*blk_sz_A;
    rec_scl->run();

    for (i=0; i<num_dim; i++){
      off_A -= ilda_A[i]*idx_arr[i];
      idx_arr[i]++;
      if (idx_arr[i] >= virt_dim[i])
        idx_arr[i] = 0;
      off_A += ilda_A[i]*idx_arr[i];
      if (idx_arr[i] != 0) break;
    }
    if (i==num_dim) break;
  }
  if (alloced){
    free(idx_arr);
  }
  TAU_FSTOP(scl_virt);
}




