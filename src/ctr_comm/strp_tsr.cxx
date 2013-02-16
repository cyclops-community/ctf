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
#include "strp_tsr.h"


/**
 * \brief copies strp_tsr object
 */
template<typename dtype>
strp_tsr<dtype>::strp_tsr(strp_tsr<dtype> * o) {
  alloced       = o->alloced;
  ndim          = o->ndim;
  blk_sz        = o->blk_sz;
  edge_len      = o->edge_len;
  strip_dim     = o->strip_dim;
  strip_idx     = o->strip_idx;
  A             = o->A;
  buffer = NULL;
}

/**
 * \brief copies strp_tsr object
 */
template<typename dtype>
strp_tsr<dtype>* strp_tsr<dtype>::clone(){
  return new strp_tsr<dtype>(this);
}

/**
 * \brief returns the number of bytes of buffer space
   we need
 * \return bytes needed
 */
template<typename dtype>
long_int strp_tsr<dtype>::mem_fp(){
  int i;
  long_int sub_sz;
  sub_sz = blk_sz;
  for (i=0; i<ndim; i++){
    sub_sz = sub_sz * edge_len[i] / strip_dim[i];
  }
  return sub_sz*sizeof(dtype);
}

/**
 * \brief strips out part of tensor to be operated on
 * param[in] dir (0 if writing from, 1 if writing to)
 */
template<typename dtype>
void strp_tsr<dtype>::run(int const dir){
  TAU_FSTART(strp_tsr);
  int i, ilda, toff, boff, ret;
  int * idx_arr, * lda;
 
  if (dir == 0)  {
    if (buffer != NULL){        
      alloced = 0;
    } else {
      alloced = 1;
      ret = posix_memalign((void**)&buffer,
                           ALIGN_BYTES,
                           mem_fp());
      LIBT_ASSERT(ret==0);
    }
  } 
  idx_arr = (int*)malloc(sizeof(int)*ndim);
  lda = (int*)malloc(sizeof(int)*ndim);
  memset(idx_arr, 0, sizeof(int)*ndim);

  ilda = 1, toff = 0;
  for (i=0; i<ndim; i++){
    lda[i] = ilda;
    ilda *= edge_len[i];
    idx_arr[i] = strip_idx[i]*(edge_len[i]/strip_dim[i]);
    toff += idx_arr[i]*lda[i];
    DEBUG_PRINTF("[%d] sidx = %d, sdim = %d, edge_len = %d\n", i, strip_idx[i], strip_dim[i], edge_len[i]);
  }
  
  boff = 0;
  for (;;){
    if (dir)
      memcpy(A+toff*blk_sz, buffer+boff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz*sizeof(dtype));
    else {
      memcpy(buffer+boff*blk_sz, A+toff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz*sizeof(dtype));
    }
    boff += (edge_len[0]/strip_dim[0]);

    for (i=1; i<ndim; i++){
      toff -= idx_arr[i]*lda[i];
      idx_arr[i]++;
      if (idx_arr[i] >= (strip_idx[i]+1)*(edge_len[i]/strip_dim[i]))
        idx_arr[i] = strip_idx[i]*(edge_len[i]/strip_dim[i]);
      toff += idx_arr[i]*lda[i];
      if (idx_arr[i] != strip_idx[i]*(edge_len[i]/strip_dim[i])) break;
    }
    if (i==ndim) break;    
  }
  

  if (dir == 1) {
    if (alloced){
      free(buffer);
      buffer = NULL;
    }
  }
  free(idx_arr);
  free(lda);
  TAU_FSTOP(strp_tsr);
}


/**
 * \brief deallocates buffer
 */
template<typename dtype>
void strp_tsr<dtype>::free_exp(){
  if (alloced){
    free(buffer);
    buffer = NULL;
  }
}

/**
 * \brief deconstructor
 */
template<typename dtype>
strp_sum<dtype>::~strp_sum(){
  delete rec_tsum;
  if (strip_A)
    delete rec_strp_A;
  if (strip_B)
    delete rec_strp_B;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
strp_sum<dtype>::strp_sum(tsum<dtype> * other) : tsum<dtype>(other) {
  strp_sum<dtype> * o   = (strp_sum<dtype>*)other;
  rec_tsum      = o->rec_tsum->clone();
  rec_strp_A    = o->rec_strp_A->clone();
  rec_strp_B    = o->rec_strp_B->clone();
  strip_A       = o->strip_A;
  strip_B       = o->strip_B;
}

/**
 * \brief copies strp_sum object
 */
template<typename dtype>
tsum<dtype>* strp_sum<dtype>::clone() {
  return new strp_sum<dtype>(this);
}

/**
 * \brief gets memory usage of op
 */
template<typename dtype>
long_int strp_sum<dtype>::mem_fp(){
  return 0;
}

/**
 * \brief runs strip for sum of tensors
 */
template<typename dtype>
void strp_sum<dtype>::run(){
  dtype * bA, * bB;

  if (strip_A) {
    rec_strp_A->A = this->A;
    rec_strp_A->run(0);
    bA = rec_strp_A->buffer;
  } else {
    bA = this->A;
  }
  if (strip_B) {
    rec_strp_B->A = this->B;
    rec_strp_B->run(0);
    bB = rec_strp_B->buffer;
  } else {
    bB = this->B;
  }

  rec_tsum->A = bA;
  rec_tsum->B = bB;
  rec_tsum->alpha = this->alpha;
  rec_tsum->beta = this->beta;
  rec_tsum->run();
  
  if (strip_A) rec_strp_A->free_exp();
  if (strip_B) rec_strp_B->run(1); 

}


/**
 * \brief deconstructor
 */
template<typename dtype>
strp_ctr<dtype>::~strp_ctr(){
  delete rec_ctr;
  if (strip_A)
    delete rec_strp_A;
  if (strip_B)
    delete rec_strp_B;
  if (strip_C)
    delete rec_strp_C;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
strp_ctr<dtype>::strp_ctr(ctr<dtype> * other) : ctr<dtype>(other) {
  strp_ctr<dtype> * o   = (strp_ctr<dtype>*)other;
  rec_ctr       = o->rec_ctr->clone();
  rec_strp_A    = o->rec_strp_A->clone();
  rec_strp_B    = o->rec_strp_B->clone();
  rec_strp_C    = o->rec_strp_C->clone();
  strip_A       = o->strip_A;
  strip_B       = o->strip_B;
  strip_C       = o->strip_C;
}

/**
 * \brief copies strp_ctr object
 */
template<typename dtype>
ctr<dtype>* strp_ctr<dtype>::clone() {
  return new strp_ctr<dtype>(this);
}

/**
 * \brief gets memory usage of op
 */
template<typename dtype>
long_int strp_ctr<dtype>::mem_fp(){
  return 0;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int strp_ctr<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}

/**
 * \brief returns the number of bytes sent recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
uint64_t strp_ctr<dtype>::comm_rec(int nlyr) {
  return rec_ctr->comm_rec(nlyr);
}


/**
 * \brief runs strip for contraction of tensors
 */
template<typename dtype>
void strp_ctr<dtype>::run(){
  dtype * bA, * bB, * bC;

  if (strip_A) {
    rec_strp_A->A = this->A;
    rec_strp_A->run(0);
    bA = rec_strp_A->buffer;
  } else {
    bA = this->A;
  }
  if (strip_B) {
    rec_strp_B->A = this->B;
    rec_strp_B->run(0);
    bB = rec_strp_B->buffer;
  } else {
    bB = this->B;
  }
  if (strip_C) {
    rec_strp_C->A = this->C;
    rec_strp_C->run(0);
    bC = rec_strp_C->buffer;
  } else {
    bC = this->C;
  }

  
  rec_ctr->A = bA;
  rec_ctr->B = bB;
  rec_ctr->C = bC;
  rec_ctr->num_lyr      = this->num_lyr;
  rec_ctr->idx_lyr      = this->idx_lyr;
  rec_ctr->beta = this->beta;
  rec_ctr->run();
  
  if (strip_A) rec_strp_A->free_exp();
  if (strip_B) rec_strp_B->free_exp();
  if (strip_C) rec_strp_C->run(1);

}

/**
 * \brief deconstructor
 */
template<typename dtype>
strp_scl<dtype>::~strp_scl(){
  delete rec_scl;
  delete rec_strp;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
strp_scl<dtype>::strp_scl(scl<dtype> * other) : scl<dtype>(other) {
  strp_scl<dtype> * o   = (strp_scl<dtype>*)other;
  rec_scl       = o->rec_scl->clone();
  rec_strp      = o->rec_strp->clone();
}

/**
 * \brief copies strp_scl object
 */
template<typename dtype>
scl<dtype>* strp_scl<dtype>::clone() {
  return new strp_scl<dtype>(this);
}

/**
 * \brief gets memory usage of op
 */
template<typename dtype>
long_int strp_scl<dtype>::mem_fp(){
  return 0;
}


/**
 * \brief runs strip for scale of tensor
 */
template<typename dtype>
void strp_scl<dtype>::run(){
  dtype * bA;

  rec_strp->run(0);
  bA = rec_strp->buffer;
  
  rec_scl->A = bA;
  rec_scl->alpha = this->alpha;
  rec_scl->run();
  
  rec_strp->run(1);
}







