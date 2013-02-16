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

#include "ctr_comm.h"
#include "../shared/util.h"

/**
 * \brief deallocates ctr_2d_rect_bcast object
 */
template<typename dtype>
ctr_2d_sqr_bcast<dtype>::~ctr_2d_sqr_bcast() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_2d_sqr_bcast<dtype>::ctr_2d_sqr_bcast(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_2d_sqr_bcast<dtype> * o = (ctr_2d_sqr_bcast<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  sz_A = o->sz_A;
  sz_B = o->sz_B;
  cdt_x = o->cdt_x;
  cdt_y = o->cdt_y;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_2d_sqr_bcast<dtype>::clone() {
  return new ctr_2d_sqr_bcast<dtype>(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_2d_sqr_bcast<dtype>::mem_fp(){
  return (sz_A+sz_B)*sizeof(dtype);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_2d_sqr_bcast<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief runs a SUMMA algorithm with on a 2D square processor grid
 */
template<typename dtype>
void ctr_2d_sqr_bcast<dtype>::run(){
  int i, alloced, ret;
  dtype * buf_A, * buf_B; 

  int const x_rank = cdt_x->rank;
  int const y_rank = cdt_y->rank;

  TAU_FSTART(ctr_2d_sqr_bcast);
  
  LIBT_ASSERT(cdt_x->np == cdt_y->np);
  LIBT_ASSERT(cdt_x->nbcast >= 2);
  LIBT_ASSERT(cdt_y->nbcast >= 2);


  rec_ctr->beta         = this->beta;
  rec_ctr->num_lyr      = 1;
  rec_ctr->idx_lyr      = 0;
 
  if (this->buffer != NULL){    
    alloced = 0;
  } else {
    alloced = 1;
    ret = posix_memalign((void**)&this->buffer,
                         ALIGN_BYTES,
                         mem_fp());
    LIBT_ASSERT(ret==0);
  }
  
  buf_A   = this->buffer;
  buf_B   = buf_A+sz_A;

  int * bid = (int*)malloc(sizeof(int));
  bid[0] = 0;
  for (i=this->idx_lyr; i<cdt_x->np; i+=this->num_lyr){
    DEBUG_PRINTF("[%d][%d] owns %lf by %lf\n",
                 cdt_x->rank, cdt_y->rank,
                 this->A[0], this->B[0]);

    COMM_BARRIER(cdt_x); 
    COMM_BARRIER(cdt_y); 

    if (x_rank == i){
      buf_A = this->A;
      DEBUG_PRINTF("[%d][%d] sending A = %lf\n",
                   cdt_x->rank, cdt_y->rank,
                   buf_A[0]);
    }
      
    if (y_rank == i){
      buf_B = this->B;
      DEBUG_PRINTF("[%d][%d] sending B = %lf\n",
                   cdt_x->rank, cdt_y->rank,
                   buf_B[0]);
    }

    POST_BCAST(buf_A, sz_A*sizeof(dtype), COMM_CHAR_T, i, cdt_x, 0);
    POST_BCAST(buf_B, sz_B*sizeof(dtype), COMM_CHAR_T, i, cdt_y, 0);

    WAIT_BCAST(cdt_x, 1, bid);
    WAIT_BCAST(cdt_y, 1, bid);

    DEBUG_PRINTF("[%d][%d] multiplying %lf by %lf\n",
                 cdt_x->rank, cdt_y->rank,
                 buf_A[0], buf_B[0]);

    rec_ctr->A = buf_A;
    rec_ctr->B = buf_B;
    rec_ctr->C = this->C;

    rec_ctr->run();

    buf_A   = this->buffer;
    buf_B   = buf_A+sz_A;
  
    rec_ctr->beta = 1.0;
  }
  free(bid);
  /* FIXME: reuse that shit */
  if (alloced){
    free(this->buffer);
    this->buffer = NULL;
  }
  TAU_FSTOP(ctr_2d_sqr_bcast);
}


