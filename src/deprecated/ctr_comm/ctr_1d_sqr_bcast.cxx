/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "ctr_comm.h"
#include "../shared/util.h"

/**
 * \brief deallocates ctr_1d_sqr_bcast object
 */
template<typename dtype>
ctr_1d_sqr_bcast<dtype>::~ctr_1d_sqr_bcast() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_1d_sqr_bcast<dtype>::ctr_1d_sqr_bcast(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_1d_sqr_bcast<dtype> * o = (ctr_1d_sqr_bcast<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  ctr_lda = o->ctr_lda;
  ctr_sub_lda = o->ctr_sub_lda;
  sz = o->sz;
  cdt = o->cdt;
  cdt_dir = o->cdt_dir;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_1d_sqr_bcast<dtype>::clone() {
  return new ctr_1d_sqr_bcast<dtype>(this);
}


/**
 * \brief returns the number of bytes of buffer space we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_1d_sqr_bcast<dtype>::mem_fp() {
  return (ctr_lda*ctr_sub_lda+sz)*sizeof(dtype);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_1d_sqr_bcast<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}

/**
 * \brief runs a SUMMA algorithm with one dimension virtualized (local) 
 */
template<typename dtype>
void ctr_1d_sqr_bcast<dtype>::run() {
  int i, alloced, ret;
  dtype * buf_A, * buf_B; 
  TAU_FSTART(ctr_1d_sqr_bcast);

  LIBT_ASSERT(cdt->nbcast >= 2);

  dtype * buf_comm, * buf_loc;
  if (this->buffer != NULL || sz == 0){ 
    alloced = 0;
  } else {
    alloced = 1;
    ret = posix_memalign((void**)&this->buffer,
                         ALIGN_BYTES,
                         mem_fp());
    LIBT_ASSERT(ret==0);
  }
  buf_comm = this->buffer;
  buf_loc  = buf_comm+sz;
 
  rec_ctr->beta         = this->beta;
  rec_ctr->num_lyr      = 1;
  rec_ctr->idx_lyr      = 0;
  
  int * bid = (int*)malloc(sizeof(int));
  bid[0] = 0;

  if (cdt_dir == 0){
    for (i=this->idx_lyr; i<cdt->np; i+=this->num_lyr){
      COMM_BARRIER(cdt); 
      if (cdt->rank == i)
        buf_A = this->A;
      else 
        buf_A = buf_comm;
      POST_BCAST(buf_A, sz*sizeof(dtype), COMM_CHAR_T, i, cdt, 0);
      WAIT_BCAST(cdt, 1, bid);

      if (ctr_lda == 1)
        buf_B = this->B+i*ctr_sub_lda;
      else {
        lda_cpy<dtype>(ctr_sub_lda, ctr_lda,
                ctr_sub_lda*cdt->np, ctr_sub_lda,
                this->B+i*ctr_sub_lda, buf_loc);
        buf_B = buf_loc;
      }

      DEBUG_PRINTF("[%d] multiplying (sz_B=%d) %lf by %lf\n",
                   cdt->rank, sz,
                   buf_A[0], buf_B[0]);

      rec_ctr->A = buf_A;
      rec_ctr->B = buf_B;
      rec_ctr->C = this->C;

      rec_ctr->run();
      rec_ctr->beta = 1.0;
    }
  } else {
    for (i=this->idx_lyr; i<cdt->np; i+=this->num_lyr){
      COMM_BARRIER(cdt); 

      if (cdt->rank == i)
        buf_B = this->B;
      else 
        buf_B = buf_comm;
      POST_BCAST(buf_B, sz*sizeof(dtype), COMM_CHAR_T, i, cdt, 0);
      WAIT_BCAST(cdt, 1, bid);

      if (ctr_lda == 1)
        buf_A = this->A+i*ctr_sub_lda;
      else {
        lda_cpy<dtype>(ctr_sub_lda, ctr_lda,
                ctr_sub_lda*cdt->np, ctr_sub_lda,
                this->A+i*ctr_sub_lda, buf_loc);
        buf_A = buf_loc;
      }

      DEBUG_PRINTF("[%d] multiplying (sz_A=%d) [%lf %lf] by [%lf %lf]\n",
                   cdt->rank, sz,
                   buf_A[0], buf_A[1], buf_B[0], buf_B[1]);

      rec_ctr->A = buf_A;
      rec_ctr->B = buf_B;
      rec_ctr->C = this->C;

      rec_ctr->run();
      rec_ctr->beta = 1.0;
    }
  }
  /* FIXME: reuse that shit */
  if (alloced){
    free(this->buffer);
    this->buffer = NULL;
  }
  TAU_FSTOP(ctr_1d_sqr_bcast);
}


