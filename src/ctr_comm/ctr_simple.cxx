/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"

/**
 * \brief deallocates generic ctr object
 */
template<typename dtype>
ctr<dtype>::~ctr(){
  if (buffer != NULL) CTF_free(buffer);
}

/**
 * \brief copies generic ctr object
 */
template<typename dtype>
ctr<dtype>::ctr(ctr<dtype> * other){
  A = other->A;
  B = other->B;
  C = other->C;
  beta = other->beta;
  num_lyr = other->num_lyr;
  idx_lyr = other->idx_lyr;
  buffer = NULL;
}

/**
 * \brief deallocates ctr_dgemm object
 */
template<typename dtype>
ctr_dgemm<dtype>::~ctr_dgemm() { }

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_dgemm<dtype>::ctr_dgemm(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_dgemm<dtype> * o = (ctr_dgemm<dtype>*)other;
  n = o->n;
  m = o->m;
  k = o->k;
  alpha = o->alpha;
  transp_A = o->transp_A;
  transp_B = o->transp_B;
}
/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_dgemm<dtype>::clone() {
  return new ctr_dgemm<dtype>(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_dgemm<dtype>::mem_fp(){
  return 0;
}

/**
 * \brief a wrapper for zgemm
 */
template<> inline
void ctr_dgemm< std::complex<double> >::run(){
  const int lda_A = transp_A == 'n' ? m : k;
  const int lda_B = transp_B == 'n' ? k : n;
  const int lda_C = m;
  if (this->idx_lyr == 0){
    czgemm(transp_A,
           transp_B,
           m,
           n,
           k,
           alpha,
           this->A,
           lda_A,
           this->B,
           lda_B,
           this->beta,
           this->C,
           lda_C);
  }
}

/**
 * \brief a wrapper for dgemm
 */
template<typename dtype>
void ctr_dgemm<dtype>::run(){
  const int lda_A = transp_A == 'n' ? m : k;
  const int lda_B = transp_B == 'n' ? k : n;
  const int lda_C = m;
  if (this->idx_lyr == 0){
    cdgemm(transp_A,
           transp_B,
           m,
           n,
           k,
           alpha,
           this->A,
           lda_A,
           this->B,
           lda_B,
           this->beta,
           this->C,
           lda_C);
  }
}

/**
 * \brief deallocates ctr_lyr object
 */
template<typename dtype>
ctr_lyr<dtype>::~ctr_lyr() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_lyr<dtype>::ctr_lyr(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_lyr<dtype> * o = (ctr_lyr<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  cdt = o->cdt;
  sz_C = o->sz_C;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_lyr<dtype>::clone() {
  return new ctr_lyr<dtype>(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_lyr<dtype>::mem_fp(){
  return 0;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_lyr<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
template<typename dtype>
void ctr_lyr<dtype>::run(){
  rec_ctr->A            = this->A;
  rec_ctr->B            = this->B;
  rec_ctr->C            = this->C;
  rec_ctr->beta         = cdt->rank > 0 ? 0.0 : this->beta;
  rec_ctr->num_lyr      = cdt->np;
  rec_ctr->idx_lyr      = cdt->rank;

  rec_ctr->run();
  
  /* FIXME: unnecessary except for current DCMF wrapper */
  COMM_BARRIER(cdt);
  /* FIXME Won't work for single precision */
  ALLREDUCE(MPI_IN_PLACE, this->C, sz_C*(sizeof(dtype)/sizeof(double)), MPI_DOUBLE, MPI_SUM, cdt);

}


/**
 * \brief deallocates ctr_replicate object
 */
template<typename dtype>
ctr_replicate<dtype>::~ctr_replicate() {
  delete rec_ctr;
  if (ncdt_A > 0)
    CTF_free(cdt_A);
  if (ncdt_B > 0)
    CTF_free(cdt_B);
  if (ncdt_C > 0)
    CTF_free(cdt_C);
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_replicate<dtype>::ctr_replicate(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_replicate<dtype> * o = (ctr_replicate<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  size_A = o->size_A;
  size_B = o->size_B;
  size_C = o->size_C;
  ncdt_A = o->ncdt_A;
  ncdt_B = o->ncdt_B;
  ncdt_C = o->ncdt_C;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_replicate<dtype>::clone() {
  return new ctr_replicate<dtype>(this);
}

/**
 * \brief print ctr object
 */
template<typename dtype>
void ctr_replicate<dtype>::print() {
  int i;
  printf("ctr_replicate: \n");
  printf("cdt_A = %p, size_A = %lld, ncdt_A = %d\n",
          cdt_A, size_A, ncdt_A);
  for (i=0; i<ncdt_A; i++){
    printf("cdt_A[%d] length = %d\n",i,cdt_A[i]->np);
  }
  printf("cdt_B = %p, size_B = %lld, ncdt_B = %d\n",
          cdt_B, size_B, ncdt_B);
  for (i=0; i<ncdt_B; i++){
    printf("cdt_B[%d] length = %d\n",i,cdt_B[i]->np);
  }
  printf("cdt_C = %p, size_C = %lld, ncdt_C = %d\n",
          cdt_C, size_C, ncdt_C);
  for (i=0; i<ncdt_C; i++){
    printf("cdt_C[%d] length = %d\n",i,cdt_C[i]->np);
  }
  rec_ctr->print();
}

/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes needed
 */
template<typename dtype>
uint64_t ctr_replicate<dtype>::comm_fp(int nlyr){
  int i;
  long_int tot_sz;
  tot_sz = 0;
  for (i=0; i<ncdt_A; i++){
    LIBT_ASSERT(cdt_A[i]->np > 0);
    tot_sz += size_A*log(cdt_A[i]->np);
  }
  for (i=0; i<ncdt_B; i++){
    LIBT_ASSERT(cdt_B[i]->np > 0);
    tot_sz += size_B*log(cdt_B[i]->np);
  }
  for (i=0; i<ncdt_C; i++){
    LIBT_ASSERT(cdt_C[i]->np > 0);
    tot_sz += size_C*log(cdt_C[i]->np);
  }
  return ((uint64_t)tot_sz)*sizeof(dtype);
}

/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
uint64_t ctr_replicate<dtype>::comm_rec(int nlyr) {
  return rec_ctr->comm_rec(nlyr) + comm_fp(nlyr);
}

/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_replicate<dtype>::mem_fp(){
  return 0;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_replicate<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
template<typename dtype>
void ctr_replicate<dtype>::run(){
  int arank, brank, crank, i;

  arank = 0, brank = 0, crank = 0;
  for (i=0; i<ncdt_A; i++){
    arank += cdt_A[i]->rank;
    POST_BCAST(this->A, size_A*sizeof(dtype), COMM_CHAR_T, 0, cdt_A[i], 0);
  }
  for (i=0; i<ncdt_B; i++){
    brank += cdt_B[i]->rank;
    POST_BCAST(this->B, size_B*sizeof(dtype), COMM_CHAR_T, 0, cdt_B[i], 0);
  }
  for (i=0; i<ncdt_C; i++){
    crank += cdt_C[i]->rank;
  }
  if (crank != 0) std::fill(this->C, this->C+size_C, get_zero<dtype>());
  else {
    for (i=0; i<size_C; i++){
      this->C[i] = this->beta * this->C[i];
    }
  }

  rec_ctr->A            = this->A;
  rec_ctr->B            = this->B;
  rec_ctr->C            = this->C;
  rec_ctr->beta         = crank != 0 ? 0.0 : 1.0;
  rec_ctr->num_lyr      = this->num_lyr;
  rec_ctr->idx_lyr      = this->idx_lyr;

  rec_ctr->run();
  
  for (i=0; i<ncdt_C; i++){
    /* FIXME Won't work for single precision */
    ALLREDUCE(MPI_IN_PLACE, this->C, size_C*(sizeof(dtype)/sizeof(double)), MPI_DOUBLE, MPI_SUM, cdt_C[i]);
  }

  if (arank != 0){
    std::fill(this->A, this->A+size_A, get_zero<dtype>());
  }
  if (brank != 0){
    std::fill(this->B, this->B+size_B, get_zero<dtype>());
  }


}
