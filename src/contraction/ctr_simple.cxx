/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"

namespace CTF_int {
/**
 * \brief deallocates generic ctr object
 */
ctr::~ctr(){
  if (buffer != NULL) CTF_free(buffer);
}

/**
 * \brief copies generic ctr object
 */
ctr::ctr(ctr * other){
  A = other->A;
  B = other->B;
  C = other->C;
  sr_A = other->sr_A;
  sr_B = other->sr_B;
  sr_C = other->sr_C;
  beta = other->beta;
  num_lyr = other->num_lyr;
  idx_lyr = other->idx_lyr;
  buffer = NULL;
}

/**
 * \brief deallocates ctr_dgemm object
 */
ctr_dgemm::~ctr_dgemm() { }

/**
 * \brief copies ctr object
 */
ctr_dgemm::ctr_dgemm(ctr * other) : ctr(other) {
  ctr_dgemm * o = (ctr_dgemm*)other;
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
ctr * ctr_dgemm::clone() {
  return new ctr_dgemm(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int64_t ctr_dgemm::mem_fp(){
  return 0;
}


/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes sent
 */
double ctr_dgemm::est_time_fp(int nlyr) {
  /* FIXME make cost proper, for now return sizes of each submatrix scaled by .2 */
  ASSERT(0);
  return n*m+m*k+n*k;
}

/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
double ctr_dgemm::est_time_rec(int nlyr) {
  return est_time_fp(nlyr);
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
void ctr_dgemm::run(){
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
ctr_lyr::~ctr_lyr() {
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
ctr_lyr::ctr_lyr(ctr * other) : ctr(other) {
  ctr_lyr * o = (ctr_lyr*)other;
  rec_ctr = o->rec_ctr->clone();
  k = o->k;
  cdt = o->cdt;
  sz_C = o->sz_C;
}

/**
 * \brief copies ctr object
 */
ctr * ctr_lyr::clone() {
  return new ctr_lyr(this);
}


/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int64_t ctr_lyr::mem_fp(){
  return 0;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
int64_t ctr_lyr::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
void ctr_lyr::run(){
  rec_ctr->A            = this->A;
  rec_ctr->B            = this->B;
  rec_ctr->C            = this->C;
  rec_ctr->beta         = cdt.rank > 0 ? 0.0 : this->beta;
  rec_ctr->num_lyr      = cdt.np;
  rec_ctr->idx_lyr      = cdt.rank;

  rec_ctr->run();
  
  /* FIXME: unnecessary except for current DCMF wrapper */
  COMM_BARRIER(cdt);
  /* FIXME Won't work for single precision */
  ALLREDUCE(MPI_IN_PLACE, this->C, sz_C*(sizeof(dtype)/sizeof(double)), MPI_DOUBLE, MPI_SUM, cdt);

}


/**
 * \brief deallocates ctr_replicate object
 */
ctr_replicate::~ctr_replicate() {
  delete rec_ctr;
  for (int i=0; i<ncdt_A; i++){
    FREE_CDT(cdt_A[i]);
  }
  if (ncdt_A > 0)
    CTF_free(cdt_A);
  for (int i=0; i<ncdt_B; i++){
    FREE_CDT(cdt_B[i]);
  }
  if (ncdt_B > 0)
    CTF_free(cdt_B);
  for (int i=0; i<ncdt_C; i++){
    FREE_CDT(cdt_C[i]);
  }
  if (ncdt_C > 0)
    CTF_free(cdt_C);
}

/**
 * \brief copies ctr object
 */
ctr_replicate::ctr_replicate(ctr * other) : ctr(other) {
  ctr_replicate * o = (ctr_replicate*)other;
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
ctr * ctr_replicate::clone() {
  return new ctr_replicate(this);
}

/**
 * \brief print ctr object
 */
void ctr_replicate::print() {
  int i;
  printf("ctr_replicate: \n");
  printf("cdt_A = %p, size_A = " PRId64 ", ncdt_A = %d\n",
          cdt_A, size_A, ncdt_A);
  for (i=0; i<ncdt_A; i++){
    printf("cdt_A[%d] length = %d\n",i,cdt_A[i].np);
  }
  printf("cdt_B = %p, size_B = " PRId64 ", ncdt_B = %d\n",
          cdt_B, size_B, ncdt_B);
  for (i=0; i<ncdt_B; i++){
    printf("cdt_B[%d] length = %d\n",i,cdt_B[i].np);
  }
  printf("cdt_C = %p, size_C = " PRId64 ", ncdt_C = %d\n",
          cdt_C, size_C, ncdt_C);
  for (i=0; i<ncdt_C; i++){
    printf("cdt_C[%d] length = %d\n",i,cdt_C[i].np);
  }
  rec_ctr->print();
}

/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes needed
 */
double ctr_replicate::est_time_fp(int nlyr){
  int i;
  double tot_sz;
  tot_sz = 0.0;
  for (i=0; i<ncdt_A; i++){
    ASSERT(cdt_A[i].np > 0);
    tot_sz += cdt_A[i].estimate_bcast_time(size_A*sr_A.el_size);
  }
  for (i=0; i<ncdt_B; i++){
    ASSERT(cdt_B[i].np > 0);
    tot_sz += cdt_B[i].estimate_bcast_time(size_B*sr_B.el_size);
  }
  for (i=0; i<ncdt_C; i++){
    ASSERT(cdt_C[i].np > 0);
    tot_sz += cdt_C[i].estimate_allred_time(size_C*sr_C.el_size);
  }
  return tot_sz;
}

/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
double ctr_replicate::est_time_rec(int nlyr) {
  return rec_ctr->est_time_rec(nlyr) + est_time_fp(nlyr);
}

/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int64_t ctr_replicate::mem_fp(){
  return 0;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
int64_t ctr_replicate::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief performs replication along a dimension, generates 2.5D algs
 */
void ctr_replicate::run(){
  int arank, brank, crank, i;

  arank = 0, brank = 0, crank = 0;
  for (i=0; i<ncdt_A; i++){
    arank += cdt_A[i].rank;
    POST_BCAST(this->A, size_A*sr_A.el_size, COMM_CHAR_T, 0, cdt_A[i], 0);
  }
  for (i=0; i<ncdt_B; i++){
    brank += cdt_B[i].rank;
    POST_BCAST(this->B, size_B*sr_B.el_size, COMM_CHAR_T, 0, cdt_B[i], 0);
  }
  for (i=0; i<ncdt_C; i++){
    crank += cdt_C[i].rank;
  }
  if (crank != 0) this->sr_C.set(this->C, this->sr_C.addid, size_C);
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
    ALLREDUCE(MPI_IN_PLACE, this->C, size_C, sr_C.mdtype, sr_C.addmop, cdt_C[i]);
  }

  if (arank != 0){
    this->sr_A.set(this->A, this->sr_A.addid, size_A);
  }
  if (brank != 0){
    this->sr_B.set(this->B, this->sr_B.addid, size_B);
  }
}

