/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

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
  CTF_free(virt_dim);
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
  virt_dim      = (int*)CTF_alloc(sizeof(int)*num_dim);
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
    ret = CTF_alloc_ptr(mem_fp(), (void**)&idx_arr);
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
    CTF_free(idx_arr);
  }
  TAU_FSTOP(scl_virt);
}




