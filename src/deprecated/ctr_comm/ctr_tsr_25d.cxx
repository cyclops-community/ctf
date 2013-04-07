/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

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
 * \brief deallocates ctr_virt_25d object
 */
template<typename dtype>
ctr_virt_25d<dtype>::~ctr_virt_25d() {
  free(virt_dim);
  delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_virt_25d<dtype>::ctr_virt_25d(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_virt_25d<dtype> * o       = (ctr_virt_25d<dtype>*)other;
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
ctr<dtype> * ctr_virt_25d<dtype>::clone() {
  return new ctr_virt_25d<dtype>(this);
}

/**
 * \brief prints ctr object
 */
template<typename dtype>
void ctr_virt_25d<dtype>::print() {
  int i;
  printf("ctr_virt_25d:\n");
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
long_int ctr_virt_25d<dtype>::mem_fp(){
  int i, nb_A, nb_B, nb_C, ntd, ncopy, ntd_A, ntd_B, ntd_C;
  nb_A = 1;
  for (i=0; i<ndim_A; i++){
    nb_A = nb_A*virt_dim[idx_map_A[i]];
  }
  nb_B = 1;
  for (i=0; i<ndim_B; i++){
    nb_B = nb_B*virt_dim[idx_map_B[i]];
  }
  nb_C = 1;
  for (i=0; i<ndim_C; i++){
    nb_C = nb_C*virt_dim[idx_map_C[i]];
  }
#ifdef USE_OMP
  ntd = MIN(nb_A*nb_B*nb_C,MIN(VIRT_NTD, omp_get_max_threads()));
#else
  ntd = MIN(nb_A*nb_B*nb_C,VIRT_NTD);
#endif
  ntd_C = MIN(nb_C, ntd);
  ntd_A = ntd/ntd_C;
  ntd_A = MIN(nb_A, ntd_A);
  ntd_B = ntd/(ntd_C*ntd_A);
  ntd_B = MIN(nb_B,ntd_B);
  ncopy = ntd_A*ntd_B;
  return (6+VIRT_NTD)*num_dim*sizeof(int) + ncopy*nb_C*blk_sz_C*sizeof(dtype);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_virt_25d<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}


/**
 * \brief iterates over the dense virtualization block grid and contracts
 */
template<typename dtype>
void ctr_virt_25d<dtype>::run(){
  TAU_FSTART(ctr_virt_25d);
  int * idx_arr, * tidx_arr, * lda_A, * lda_B, * lda_C;
  int * ilda_A, * ilda_B, * ilda_C;
  long_int i, j, k, off_A, off_B, off_C;
  int nb_A, nb_B, nb_C, alloced, ret; 
  dtype * C_buf, * my_C_buf;
  dtype dbeta;

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
  C_buf = (dtype*)(ilda_C + num_dim);
  

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
 
  int * beta_arr = (int*)malloc(sizeof(int)*nb_C);
  memset(beta_arr, 0, nb_C*sizeof(int));
  /* dynammically determined size */ 
#if (VIRT_NTD>1)
#pragma omp parallel private(off_A,off_B,off_C,tidx_arr,i,j,k,my_C_buf,dbeta) 
#endif
  {
    int tid, ntd, ntd_C, ntd_B, ntd_A;
    int tid_C, tid_B, tid_A;
    int st_off_A, end_off_A, st_off_B, end_off_B;
    int st_off_C, end_off_C;
    st_off_C = 0, end_off_C = 0;
#if (VIRT_NTD>1)
    tid = omp_get_thread_num();
    ntd = MIN(nb_C*nb_A*nb_B,MIN(VIRT_NTD, omp_get_num_threads()));
#else
    tid = 0;
    ntd = 1;
#endif
#if (VIRT_NTD>1)
    DPRINTF(2,"%d/%d %d %d\n",tid,ntd,VIRT_NTD,omp_get_num_threads());
#endif
    // prevent compiler warning
    my_C_buf = NULL;
    ntd_C = MIN(nb_C, ntd);
    ntd_A = ntd/ntd_C;
    ntd_A = MIN(nb_A, ntd_A);
    ntd_B = ntd/(ntd_C*ntd_A);
    ntd_B = MIN(nb_B,ntd_B);
    tid_C = tid%ntd_C;
    tid_A = (tid/ntd_C)%ntd_A;
    tid_B = (tid/(ntd_C*ntd_A))%ntd_B;
    if (tid < ntd_A*ntd_B*ntd_C){
      tidx_arr = idx_arr + tid*num_dim;
      memset(tidx_arr, 0, num_dim*sizeof(int));
      
      st_off_A = (nb_A/ntd_A)*tid_A;
      if (tid_A < nb_A%ntd_A){
        st_off_A += tid_A;
        end_off_A = st_off_A + nb_A/ntd_A + 1;
      } else {
        st_off_A += nb_A%ntd_A;
        end_off_A = st_off_A + nb_A/ntd_A;
      }

      st_off_B = (nb_B/ntd_B)*tid_B;
      if (tid_B < nb_B%ntd_B){
        st_off_B += tid_B;
        end_off_B = st_off_B + nb_B/ntd_B + 1;
      } else {
        st_off_B += nb_B%ntd_B;
        end_off_B = st_off_B + nb_B/ntd_B;
      }

      st_off_C = (nb_C/ntd_C)*tid_C;
      if (tid_C < nb_C%ntd_C){
        st_off_C += tid_C;
        end_off_C = st_off_C + nb_C/ntd_C + 1;
      } else {
        st_off_C += nb_C%ntd_C;
        end_off_C = st_off_C + nb_C/ntd_C;
      }
      /*printf("[%d] tA = %d/%d, tB = %d/%d, tC = %d/%d\n",
              tid, tid_A, ntd_A, tid_B, ntd_B, tid_C, ntd_C);
      printf("[%d] oA = %d-%d, oB = %d-%d, iC = %d-%d\n",
              tid, st_off_A, end_off_A, st_off_B, end_off_B, st_off_C, end_off_C);*/

      my_C_buf = C_buf + blk_sz_C*(st_off_C+(tid_A+ntd_A*tid_B)*nb_C);
      std::fill(my_C_buf, my_C_buf+(end_off_C-st_off_C)*blk_sz_C, 0.0);
      dbeta = 1.0;
/*      } else {
        my_C_buf = this->C+st_off_C*blk_sz_C;
        dbeta = this->beta;
      }*/

      ctr<dtype> * tid_rec_ctr;
      if (tid > 0)
        tid_rec_ctr = rec_ctr->clone();
      else
        tid_rec_ctr = rec_ctr;
      
      tid_rec_ctr->num_lyr = this->num_lyr;
      tid_rec_ctr->idx_lyr = this->idx_lyr;

      off_A = 0, off_B = 0, off_C = 0;
      for (;;){
        if (off_C >= st_off_C && off_C < end_off_C &&
            off_B >= st_off_B && off_B < end_off_B &&
            off_A >= st_off_A && off_A < end_off_A) {
          tid_rec_ctr->A        = this->A + off_A*blk_sz_A;
          tid_rec_ctr->B        = this->B + off_B*blk_sz_B;
          tid_rec_ctr->C        = my_C_buf + (off_C-st_off_C)*blk_sz_C;
          tid_rec_ctr->beta     = dbeta;
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
    for (i=0; i<ntd_A*ntd_B; i++){
      if (tid < ntd_A*ntd_B*ntd_C && tid_A == i%ntd_A && tid_B == i/ntd_A){
        for (j=0; j<end_off_C-st_off_C; j++){
          if (beta_arr[st_off_C+j] == 0){
            dbeta = this->beta;
            beta_arr[st_off_C+j] = 1;
          } else
            dbeta = 1.0;
          for (k=0; k<blk_sz_C; k++){
            this->C[(st_off_C+j)*blk_sz_C+k] = 
                  dbeta*this->C[(st_off_C+j)*blk_sz_C+k] 
                  + my_C_buf[j*blk_sz_C+k];
          }
        }
      }
#ifdef USE_OMP
      #pragma omp barrier
#endif
    }
  }
  if (alloced){
    free(idx_arr);
    this->buffer = NULL;
  }
  free(beta_arr);
  TAU_FSTOP(ctr_virt_25d);
}




