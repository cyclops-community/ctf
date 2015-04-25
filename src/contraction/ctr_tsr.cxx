/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"
#include "ctr_tsr.h"

namespace CTF_int {

  #ifdef USE_OMP
  #include <omp.h>
  #endif
  #ifndef VIRT_NTD
  #define VIRT_NTD        1
  #endif

  /**
   * \brief deallocates ctr_virt object
   */
  ctr_virt::~ctr_virt() {
    CTF_int::cdealloc(virt_dim);
    delete rec_ctr;
  }

  /**
   * \brief copies ctr object
   */
  ctr_virt::ctr_virt(ctr * other) : ctr(other) {
    ctr_virt * o   = (ctr_virt*)other;
    rec_ctr       = o->rec_ctr->clone();
    num_dim       = o->num_dim;
    virt_dim      = (int*)CTF_int::alloc(sizeof(int)*num_dim);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

    order_A        = o->order_A;
    blk_sz_A      = o->blk_sz_A;
    idx_map_A     = o->idx_map_A;

    order_B        = o->order_B;
    blk_sz_B      = o->blk_sz_B;
    idx_map_B     = o->idx_map_B;

    order_C        = o->order_C;
    blk_sz_C      = o->blk_sz_C;
    idx_map_C     = o->idx_map_C;
  }

  /**
   * \brief copies ctr object
   */
  ctr * ctr_virt::clone() {
    return new ctr_virt(this);
  }

  /**
   * \brief prints ctr object
   */
  void ctr_virt::print() {
    int i;
    printf("ctr_virt:\n");
    printf("blk_sz_A = %ld, blk_sz_B = %ld, blk_sz_C = %ld\n",
            blk_sz_A, blk_sz_B, blk_sz_C);
    for (i=0; i<num_dim; i++){
      printf("virt_dim[%d] = %d\n", i, virt_dim[i]);
    }
    rec_ctr->print();
  }


  /**
   * \brief returns the number of bytes send by each proc recursively 
   * \return bytes needed for recursive contraction
   */
  double ctr_virt::est_time_rec(int nlyr) {
    /* FIXME: for now treat flops like comm, later make proper cost */
    int64_t nvirt = 1;
    for (int dim=0; dim<num_dim; dim++){
      nvirt *= virt_dim[dim];
    }
    return nvirt*rec_ctr->est_time_rec(nlyr);
  }


  /**
   * \brief returns the number of bytes of buffer space
     we need
   * \return bytes needed
   */
  int64_t ctr_virt::mem_fp(){
    return (order_A+order_B+order_C+(3+VIRT_NTD)*num_dim)*sizeof(int);
  }

  /**
   * \brief returns the number of bytes of buffer space we need recursively 
   * \return bytes needed for recursive contraction
   */
  int64_t ctr_virt::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }


  /**
   * \brief iterates over the dense virtualization block grid and contracts
   */
  void ctr_virt::run(){
    TAU_FSTART(ctr_virt);
    int * idx_arr, * tidx_arr, * lda_A, * lda_B, * lda_C, * beta_arr;
    int * ilda_A, * ilda_B, * ilda_C;
    int64_t i, off_A, off_B, off_C;
    int nb_A, nb_B, nb_C, alloced, ret; 

    /*if (this->buffer != NULL){    
      alloced = 0;
      idx_arr = (int*)this->buffer;
    } else {*/
      alloced = 1;
      ret = CTF_int::alloc_ptr(mem_fp(), (void**)&idx_arr);
      ASSERT(ret==0);
//    }

    
    lda_A = idx_arr + VIRT_NTD*num_dim;
    lda_B = lda_A + order_A;
    lda_C = lda_B + order_B;
    ilda_A = lda_C + order_C;
    ilda_B = ilda_A + num_dim;
    ilda_C = ilda_B + num_dim;

  #define SET_LDA_X(__X)                                                  \
  do {                                                                    \
    nb_##__X = 1;                                                         \
    for (i=0; i<order_##__X; i++){                                         \
      lda_##__X[i] = nb_##__X;                                            \
      nb_##__X = nb_##__X*virt_dim[idx_map_##__X[i]];                     \
    }                                                                     \
    memset(ilda_##__X, 0, num_dim*sizeof(int));                           \
    for (i=0; i<order_##__X; i++){                                         \
      ilda_##__X[idx_map_##__X[i]] += lda_##__X[i];                       \
    }                                                                     \
  } while (0)
    SET_LDA_X(A);
    SET_LDA_X(B);
    SET_LDA_X(C);
  #undef SET_LDA_X
   
    /* dynammically determined size */ 
    beta_arr = (int*)CTF_int::alloc(sizeof(int)*nb_C);
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

        ctr * tid_rec_ctr;
        if (tid > 0)
          tid_rec_ctr = rec_ctr->clone();
        else
          tid_rec_ctr = rec_ctr;
        
        tid_rec_ctr->num_lyr = this->num_lyr;
        tid_rec_ctr->idx_lyr = this->idx_lyr;

        off_A = 0, off_B = 0, off_C = 0;
        for (;;){
          if (off_C >= start_off && off_C < end_off) {
            tid_rec_ctr->A        = this->A + off_A*blk_sz_A*sr_A->el_size;
            tid_rec_ctr->B        = this->B + off_B*blk_sz_B*sr_A->el_size;
            tid_rec_ctr->C        = this->C + off_C*blk_sz_C*sr_A->el_size;
            if (beta_arr[off_C]>0)
              rec_ctr->beta = sr_B->mulid();
            else
              rec_ctr->beta = this->beta; 
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
      CTF_int::cdealloc(idx_arr);
    }
    CTF_int::cdealloc(beta_arr);
    TAU_FSTOP(ctr_virt);
  }

}

