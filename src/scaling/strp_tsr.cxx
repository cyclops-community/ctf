/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "strp_tsr.h"

namespace CTF_int {

  strp_tsr::strp_tsr(strp_tsr * o) {
    alloced       = o->alloced;
    order          = o->order;
    blk_sz        = o->blk_sz;
    edge_len      = o->edge_len;
    strip_dim     = o->strip_dim;
    strip_idx     = o->strip_idx;
    A             = o->A;
    buffer = NULL;
  }

  strp_tsr* strp_tsr::clone(){
    return new strp_tsr(this);
  }

  int64_t strp_tsr::mem_fp(){
    int i;
    int64_t sub_sz;
    sub_sz = blk_sz;
    for (i=0; i<order; i++){
      sub_sz = sub_sz * edge_len[i] / strip_dim[i];
    }
    return sub_sz*sr_A.el_size;
  }

  void strp_tsr::run(int const dir){
    TAU_FSTART(strp_tsr);
    int i, ilda, toff, boff, ret;
    int * idx_arr, * lda;
   
    if (dir == 0)  {
      if (buffer != NULL){        
        alloced = 0;
      } else {
        alloced = 1;
        ret = CTF_alloc_ptr(mem_fp(), (void**)&this->buffer);
        ASSERT(ret==0);
      }
    } 
    idx_arr = (int*)CTF_alloc(sizeof(int)*order);
    lda = (int*)CTF_alloc(sizeof(int)*order);
    memset(idx_arr, 0, sizeof(int)*order);

    ilda = 1, toff = 0;
    for (i=0; i<order; i++){
      lda[i] = ilda;
      ilda *= edge_len[i];
      idx_arr[i] = strip_idx[i]*(edge_len[i]/strip_dim[i]);
      toff += idx_arr[i]*lda[i];
      DPRINTF(3,"[%d] sidx = %d, sdim = %d, edge_len = %d\n", i, strip_idx[i], strip_dim[i], edge_len[i]);
    }
    
    boff = 0;
    for (;;){
      if (dir)
        sr_A.copy(A+toff*blk_sz, buffer+boff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz);
      else {
    /*    printf("boff = %d, toff = %d blk_sz = " PRId64 " mv_ez=" PRId64 "\n",boff,toff,blk_sz,
                (edge_len[0]/strip_dim[0])*blk_sz*sr_A.el_size);*/
        sr_A.copy(buffer+boff*blk_sz, A+toff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz);
      }
      boff += (edge_len[0]/strip_dim[0]);

      for (i=1; i<order; i++){
        toff -= idx_arr[i]*lda[i];
        idx_arr[i]++;
        if (idx_arr[i] >= (strip_idx[i]+1)*(edge_len[i]/strip_dim[i]))
          idx_arr[i] = strip_idx[i]*(edge_len[i]/strip_dim[i]);
        toff += idx_arr[i]*lda[i];
        if (idx_arr[i] != strip_idx[i]*(edge_len[i]/strip_dim[i])) break;
      }
      if (i==order) break;    
    }
    

    if (dir == 1) {
      if (alloced){
        CTF_free(buffer);
        buffer = NULL;
      }
    }
    CTF_free(idx_arr);
    CTF_free(lda);
    TAU_FSTOP(strp_tsr);
  }

  void strp_tsr::free_exp(){
    if (alloced){
      CTF_free(buffer);
      buffer = NULL;
    }
  }

  strp_sum::~strp_sum(){
    delete rec_tsum;
    if (strip_A)
      delete rec_strp_A;
    if (strip_B)
      delete rec_strp_B;
  }

  strp_sum::strp_sum(tsum * other) : tsum(other) {
    strp_sum * o   = (strp_sum*)other;
    rec_tsum      = o->rec_tsum->clone();
    rec_strp_A    = o->rec_strp_A->clone();
    rec_strp_B    = o->rec_strp_B->clone();
    strip_A       = o->strip_A;
    strip_B       = o->strip_B;
  }

  tsum* strp_sum::clone() {
    return new strp_sum(this);
  }

  int64_t strp_sum::mem_fp(){
    return 0;
  }

  void strp_sum::run(){
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


  strp_ctr::~strp_ctr(){
    delete rec_ctr;
    if (strip_A)
      delete rec_strp_A;
    if (strip_B)
      delete rec_strp_B;
    if (strip_C)
      delete rec_strp_C;
  }

  strp_ctr::strp_ctr(ctr * other) : ctr(other) {
    strp_ctr * o   = (strp_ctr*)other;
    rec_ctr       = o->rec_ctr->clone();
    rec_strp_A    = o->rec_strp_A->clone();
    rec_strp_B    = o->rec_strp_B->clone();
    rec_strp_C    = o->rec_strp_C->clone();
    strip_A       = o->strip_A;
    strip_B       = o->strip_B;
    strip_C       = o->strip_C;
  }

  ctr* strp_ctr::clone() {
    return new strp_ctr(this);
  }

  int64_t strp_ctr::mem_fp(){
    return 0;
  }

  int64_t strp_ctr::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  double strp_ctr::est_time_rec(int nlyr) {
    return rec_ctr->est_time_rec(nlyr);
  }


  void strp_ctr::run(){
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

    strp_scl::~strp_scl(){
    delete rec_scl;
    delete rec_strp;
  }

  strp_scl::strp_scl(scl * other) : scl(other) {
    strp_scl * o   = (strp_scl*)other;
    rec_scl       = o->rec_scl->clone();
    rec_strp      = o->rec_strp->clone();
  }

  scl* strp_scl::clone() {
    return new strp_scl(this);
  }

  int64_t strp_scl::mem_fp(){
    return 0;
  }

  void strp_scl::run(){
    dtype * bA;


    rec_strp->A = this->A;
    rec_strp->run(0);
    bA = rec_strp->buffer;

  /*  printf("alpha = %lf %lf\n",
            ((std::complex<double>)this->alpha).real(),
            ((std::complex<double>)this->alpha).imag());
    printf("A[0] = %lf %lf\n",
            ((std::complex<double>)bA[0]).real(),
            ((std::complex<double>)bA[0]).imag());*/
    
    rec_scl->A = bA;
    rec_scl->alpha = this->alpha;
    rec_scl->run();
    
    rec_strp->run(1);
  }
}
