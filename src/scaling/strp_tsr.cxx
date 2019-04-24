/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "strp_tsr.h"

namespace CTF_int {

  strp_tsr::strp_tsr(strp_tsr * o) {
    alloced   = o->alloced;
    order     = o->order;
    blk_sz    = o->blk_sz;
    edge_len  = o->edge_len;
    strip_dim = o->strip_dim;
    strip_idx = o->strip_idx;
    A         = o->A;
    buffer    = NULL;
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
    return sub_sz*sr_A->el_size;
  }

  void strp_tsr::run(int const dir){
    TAU_FSTART(strp_tsr);
    int i, toff, boff, ret;
    int64_t ilda;
    int64_t * idx_arr, * lda;
   
    if (dir == 0)  {
      if (buffer != NULL){        
        alloced = 0;
      } else {
        alloced = 1;
        ret = CTF_int::alloc_ptr(mem_fp(), (void**)&this->buffer);
        ASSERT(ret==0);
      }
    } 
    idx_arr = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order);
    lda = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order);
    memset(idx_arr, 0, sizeof(int64_t)*order);

    ilda = 1, toff = 0;
    for (i=0; i<order; i++){
      lda[i] = ilda;
      ilda *= edge_len[i];
      idx_arr[i] = strip_idx[i]*(edge_len[i]/strip_dim[i]);
      toff += idx_arr[i]*lda[i];
      DPRINTF(3,"[%d] sidx = %ld, sdim = %ld, edge_len = %ld\n", i, strip_idx[i], strip_dim[i], edge_len[i]);
    }
    
    boff = 0;
    for (;;){
      if (dir)
        sr_A->copy(A+sr_A->el_size*toff*blk_sz, buffer+sr_A->el_size*boff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz);
      else {
    /*    printf("boff = %d, toff = %d blk_sz = " PRId64 " mv_ez=" PRId64 "\n",boff,toff,blk_sz,
                (edge_len[0]/strip_dim[0])*blk_sz*sr_A->el_size);*/
        sr_A->copy(buffer+sr_A->el_size*boff*blk_sz, A+sr_A->el_size*toff*blk_sz, (edge_len[0]/strip_dim[0])*blk_sz);
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
        CTF_int::cdealloc(buffer);
        buffer = NULL;
      }
    }
    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(lda);
    TAU_FSTOP(strp_tsr);
  }

  void strp_tsr::free_exp(){
    if (alloced){
      CTF_int::cdealloc(buffer);
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
    strp_sum * o = (strp_sum*)other;
    rec_tsum     = o->rec_tsum->clone();
    rec_strp_A   = o->rec_strp_A->clone();
    rec_strp_B   = o->rec_strp_B->clone();
    strip_A      = o->strip_A;
    strip_B      = o->strip_B;
  }

  strp_sum::strp_sum(summation const * s) : tsum(s) { }

  tsum* strp_sum::clone() {
    return new strp_sum(this);
  }

  int64_t strp_sum::mem_fp(){
    return 0;
  }

  void strp_sum::run(){
    char * bA, * bB;

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
    strp_ctr * o = (strp_ctr*)other;
    rec_ctr      = o->rec_ctr->clone();
    rec_strp_A   = o->rec_strp_A->clone();
    rec_strp_B   = o->rec_strp_B->clone();
    rec_strp_C   = o->rec_strp_C->clone();
    strip_A      = o->strip_A;
    strip_B      = o->strip_B;
    strip_C      = o->strip_C;
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


  void strp_ctr::run(char * A, char * B, char * C){
    char * bA, * bB, * bC;

    if (strip_A) {
      rec_strp_A->A = A;
      rec_strp_A->run(0);
      bA = rec_strp_A->buffer;
    } else {
      bA = A;
    }
    if (strip_B) {
      rec_strp_B->A = B;
      rec_strp_B->run(0);
      bB = rec_strp_B->buffer;
    } else {
      bB = B;
    }
    if (strip_C) {
      rec_strp_C->A = C;
      rec_strp_C->run(0);
      bC = rec_strp_C->buffer;
    } else {
      bC = C;
    }

    
    rec_ctr->num_lyr      = this->num_lyr;
    rec_ctr->idx_lyr      = this->idx_lyr;
    rec_ctr->beta = this->beta;
    rec_ctr->run(bA, bB, bC);
    
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
    char * bA;

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

  int strip_diag(int              order,
                 int              order_tot,
                 int const *      idx_map,
                 int64_t          vrt_sz,
                 mapping const *  edge_map,
                 topology const * topo,
                 algstrct const * sr,
                 int64_t *        blk_edge_len,
                 int64_t *        blk_sz,
                 strp_tsr **      stpr){
    int64_t i;
    int need_strip;
    int * pmap;
    int64_t * edge_len, * sdim, * sidx;
    strp_tsr * stripper;

    CTF_int::alloc_ptr(order_tot*sizeof(int), (void**)&pmap);

    std::fill(pmap, pmap+order_tot, -1);

    need_strip = 0;

    for (i=0; i<order; i++){
      if (edge_map[i].type == PHYSICAL_MAP) {
        ASSERT(pmap[idx_map[i]] == -1);
        pmap[idx_map[i]] = i;
      }
    }
    for (i=0; i<order; i++){
      if (edge_map[i].type == VIRTUAL_MAP && pmap[idx_map[i]] != -1)
        need_strip = 1;
    }
    if (need_strip == 0) {
      CTF_int::cdealloc(pmap);
      return 0;
    }

    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&edge_len);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&sdim);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&sidx);
    stripper = new strp_tsr;

    std::fill(sdim, sdim+order, 1);
    std::fill(sidx, sidx+order, 0);

    for (i=0; i<order; i++){
      edge_len[i] = edge_map[i].calc_phase()/edge_map[i].calc_phys_phase();
      //if (edge_map[i].type == VIRTUAL_MAP) {
      //  edge_len[i] = edge_map[i].np;
      //}
      //if (edge_map[i].type == PHYSICAL_MAP && edge_map[i].has_child) {
        //dont allow recursive mappings for self indices
        // or things get weird here
        //ASSERT(edge_map[i].child->type == VIRTUAL_MAP);
      //  edge_len[i] = edge_map[i].child->np;
     // }
      if (edge_map[i].type == VIRTUAL_MAP && pmap[idx_map[i]] != -1) {
        sdim[i] = edge_len[i];
        sidx[i] = edge_map[pmap[idx_map[i]]].calc_phys_rank(topo);
        ASSERT(edge_map[i].np == edge_map[pmap[idx_map[i]]].np);
      }
      blk_edge_len[i] = blk_edge_len[i] / sdim[i];
      *blk_sz = (*blk_sz) / sdim[i];
    }

    stripper->alloced   = 0;
    stripper->order     = order;
    stripper->edge_len  = edge_len;
    stripper->strip_dim = sdim;
    stripper->strip_idx = sidx;
    stripper->buffer    = NULL;
    stripper->blk_sz    = vrt_sz;
    stripper->sr_A      = sr;

    *stpr               = stripper;

    CTF_int::cdealloc(pmap);

    return 1;
  }

  
}
