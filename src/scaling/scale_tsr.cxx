/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "scale_tsr.h"

namespace CTF_int { 
  /**
   * \brief copies generic scl object
   */
  scl::scl(scl * other){
    A = other->A;
    alpha = other->alpha;
    buffer = NULL;
  }

  /**
   * \brief deallocates scl_virt object
   */
  scl_virt::~scl_virt() {
    CTF_int::cdealloc(virt_dim);
    delete rec_scl;
  }

  /**
   * \brief copies scl object
   */
  scl_virt::scl_virt(scl * other) : scl(other) {
    scl_virt * o = (scl_virt*)other;
    rec_scl      = o->rec_scl->clone();
    num_dim      = o->num_dim;
    virt_dim     = (int*)CTF_int::alloc(sizeof(int)*num_dim);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

    order_A      = o->order_A;
    blk_sz_A     = o->blk_sz_A;
    idx_map_A    = o->idx_map_A;
  }

  /**
   * \brief copies scl object
   */
  scl * scl_virt::clone() {
    return new scl_virt(this);
  }


  /**
   * \brief returns the number of bytes of buffer space
     we need
   * \return bytes needed
   */
  int64_t scl_virt::mem_fp(){
    return (order_A+2*num_dim)*sizeof(int);
  }

  /**
   * \brief iterates over the dense virtualization block grid and contracts
   */
  void scl_virt::run(){
    int * idx_arr, * lda_A;
    int * ilda_A;
    int i, off_A, nb_A, alloced, ret; 
    TAU_FSTART(scl_virt);

    if (this->buffer != NULL){    
      alloced = 0;
      idx_arr = (int*)this->buffer;
    } else {
      alloced = 1;
      ret = CTF_int::alloc_ptr(mem_fp(), (void**)&idx_arr);
      ASSERT(ret==0);
    }
    
    lda_A = idx_arr + num_dim;
    ilda_A = lda_A + order_A;
    

  #define SET_LDA_X(__X)                              \
  do {                                                \
    nb_##__X = 1;                                     \
    for (i=0; i<order_##__X; i++){                    \
      lda_##__X[i] = nb_##__X;                        \
      nb_##__X = nb_##__X*virt_dim[idx_map_##__X[i]]; \
    }                                                 \
    memset(ilda_##__X, 0, num_dim*sizeof(int));       \
    for (i=0; i<order_##__X; i++){                    \
      ilda_##__X[idx_map_##__X[i]] += lda_##__X[i];   \
    }                                                 \
  } while (0)
    SET_LDA_X(A);
  #undef SET_LDA_X
   
  /*  for (i=0; i<order_A; i++){
      printf("lda[%d] = %d idx_map_A[%d] = %d\n",i,lda_A[i],i,idx_map_A[i]);
    }
    for (i=0; i<num_dim; i++){
      printf("ilda[%d] = %d virt_dim[%d] = %d\n",i,ilda_A[i],i,virt_dim[i]);
    }*/
    memset(idx_arr, 0, num_dim*sizeof(int));
    rec_scl->alpha = this->alpha;
    off_A = 0;
    for (;;){
   /*   for (i=0; i<num_dim; i++){
        for (j=0; j<num_dim; j++){
          if (i!=j && idx_arr[i] != idx_arr[j] && idx_map[i] */
      rec_scl->A = this->A + off_A*blk_sz_A*sr_A->el_size;
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
      CTF_int::cdealloc(idx_arr);
    }
    TAU_FSTOP(scl_virt);
  }


  seq_tsr_scl::seq_tsr_scl(scl * other) : scl(other) {
    seq_tsr_scl * o = (seq_tsr_scl*)other;
    
    order           = o->order;
    idx_map         = o->idx_map;
    sym             = o->sym;
    edge_len        = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order);
    memcpy(edge_len, o->edge_len, sizeof(int64_t)*order);
    is_custom       = o->is_custom;
    func            = o->func;
  }

  scl * seq_tsr_scl::clone() {
    return new seq_tsr_scl(this);
  }

  int64_t seq_tsr_scl::mem_fp(){ return 0; }

  void seq_tsr_scl::run(){
    if (is_custom)
      sym_seq_scl_cust(alpha,
                       this->A,
                       sr_A,
                       order,
                       edge_len,
                       sym,
                       idx_map,
                       func);
    else
      sym_seq_scl_ref(alpha,
                      this->A,
                      sr_A,
                      order,
                      edge_len,
                      sym,
                      idx_map);
  }

  void seq_tsr_scl::print(){
    int i;
    printf("seq_tsr_scl:\n");
    printf("is_custom = %d\n",is_custom);
    for (i=0; i<order; i++){
      printf("edge_len[%d]=%ld\n",i,edge_len[i]);
    }
  }

}
