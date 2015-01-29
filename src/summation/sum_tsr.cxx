/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "sum_tsr.h"
#include "sym_seq_sum.h"

namespace CTF_int {
  tsum::tsum(tsum * other){
    A      = other->A;
    alpha  = other->alpha;
    sr_A   = other->sr_A;
    B      = other->B;
    beta   = other->beta;
    sr_B   = other->sr_B;
    buffer = NULL;
  }

  tsum_virt::~tsum_virt() {
    CTF_free(virt_dim);
    delete rec_tsum;
  }

  tsum_virt::tsum_virt(tsum * other) : tsum(other) {
    tsum_virt * o = (tsum_virt*)other;
    rec_tsum      = o->rec_tsum->clone();
    num_dim       = o->num_dim;
    virt_dim      = (int*)CTF_alloc(sizeof(int)*num_dim);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

    order_A       = o->order_A;
    blk_sz_A      = o->blk_sz_A;
    idx_map_A     = o->idx_map_A;

    order_B       = o->order_B;
    blk_sz_B      = o->blk_sz_B;
    idx_map_B     = o->idx_map_B;
  }

  tsum * tsum_virt::clone() {
    return new tsum_virt(this);
  }

  int64_t tsum_virt::mem_fp(){
    return (order_A+order_B+3*num_dim)*sizeof(int);
  }

  void tsum_virt::run(){
    int * idx_arr, * lda_A, * lda_B, * beta_arr;
    int * ilda_A, * ilda_B;
    int64_t i, off_A, off_B;
    int nb_A, nb_B, alloced, ret; 
    TAU_FSTART(sum_virt);

    if (this->buffer != NULL){    
      alloced = 0;
      idx_arr = (int*)this->buffer;
    } else {
      alloced = 1;
      ret = CTF_alloc_ptr(mem_fp(), (void**)&idx_arr);
      ASSERT(ret==0);
    }
    
    lda_A = idx_arr + num_dim;
    lda_B = lda_A + order_A;
    ilda_A = lda_B + order_B;
    ilda_B = ilda_A + num_dim;
    

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
    SET_LDA_X(B);
  #undef SET_LDA_X
    
    /* dynammically determined size */ 
    beta_arr = (int*)CTF_alloc(sizeof(int)*nb_B);
   
    memset(idx_arr, 0, num_dim*sizeof(int));
    memset(beta_arr, 0, nb_B*sizeof(int));
    off_A = 0, off_B = 0;
    rec_tsum->alpha = this->alpha;
    rec_tsum->beta = this->beta;
    for (;;){
      rec_tsum->A = this->A + off_A*blk_sz_A*this->sr_A.el_size;
      rec_tsum->B = this->B + off_B*blk_sz_B*this->sr_B.el_size;
//        sr_B.copy(rec_tsum->beta, sr_B.mulid);
      if (beta_arr[off_B]>0)
        rec_tsum->beta = sr_B.mulid;
      else
        rec_tsum->beta = this->beta; 
//        sr_B.copy(rec_tsum->beta, this->beta);
      beta_arr[off_B] = 1;
      rec_tsum->run();

      for (i=0; i<num_dim; i++){
        off_A -= ilda_A[i]*idx_arr[i];
        off_B -= ilda_B[i]*idx_arr[i];
        idx_arr[i]++;
        if (idx_arr[i] >= virt_dim[i])
          idx_arr[i] = 0;
        off_A += ilda_A[i]*idx_arr[i];
        off_B += ilda_B[i]*idx_arr[i];
        if (idx_arr[i] != 0) break;
      }
      if (i==num_dim) break;
    }
    if (alloced){
      CTF_free(idx_arr);
    }
    CTF_free(beta_arr);
    TAU_FSTOP(sum_virt);
  }


  tsum_replicate::~tsum_replicate() {
    delete rec_tsum;
    for (int i=0; i<ncdt_A; i++){
      cdt_A[i].deactivate();
    }
    if (ncdt_A > 0)
      CTF_free(cdt_A);
    for (int i=0; i<ncdt_B; i++){
      cdt_B[i].deactivate();
    }
    if (ncdt_B > 0)
      CTF_free(cdt_B);
  }

  tsum_replicate::tsum_replicate(tsum * other) : tsum(other) {
    tsum_replicate * o = (tsum_replicate*)other;
    rec_tsum = o->rec_tsum->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
  }

  tsum * tsum_replicate::clone() {
    return new tsum_replicate(this);
  }

  int64_t tsum_replicate::mem_fp(){
    return 0;
  }

  void tsum_replicate::run(){
    int brank, i;

    for (i=0; i<ncdt_A; i++){
      MPI_Bcast(this->A, size_A*sr_A.el_size, MPI_CHAR, 0, cdt_A[i].cm);
    }
   /* for (i=0; i<ncdt_B; i++){
      POST_BCAST(this->B, size_B*sizeof(dtype), COMM_CHAR_T, 0, cdt_B[i], 0);
    }*/
    brank = 0;
    for (i=0; i<ncdt_B; i++){
      brank += cdt_B[i].rank;
    }
    if (brank != 0) sr_B.set(this->B, sr_B.addid, size_B);

    rec_tsum->A           = this->A;
    rec_tsum->B           = this->B;
    rec_tsum->alpha       = this->alpha;
    if (brank != 0)
      rec_tsum->beta = sr_B.addid;
    else
      rec_tsum->beta = this->beta; 

    rec_tsum->run();
    
    for (i=0; i<ncdt_B; i++){
      /* FIXME Won't work for single precision */
      MPI_Allreduce(MPI_IN_PLACE, this->B, size_B, sr_B.mdtype, sr_B.addmop, cdt_B[i].cm);
    }

  }


  seq_tsr_sum::seq_tsr_sum(tsum * other) : tsum(other) {
    seq_tsr_sum * o = (seq_tsr_sum*)other;
    
    order_A    = o->order_A;
    idx_map_A  = o->idx_map_A;
    sym_A      = o->sym_A;
    edge_len_A = (int*)CTF_alloc(sizeof(int)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int)*order_A);

    order_B    = o->order_B;
    idx_map_B  = o->idx_map_B;
    sym_B      = o->sym_B;
    edge_len_B = (int*)CTF_alloc(sizeof(int)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int)*order_B);
    
    is_inner   = o->is_inner;
    inr_stride = o->inr_stride;

    is_custom  = o->is_custom;
    func       = o->func;
  }

  void seq_tsr_sum::print(){
    int i;
    printf("seq_tsr_sum:\n");
    for (i=0; i<order_A; i++){
      printf("edge_len_A[%d]=%d\n",i,edge_len_A[i]);
    }
    for (i=0; i<order_B; i++){
      printf("edge_len_B[%d]=%d\n",i,edge_len_B[i]);
    }
    printf("is inner = %d\n", is_inner);
    if (is_inner) printf("inner stride = %d\n", inr_stride);
  }

  tsum * seq_tsr_sum::clone() {
    return new seq_tsr_sum(this);
  }

  int64_t seq_tsr_sum::mem_fp(){ return 0; }

  void seq_tsr_sum::run(){
    if (is_custom){
      ASSERT(is_inner == 0);
      sym_seq_sum_cust(
                      this->A,
                      this->sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->B,
                      this->sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      func);
    } else if (is_inner){
      sym_seq_sum_inr(this->alpha,
                      this->A,
                      this->sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->beta,
                      this->B,
                      this->sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      inr_stride);
    } else {
      sym_seq_sum_ref(this->alpha,
                      this->A,
                      this->sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->beta,
                      this->B,
                      this->sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B);
    }
  }

  void inv_idx(int                order_A,
               int const *        idx_A,
               int                order_B,
               int const *        idx_B,
               int *              order_tot,
               int **             idx_arr){
    int i, dim_max;

    dim_max = -1;
    for (i=0; i<order_A; i++){
      if (idx_A[i] > dim_max) dim_max = idx_A[i];
    }
    for (i=0; i<order_B; i++){
      if (idx_B[i] > dim_max) dim_max = idx_B[i];
    }
    dim_max++;
    *order_tot = dim_max;
    *idx_arr = (int*)CTF_alloc(sizeof(int)*2*dim_max);
    std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[2*idx_A[i]] = i;
    }
    for (i=0; i<order_B; i++){
      (*idx_arr)[2*idx_B[i]+1] = i;
    }
  }

}
