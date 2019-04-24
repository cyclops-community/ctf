/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "sum_tsr.h"
#include "sym_seq_sum.h"
#include "../interface/fun_term.h"
#include "../interface/idx_tensor.h"

namespace CTF_int {
  Unifun_Term univar_function::operator()(Term const & A) const {
    return Unifun_Term(A.clone(), this);
  }

  void univar_function::operator()(Term const & A, Term const & B) const {
    Unifun_Term ft(A.clone(), this);
    ft.execute(B.execute(B.get_uniq_inds()));
  }

  tsum::tsum(tsum * other){
    A           = other->A;
    sr_A        = other->sr_A;
    alpha       = other->alpha;
    B           = other->B;
    beta        = other->beta;
    sr_B        = other->sr_B;

    buffer      = NULL;
  }
  
  tsum::~tsum(){
    if (buffer != NULL) cdealloc(buffer); 
  }

  tsum::tsum(summation const * s){
    A           = s->A->data;
    sr_A        = s->A->sr;
    alpha       = s->alpha;

    B           = s->B->data;
    sr_B        = s->B->sr;
    beta        = s->beta;

    buffer      = NULL;
  }

  tsum_virt::~tsum_virt() {
    cdealloc(virt_dim);
    delete rec_tsum;
  }

  tsum_virt::tsum_virt(tsum * other) : tsum(other) {
    tsum_virt * o = (tsum_virt*)other;
    rec_tsum      = o->rec_tsum->clone();
    num_dim       = o->num_dim;
    virt_dim      = (int*)alloc(sizeof(int)*num_dim);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);
  }

  tsum_virt::tsum_virt(summation const * s) : tsum(s) {
    order_A   = s->A->order;
    idx_map_A = s->idx_A;
    order_B   = s->B->order;
    idx_map_B = s->idx_B;
  }

  tsum * tsum_virt::clone() {
    return new tsum_virt(this);
  }

  void tsum_virt::print(){
    int i;
    printf("tsum_virt:\n");
    printf("blk_sz_A = %ld, blk_sz_B = %ld\n",
            blk_sz_A, blk_sz_B);
    for (i=0; i<num_dim; i++){
      printf("virt_dim[%d] = %d\n", i, virt_dim[i]);
    }
    rec_tsum->print();
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
      ret = alloc_ptr(mem_fp(), (void**)&idx_arr);
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
    beta_arr = (int*)alloc(sizeof(int)*nb_B);
   
    memset(idx_arr, 0, num_dim*sizeof(int));
    memset(beta_arr, 0, nb_B*sizeof(int));
    off_A = 0, off_B = 0;
    rec_tsum->alpha = this->alpha;
    rec_tsum->beta = this->beta;
    for (;;){
      rec_tsum->A = this->A + off_A*blk_sz_A*this->sr_A->el_size;
      rec_tsum->B = this->B + off_B*blk_sz_B*this->sr_B->el_size;
//        sr_B->copy(rec_tsum->beta, sr_B->mulid());
      if (beta_arr[off_B]>0)
        rec_tsum->beta = sr_B->mulid();
      else
        rec_tsum->beta = this->beta; 
  
      rec_tsum->run();
       beta_arr[off_B] = 1;

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
      cdealloc(idx_arr);
    }
    cdealloc(beta_arr);
    TAU_FSTOP(sum_virt);
  }

  void tsum_replicate::print(){
    int i;
    printf("tsum_replicate: \n");
    printf("cdt_A = %p, size_A = %ld, ncdt_A = %d\n",
            cdt_A, size_A, ncdt_A);
    for (i=0; i<ncdt_A; i++){
      printf("cdt_A[%d] length = %d\n",i,cdt_A[i]->np);
    }
    printf("cdt_B = %p, size_B = %ld, ncdt_B = %d\n",
            cdt_B, size_B, ncdt_B);
    for (i=0; i<ncdt_B; i++){
      printf("cdt_B[%d] length = %d\n",i,cdt_B[i]->np);
    }

    rec_tsum->print();
  }

  tsum_replicate::~tsum_replicate() {
    delete rec_tsum;
/*    for (int i=0; i<ncdt_A; i++){
      cdt_A[i]->deactivate();
    }*/
    if (ncdt_A > 0)
      cdealloc(cdt_A);
/*    for (int i=0; i<ncdt_B; i++){
      cdt_B[i]->deactivate();
    }*/
    if (ncdt_B > 0)
      cdealloc(cdt_B);
  }

  tsum_replicate::tsum_replicate(tsum * other) : tsum(other) {
    tsum_replicate * o = (tsum_replicate*)other;
    rec_tsum = o->rec_tsum->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
  }


  tsum_replicate::tsum_replicate(summation const *           s,
                                 int const *                 phys_mapped,
                                 int64_t                     blk_sz_A,
                                 int64_t blk_sz_B) : tsum(s) {
    int i;
    int nphys_dim = s->A->topo->order;
    this->ncdt_A = 0;
    this->ncdt_B = 0;
    this->size_A = blk_sz_A;
    this->size_B = blk_sz_B;
    this->cdt_A  = NULL;
    this->cdt_B  = NULL;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
        this->ncdt_A++;
      }
      if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
        this->ncdt_B++;
      }
    }
    if (this->ncdt_A > 0)
      CTF_int::alloc_ptr(sizeof(CommData*)*this->ncdt_A, (void**)&this->cdt_A);
    if (this->ncdt_B > 0)
      CTF_int::alloc_ptr(sizeof(CommData*)*this->ncdt_B, (void**)&this->cdt_B);
    this->ncdt_A = 0;
    this->ncdt_B = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
        this->cdt_A[this->ncdt_A] = &s->A->topo->dim_comm[i];
        this->ncdt_A++;
      }
      if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
        this->cdt_B[this->ncdt_B] = &s->B->topo->dim_comm[i];
        this->ncdt_B++;
      }
    }
    ASSERT(this->ncdt_A == 0 || this->cdt_B == 0);

  }

  tsum * tsum_replicate::clone() {
    return new tsum_replicate(this);
  }

  int64_t tsum_replicate::mem_fp(){
    return 0;
  }

  void tsum_replicate::run(){
    int brank, i;
    char * buf = this->A;
    for (i=0; i<ncdt_A; i++){
      cdt_A[i]->bcast(this->A, size_A, sr_A->mdtype(), 0);
    }

   /* for (i=0; i<ncdt_B; i++){
      POST_BCAST(this->B, size_B*sizeof(dtype), COMM_CHAR_T, 0, cdt_B[i]-> 0);
    }*/
    brank = 0;
    for (i=0; i<ncdt_B; i++){
      brank += cdt_B[i]->rank;
    }
    if (brank != 0) sr_B->set(this->B, sr_B->addid(), size_B);

    rec_tsum->A         = buf;
    rec_tsum->B         = this->B;
    rec_tsum->alpha     = this->alpha;
    if (brank != 0)
      rec_tsum->beta = sr_B->mulid();
    else
      rec_tsum->beta = this->beta; 

    rec_tsum->run();
    
    if (buf != this->A) cdealloc(buf);

    for (i=0; i<ncdt_B; i++){
      cdt_B[i]->allred(MPI_IN_PLACE, this->B, size_B, sr_B->mdtype(), sr_B->addmop());
    }

  }


  seq_tsr_sum::seq_tsr_sum(tsum * other) : tsum(other) {
    seq_tsr_sum * o = (seq_tsr_sum*)other;
    
    order_A    = o->order_A;
    idx_map_A  = o->idx_map_A;
    sym_A      = o->sym_A;
    edge_len_A = (int64_t*)alloc(sizeof(int64_t)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int64_t)*order_A);

    order_B    = o->order_B;
    idx_map_B  = o->idx_map_B;
    sym_B      = o->sym_B;
    edge_len_B = (int64_t*)alloc(sizeof(int64_t)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int64_t)*order_B);
    
    is_inner   = o->is_inner;
    inr_stride = o->inr_stride;
    
    map_pfx    = o->map_pfx;

    is_custom  = o->is_custom;
    func       = o->func;
  }
  
  seq_tsr_sum::seq_tsr_sum(summation const * s) : tsum(s) {
    order_A   = s->A->order;
    sym_A     = s->A->sym;
    idx_map_A = s->idx_A;
    order_B   = s->B->order;
    sym_B     = s->B->sym;
    idx_map_B = s->idx_B;
    is_custom = s->is_custom;

    map_pfx = 1;
  }

  void seq_tsr_sum::print(){
    int i;
    printf("seq_tsr_sum:\n");
    for (i=0; i<order_A; i++){
      printf("edge_len_A[%d]=%ld\n",i,edge_len_A[i]);
    }
    for (i=0; i<order_B; i++){
      printf("edge_len_B[%d]=%ld\n",i,edge_len_B[i]);
    }
    printf("is inner = %d\n", is_inner);
    if (is_inner) printf("inner stride = %d\n", inr_stride);
    printf("map_pfx = %ld\n", map_pfx);
  }

  tsum * seq_tsr_sum::clone() {
    return new seq_tsr_sum(this);
  }

  int64_t seq_tsr_sum::mem_fp(){ return 0; }

  void seq_tsr_sum::run(){
    if (is_custom){
      ASSERT(is_inner == 0);
      sym_seq_sum_cust(this->alpha,
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
}
