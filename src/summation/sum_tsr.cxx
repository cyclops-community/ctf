/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "sum_tsr.h"
#include "sym_seq_sum.h"
#include "spr_seq_sum.h"
#include "../interface/fun_term.h"
#include "../interface/idx_tensor.h"

namespace CTF_int {
  Fun_Term univar_function::operator()(Term const & A) const {
    return Fun_Term(A.clone(), this);
  }

  void univar_function::operator()(Term const & A, Term const & B) const {
    Fun_Term ft(A.clone(), this);
    ft.execute(B.execute());
  }

  tsum::tsum(tsum * other){
    A           = other->A;
    sr_A        = other->sr_A;
    alpha       = other->alpha;
    B           = other->B;
    beta        = other->beta;
    sr_B        = other->sr_B;

    buffer      = NULL;

    is_sparse_A = other->is_sparse_A;
    nnz_A       = other->nnz_A;
    nvirt_A     = other->nvirt_A;
    is_sparse_B = other->is_sparse_B;
    nnz_B       = other->nnz_B;
    nvirt_B     = other->nvirt_B;
    new_nnz_B   = other->new_nnz_B;
    new_B       = other->new_B;

    //nnz_blk_B should be copied by pointer, they are the same pointer as in tensor object
    nnz_blk_B   = other->nnz_blk_B;
    //nnz_blk_A should be copied by value, since it needs to be potentially set in replicate and deallocated later
    if (is_sparse_A){
      nnz_blk_A   = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, other->nnz_blk_A, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;
  }
  
  tsum::tsum(summation const * s){
    A           = s->A->data;
    sr_A        = s->A->sr;
    alpha       = s->alpha;

    B           = s->B->data;
    sr_B        = s->B->sr;
    beta        = s->beta;

    buffer      = NULL;

    is_sparse_A = s->A->is_sparse;
    nnz_A       = s->A->nnz_loc;
    nvirt_A     = s->A->calc_nvirt();

    is_sparse_B = s->B->is_sparse;
    nnz_B       = s->B->nnz_loc;
    nvirt_B     = s->B->calc_nvirt();

    if (is_sparse_A){
      nnz_blk_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, s->A->nnz_blk, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;

    nnz_blk_B   = s->B->nnz_blk;
    new_nnz_B   = nnz_B;
    new_B       = NULL;
  }

  tsum_virt::~tsum_virt() {
    CTF_int::cdealloc(virt_dim);
    delete rec_tsum;
  }

  tsum_virt::tsum_virt(tsum * other) : tsum(other) {
    tsum_virt * o = (tsum_virt*)other;
    rec_tsum      = o->rec_tsum->clone();
    num_dim       = o->num_dim;
    virt_dim      = (int*)CTF_int::alloc(sizeof(int)*num_dim);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*num_dim);

    order_A   = o->order_A;
    blk_sz_A  = o->blk_sz_A;
    nnz_blk_A = o->nnz_blk_A;
    idx_map_A = o->idx_map_A;

    order_B   = o->order_B;
    blk_sz_B  = o->blk_sz_B;
    nnz_blk_B = o->nnz_blk_B;
    idx_map_B = o->idx_map_B;
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
      ret = CTF_int::alloc_ptr(mem_fp(), (void**)&idx_arr);
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
    beta_arr = (int*)CTF_int::alloc(sizeof(int)*nb_B);
    
    int64_t * sp_offsets_A;
    if (is_sparse_A){
      sp_offsets_A = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nb_A);
      sp_offsets_A[0] = 0;
      for (int i=1; i<nb_A; i++){
        sp_offsets_A[i] = sp_offsets_A[i-1]+nnz_blk_A[i-1];
      }      
    }
 
    int64_t * sp_offsets_B;
    int64_t * new_sp_szs_B;
    char ** buckets_B;
    if (is_sparse_B){
      sp_offsets_B = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nb_B);
      new_sp_szs_B = nnz_blk_B; //(int64_t*)CTF_int::alloc(sizeof(int64_t)*nb_B);
//      memcpy(new_sp_szs_B, blk_sz_B, sizeof(int64_t)*nb_B);
      buckets_B = (char**)CTF_int::alloc(sizeof(char*)*nb_B);
      for (int i=0; i<nb_B; i++){
        if (i==0)
          sp_offsets_B[0] = 0;
        else
          sp_offsets_B[i] = sp_offsets_B[i-1]+nnz_blk_B[i-1];
        buckets_B[i] = this->B + sp_offsets_B[i]*this->sr_B->pair_size();
      }      
    }

   
    memset(idx_arr, 0, num_dim*sizeof(int));
    memset(beta_arr, 0, nb_B*sizeof(int));
    off_A = 0, off_B = 0;
    rec_tsum->alpha = this->alpha;
    rec_tsum->beta = this->beta;
    for (;;){
      if (is_sparse_A){
        rec_tsum->nnz_A = nnz_blk_A[off_A];
        rec_tsum->A = this->A + sp_offsets_A[off_A]*this->sr_A->pair_size();
      } else
        rec_tsum->A = this->A + off_A*blk_sz_A*this->sr_A->el_size;
      if (is_sparse_B){
        rec_tsum->nnz_B = new_sp_szs_B[off_B];
        rec_tsum->B = buckets_B[off_B];
      } else
        rec_tsum->B = this->B + off_B*blk_sz_B*this->sr_B->el_size;
//        sr_B->copy(rec_tsum->beta, sr_B->mulid());
      if (beta_arr[off_B]>0)
        rec_tsum->beta = sr_B->mulid();
      else
        rec_tsum->beta = this->beta; 
  
      rec_tsum->run();
      if (is_sparse_B){
        new_sp_szs_B[off_B] = rec_tsum->new_nnz_B;
        if (beta_arr[off_B] > 0) cdealloc(buckets_B[off_B]);
        buckets_B[off_B] = rec_tsum->new_B;
      }
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
    if (this->is_sparse_B){
      this->new_nnz_B = 0;
      for (int i=0; i<nb_B; i++){
        this->new_nnz_B += new_sp_szs_B[i];
      }
      new_B = (char*)alloc(this->new_nnz_B*this->sr_B->pair_size());
      int64_t pfx = 0;
      for (int i=0; i<nb_B; i++){
        memcpy(new_B+pfx, buckets_B[i], new_sp_szs_B[i]*this->sr_B->pair_size());
        pfx += new_sp_szs_B[i]*this->sr_B->pair_size();
        if (beta_arr[i] > 0) cdealloc(buckets_B[i]);
      }
      //FIXME: how to pass B back generally
      //cdealloc(this->B);
      cdealloc(buckets_B);
    }
    if (is_sparse_A) cdealloc(sp_offsets_A);
    if (is_sparse_B) cdealloc(sp_offsets_B);
    if (alloced){
      CTF_int::cdealloc(idx_arr);
    }
    CTF_int::cdealloc(beta_arr);
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
      CTF_int::cdealloc(cdt_A);
/*    for (int i=0; i<ncdt_B; i++){
      cdt_B[i]->deactivate();
    }*/
    if (ncdt_B > 0)
      CTF_int::cdealloc(cdt_B);
  }

  tsum_replicate::tsum_replicate(tsum * other) : tsum(other) {
    tsum_replicate * o = (tsum_replicate*)other;
    rec_tsum = o->rec_tsum->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
  }


  tsum_replicate::tsum_replicate(summation const * s) : tsum(s) {}

  tsum * tsum_replicate::clone() {
    return new tsum_replicate(this);
  }

  int64_t tsum_replicate::mem_fp(){
    return 0;
  }

  void tsum_replicate::run(){
    int brank, i;
    char * buf = this->A;
//    int64_t * save_nnz_blk_A = NULL;
    if (is_sparse_A){
      /*if (ncdt_A > 0){
        save_nnz_blk_A = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
        memcpy(save_nnz_blk_A,nnz_blk_A,sizeof(int64_t)*nvirt_A);
      }*/
      size_A = nnz_A;
      for (i=0; i<ncdt_A; i++){
        MPI_Bcast(&size_A, 1, MPI_INT64_T, 0, cdt_A[i]->cm);
        MPI_Bcast(nnz_blk_A, nvirt_A, MPI_INT64_T, 0, cdt_A[i]->cm);
      }
      MPI_Datatype md;
      bool need_free = get_mpi_dt(size_A, sr_A->pair_size(), md);
      
      if (nnz_A != size_A) 
        buf = (char*)alloc(sr_A->pair_size()*size_A);
      for (i=0; i<ncdt_A; i++){
        MPI_Bcast(buf, size_A, md, 0, cdt_A[i]->cm);
      }
      if (need_free) MPI_Type_free(&md);
    } else {
      for (i=0; i<ncdt_A; i++){
        MPI_Bcast(this->A, size_A, sr_A->mdtype(), 0, cdt_A[i]->cm);
      }
    }
    if (is_sparse_B){
      //FIXME: need to replicate nnz_blk_B for this
      assert(ncdt_B == 0);
      size_B = nnz_B;
      for (i=0; i<ncdt_B; i++){
        MPI_Bcast(&size_B, 1, MPI_INT64_T, 0, cdt_B[i]->cm);
        MPI_Bcast(nnz_blk_B, nvirt_B, MPI_INT64_T, 0, cdt_A[i]->cm);
      }
    }

   /* for (i=0; i<ncdt_B; i++){
      POST_BCAST(this->B, size_B*sizeof(dtype), COMM_CHAR_T, 0, cdt_B[i]-> 0);
    }*/
    ASSERT(ncdt_B == 0 || !is_sparse_B);
    brank = 0;
    for (i=0; i<ncdt_B; i++){
      brank += cdt_B[i]->rank;
    }
    if (brank != 0) sr_B->set(this->B, sr_B->addid(), size_B);

    rec_tsum->set_nnz_blk_A(this->nnz_blk_A);;
    rec_tsum->A         = buf;
    rec_tsum->nnz_A     = size_A;
    rec_tsum->B         = this->B;
    rec_tsum->nnz_B     = nnz_A;
    rec_tsum->nnz_blk_B = this->nnz_blk_B;
    rec_tsum->alpha     = this->alpha;
    if (brank != 0)
      rec_tsum->beta = sr_B->mulid();
    else
      rec_tsum->beta = this->beta; 

    rec_tsum->run();
    
    new_nnz_B = rec_tsum->new_nnz_B;
    new_B = rec_tsum->new_B;
    //printf("new_nnz_B = %ld\n",new_nnz_B);
    if (buf != this->A) cdealloc(buf);

    for (i=0; i<ncdt_B; i++){
      MPI_Allreduce(MPI_IN_PLACE, this->B, size_B, sr_B->mdtype(), sr_B->addmop(), cdt_B[i]->cm);
    }

/*    if (save_nnz_blk_A != NULL){
      memcpy(nnz_blk_A,save_nnz_blk_A,sizeof(int64_t)*nvirt_A);
    }*/

  }


  seq_tsr_sum::seq_tsr_sum(tsum * other) : tsum(other) {
    seq_tsr_sum * o = (seq_tsr_sum*)other;
    
    order_A    = o->order_A;
    idx_map_A  = o->idx_map_A;
    sym_A      = o->sym_A;
    edge_len_A = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int)*order_A);

    order_B    = o->order_B;
    idx_map_B  = o->idx_map_B;
    sym_B      = o->sym_B;
    edge_len_B = (int*)CTF_int::alloc(sizeof(int)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int)*order_B);
    
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
   
    //printf("A is sparse = %d, B is sparse = %d\n",  s->A->is_sparse, s->B->is_sparse);
    if (s->A->is_sparse && s->B->is_sparse){
      for (int i=0; i<s->B->order; i++){
        bool mapped = true;
        for (int j=0; j<s->A->order; j++){
          if (s->idx_B[i] == s->idx_A[j]){
            mapped = false;
          }
        }
        if (mapped)
          map_pfx *= s->B->pad_edge_len[i]/s->B->edge_map[i].calc_phase();
      } 
    }
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
    printf("map_pfx = %ld\n", map_pfx);
  }

  tsum * seq_tsr_sum::clone() {
    return new seq_tsr_sum(this);
  }

  int64_t seq_tsr_sum::mem_fp(){ return 0; }

  void seq_tsr_sum::run(){
    if (is_sparse_A && !is_sparse_B){
      spA_dnB_seq_sum(this->alpha,
                      this->A,
                      this->nnz_A,
                      this->sr_A,
                      this->beta,
                      this->B,
                      this->sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      func);
    } else if (!is_sparse_A && is_sparse_B){
      dnA_spB_seq_sum(this->alpha,
                      this->A,
                      this->sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      this->beta,
                      this->B,
                      this->nnz_B,
                      this->new_B,
                      this->new_nnz_B,
                      this->sr_B,
                      func);
    } else if (is_sparse_A && is_sparse_B){
      spA_spB_seq_sum(this->alpha,
                      this->A,
                      this->nnz_A,
                      this->sr_A,
                      this->beta,
                      this->B,
                      this->nnz_B,
                      this->new_B,
                      this->new_nnz_B,
                      this->sr_B,
                      func,
                      this->map_pfx);
    } else {
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

  tsum_sp_map::~tsum_sp_map() {
    cdealloc(map_idx_len);
    cdealloc(map_idx_lda);
  }


  tsum_sp_map::tsum_sp_map(tsum * other) : tsum(other) {
    tsum_sp_map * o = (tsum_sp_map*)other;
    rec_tsum    = o->rec_tsum->clone();
    nmap_idx    = o->nmap_idx;
    map_idx_len = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nmap_idx);
    map_idx_lda = (int64_t*)CTF_int::alloc(sizeof(int64_t)*nmap_idx);
    memcpy(map_idx_len, o->map_idx_len, sizeof(int64_t)*nmap_idx);
    memcpy(map_idx_lda, o->map_idx_lda, sizeof(int64_t)*nmap_idx);
  }

  tsum_sp_map::tsum_sp_map(summation const * s) : tsum(s) {
    nmap_idx = 0;
    map_idx_len = (int64_t*)CTF_int::alloc(sizeof(int64_t)*s->B->order);
    map_idx_lda = (int64_t*)CTF_int::alloc(sizeof(int64_t)*s->B->order);
    int map_idx_rev[s->B->order];

    int64_t lda_B[s->B->order];
    lda_B[0] = 1;
    for (int o=1; o<s->B->order; o++){
      if (s->B->is_sparse)
        lda_B[o] = lda_B[o-1]*s->B->lens[o];
      else
        lda_B[o] = lda_B[o-1]*s->B->pad_edge_len[o]/s->B->edge_map[o].calc_phase();
    }

    for (int oB=0; oB<s->B->order; oB++){
      bool inA = false;
      for (int oA=0; oA<s->A->order; oA++){
        if (s->idx_A[oA] == s->idx_B[oB]){
          inA = true;
        }
      }
      if (!inA){ 
        bool is_rep=false;
        for (int ooB=0; ooB<oB; ooB++){
          if (s->idx_B[ooB] == s->idx_B[oB]){
            is_rep = true;
            map_idx_lda[map_idx_rev[ooB]] += lda_B[oB];
            break;
          }
        }
        if (!is_rep){
          map_idx_len[nmap_idx] = s->B->lens[oB]/s->B->edge_map[oB].calc_phase() + (s->B->lens[oB]/s->B->edge_map[oB].calc_phase() > s->B->edge_map[oB].calc_phase());
          map_idx_lda[nmap_idx] = lda_B[oB];
          map_idx_rev[nmap_idx] = oB;
          nmap_idx++;
        }
      }
    }
  }

  tsum * tsum_sp_map::clone() {
    return new tsum_sp_map(this);
  }

  void tsum_sp_map::print(){
    printf("tsum_sp_map:\n");
    printf("namp_idx = %d\n",nmap_idx);
    rec_tsum->print();
  }
  
  int64_t tsum_sp_map::mem_fp(){
    int64_t mem = nnz_A*this->sr_A->pair_size();
    if (nmap_idx > 0){
      int64_t tot_rep=1;
      for (int midx=0; midx<nmap_idx; midx++){
        tot_rep *= map_idx_len[midx];
      }
      return tot_rep*mem;
    } else return mem;
  }

  void tsum_sp_map::run(){
    int64_t tot_rep=1;
    for (int midx=0; midx<nmap_idx; midx++){
      tot_rep *= map_idx_len[midx];
    }
    PairIterator pi(this->sr_A, A);
    char * buf;
    alloc_ptr(this->sr_A->pair_size()*nnz_A*tot_rep, (void**)&buf);
    //printf("pair size is %d, nnz is %ld\n",this->sr_A->pair_size(), nnz_A);
    PairIterator pi_new(this->sr_A, buf);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nnz_A; i++){
      for (int64_t r=0; r<tot_rep; r++){
        memcpy(pi_new[i*tot_rep+r].ptr, pi[i].ptr, this->sr_A->pair_size());
      }
    }
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nnz_A; i++){
      int64_t phase=1;
      for (int midx=0; midx<nmap_idx; midx++){
        int64_t stride=phase;
        phase *= map_idx_len[midx];
        for (int64_t r=0; r<tot_rep/phase; r++){
          for (int64_t m=1; m<map_idx_len[midx]; m++){
            for (int64_t s=0; s<stride; s++){
              ((int64_t*)(pi_new[i*tot_rep + r*phase + m*stride + s].ptr))[0] += m*map_idx_lda[midx];
            }
          }
        }
      }
    }
    nnz_A *= tot_rep;
    rec_tsum->nnz_A = nnz_A;
    rec_tsum->A = buf;
    rec_tsum->nnz_B = nnz_B;
    rec_tsum->B = B;
    for (int v=0; v<nvirt_A; v++){
      nnz_blk_A[v] *= tot_rep;
    }
    rec_tsum->set_nnz_blk_A(nnz_blk_A);
    rec_tsum->run();
    new_nnz_B = rec_tsum->new_nnz_B;
    new_B = rec_tsum->new_B;
    cdealloc(buf);
  }

  tsum_sp_permute::~tsum_sp_permute() {
    cdealloc(p);
    cdealloc(lens_new);
    cdealloc(lens_old);
  }

  tsum_sp_permute::tsum_sp_permute(tsum * other) : tsum(other) {
    tsum_sp_permute * o = (tsum_sp_permute*)other;
    rec_tsum = o->rec_tsum->clone();
    A_or_B   = o->A_or_B;
    order    = o->order;
    p        = (int*)CTF_int::alloc(sizeof(int)*order);
    lens_old = (int*)CTF_int::alloc(sizeof(int)*order);
    lens_new = (int*)CTF_int::alloc(sizeof(int)*order);
    memcpy(p, o->p, sizeof(int)*order);
    memcpy(lens_old, o->lens_old, sizeof(int)*order);
    memcpy(lens_new, o->lens_new, sizeof(int)*order);
  }

  tsum_sp_permute::tsum_sp_permute(summation const * s, bool A_or_B_, int const * lens) : tsum(s) {
    tensor * X, * Y;
    int const * idx_X, * idx_Y;
    A_or_B = A_or_B_;
    if (A_or_B){
      X = s->A;
      Y = s->B;
      idx_X = s->idx_A;
      idx_Y = s->idx_B;
    } else {
      X = s->B;
      Y = s->A;
      idx_X = s->idx_B;
      idx_Y = s->idx_A;
    }
    order = X->order;

    p           = (int*)CTF_int::alloc(sizeof(int)*order);
    lens_old    = (int*)CTF_int::alloc(sizeof(int)*order);
    lens_new    = (int*)CTF_int::alloc(sizeof(int)*order);

    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&lens_old);
    memcpy(lens_old, lens, sizeof(int)*this->order);
    CTF_int::alloc_ptr(sizeof(int)*this->order, (void**)&lens_new);
    memcpy(lens_new, lens, sizeof(int)*this->order);
/*    for (int i=0; i<this->order; i++){
      memcpy(lens_new, lens, sizeof(int)*this->order);
    }
    if (Y->is_sparse){
    } else {
      lens_new[i] = X->pad_edge_len[i]/X->edge_map[i].calc_phase();
    }*/
    if (A_or_B){
      // if A then ignore 'reduced' indices
      for (int i=0; i<this->order; i++){
        p[i] = -1;
        for (int j=0; j<Y->order; j++){
          if (idx_X[i] == idx_Y[j]){
            ASSERT(p[i] == -1); // no repeating indices allowed here!
            p[i] = j;
          }
        }       
      } 
    } else {
      // if B then put 'map' indices first
      int nmap_idx = 0;
      for (int i=0; i<this->order; i++){
        bool mapped = true;
        for (int j=0; j<Y->order; j++){
          if (idx_X[i] == idx_Y[j]){
            mapped = false;
          }
        }
        if (mapped) nmap_idx++;
      } 

      int nm = 0;
      for (int i=0; i<this->order; i++){
        p[i] = nm;
        for (int j=0; j<Y->order; j++){
          if (idx_X[i] == idx_Y[j]){
            ASSERT(p[i] == nm); // no repeating indices allowed here!
            p[i] = j+nmap_idx;
          }
        }
        if (p[i] == nm) nm++;
      } 
    }
  }

  tsum * tsum_sp_permute::clone() {
    return new tsum_sp_permute(this);
  }

  void tsum_sp_permute::print(){
    printf("tsum_sp_permute:\n");
    if (A_or_B) printf("permuting A\n");
    else        printf("permuting B\n");
    rec_tsum->print();
  }
  
  int64_t tsum_sp_permute::mem_fp(){
    int64_t mem = 0;
    if (A_or_B) mem+=nnz_A*sr_A->pair_size();
    else mem+=nnz_B*sr_B->pair_size();
    return mem;
  }

  void tsum_sp_permute::run(){
    char * buf;

    if (A_or_B){
      CTF_int::alloc_ptr(nnz_A*sr_A->pair_size(), (void**)&buf);
      rec_tsum->A = buf;
      rec_tsum->B = B;
      memcpy(buf, A, nnz_A*sr_A->pair_size());
      int64_t new_lda_A[order];
      memset(new_lda_A, 0, order*sizeof(int64_t));
      int64_t lda=1;
      for (int i=0; i<order; i++){
        for (int j=0; j<order; j++){
          if (p[j] == i){ 
            new_lda_A[j] = lda;
            lda *= lens_new[j];
          }
        }
      }
      ConstPairIterator rA(sr_A, A);
      PairIterator wA(sr_A, buf);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<nnz_A; i++){
        int64_t k = rA[i].k();
        int64_t k_new = 0;
        for (int j=0; j<order; j++){
          k_new += (k%lens_old[j])*new_lda_A[j];
          k = k/lens_old[j];
        }
        ((int64_t*)wA[i].ptr)[0] = k_new;
        memcpy(wA[i].d(), rA[i].d(), sr_A->el_size);
        //printf("value %lf old key %ld new key %ld\n",((double*)wA[i].d())[0], rA[i].k(), wA[i].k());
      }
      
      PairIterator mwA = wA;
      for (int v=0; v<nvirt_A; v++){
        mwA.sort(nnz_blk_A[v]);
        mwA = mwA[nnz_blk_A[v]];
      }
      rec_tsum->A = buf;
    } else {
      CTF_int::alloc_ptr(nnz_B*sr_B->pair_size(), (void**)&buf);
      rec_tsum->A = A;
      rec_tsum->B = buf;
      memcpy(buf, B, nnz_B*sr_B->pair_size());
      int64_t new_lda_B[order];
      int64_t lda=1;
      for (int i=0; i<order; i++){
        new_lda_B[p[i]] = lda;
        lda *= lens_new[i];
      }
      ConstPairIterator rB(sr_B, B);
      PairIterator wB(sr_B, buf);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<nnz_B; i++){
        int64_t k = rB[i].k();
        int64_t k_new = 0;
        for (int j=0; j<order; j++){
          k_new += (k%lens_old[j])*new_lda_B[j];
          k = k/lens_old[j];
        }
        ((int64_t*)wB[i].ptr)[0] = k_new;
        memcpy(wB[i].d(), rB[i].d(), sr_B->el_size);
      }

      PairIterator mwB = wB;
      for (int v=0; v<nvirt_B; v++){
        mwB.sort(nnz_blk_B[v]);
        mwB = mwB[nnz_blk_B[v]];
      }

      rec_tsum->B = buf;
    }

    rec_tsum->nnz_A = nnz_A;
    rec_tsum->nnz_B = nnz_B;
    rec_tsum->run();

    new_nnz_B = rec_tsum->new_nnz_B;
    if (A_or_B){
      new_B = rec_tsum->new_B;
      cdealloc(buf);
    } else {
      if (nnz_B == new_nnz_B){
        new_B = B;
      } else {
        new_B = (char*)alloc(new_nnz_B*sr_B->pair_size());
      }
      int inv_p[order];
      for (int i=0; i<order; i++){
        inv_p[p[i]] = i;
      }
      int64_t new_lda_B[order];
      int64_t lda=1;
      for (int i=0; i<order; i++){
        new_lda_B[inv_p[i]] = lda;
        lda *= lens_old[i];
      }
      ConstPairIterator rB(sr_B, rec_tsum->new_B);
      PairIterator wB(sr_B, new_B);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<new_nnz_B; i++){
        int64_t k = rB[i].k();
        int64_t k_new = 0;
        for (int j=0; j<order; j++){
          k_new += (k%lens_new[j])*new_lda_B[j];
          k = k/lens_new[j];
        }
        ((int64_t*)wB[i].ptr)[0] = k_new;
        memcpy(wB[i].d(), rB[i].d(), sr_B->el_size);
        //printf("value %lf old key %ld new key %ld\n",((double*)wB[i].d())[0], rB[i].k(), wB[i].k());
      }
      PairIterator mwB = wB;
      for (int v=0; v<nvirt_B; v++){
        mwB.sort(nnz_blk_B[v]);
        mwB = mwB[nnz_blk_B[v]];
      }


      if (buf != rec_tsum->new_B){
        cdealloc(rec_tsum->new_B);
      }
      cdealloc(buf);
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
    *idx_arr = (int*)CTF_int::alloc(sizeof(int)*2*dim_max);
    std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[2*idx_A[i]] = i;
    }
    for (i=0; i<order_B; i++){
      (*idx_arr)[2*idx_B[i]+1] = i;
    }
  }

  tsum_sp_pin_keys::~tsum_sp_pin_keys(){
    cdealloc(divisor);
    cdealloc(virt_dim);
    delete rec_tsum;
  }
  
  tsum_sp_pin_keys::tsum_sp_pin_keys(tsum * other) : tsum(other) {
    tsum_sp_pin_keys * o = (tsum_sp_pin_keys*)other;

    rec_tsum  = o->rec_tsum->clone();
    A_or_B    = o->A_or_B;
    order     = o->order;
    lens      = o->lens;
    divisor   = (int*)CTF_int::alloc(sizeof(int)*order);
    phys_rank = (int*)CTF_int::alloc(sizeof(int)*order);
    virt_dim  = (int*)CTF_int::alloc(sizeof(int)*order);
    memcpy(divisor, o->divisor, sizeof(int)*order);
    memcpy(phys_rank, o->phys_rank, sizeof(int)*order);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*order);
  }

  tsum * tsum_sp_pin_keys::clone() {
    return new tsum_sp_pin_keys(this);
  }

  void tsum_sp_pin_keys::print(){
    printf("tsum_sp_pin_keys:\n");
    if (A_or_B) printf("transforming global keys of A to local keys\n");
    else        printf("transforming global keys of B to local keys\n");
    rec_tsum->print();
  }

  int64_t tsum_sp_pin_keys::mem_fp(){
    return 3*order*sizeof(int);
  }

  tsum_sp_pin_keys::tsum_sp_pin_keys(summation const * s, bool A_or_B_) : tsum(s) {
    tensor * X;
    A_or_B = A_or_B_;
    if (A_or_B){
      X = s->A;
    } else {
      X = s->B;
    }
    order = X->order;
    lens = X->lens;

    divisor = (int*)CTF_int::alloc(sizeof(int)*order);
    phys_rank = (int*)CTF_int::alloc(sizeof(int)*order);
    virt_dim = (int*)CTF_int::alloc(sizeof(int*)*order);

    for (int i=0; i<order; i++){
      divisor[i] = X->edge_map[i].calc_phase();
      phys_rank[i] = X->edge_map[i].calc_phys_rank(X->topo);
      virt_dim[i] = divisor[i]/X->edge_map[i].calc_phys_phase();
    }
  }

  void tsum_sp_pin_keys::run(){
    char * X;
//    char * buf;
    algstrct const * sr;
    int64_t nnz;
    int64_t * nnz_blk;
    int nvirt;
    if (A_or_B){
      X = this->A;
      sr = this->sr_A;
      nnz = this->nnz_A;
    } else {
      X = this->B;
      sr = this->sr_B;
      nnz = this->nnz_B;
    }
//    CTF_int::alloc_ptr(nnz*sr->pair_size(), (void**)&buf);
//    memcpy(buf, X, nnz*sr->pair_size());

    int * div_lens;
    CTF_int::alloc_ptr(order*sizeof(int), (void**)&div_lens);
    for (int64_t j=0; j<order; j++){
      div_lens[j] = (lens[j]/divisor[j] + (lens[j]%divisor[j] > 0));
    }

    ConstPairIterator pi(sr, X);
    PairIterator pi_new(sr, X);

    for (int64_t i=0; i<nnz; i++){
      int64_t key = pi[i].k();
      int64_t new_key = 0;
      int64_t lda = 1;
      for (int64_t j=0; j<order; j++){
        new_key += ((key%lens[j])/divisor[j])*lda;
        lda *= div_lens[j];
        key = key/lens[j];
      }
      ((int64_t*)pi_new[i].ptr)[0] = new_key;
        //printf("in key = %ld, new_key = %ld, val = %lf\n", pi[i].k(), new_key, ((double*)pi_new[i].d())[0]);
    }

    if (A_or_B){
      rec_tsum->A = X;
      rec_tsum->B = B;
    } else {
      rec_tsum->A = A;
      rec_tsum->B = X;
    }
    rec_tsum->nnz_A = nnz_A;
    rec_tsum->nnz_B = nnz_B;
    rec_tsum->run();

    new_nnz_B = rec_tsum->new_nnz_B;
    new_B = rec_tsum->new_B;

    if (!A_or_B){
      X = new_B;
      nnz_blk = nnz_blk_B;
      nvirt = nvirt_B;
    } else {
      nvirt = nvirt_A;
      nnz_blk = nnz_blk_A;
    }

    int * virt_offset;
    CTF_int::alloc_ptr(order*sizeof(int), (void**)&virt_offset);
    int64_t nnz_off = 0;
    for (int v=0; v<nvirt; v++){
      //printf("%d %p new_B %p pin %p new_blk_nnz_B[%d] = %ld\n",A_or_B,this,new_B,nnz_blk,v,nnz_blk[v]);
      int vv=v;
      for (int64_t j=0; j<order; j++){
        virt_offset[j] = (vv%virt_dim[j])*(divisor[j]/virt_dim[j])+phys_rank[j];
        vv=vv/virt_dim[j];
      }
      ConstPairIterator vpi(sr, X+nnz_off*sr->pair_size());
      PairIterator vpi_new(sr, X+nnz_off*sr->pair_size());
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int64_t i=0; i<nnz_blk[v]; i++){
        int64_t key = vpi[i].k();
        int64_t new_key = 0;
        int64_t lda = 1;
        for (int64_t j=0; j<order; j++){
          new_key += ((key%div_lens[j])*divisor[j]+virt_offset[j])*lda;
          lda *= lens[j];
          key = key/div_lens[j];
        }
        //printf("key = %ld, new_key = %ld, val = %lf\n", vpi[i].k(), new_key, ((double*)vpi_new[i].d())[0]);
        ((int64_t*)vpi_new[i].ptr)[0] = new_key;
      }
      nnz_off += nnz_blk[v];
    }
    cdealloc(virt_offset);
    cdealloc(div_lens);

  }
}
