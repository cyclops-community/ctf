/*Copyright (c) 2015, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "spsum_tsr.h"
#include "spr_seq_sum.h"
#include "../interface/fun_term.h"
#include "../interface/idx_tensor.h"


namespace CTF_int {
  tspsum::tspsum(tspsum * other) : tsum(other) {
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
      nnz_blk_A   = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, other->nnz_blk_A, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;
  }
  
  tspsum::~tspsum(){
    if (buffer != NULL) cdealloc(buffer); 
    if (nnz_blk_A != NULL) cdealloc(nnz_blk_A);
  }

  tspsum::tspsum(summation const * s) : tsum(s) {
    is_sparse_A = s->A->is_sparse;
    nnz_A       = s->A->nnz_loc;
    nvirt_A     = s->A->calc_nvirt();

    is_sparse_B = s->B->is_sparse;
    nnz_B       = s->B->nnz_loc;
    nvirt_B     = s->B->calc_nvirt();

    if (is_sparse_A){
      nnz_blk_A = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, s->A->nnz_blk, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;

    nnz_blk_B   = s->B->nnz_blk;
    new_nnz_B   = nnz_B;
    new_B       = NULL;
  }

  tspsum_virt::~tspsum_virt() {
    cdealloc(virt_dim);
    delete rec_tsum;
  }

  tspsum_virt::tspsum_virt(tspsum * other) : tspsum(other) {
    tspsum_virt * o = (tspsum_virt*)other;
    rec_tsum      = o->rec_tsum->clone();
    num_dim       = o->num_dim;
    virt_dim      = (int*)alloc(sizeof(int)*num_dim);
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

  tspsum_virt::tspsum_virt(summation const * s) : tspsum(s) {
    order_A   = s->A->order;
    idx_map_A = s->idx_A;
    order_B   = s->B->order;
    idx_map_B = s->idx_B;
  }

  tspsum * tspsum_virt::clone() {
    return new tspsum_virt(this);
  }

  void tspsum_virt::print(){
    int i;
    printf("tspsum_virt:\n");
    printf("blk_sz_A = %ld, blk_sz_B = %ld\n",
            blk_sz_A, blk_sz_B);
    for (i=0; i<num_dim; i++){
      printf("virt_dim[%d] = %d\n", i, virt_dim[i]);
    }
    rec_tsum->print();
  }

  int64_t tspsum_virt::mem_fp(){
    return (order_A+order_B+3*num_dim)*sizeof(int);
  }

  void tspsum_virt::run(){
    int * idx_arr, * lda_A, * lda_B, * beta_arr;
    int * ilda_A, * ilda_B;
    int64_t i, off_A, off_B;
    int nb_A, nb_B, alloced, ret; 
    TAU_FSTART(spsum_virt);

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

    int64_t * sp_offsets_A = NULL;
    if (is_sparse_A){
      sp_offsets_A = (int64_t*)alloc(sizeof(int64_t)*nb_A);
      sp_offsets_A[0] = 0;
      for (int i=1; i<nb_A; i++){
        sp_offsets_A[i] = sp_offsets_A[i-1]+nnz_blk_A[i-1];
      }
    }

    int64_t * sp_offsets_B = NULL;
    int64_t * new_sp_szs_B = NULL;
    char ** buckets_B = NULL;
    if (is_sparse_B){
      sp_offsets_B = (int64_t*)alloc(sizeof(int64_t)*nb_B);
      new_sp_szs_B = nnz_blk_B; //(int64_t*)alloc(sizeof(int64_t)*nb_B);
//      memcpy(new_sp_szs_B, blk_sz_B, sizeof(int64_t)*nb_B);
      buckets_B = (char**)alloc(sizeof(char*)*nb_B);
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
        if (beta_arr[off_B] > 0) sr_B->pair_dealloc(buckets_B[off_B]);
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
    if (is_sparse_B){
      this->new_nnz_B = 0;
      for (int i=0; i<nb_B; i++){
        this->new_nnz_B += new_sp_szs_B[i];
      }
      new_B = sr_B->pair_alloc(this->new_nnz_B);
      int64_t pfx = 0;
      for (int i=0; i<nb_B; i++){
        //memcpy(new_B+pfx, buckets_B[i], new_sp_szs_B[i]*this->sr_B->pair_size());
        //printf("copying %ld pairs\n",new_sp_szs_B[i]);
        sr_B->copy_pairs(new_B+pfx, buckets_B[i], new_sp_szs_B[i]);
        pfx += new_sp_szs_B[i]*this->sr_B->pair_size();
        if (beta_arr[i] > 0) sr_B->pair_dealloc(buckets_B[i]);
      }
      //FIXME: how to pass B back generally
      //cdealloc(this->B);
      cdealloc(buckets_B);
    }
    if (is_sparse_A) cdealloc(sp_offsets_A);
    if (is_sparse_B) cdealloc(sp_offsets_B);
    if (alloced){
      cdealloc(idx_arr);
    }
    cdealloc(beta_arr);
    TAU_FSTOP(spsum_virt);
  }

  void tspsum_replicate::print(){
    int i;
    printf("tspsum_replicate: \n");
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

  tspsum_replicate::~tspsum_replicate() {
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

  tspsum_replicate::tspsum_replicate(tspsum * other) : tspsum(other) {
    tspsum_replicate * o = (tspsum_replicate*)other;
    rec_tsum = o->rec_tsum->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
  }


  tspsum_replicate::tspsum_replicate(summation const * s,
                                     int const *       phys_mapped,
                                     int64_t           blk_sz_A,
                                     int64_t           blk_sz_B)
       : tspsum(s) {
    //FIXME: might be smarter to use virtual inheritance and not replicate all the code from tsum_replicdate
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

  tspsum * tspsum_replicate::clone() {
    return new tspsum_replicate(this);
  }

  int64_t tspsum_replicate::mem_fp(){
    return 0;
  }

  void tspsum_replicate::run(){
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
        cdt_A[i]->bcast(&size_A, 1, MPI_INT64_T, 0);
        cdt_A[i]->bcast(nnz_blk_A, nvirt_A, MPI_INT64_T, 0);
      }
      //get mpi dtype for pair object
      MPI_Datatype md;
      bool need_free = get_mpi_dt(size_A, sr_A->pair_size(), md);
      
      if (nnz_A != size_A) 
        buf = (char*)alloc(sr_A->pair_size()*size_A);
      for (i=0; i<ncdt_A; i++){
        cdt_A[i]->bcast(buf, size_A, md, 0);
      }
      if (need_free) MPI_Type_free(&md);
    } else {
      for (i=0; i<ncdt_A; i++){
        cdt_A[i]->bcast(this->A, size_A, sr_A->mdtype(), 0);
      }
    }
    if (is_sparse_B){
      //FIXME: need to replicate nnz_blk_B for this
      assert(ncdt_B == 0);
      size_B = nnz_B;
      for (i=0; i<ncdt_B; i++){
        cdt_B[i]->bcast(&size_B, 1, MPI_INT64_T, 0);
        cdt_B[i]->bcast(nnz_blk_B, nvirt_B, MPI_INT64_T, 0);
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

    rec_tsum->set_nnz_blk_A(this->nnz_blk_A);
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
      cdt_B[i]->allred(MPI_IN_PLACE, this->B, size_B, sr_B->mdtype(), sr_B->addmop());
    }

/*    if (save_nnz_blk_A != NULL){
      memcpy(nnz_blk_A,save_nnz_blk_A,sizeof(int64_t)*nvirt_A);
    }*/

  }


  seq_tsr_spsum::seq_tsr_spsum(tspsum * other) : tspsum(other) {
    seq_tsr_spsum * o = (seq_tsr_spsum*)other;
    
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
  
  seq_tsr_spsum::seq_tsr_spsum(summation const * s) : tspsum(s) {
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

  void seq_tsr_spsum::print(){
    int i;
    printf("seq_tsr_spsum:\n");
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

  tspsum * seq_tsr_spsum::clone() {
    return new seq_tsr_spsum(this);
  }

  int64_t seq_tsr_spsum::mem_fp(){ return 0; }

  void seq_tsr_spsum::run(){
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
      assert(0); //we should be doing dense summation then
    }
  }

  tspsum_map::~tspsum_map() {
    delete rec_tsum;
    cdealloc(map_idx_len);
    cdealloc(map_idx_lda);
  }


  tspsum_map::tspsum_map(tspsum * other) : tspsum(other) {
    tspsum_map * o = (tspsum_map*)other;
    rec_tsum    = o->rec_tsum->clone();
    nmap_idx    = o->nmap_idx;
    map_idx_len = (int64_t*)alloc(sizeof(int64_t)*nmap_idx);
    map_idx_lda = (int64_t*)alloc(sizeof(int64_t)*nmap_idx);
    memcpy(map_idx_len, o->map_idx_len, sizeof(int64_t)*nmap_idx);
    memcpy(map_idx_lda, o->map_idx_lda, sizeof(int64_t)*nmap_idx);
  }

  tspsum_map::tspsum_map(summation const * s) : tspsum(s) {
    nmap_idx = 0;
    map_idx_len = (int64_t*)alloc(sizeof(int64_t)*s->B->order);
    map_idx_lda = (int64_t*)alloc(sizeof(int64_t)*s->B->order);
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

  tspsum * tspsum_map::clone() {
    return new tspsum_map(this);
  }

  void tspsum_map::print(){
    printf("tspsum_map:\n");
    printf("namp_idx = %d\n",nmap_idx);
    rec_tsum->print();
  }
  
  int64_t tspsum_map::mem_fp(){
    int64_t mem = nnz_A*this->sr_A->pair_size();
    if (nmap_idx > 0){
      int64_t tot_rep=1;
      for (int midx=0; midx<nmap_idx; midx++){
        tot_rep *= map_idx_len[midx];
      }
      return tot_rep*mem;
    } else return mem;
  }

  void tspsum_map::run(){
    TAU_FSTART(tspsum_map);
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
        this->sr_A->copy(pi_new[i*tot_rep+r].ptr, pi[i].ptr);
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
    TAU_FSTOP(tspsum_map);
    rec_tsum->run();
    TAU_FSTART(tspsum_map);
    new_nnz_B = rec_tsum->new_nnz_B;
    new_B = rec_tsum->new_B;
    cdealloc(buf);
    TAU_FSTOP(tspsum_map);
  }

  tspsum_permute::~tspsum_permute() {
    delete rec_tsum;
    cdealloc(p);
    cdealloc(lens_new);
    cdealloc(lens_old);
  }

  tspsum_permute::tspsum_permute(tspsum * other) : tspsum(other) {
    tspsum_permute * o = (tspsum_permute*)other;
    rec_tsum = o->rec_tsum->clone();
    A_or_B   = o->A_or_B;
    order    = o->order;
    skip     = o->skip;
    p        = (int*)alloc(sizeof(int)*order);
    lens_old = (int64_t*)alloc(sizeof(int64_t)*order);
    lens_new = (int64_t*)alloc(sizeof(int64_t)*order);
    memcpy(p, o->p, sizeof(int)*order);
    memcpy(lens_old, o->lens_old, sizeof(int64_t)*order);
    memcpy(lens_new, o->lens_new, sizeof(int64_t)*order);
  }

  tspsum_permute::tspsum_permute(summation const * s, bool A_or_B_, int64_t const * lens) : tspsum(s) {
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

    p           = (int*)alloc(sizeof(int)*order);
    lens_old    = (int64_t*)alloc(sizeof(int64_t)*order);
    lens_new    = (int64_t*)alloc(sizeof(int64_t)*order);

    memcpy(lens_old, lens, sizeof(int64_t)*this->order);
    memcpy(lens_new, lens, sizeof(int64_t)*this->order);
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
      //adjust p in case there are mapped indices in B 
      int lesser[this->order];
      for (int i=0; i<this->order; i++){
        if (p[i] != -1){
          lesser[i] = 0;
          for (int j=0; j<this->order; j++){
            if (i!=j && p[j] != -1 && p[j] < p[i]) lesser[i]++; 
          }
        }
      }
      for (int i=0; i<this->order; i++){
        if (p[i] != -1)
          p[i] = lesser[i];
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
      int nnm = 0;
      for (int i=0; i<this->order; i++){
        p[i] = nm;
        for (int j=0; j<Y->order; j++){
          if (idx_X[i] == idx_Y[j]){
            ASSERT(p[i] == nm); // no repeating indices allowed here!
            p[i] = nnm+nmap_idx;
            nnm++;
          }
        }
        if (p[i] == nm) nm++;
      } 
    }
    skip = true;
    for (int i=0; i<this->order; i++){
      if (p[i] != i) skip = false;
//      printf("p[%d] = %d order = %d\n", i, p[i], this->order);
      if (p[i] != -1) lens_new[p[i]] = lens[i];
    }
  }

  tspsum * tspsum_permute::clone() {
    return new tspsum_permute(this);
  }

  void tspsum_permute::print(){
    printf("tspsum_permute:\n");
    if (A_or_B) printf("permuting A\n");
    else        printf("permuting B\n");
    for (int i=0; i<order; i++){
      printf("p[%d] = %d ",i,p[i]);
    }
    printf("\n");
    rec_tsum->print();
  }
  
  int64_t tspsum_permute::mem_fp(){
    int64_t mem = 0;
    if (A_or_B) mem+=nnz_A*sr_A->pair_size();
    else mem+=nnz_B*sr_B->pair_size();
    return mem;
  }

  void tspsum_permute::run(){
    char * buf;

    if (skip){
      rec_tsum->A = A;
      rec_tsum->B = B;
      rec_tsum->nnz_A = nnz_A;
      rec_tsum->nnz_B = nnz_B;

      rec_tsum->run();
      new_nnz_B = rec_tsum->new_nnz_B;
      new_B = rec_tsum->new_B;
      return;
    }

    TAU_FSTART(spsum_permute);
    if (A_or_B){
      alloc_ptr(nnz_A*sr_A->pair_size(), (void**)&buf);
      rec_tsum->A = buf;
      rec_tsum->B = B;
      sr_A->copy_pairs(buf, A, nnz_A);
      int64_t new_lda_A[order];
      memset(new_lda_A, 0, order*sizeof(int64_t));
      int64_t lda=1;
      for (int i=0; i<order; i++){
        for (int j=0; j<order; j++){
          if (p[j] == i){ 
            new_lda_A[j] = lda;
            lda *= lens_new[i];
          }
        }
      }
      ConstPairIterator rA(sr_A, A);
      PairIterator wA(sr_A, buf);
      rA.permute(nnz_A, order, lens_old, new_lda_A, wA);
     
      PairIterator mwA = wA;
      for (int v=0; v<nvirt_A; v++){
        mwA.sort(nnz_blk_A[v]);
        mwA = mwA[nnz_blk_A[v]];
      }
      rec_tsum->A = buf;
    } else {
      alloc_ptr(nnz_B*sr_B->pair_size(), (void**)&buf);
      rec_tsum->A = A;
      rec_tsum->B = buf;
      sr_B->copy(buf, B, nnz_B);
      int64_t old_lda_B[order];
      int64_t lda=1;
      for (int i=0; i<order; i++){
        old_lda_B[i] = lda;
        lda *= lens_new[i];
      }
      int64_t new_lda_B[order];
      std::fill(new_lda_B, new_lda_B+order, 0);
      for (int i=0; i<order; i++){
        new_lda_B[i] = old_lda_B[p[i]];
      }
      ConstPairIterator rB(sr_B, B);
      PairIterator wB(sr_B, buf);
      rB.permute(nnz_B, order, lens_old, new_lda_B, wB);

      PairIterator mwB = wB;
      for (int v=0; v<nvirt_B; v++){
        mwB.sort(nnz_blk_B[v]);
        mwB = mwB[nnz_blk_B[v]];
      }

      rec_tsum->B = buf;
    }

    rec_tsum->nnz_A = nnz_A;
    rec_tsum->nnz_B = nnz_B;
    TAU_FSTOP(spsum_permute);
    rec_tsum->run();
    TAU_FSTART(spsum_permute);

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
      int64_t old_lda_B[order];
      int64_t lda=1;
      for (int i=0; i<order; i++){
        old_lda_B[i] = lda;
        lda *= lens_old[i];
      }
      for (int i=0; i<order; i++){
        new_lda_B[i] = old_lda_B[inv_p[i]];
      }
      ConstPairIterator rB(sr_B, rec_tsum->new_B);
      PairIterator wB(sr_B, new_B);

      rB.permute(new_nnz_B, order, lens_new, new_lda_B, wB);
      PairIterator mwB = wB;
      for (int v=0; v<nvirt_B; v++){
        /*for (int i=0; i<nnz_blk_B[v]; i++){
          printf("i=%d/%ld\n",i,nnz_blk_B[v]);
          sr_B->print(mwB[i].d());
        }*/
        mwB.sort(nnz_blk_B[v]);
        mwB = mwB[nnz_blk_B[v]];
      }

      if (buf != rec_tsum->new_B && new_B != rec_tsum->new_B){
        sr_B->pair_dealloc(rec_tsum->new_B);
      }
      cdealloc(buf);
    }
    TAU_FSTOP(spsum_permute);
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
    *idx_arr = (int*)alloc(sizeof(int)*2*dim_max);
    std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[2*idx_A[i]] = i;
    }
    for (i=0; i<order_B; i++){
      (*idx_arr)[2*idx_B[i]+1] = i;
    }
  }

  tspsum_pin_keys::~tspsum_pin_keys(){
    delete rec_tsum;
    cdealloc(divisor);
    cdealloc(virt_dim);
    cdealloc(phys_rank);
  }
  
  tspsum_pin_keys::tspsum_pin_keys(tspsum * other) : tspsum(other) {
    tspsum_pin_keys * o = (tspsum_pin_keys*)other;

    rec_tsum  = o->rec_tsum->clone();
    A_or_B    = o->A_or_B;
    order     = o->order;
    lens      = o->lens;
    divisor   = (int*)alloc(sizeof(int)*order);
    phys_rank = (int*)alloc(sizeof(int)*order);
    virt_dim  = (int*)alloc(sizeof(int)*order);
    memcpy(divisor, o->divisor, sizeof(int)*order);
    memcpy(phys_rank, o->phys_rank, sizeof(int)*order);
    memcpy(virt_dim, o->virt_dim, sizeof(int)*order);
  }

  tspsum * tspsum_pin_keys::clone() {
    return new tspsum_pin_keys(this);
  }

  void tspsum_pin_keys::print(){
    printf("tspsum_pin_keys:\n");
    if (A_or_B) printf("transforming global keys of A to local keys\n");
    else        printf("transforming global keys of B to local keys\n");
    rec_tsum->print();
  }

  int64_t tspsum_pin_keys::mem_fp(){
    return 3*order*sizeof(int);
  }

  tspsum_pin_keys::tspsum_pin_keys(summation const * s, bool A_or_B_) : tspsum(s) {
    tensor * X;
    A_or_B = A_or_B_;
    if (A_or_B){
      X = s->A;
    } else {
      X = s->B;
    }
    order = X->order;
    lens = X->lens;

    divisor = (int*)alloc(sizeof(int)*order);
    phys_rank = (int*)alloc(sizeof(int)*order);
    virt_dim = (int*)alloc(sizeof(int*)*order);

    for (int i=0; i<order; i++){
      divisor[i] = X->edge_map[i].calc_phase();
      phys_rank[i] = X->edge_map[i].calc_phys_rank(X->topo);
      virt_dim[i] = divisor[i]/X->edge_map[i].calc_phys_phase();
    }
  }

  void tspsum_pin_keys::run(){
    TAU_FSTART(spsum_pin);
    char * X;
    algstrct const * sr;
    int64_t nnz;
    if (A_or_B){
      X = this->A;
      sr = this->sr_A;
      nnz = this->nnz_A;
    } else {
      X = this->B;
      sr = this->sr_B;
      nnz = this->nnz_B;
    }

/*    int * div_lens;
    alloc_ptr(order*sizeof(int), (void**)&div_lens);
    for (int j=0; j<order; j++){
      div_lens[j] = (lens[j]/divisor[j] + (lens[j]%divisor[j] > 0));
//      printf("lens[%d] = %d divisor[%d] = %d div_lens[%d] = %d\n",j,lens[j],j,divisor[j],j,div_lens[j]);
    }*/

    ConstPairIterator pi(sr, X);
    PairIterator pi_new(sr, X);

    pi.pin(nnz, order, lens, divisor, pi_new);

    if (A_or_B){
      rec_tsum->A = X;
      rec_tsum->B = B;
    } else {
      rec_tsum->A = A;
      rec_tsum->B = X;
    }
    rec_tsum->nnz_A = nnz_A;
    rec_tsum->nnz_B = nnz_B;
    TAU_FSTOP(spsum_pin);
    rec_tsum->run();
    TAU_FSTART(spsum_pin);

    new_nnz_B = rec_tsum->new_nnz_B;
    new_B = rec_tsum->new_B;
    if (A_or_B){
      depin(sr_A, order, lens, divisor, nvirt_A, virt_dim, phys_rank, A, nnz_A, (int64_t*)nnz_blk_A, A, false);
    } else {
      char * old_B = new_B;
      depin(sr_B, order, lens, divisor, nvirt_B, virt_dim, phys_rank, new_B, new_nnz_B, nnz_blk_B, new_B, true);
      if (old_B != new_B && old_B != B) sr->pair_dealloc(old_B);
    }
    TAU_FSTOP(spsum_pin);
  }
}
