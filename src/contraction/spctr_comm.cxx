
/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "spctr_comm.h"
#include "sp_seq_ctr.h"
#include "contraction.h"
#include "../tensor/untyped_tensor.h"

namespace CTF_int {  
  spctr::spctr(contraction const * c)
    : ctr(c) {
    is_sparse_A = c->A->is_sparse;
    nnz_A       = c->A->nnz_loc;
    nvirt_A     = c->A->calc_nvirt();

    is_sparse_B = c->B->is_sparse;
    nnz_B       = c->B->nnz_loc;
    nvirt_B     = c->B->calc_nvirt();

    is_sparse_C = c->C->is_sparse;
    nnz_C       = c->C->nnz_loc;
    nvirt_C     = c->C->calc_nvirt();

    if (is_sparse_A){
      nnz_blk_A = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, c->A->nnz_blk, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;

    if (is_sparse_B){
      nnz_blk_B = (int64_t*)alloc(sizeof(int64_t)*nvirt_B);
      memcpy(nnz_blk_B, c->B->nnz_blk, sizeof(int64_t)*nvirt_B);
    } else nnz_blk_B = NULL;

    nnz_blk_C   = c->C->nnz_blk;
    new_nnz_C   = nnz_C;
    new_C       = NULL;

  }

  spctr::spctr(spctr * other)
    : ctr(other) {
    is_sparse_A = other->is_sparse_A;
    nnz_A       = other->nnz_A;
    nvirt_A     = other->nvirt_A;

    is_sparse_B = other->is_sparse_B;
    nnz_B       = other->nnz_B;
    nvirt_B     = other->nvirt_B;

    is_sparse_C = other->is_sparse_C;
    nnz_C       = other->nnz_C;
    nvirt_C     = other->nvirt_C;

    new_nnz_C   = other->new_nnz_C;
    new_C       = other->new_C;

    //nnz_blk_B should be copied by pointer, they are the same pointer as in tensor object
    nnz_blk_C   = other->nnz_blk_C;
    //nnz_blk_A should be copied by value, since it needs to be potentially set in replicate and deallocated later
    if (is_sparse_A){
      nnz_blk_A   = (int64_t*)alloc(sizeof(int64_t)*nvirt_A);
      memcpy(nnz_blk_A, other->nnz_blk_A, sizeof(int64_t)*nvirt_A);
    } else nnz_blk_A = NULL;

    if (is_sparse_B){
      nnz_blk_B   = (int64_t*)alloc(sizeof(int64_t)*nvirt_B);
      memcpy(nnz_blk_B, other->nnz_blk_B, sizeof(int64_t)*nvirt_B);
    } else nnz_blk_B = NULL;
  }

  spctr::~spctr(){
    if (nnz_blk_A != NULL) cdealloc(nnz_blk_A);
    if (nnz_blk_B != NULL) cdealloc(nnz_blk_B);
  }


  seq_tsr_spctr::seq_tsr_spctr(contraction const * c,
                               bool                is_inner,
                               iparam const *      inner_params,
                               int *               virt_blk_len_A,
                               int *               virt_blk_len_B,
                               int *               virt_blk_len_C,
                               int64_t             vrt_sz_C)
        : spctr(c) {
     
    int * new_sym_A, * new_sym_B, * new_sym_C;
    CTF_int::alloc_ptr(sizeof(int)*c->A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, c->A->sym, sizeof(int)*c->A->order);
    CTF_int::alloc_ptr(sizeof(int)*c->B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, c->B->sym, sizeof(int)*c->B->order);
    CTF_int::alloc_ptr(sizeof(int)*c->C->order, (void**)&new_sym_C);
    memcpy(new_sym_C, c->C->sym, sizeof(int)*c->C->order);

    ASSERT(!is_inner);
    this->is_inner  = 0;
    this->is_custom  = c->is_custom;
    this->alpha      = c->alpha;
    if (is_custom){
      this->func     = c->func;
    } else {
      this->func     = NULL;
    }
    this->order_A    = c->A->order;
    this->idx_map_A  = c->idx_A;
    this->edge_len_A = virt_blk_len_A;
    this->sym_A      = new_sym_A;
    this->order_B    = c->B->order;
    this->idx_map_B  = c->idx_B;
    this->edge_len_B = virt_blk_len_B;
    this->sym_B      = new_sym_B;
    this->order_C    = c->C->order;
    this->idx_map_C  = c->idx_C;
    this->edge_len_C = virt_blk_len_C;
    this->sym_C      = new_sym_C;

  }


  void seq_tsr_spctr::print(){
    int i;
    printf("seq_tsr_spctr:\n");
    for (i=0; i<order_A; i++){
      printf("edge_len_A[%d]=%d\n",i,edge_len_A[i]);
    }
    for (i=0; i<order_B; i++){
      printf("edge_len_B[%d]=%d\n",i,edge_len_B[i]);
    }
    for (i=0; i<order_C; i++){
      printf("edge_len_C[%d]=%d\n",i,edge_len_C[i]);
    }
    printf("is inner = %d\n", is_inner);
    if (is_inner) printf("inner n = %d m= %d k = %d\n",
                          inner_params.n, inner_params.m, inner_params.k);
  }

  seq_tsr_spctr::seq_tsr_spctr(spctr * other) : spctr(other) {
    seq_tsr_spctr * o = (seq_tsr_spctr*)other;
    alpha = o->alpha;
    
    order_A        = o->order_A;
    idx_map_A     = o->idx_map_A;
    sym_A         = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(sym_A, o->sym_A, sizeof(int)*order_A);
    edge_len_A    = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int)*order_A);

    order_B        = o->order_B;
    idx_map_B     = o->idx_map_B;
    sym_B         = (int*)CTF_int::alloc(sizeof(int)*order_B);
    memcpy(sym_B, o->sym_B, sizeof(int)*order_B);
    edge_len_B    = (int*)CTF_int::alloc(sizeof(int)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int)*order_B);

    order_C      = o->order_C;
    idx_map_C    = o->idx_map_C;
    sym_C        = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(sym_C, o->sym_C, sizeof(int)*order_C);
    edge_len_C   = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(edge_len_C, o->edge_len_C, sizeof(int)*order_C);

    is_inner     = o->is_inner;
    inner_params = o->inner_params;
    is_custom    = o->is_custom;
    func         = o->func;
  }

  spctr * seq_tsr_spctr::clone() {
    return new seq_tsr_spctr(this);
  }


  int64_t seq_tsr_spctr::mem_fp(){ return 0; }

  double seq_tsr_spctr::est_time_fp(int nlyr){ 
    uint64_t size_A = sy_packed_size(order_A, edge_len_A, sym_A)*sr_A->el_size;
    uint64_t size_B = sy_packed_size(order_B, edge_len_B, sym_B)*sr_B->el_size;
    uint64_t size_C = sy_packed_size(order_C, edge_len_C, sym_C)*sr_C->el_size;
    if (is_inner) size_A *= inner_params.m*inner_params.k;
    if (is_inner) size_B *= inner_params.n*inner_params.k;
    if (is_inner) size_C *= inner_params.m*inner_params.n;
    if (is_sparse_A) size_A = nnz_A*sr_A->pair_size();
    if (is_sparse_B) size_B = nnz_B*sr_B->pair_size();
    if (is_sparse_C) size_C = nnz_C*sr_C->pair_size();
   
    /*ASSERT(size_A > 0);
    ASSERT(size_B > 0);
    ASSERT(size_C > 0);*/

    int idx_max, * rev_idx_map; 
    inv_idx(order_A,       idx_map_A,
            order_B,       idx_map_B,
            order_C,       idx_map_C,
            &idx_max,     &rev_idx_map);

    double flops = 2.0;
    if (is_inner) {
      flops *= inner_params.m;
      flops *= inner_params.n;
      flops *= inner_params.k;
    } else {
      for (int i=0; i<idx_max; i++){
        if (rev_idx_map[3*i+0] != -1) flops*=edge_len_A[rev_idx_map[3*i+0]];
        else if (rev_idx_map[3*i+1] != -1) flops*=edge_len_B[rev_idx_map[3*i+1]];
        else if (rev_idx_map[3*i+2] != -1) flops*=edge_len_C[rev_idx_map[3*i+2]];
      }
    }
    ASSERT(flops >= 0.0);
    CTF_int::cdealloc(rev_idx_map);
    return COST_MEMBW*(size_A+size_B+size_C)+COST_FLOP*flops;
  }

  double seq_tsr_spctr::est_time_rec(int nlyr){ 
    return est_time_fp(nlyr);
  }

  void seq_tsr_spctr::run(){
    ASSERT(idx_lyr == 0 && num_lyr == 1);
    ASSERT( is_sparse_A);
    ASSERT(!is_sparse_B);
    ASSERT(!is_sparse_C);
    ASSERT(is_inner == 0);


    spA_dnB_dnC_seq_ctr(this->alpha,
                        this->A,
                        nnz_A,
                        sr_A,
                        order_A,
                        edge_len_A,
                        sym_A,
                        idx_map_A,
                        this->B,
                        sr_B,
                        order_B,
                        edge_len_B,
                        sym_B,
                        idx_map_B,
                        this->beta,
                        this->C,
                        sr_C,
                        order_C,
                        edge_len_C,
                        sym_C,
                        idx_map_C,
                        func);
  }

}
