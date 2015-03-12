/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"
#include "sym_seq_ctr.h"

namespace CTF_int {
  ctr::~ctr(){
  }

  ctr::ctr(ctr * other){
    A = other->A;
    B = other->B;
    C = other->C;
    sr_A = other->sr_A;
    sr_B = other->sr_B;
    sr_C = other->sr_C;
    beta = other->beta;
    num_lyr = other->num_lyr;
    idx_lyr = other->idx_lyr;
  }

/*  ctr_dgemm::~ctr_dgemm() { }

  ctr_dgemm::ctr_dgemm(ctr * other) : ctr(other) {
    ctr_dgemm * o = (ctr_dgemm*)other;
    n = o->n;
    m = o->m;
    k = o->k;
    alpha = o->alpha;
    transp_A = o->transp_A;
    transp_B = o->transp_B;
  }
  ctr * ctr_dgemm::clone() {
    return new ctr_dgemm(this);
  }


  int64_t ctr_dgemm::mem_fp(){
    return 0;
  }


  double ctr_dgemm::est_time_fp(int nlyr) {
    // FIXME make cost proper, for now return sizes of each submatrix scaled by .2 
    ASSERT(0);
    return n*m+m*k+n*k;
  }

  double ctr_dgemm::est_time_rec(int nlyr) {
    return est_time_fp(nlyr);
  }*/
/*
  template<> inline
  void ctr_dgemm< std::complex<double> >::run(){
    const int lda_A = transp_A == 'n' ? m : k;
    const int lda_B = transp_B == 'n' ? k : n;
    const int lda_C = m;
    if (this->idx_lyr == 0){
      czgemm(transp_A,
             transp_B,
             m,
             n,
             k,
             alpha,
             this->A,
             lda_A,
             this->B,
             lda_B,
             this->beta,
             this->C,
             lda_C);
    }
  }

  void ctr_dgemm::run(){
    const int lda_A = transp_A == 'n' ? m : k;
    const int lda_B = transp_B == 'n' ? k : n;
    const int lda_C = m;
    if (this->idx_lyr == 0){
      cdgemm(transp_A,
             transp_B,
             m,
             n,
             k,
             alpha,
             this->A,
             lda_A,
             this->B,
             lda_B,
             this->beta,
             this->C,
             lda_C);
    }
  }*/

  ctr_lyr::~ctr_lyr() {
    delete rec_ctr;
  }

  ctr_lyr::ctr_lyr(ctr * other) : ctr(other) {
    ctr_lyr * o = (ctr_lyr*)other;
    rec_ctr = o->rec_ctr->clone();
    k = o->k;
    cdt = o->cdt;
    sz_C = o->sz_C;
  }

  /**
   * \brief copies ctr object
   */
  ctr * ctr_lyr::clone() {
    return new ctr_lyr(this);
  }


  int64_t ctr_lyr::mem_fp(){
    return 0;
  }

  int64_t ctr_lyr::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }


  void ctr_lyr::run(){
    rec_ctr->A            = this->A;
    rec_ctr->B            = this->B;
    rec_ctr->C            = this->C;
    if (cdt->rank != 0)
      rec_ctr->beta = sr_C->addid();
    else
      rec_ctr->beta = this->beta; 
    rec_ctr->num_lyr      = cdt->np;
    rec_ctr->idx_lyr      = cdt->rank;

    rec_ctr->run();
    
    /* FIXME: unnecessary except for current DCMF wrapper */
    //COMM_BARRIER(cdt);
    /* FIXME Won't work for single precision */
    //ALLREDUCE(MPI_IN_PLACE, this->C, sz_C*(sizeof(dtype)/sizeof(double)), MPI_DOUBLE, MPI_SUM, cdt);
    MPI_Allreduce(MPI_IN_PLACE, this->C, sz_C, sr_C->mdtype(), sr_C->addmop(), cdt->cm);

  }

  ctr_replicate::~ctr_replicate() {
    delete rec_ctr;
/*    for (int i=0; i<ncdt_A; i++){
      cdt_A[i]->deactivate();
    }*/
    if (ncdt_A > 0)
      CTF_int::cfree(cdt_A);
/*    for (int i=0; i<ncdt_B; i++){
      cdt_B[i]->deactivate();
    }*/
    if (ncdt_B > 0)
      CTF_int::cfree(cdt_B);
/*    for (int i=0; i<ncdt_C; i++){
      cdt_C[i]->deactivate();
    }*/
    if (ncdt_C > 0)
      CTF_int::cfree(cdt_C);
  }

  ctr_replicate::ctr_replicate(ctr * other) : ctr(other) {
    ctr_replicate * o = (ctr_replicate*)other;
    rec_ctr = o->rec_ctr->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    size_C = o->size_C;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
    ncdt_C = o->ncdt_C;
  }

  ctr * ctr_replicate::clone() {
    return new ctr_replicate(this);
  }

  void ctr_replicate::print() {
    int i;
    printf("ctr_replicate: \n");
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
    printf("cdt_C = %p, size_C = %ld, ncdt_C = %d\n",
            cdt_C, size_C, ncdt_C);
    for (i=0; i<ncdt_C; i++){
      printf("cdt_C[%d] length = %d\n",i,cdt_C[i]->np);
    }
    rec_ctr->print();
  }

  double ctr_replicate::est_time_fp(int nlyr){
    int i;
    double tot_sz;
    tot_sz = 0.0;
    for (i=0; i<ncdt_A; i++){
      ASSERT(cdt_A[i]->np > 0);
      tot_sz += cdt_A[i]->estimate_bcast_time(size_A*sr_A->el_size);
    }
    for (i=0; i<ncdt_B; i++){
      ASSERT(cdt_B[i]->np > 0);
      tot_sz += cdt_B[i]->estimate_bcast_time(size_B*sr_B->el_size);
    }
    for (i=0; i<ncdt_C; i++){
      ASSERT(cdt_C[i]->np > 0);
      tot_sz += cdt_C[i]->estimate_allred_time(size_C*sr_C->el_size);
    }
    return tot_sz;
  }

  double ctr_replicate::est_time_rec(int nlyr) {
    return rec_ctr->est_time_rec(nlyr) + est_time_fp(nlyr);
  }

  int64_t ctr_replicate::mem_fp(){
    return 0;
  }

  int64_t ctr_replicate::mem_rec(){
    return rec_ctr->mem_rec() + mem_fp();
  }


  void ctr_replicate::run(){
    int arank, brank, crank, i;

    arank = 0, brank = 0, crank = 0;
    for (i=0; i<ncdt_A; i++){
      arank += cdt_A[i]->rank;
//      POST_BCAST(this->A, size_A*sr_A->el_size, COMM_CHAR_T, 0, cdt_A[i]-> 0);
      MPI_Bcast(this->A, size_A*sr_A->el_size, MPI_CHAR, 0, cdt_A[i]->cm);
    }
    for (i=0; i<ncdt_B; i++){
      brank += cdt_B[i]->rank;
//      POST_BCAST(this->B, size_B*sr_B->el_size, COMM_CHAR_T, 0, cdt_B[i]-> 0);
      MPI_Bcast(this->B, size_B*sr_B->el_size, MPI_CHAR, 0, cdt_B[i]->cm);
    }
    for (i=0; i<ncdt_C; i++){
      crank += cdt_C[i]->rank;
    }
    if (crank != 0) this->sr_C->set(this->C, this->sr_C->addid(), size_C);
    else {
      for (i=0; i<size_C; i++){
        sr_C->mul(this->beta, this->C+i*sr_C->el_size, this->C+i*sr_C->el_size);
      }
    }

    rec_ctr->A            = this->A;
    rec_ctr->B            = this->B;
    rec_ctr->C            = this->C;
    if (crank != 0)
      rec_ctr->beta = sr_C->addid();
    else
      rec_ctr->beta = sr_C->mulid(); 
    rec_ctr->num_lyr      = this->num_lyr;
    rec_ctr->idx_lyr      = this->idx_lyr;

    rec_ctr->run();
    
    for (i=0; i<ncdt_C; i++){
      //ALLREDUCE(MPI_IN_PLACE, this->C, size_C, sr_C->mdtype(), sr_C->addmop(), cdt_C[i]->;
      MPI_Allreduce(MPI_IN_PLACE, this->C, size_C, sr_C->mdtype(), sr_C->addmop(), cdt_C[i]->cm);
    }

    if (arank != 0){
      this->sr_A->set(this->A, this->sr_A->addid(), size_A);
    }
    if (brank != 0){
      this->sr_B->set(this->B, this->sr_B->addid(), size_B);
    }
  }

  void seq_tsr_ctr::print(){
    int i;
    printf("seq_tsr_ctr:\n");
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

  seq_tsr_ctr::seq_tsr_ctr(ctr * other) : ctr(other) {
    seq_tsr_ctr * o = (seq_tsr_ctr*)other;
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

  ctr * seq_tsr_ctr::clone() {
    return new seq_tsr_ctr(this);
  }


  int64_t seq_tsr_ctr::mem_fp(){ return 0; }

  double seq_tsr_ctr::est_time_fp(int nlyr){ 
    uint64_t size_A = sy_packed_size(order_A, edge_len_A, sym_A);
    uint64_t size_B = sy_packed_size(order_B, edge_len_B, sym_B);
    uint64_t size_C = sy_packed_size(order_C, edge_len_C, sym_C);
    if (is_inner) size_A *= inner_params.m*inner_params.k*sr_A->el_size;
    if (is_inner) size_B *= inner_params.n*inner_params.k*sr_B->el_size;
    if (is_inner) size_C *= inner_params.m*inner_params.n*sr_C->el_size;
   
    ASSERT(size_A > 0);
    ASSERT(size_B > 0);
    ASSERT(size_C > 0);

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
    }
    for (int i=0; i<idx_max; i++){
      if (rev_idx_map[3*i+0] != -1) flops*=edge_len_A[rev_idx_map[3*i+0]];
      else if (rev_idx_map[3*i+1] != -1) flops*=edge_len_B[rev_idx_map[3*i+1]];
      else if (rev_idx_map[3*i+2] != -1) flops*=edge_len_C[rev_idx_map[3*i+2]];
    }
    ASSERT(flops >= 0.0);
    CTF_int::cfree(rev_idx_map);
    return COST_MEMBW*(size_A+size_B+size_C)+COST_FLOP*flops;
  }

  double seq_tsr_ctr::est_time_rec(int nlyr){ 
    return est_time_fp(nlyr);
  }

  void seq_tsr_ctr::run(){
    if (is_custom){
      ASSERT(is_inner == 0);
      sym_seq_ctr_cust(this->alpha,
                       this->A,
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
    } else if (is_inner){
      sym_seq_ctr_inr(this->alpha,
                      this->A,
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
                      &inner_params);
    } else {
      sym_seq_ctr_ref(this->alpha,
                      this->A,
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
                      idx_map_C);
    }
  }

  void inv_idx(int                order_A,
               int const *        idx_A,
               int                order_B,
               int const *        idx_B,
               int                order_C,
               int const *        idx_C,
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
    for (i=0; i<order_C; i++){
      if (idx_C[i] > dim_max) dim_max = idx_C[i];
    }
    dim_max++;
    *order_tot = dim_max;
    *idx_arr = (int*)CTF_int::alloc(sizeof(int)*3*dim_max);
    std::fill((*idx_arr), (*idx_arr)+3*dim_max, -1);  

    for (i=0; i<order_A; i++){
      (*idx_arr)[3*idx_A[i]] = i;
    }
    for (i=0; i<order_B; i++){
      (*idx_arr)[3*idx_B[i]+1] = i;
    }
    for (i=0; i<order_C; i++){
      (*idx_arr)[3*idx_C[i]+2] = i;
    }
  }
}
