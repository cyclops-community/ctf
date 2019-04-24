/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"
#include "ctr_tsr.h"
#include "sym_seq_ctr.h"
#include "contraction.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/model.h"
#ifdef USE_OMP
#include <omp.h>
#endif

namespace CTF_int {

  #ifndef VIRT_NTD
  #define VIRT_NTD        1
  #endif

  ctr_virt::ctr_virt(contraction const * c,
                     int                 num_tot,
                     int *               virt_dim,
                     int64_t             vrt_sz_A,
                     int64_t             vrt_sz_B,
                     int64_t             vrt_sz_C)
      : ctr(c) {
    this->num_dim   = num_tot;
    this->virt_dim  = virt_dim;
    this->order_A   = c->A->order;
    this->blk_sz_A  = vrt_sz_A;
    this->idx_map_A = c->idx_A;
    this->order_B   = c->B->order;
    this->blk_sz_B  = vrt_sz_B;
    this->idx_map_B = c->idx_B;
    this->order_C   = c->C->order;
    this->blk_sz_C  = vrt_sz_C;
    this->idx_map_C = c->idx_C;
  }


  ctr_virt::~ctr_virt() {
    CTF_int::cdealloc(virt_dim);
    delete rec_ctr;
  }

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

  ctr * ctr_virt::clone() {
    return new ctr_virt(this);
  }

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


  double ctr_virt::est_time_rec(int nlyr) {
    /* FIXME: for now treat flops like comm, later make proper cost */
    int64_t nvirt = 1;
    for (int dim=0; dim<num_dim; dim++){
      nvirt *= virt_dim[dim];
    }
    return nvirt*rec_ctr->est_time_rec(nlyr);
  }


  int64_t ctr_virt::mem_fp(){
    return (order_A+order_B+order_C+(3+VIRT_NTD)*num_dim)*sizeof(int);
  }

  int64_t ctr_virt::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }


  void ctr_virt::run(char * A, char * B, char * C){
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
            if (beta_arr[off_C]>0)
              rec_ctr->beta = sr_C->mulid();
            else
              rec_ctr->beta = this->beta;
            beta_arr[off_C]       = 1;
            tid_rec_ctr->run(
                     A + off_A*blk_sz_A*sr_A->el_size,
                     B + off_B*blk_sz_B*sr_A->el_size,
                     C + off_C*blk_sz_C*sr_A->el_size);
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
#ifdef MICROBENCH
          break;
#else
          if (i==num_dim) break;
#endif
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



  seq_tsr_ctr::seq_tsr_ctr(contraction const * c,
                           bool                is_inner,
                           iparam const *      inner_params,
                           int64_t *           virt_blk_len_A,
                           int64_t *           virt_blk_len_B,
                           int64_t *           virt_blk_len_C,
                           int64_t             vrt_sz_C)
        : ctr(c) {

    int i, j, k;
    int * new_sym_A, * new_sym_B, * new_sym_C;
    CTF_int::alloc_ptr(sizeof(int)*c->A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, c->A->sym, sizeof(int)*c->A->order);
    CTF_int::alloc_ptr(sizeof(int)*c->B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, c->B->sym, sizeof(int)*c->B->order);
    CTF_int::alloc_ptr(sizeof(int)*c->C->order, (void**)&new_sym_C);
    memcpy(new_sym_C, c->C->sym, sizeof(int)*c->C->order);

    this->inner_params  = *inner_params;
    if (!is_inner){
      this->is_inner  = 0;
    } else if (is_inner == 1) {
      if (c->A->wrld->cdt.rank == 0){
        DPRINTF(3,"Folded tensor l=%ld n=%ld m=%ld k=%ld\n", inner_params->l, inner_params->n,
          inner_params->m, inner_params->k);
      }

      this->is_inner    = 1;
      this->inner_params.sz_C = vrt_sz_C;
      tensor * itsr;
      itsr = c->A->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = c->A->inner_ordering[i];
        for (k=0; k<c->A->order; k++){
          if (c->A->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && c->A->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
  /*        printf("inner_ordering[%d]=%d setting dim %d of A, to len %d from len %d\n",
                  i, c->A->inner_ordering[i], k, 1, virt_blk_len_A[k]);*/
          virt_blk_len_A[k] = 1;
          new_sym_A[k] = NS;
        }
      }
      itsr = c->B->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = c->B->inner_ordering[i];
        for (k=0; k<c->B->order; k++){
          if (c->B->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && c->B->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of B, to len %d from len %d\n",
                  i, c->B->inner_ordering[i], k, 1, virt_blk_len_B[k]);*/
          virt_blk_len_B[k] = 1;
          new_sym_B[k] = NS;
        }
      }
      itsr = c->C->rec_tsr;
      for (i=0; i<itsr->order; i++){
        j = c->C->inner_ordering[i];
        for (k=0; k<c->C->order; k++){
          if (c->C->sym[k] == NS) j--;
          if (j<0) break;
        }
        j = k;
        while (k>0 && c->C->sym[k-1] != NS){
          k--;
        }
        for (; k<=j; k++){
        /*  printf("inner_ordering[%d]=%d setting dim %d of C, to len %d from len %d\n",
                  i, c->C->inner_ordering[i], k, 1, virt_blk_len_C[k]);*/
          virt_blk_len_C[k] = 1;
          new_sym_C[k] = NS;
        }
      }
    }
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


  void seq_tsr_ctr::print(){
    int i;
    printf("seq_tsr_ctr:\n");
    for (i=0; i<order_A; i++){
      printf("edge_len_A[%d]=%ld\n",i,edge_len_A[i]);
    }
    for (i=0; i<order_B; i++){
      printf("edge_len_B[%d]=%ld\n",i,edge_len_B[i]);
    }
    for (i=0; i<order_C; i++){
      printf("edge_len_C[%d]=%ld\n",i,edge_len_C[i]);
    }
    printf("is inner = %d\n", is_inner);
    if (is_inner) printf("inner n = %ld m= %ld k = %ld l = %ld\n",
                          inner_params.n, inner_params.m, inner_params.k, inner_params.l);
  }

  seq_tsr_ctr::seq_tsr_ctr(ctr * other) : ctr(other) {
    seq_tsr_ctr * o = (seq_tsr_ctr*)other;
    alpha = o->alpha;

    order_A        = o->order_A;
    idx_map_A     = o->idx_map_A;
    sym_A         = (int*)CTF_int::alloc(sizeof(int)*order_A);
    memcpy(sym_A, o->sym_A, sizeof(int)*order_A);
    edge_len_A    = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_A);
    memcpy(edge_len_A, o->edge_len_A, sizeof(int64_t)*order_A);

    order_B        = o->order_B;
    idx_map_B     = o->idx_map_B;
    sym_B         = (int*)CTF_int::alloc(sizeof(int)*order_B);
    memcpy(sym_B, o->sym_B, sizeof(int)*order_B);
    edge_len_B    = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_B);
    memcpy(edge_len_B, o->edge_len_B, sizeof(int64_t)*order_B);

    order_C      = o->order_C;
    idx_map_C    = o->idx_map_C;
    sym_C        = (int*)CTF_int::alloc(sizeof(int)*order_C);
    memcpy(sym_C, o->sym_C, sizeof(int)*order_C);
    edge_len_C   = (int64_t*)CTF_int::alloc(sizeof(int64_t)*order_C);
    memcpy(edge_len_C, o->edge_len_C, sizeof(int64_t)*order_C);

    is_inner     = o->is_inner;
    inner_params = o->inner_params;
    is_custom    = o->is_custom;
    func         = o->func;
  }

  ctr * seq_tsr_ctr::clone() {
    return new seq_tsr_ctr(this);
  }

  int64_t seq_tsr_ctr::mem_fp(){ return 0; }

  //double seq_tsr_ctr_mig[] = {1e-6, 9.30e-11, 5.61e-10};
  LinModel<3> seq_tsr_ctr_mdl_cst(seq_tsr_ctr_mdl_cst_init,"seq_tsr_ctr_mdl_cst");
  LinModel<3> seq_tsr_ctr_mdl_ref(seq_tsr_ctr_mdl_ref_init,"seq_tsr_ctr_mdl_ref");
  LinModel<3> seq_tsr_ctr_mdl_inr(seq_tsr_ctr_mdl_inr_init,"seq_tsr_ctr_mdl_inr");
  LinModel<3> seq_tsr_ctr_mdl_off(seq_tsr_ctr_mdl_off_init,"seq_tsr_ctr_mdl_off");
  LinModel<3> seq_tsr_ctr_mdl_cst_inr(seq_tsr_ctr_mdl_cst_inr_init,"seq_tsr_ctr_mdl_cst_inr");
  LinModel<3> seq_tsr_ctr_mdl_cst_off(seq_tsr_ctr_mdl_cst_off_init,"seq_tsr_ctr_mdl_cst_off");

  uint64_t seq_tsr_ctr::est_membw(){
    uint64_t size_A = sy_packed_size(order_A, edge_len_A, sym_A)*sr_A->el_size;
    uint64_t size_B = sy_packed_size(order_B, edge_len_B, sym_B)*sr_B->el_size;
    uint64_t size_C = sy_packed_size(order_C, edge_len_C, sym_C)*sr_C->el_size;
    if (is_inner) size_A *= inner_params.m*inner_params.k;
    if (is_inner) size_B *= inner_params.n*inner_params.k;
    if (is_inner) size_C *= inner_params.m*inner_params.n;

    ASSERT(size_A > 0);
    ASSERT(size_B > 0);
    ASSERT(size_C > 0);
    return size_A+size_B+size_C;
  }

  double seq_tsr_ctr::est_fp(){
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
    CTF_int::cdealloc(rev_idx_map);
    return flops;
  }

  double seq_tsr_ctr::est_time_fp(int nlyr){
    //return COST_MEMBW*(size_A+size_B+size_C)+COST_FLOP*flops;
    double ps[] = {1.0, (double)est_membw(), est_fp()};
//    printf("time estimate is %lf\n", seq_tsr_ctr_mdl.est_time(ps));
    if (is_custom && !is_inner){
      return seq_tsr_ctr_mdl_cst.est_time(ps);
    } else if (is_inner){
      if (is_custom){
        if (inner_params.offload)
          return seq_tsr_ctr_mdl_cst_off.est_time(ps);
        else
          return seq_tsr_ctr_mdl_cst_inr.est_time(ps);
      } else {
        if (inner_params.offload)
          return seq_tsr_ctr_mdl_off.est_time(ps);
        else
          return seq_tsr_ctr_mdl_inr.est_time(ps);
      }
    } else
      return seq_tsr_ctr_mdl_ref.est_time(ps);
    assert(0); //wont make it here
    return 0.0;
  }

  double seq_tsr_ctr::est_time_rec(int nlyr){
    return est_time_fp(nlyr);
  }

  void seq_tsr_ctr::run(char * A, char * B, char * C){
    ASSERT(idx_lyr == 0 && num_lyr == 1);

#ifdef TUNE
    // Check if we need to execute this function for the sake of training
    bool sr;
    if (is_custom && !is_inner){
      double tps[] = {0, 1.0, (double)est_membw(), est_fp()};
      sr = seq_tsr_ctr_mdl_cst.should_observe(tps);
    } else if (is_inner){
      ASSERT(is_custom || func == NULL);
      double tps[] = {0.0, 1.0, (double)est_membw(), est_fp()};
      if (is_custom){
        if (inner_params.offload)
          sr = seq_tsr_ctr_mdl_cst_off.should_observe(tps);
        else
          sr = seq_tsr_ctr_mdl_cst_inr.should_observe(tps);
      } else {
        if (inner_params.offload)
          sr = seq_tsr_ctr_mdl_off.should_observe(tps);
        else
          sr = seq_tsr_ctr_mdl_inr.should_observe(tps);
      }

    } else {
       double tps[] = {0.0, 1.0, (double)est_membw(), est_fp()};
       sr = seq_tsr_ctr_mdl_ref.should_observe(tps);
    }

    if (!sr) return;
#endif
    if (is_custom && !is_inner){
      double st_time = MPI_Wtime();
      ASSERT(is_inner == 0);
      sym_seq_ctr_cust(this->alpha,
                       A,
                       sr_A,
                       order_A,
                       edge_len_A,
                       sym_A,
                       idx_map_A,
                       B,
                       sr_B,
                       order_B,
                       edge_len_B,
                       sym_B,
                       idx_map_B,
                       this->beta,
                       C,
                       sr_C,
                       order_C,
                       edge_len_C,
                       sym_C,
                       idx_map_C,
                       func);
      double exe_time = MPI_Wtime()-st_time;
      double tps[] = {exe_time, 1.0, (double)est_membw(), est_fp()};
      seq_tsr_ctr_mdl_cst.observe(tps);
    } else if (is_inner){
      ASSERT(is_custom || func == NULL);
//      double ps[] = {1.0, (double)est_membw(), est_fp()};
//      double est_time = seq_tsr_ctr_mdl_inr.est_time(ps);
      double st_time = MPI_Wtime();
      sym_seq_ctr_inr(this->alpha,
                      A,
                      sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      B,
                      sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      this->beta,
                      C,
                      sr_C,
                      order_C,
                      edge_len_C,
                      sym_C,
                      idx_map_C,
                      &inner_params,
                      func);
      double exe_time = MPI_Wtime()-st_time;
 //     printf("exe_time = %E est_time = %E abs_err = %e rel_err = %lf\n", exe_time,est_time,fabs(exe_time-est_time),fabs(exe_time-est_time)/exe_time);
      double tps[] = {exe_time, 1.0, (double)est_membw(), est_fp()};
      if (is_custom){
        if (inner_params.offload)
          seq_tsr_ctr_mdl_cst_off.observe(tps);
        else
          seq_tsr_ctr_mdl_cst_inr.observe(tps);
      } else {
        if (inner_params.offload)
          seq_tsr_ctr_mdl_off.observe(tps);
        else
          seq_tsr_ctr_mdl_inr.observe(tps);
      }
//      seq_tsr_ctr_mdl_inr.print_param_guess();
    } else {
      double st_time = MPI_Wtime();
      sym_seq_ctr_ref(this->alpha,
                      A,
                      sr_A,
                      order_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      B,
                      sr_B,
                      order_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      this->beta,
                      C,
                      sr_C,
                      order_C,
                      edge_len_C,
                      sym_C,
                      idx_map_C);
      double exe_time = MPI_Wtime()-st_time;
      double tps[] = {exe_time, 1.0, (double)est_membw(), est_fp()};
      seq_tsr_ctr_mdl_ref.observe(tps);
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
             A,
             lda_A,
             B,
             lda_B,
             this->beta,
             C,
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
             A,
             lda_A,
             B,
             lda_B,
             this->beta,
             C,
             lda_C);
    }
  }*/


}
