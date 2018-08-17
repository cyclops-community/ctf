/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"
#include "contraction.h"
#include "../interface/fun_term.h"
#include "../interface/idx_tensor.h"
#include "../tensor/untyped_tensor.h"

namespace CTF_int {
  Bifun_Term bivar_function::operator()(Term const & A, Term const & B) const {
    return Bifun_Term(A.clone(), B.clone(), this);
  }

  void bivar_function::operator()(Term const & A, Term const & B, Term const & C) const {
    Bifun_Term ft(A.clone(), B.clone(), this);
    ft.execute(C.execute(C.get_uniq_inds()));
  }



  ctr::ctr(contraction const * c){
    sr_A    = c->A->sr;
    sr_B    = c->B->sr;
    sr_C    = c->C->sr;
    beta    = c->beta;
    idx_lyr = 0;
    num_lyr = 1;
  }

  ctr::ctr(ctr * other){
    sr_A = other->sr_A;
    sr_B = other->sr_B;
    sr_C = other->sr_C;
    beta = other->beta;
    num_lyr = other->num_lyr;
    idx_lyr = other->idx_lyr;
  }

  ctr::~ctr(){
  }

  ctr_replicate::ctr_replicate(contraction const * c,
                               int const *         phys_mapped,
                               int64_t             blk_sz_A,
                               int64_t             blk_sz_B,
                               int64_t             blk_sz_C)
       : ctr(c) {
    int i;
    int nphys_dim = c->A->topo->order;
    this->ncdt_A = 0;
    this->ncdt_B = 0;
    this->ncdt_C = 0;
    this->size_A = blk_sz_A;
    this->size_B = blk_sz_B;
    this->size_C = blk_sz_C;
    this->cdt_A = NULL;
    this->cdt_B = NULL;
    this->cdt_C = NULL;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[3*i+0] == 0 &&
          phys_mapped[3*i+1] == 0 &&
          phys_mapped[3*i+2] == 0){
  /*      printf("ERROR: ALL-TENSOR REPLICATION NO LONGER DONE\n");
        ABORT;
        ASSERT(this->num_lyr == 1);
        hctr->idx_lyr = A->topo->dim_comm[i].rank;
        hctr->num_lyr = A->topo->dim_comm[i]->np;
        this->idx_lyr = A->topo->dim_comm[i].rank;
        this->num_lyr = A->topo->dim_comm[i]->np;*/
      } else {
        if (phys_mapped[3*i+0] == 0){
          this->ncdt_A++;
        }
        if (phys_mapped[3*i+1] == 0){
          this->ncdt_B++;
        }
        if (phys_mapped[3*i+2] == 0){
          this->ncdt_C++;
        }
      }
    }
    if (this->ncdt_A > 0)
      CTF_int::alloc_ptr(sizeof(CommData*)*this->ncdt_A, (void**)&this->cdt_A);
    if (this->ncdt_B > 0)
      CTF_int::alloc_ptr(sizeof(CommData*)*this->ncdt_B, (void**)&this->cdt_B);
    if (this->ncdt_C > 0)
      CTF_int::alloc_ptr(sizeof(CommData*)*this->ncdt_C, (void**)&this->cdt_C);
    this->ncdt_A = 0;
    this->ncdt_B = 0;
    this->ncdt_C = 0;
    for (i=0; i<nphys_dim; i++){
      if (!(phys_mapped[3*i+0] == 0 &&
          phys_mapped[3*i+1] == 0 &&
          phys_mapped[3*i+2] == 0)){
        if (phys_mapped[3*i+0] == 0){
          this->cdt_A[this->ncdt_A] = &c->A->topo->dim_comm[i];
        /*    if (is_used && this->cdt_A[this->ncdt_A].alive == 0)
            this->cdt_A[this->ncdt_A].activate(global_comm.cm);*/
          this->ncdt_A++;
        }
        if (phys_mapped[3*i+1] == 0){
          this->cdt_B[this->ncdt_B] = &c->B->topo->dim_comm[i];
  /*        if (is_used && this->cdt_B[this->ncdt_B].alive == 0)
            this->cdt_B[this->ncdt_B].activate(global_comm.cm);*/
          this->ncdt_B++;
        }
        if (phys_mapped[3*i+2] == 0){
          this->cdt_C[this->ncdt_C] = &c->C->topo->dim_comm[i];
  /*        if (is_used && this->cdt_C[this->ncdt_C].alive == 0)
            this->cdt_C[this->ncdt_C].activate(global_comm.cm);*/
          this->ncdt_C++;
        }
      }
    }
  }
 
  ctr_replicate::~ctr_replicate() {
    delete rec_ctr;
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
/*    for (int i=0; i<ncdt_C; i++){
      cdt_C[i]->deactivate();
    }*/
    if (ncdt_C > 0)
      CTF_int::cdealloc(cdt_C);
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
      tot_sz += cdt_C[i]->estimate_red_time(size_C*sr_C->el_size, sr_C->addmop());
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


  void ctr_replicate::run(char * A, char * B, char * C){
    int arank, brank, crank, i;

    arank = 0, brank = 0, crank = 0;
    for (i=0; i<ncdt_A; i++){
      arank += cdt_A[i]->rank;
//      POST_BCAST(A, size_A*sr_A->el_size, COMM_CHAR_T, 0, cdt_A[i]-> 0);
      cdt_A[i]->bcast(A, size_A, sr_A->mdtype(), 0);
    }
    for (i=0; i<ncdt_B; i++){
      brank += cdt_B[i]->rank;
//      POST_BCAST(B, size_B*sr_B->el_size, COMM_CHAR_T, 0, cdt_B[i]-> 0);
      cdt_B[i]->bcast(B, size_B, sr_B->mdtype(), 0);
    }
    for (i=0; i<ncdt_C; i++){
      crank += cdt_C[i]->rank;
    }
//    if (crank != 0) this->sr_C->set(C, this->sr_C->addid(), size_C);
//    else {
    if (crank == 0 && !sr_C->isequal(this->beta, sr_C->mulid())){
      if (sr_C->isequal(this->beta, sr_C->addid())){
        sr_C->set(C, sr_C->addid(), size_C);
      } else {
        sr_C->scal(size_C, this->beta, C, 1);
      }

/*      for (i=0; i<size_C; i++){
        sr_C->mul(this->beta, C+i*sr_C->el_size, C+i*sr_C->el_size);
      }*/
    }
//
    //sr_C->set(C, sr_C->addid(), size_C);
    if (crank != 0)
      rec_ctr->beta = sr_C->addid();
    else
      rec_ctr->beta = sr_C->mulid(); 

    rec_ctr->num_lyr      = this->num_lyr;
    rec_ctr->idx_lyr      = this->idx_lyr;

    rec_ctr->run(A, B, C);
    
    /*for (i=0; i<size_C; i++){
      printf("P%d C[%d]  = %lf\n",crank,i, ((double*)C)[i]);
    }*/
    for (i=0; i<ncdt_C; i++){
      //ALLREDUCE(MPI_IN_PLACE, C, size_C, sr_C->mdtype(), sr_C->addmop(), cdt_C[i]->;
      if (cdt_C[i]->rank == 0)
        cdt_C[i]->red(MPI_IN_PLACE, C, size_C, sr_C->mdtype(), sr_C->addmop(), 0);
      else
        cdt_C[i]->red(C, NULL, size_C, sr_C->mdtype(), sr_C->addmop(), 0);
    }

    if (arank != 0 && this->sr_A->addid() != NULL){
      this->sr_A->set(A, this->sr_A->addid(), size_A);
    }
    if (brank != 0 && this->sr_B->addid() != NULL){
      this->sr_B->set(B, this->sr_B->addid(), size_B);
    }
  }
}
