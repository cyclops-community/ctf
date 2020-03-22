
/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "spctr_comm.h"
#include "contraction.h"
#include "../tensor/untyped_tensor.h"
#include "../sparse_formats/ccsr.h"
#include <cfloat>

namespace CTF_int {
  spctr_replicate::spctr_replicate(contraction const * c,
                               int const *         phys_mapped,
                               int64_t             blk_sz_A,
                               int64_t             blk_sz_B,
                               int64_t             blk_sz_C)
       : spctr(c) {
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
        hspctr->idx_lyr = A->topo->dim_comm[i].rank;
        hspctr->num_lyr = A->topo->dim_comm[i]->np;
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
 
  spctr_replicate::~spctr_replicate() {
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

  spctr_replicate::spctr_replicate(spctr * other) : spctr(other) {
    spctr_replicate * o = (spctr_replicate*)other;
    rec_ctr = o->rec_ctr->clone();
    size_A = o->size_A;
    size_B = o->size_B;
    size_C = o->size_C;
    ncdt_A = o->ncdt_A;
    ncdt_B = o->ncdt_B;
    ncdt_C = o->ncdt_C;
  }

  spctr * spctr_replicate::clone() {
    return new spctr_replicate(this);
  }

  void spctr_replicate::print() {
    int i;
    printf("spctr_replicate: \n");
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

  double spctr_replicate::est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    int i;
    double tot_sz;
    tot_sz = 0.0;
    for (i=0; i<ncdt_A; i++){
      ASSERT(cdt_A[i]->np > 0);
      //FIXME estimates off since multiplying by element size vs pair/CSR size
      if (is_sparse_A)
        tot_sz += cdt_A[i]->estimate_bcast_time(nnz_frac_A*size_A*sr_A->pair_size());
      else
        tot_sz += cdt_A[i]->estimate_bcast_time(nnz_frac_A*size_A*sr_A->el_size);
    }
    for (i=0; i<ncdt_B; i++){
      ASSERT(cdt_B[i]->np > 0);
      if (is_sparse_B)
        tot_sz += cdt_B[i]->estimate_bcast_time(nnz_frac_B*size_B*sr_B->pair_size());
      else
        tot_sz += cdt_B[i]->estimate_bcast_time(nnz_frac_B*size_B*sr_B->el_size);
    }
    for (i=0; i<ncdt_C; i++){
      ASSERT(cdt_C[i]->np > 0);
      if (is_sparse_C)
        tot_sz += sr_C->estimate_csr_red_time(nnz_frac_C*size_C*sr_C->pair_size(), cdt_C[i]);
      else
        tot_sz += cdt_C[i]->estimate_red_time(nnz_frac_C*size_C*sr_C->el_size, sr_C->addmop());
    }
    return tot_sz;
  }

  double spctr_replicate::est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    return rec_ctr->est_time_rec(nlyr, nblk_A, nblk_B, nblk_C, nnz_frac_A, nnz_frac_B, nnz_frac_C) + est_time_fp(nlyr, nblk_A, nblk_B, nblk_C, nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  int64_t spctr_replicate::spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    int64_t mem_usage = 0;
    if (this->ncdt_A > 1)
      if (is_sparse_A) mem_usage += nnz_frac_A*(size_A*sr_A->pair_size());
    if (this->ncdt_B > 1)
      if (is_sparse_B) mem_usage += nnz_frac_B*(size_B*sr_B->pair_size());
    if (this->ncdt_C > 0)
      if (is_sparse_C) mem_usage += 3.*nnz_frac_C*(size_C*sr_C->pair_size());
    return mem_usage;
  }

  int64_t spctr_replicate::spmem_tmp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){
    int64_t mem_usage = 0;
    if (this->ncdt_C > 0){
      if (!is_sparse_C)
        mem_usage += sr_C->el_size*size_C;
    }
    return mem_usage;
  }

  int64_t spctr_replicate::spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    return std::max(spmem_tmp(nnz_frac_A, nnz_frac_B, nnz_frac_C), rec_ctr->spmem_rec(nnz_frac_A, nnz_frac_B, nnz_frac_C)) + spmem_fp(nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  void spctr_replicate::run(char * A, int nblk_A, int64_t const * size_blk_A,
                            char * B, int nblk_B, int64_t const * size_blk_B,
                            char * C, int nblk_C, int64_t * size_blk_C,
                            char *& new_C){
    int arank, brank, crank, i;
    TAU_FSTART(spctr_replicate);
    arank = 0, brank = 0, crank = 0;
    for (i=0; i<ncdt_A; i++)
      arank += cdt_A[i]->rank;
    for (i=0; i<ncdt_B; i++)
      brank += cdt_B[i]->rank;
    for (i=0; i<ncdt_C; i++)
      crank += cdt_C[i]->rank;
    char * buf_A = A;
    char * buf_B = B;
    int64_t new_size_blk_A[nblk_A];
    int64_t new_size_blk_B[nblk_B];
    int64_t new_size_blk_C[nblk_C];
    if (is_sparse_A){
      memcpy(new_size_blk_A, size_blk_A, nblk_A*sizeof(int64_t));
      /*if (ncdt_A > 0){
        save_size_blk_A = (int64_t*)alloc(sizeof(int64_t)*nblk_A);
        memcpy(save_size_blk_A,size_blk_A,sizeof(int64_t)*nblk_A);
      }*/
      for (i=0; i<ncdt_A; i++){
        cdt_A[i]->bcast(new_size_blk_A, nblk_A, MPI_INT64_T, 0);
      }
      int64_t new_size_A = 0;
      for (i=0; i<nblk_A; i++){
        new_size_A += new_size_blk_A[i];
      }      
      if (arank != 0)
        buf_A = (char*)alloc(new_size_A);
      for (i=0; i<ncdt_A; i++){
        cdt_A[i]->bcast(buf_A, new_size_A, MPI_CHAR, 0);
      }
    } else {
      for (i=0; i<ncdt_A; i++){
        cdt_A[i]->bcast(A, size_A, sr_A->mdtype(), 0);
      }
    }
    if (is_sparse_B){
      memcpy(new_size_blk_B, size_blk_B, nblk_B*sizeof(int64_t));
      /*if (ncdt_B > 0){
        save_size_blk_B = (int64_t*)alloc(sizeof(int64_t)*nblk_B);
        memcpy(save_size_blk_B,size_blk_B,sizeof(int64_t)*nblk_B);
      }*/
      for (i=0; i<ncdt_B; i++){
        cdt_B[i]->bcast(new_size_blk_B, nblk_B, MPI_INT64_T, 0);
      }
      int64_t new_size_B = 0;
      for (i=0; i<nblk_B; i++){
        new_size_B += new_size_blk_B[i];
      }      
      if (brank != 0)
        buf_B = (char*)alloc(new_size_B);
      for (i=0; i<ncdt_B; i++){
        cdt_B[i]->bcast(buf_B, new_size_B, MPI_CHAR, 0);
      }
    } else {
      for (i=0; i<ncdt_B; i++){
        cdt_B[i]->bcast(B, size_B, sr_B->mdtype(), 0);
      }
    }
    if (is_sparse_C){
      memset(new_size_blk_C, 0, sizeof(int64_t)*nblk_C);
    }
//    if (crank != 0) this->sr_C->set(C, this->sr_C->addid(), size_C);
    if (crank == 0 && !sr_C->isequal(this->beta, sr_C->mulid())){
      ASSERT(!is_sparse_C);
      if (sr_C->isequal(sr_C->addid(), beta))
        sr_C->set(C, sr_C->addid(), size_C);
      else sr_C->scal(size_C, this->beta, C, 1);
/*      for (i=0; i<size_C; i++){
        sr_C->mul(this->beta, C+i*sr_C->el_size, C+i*sr_C->el_size);
      }*/
    }
    if (crank != 0)
      rec_ctr->beta = sr_C->addid();
    else
      rec_ctr->beta = sr_C->mulid(); 

    rec_ctr->num_lyr      = this->num_lyr;
    rec_ctr->idx_lyr      = this->idx_lyr;

    TAU_FSTOP(spctr_replicate);
    rec_ctr->run(buf_A, nblk_A, new_size_blk_A,
                 buf_B, nblk_B, new_size_blk_B,
                     C, nblk_C,     size_blk_C,
                 new_C);
    TAU_FSTART(spctr_replicate);
    if (is_sparse_A && buf_A != A) cdealloc(buf_A);
    if (is_sparse_B && buf_B != B) cdealloc(buf_B);
    /*for (i=0; i<size_C; i++){
      printf("P%d C[%d]  = %lf\n",crank,i, ((double*)C)[i]);
    }*/
    for (i=0; i<ncdt_C; i++){
      if (i>0 && cdt_C[i-1]->rank != 0) break;
      //ALLREDUCE(MPI_IN_PLACE, C, size_C, sr_C->mdtype(), sr_C->addmop(), cdt_C[i]->;
      if (is_sparse_C){
        int64_t csr_sz_acc = 0;
        int64_t new_csr_sz_acc = 0;
        char * new_Cs[nblk_C];
        for (int blk=0; blk<nblk_C; blk++){
          new_Cs[blk] = sr_C->csr_reduce(new_C+csr_sz_acc, 0, cdt_C[i]->cm, this->is_ccsr_C);
  //        printf("%d out of %d reds complete, size = %ld should be %ld\n",blk,nblk_C,size_blk_C[blk],((CSR_Matrix)(new_C+csr_sz_acc)).size());
        
          csr_sz_acc += size_blk_C[blk];
          if (this->is_ccsr_C)
            size_blk_C[blk] = cdt_C[i]->rank == 0 ? ((CCSR_Matrix)(new_Cs[blk])).size() : 0;
          else
            size_blk_C[blk] = cdt_C[i]->rank == 0 ? ((CSR_Matrix)(new_Cs[blk])).size() : 0;
          new_csr_sz_acc += size_blk_C[blk];
//          printf("rank %d blk size %ld tot sz %ld\n", cdt_C[i]->rank, size_blk_C[blk], new_csr_sz_acc);
        }
        cdealloc(new_C);
        alloc_ptr(new_csr_sz_acc, (void**)&new_C);
        new_csr_sz_acc = 0;
        for (int blk=0; blk<nblk_C; blk++){
          memcpy(new_C+new_csr_sz_acc, new_Cs[blk], size_blk_C[blk]);
          cdealloc(new_Cs[blk]);
          new_csr_sz_acc += size_blk_C[blk];
        }        
      } else {
        if (cdt_C[i]->rank == 0){
          cdt_C[i]->red(MPI_IN_PLACE, new_C, size_C, sr_C->mdtype(), sr_C->addmop(), 0);
        } else {
          cdt_C[i]->red(new_C, NULL, size_C, sr_C->mdtype(), sr_C->addmop(), 0);
        }
      }
    }

    if (!is_sparse_A && arank != 0){
      this->sr_A->set(A, this->sr_A->addid(), size_A);
    }
    if (!is_sparse_B && brank != 0){
      this->sr_B->set(B, this->sr_B->addid(), size_B);
    }
    TAU_FSTOP(spctr_replicate);
  }
}
