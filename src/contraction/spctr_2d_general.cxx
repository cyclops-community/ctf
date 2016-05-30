/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "spctr_2d_general.h"
#include "../tensor/untyped_tensor.h"
#include "../mapping/mapping.h"
#include "../shared/util.h"
#include <climits>

namespace CTF_int {

  spctr_2d_general::~spctr_2d_general() {
    /*if (move_A) cdt_A->deactivate();
    if (move_B) cdt_B->deactivate();
    if (move_C) cdt_C->deactivate();*/
    if (rec_ctr != NULL)
      delete rec_ctr;
  }

  spctr_2d_general::spctr_2d_general(spctr * other) : spctr(other) {
    spctr_2d_general * o = (spctr_2d_general*)other;
    rec_ctr = o->rec_ctr->clone();
    edge_len      = o->edge_len;
    ctr_lda_A     = o->ctr_lda_A;
    ctr_sub_lda_A = o->ctr_sub_lda_A;
    cdt_A         = o->cdt_A;
    move_A        = o->move_A;
    ctr_lda_B     = o->ctr_lda_B;
    ctr_sub_lda_B = o->ctr_sub_lda_B;
    cdt_B         = o->cdt_B;
    move_B        = o->move_B;
    ctr_lda_C     = o->ctr_lda_C;
    ctr_sub_lda_C = o->ctr_sub_lda_C;
    cdt_C         = o->cdt_C;
    move_C        = o->move_C;
#if 0 //def OFFLOAD
    alloc_host_buf = o->alloc_host_buf;
#endif
  }

  void spctr_2d_general::print() {
    printf("spctr_2d_general: edge_len = %d\n", edge_len);
    printf("move_A = %d, ctr_lda_A = %ld, ctr_sub_lda_A = %ld\n",
            move_A, ctr_lda_A, ctr_sub_lda_A);
    if (move_A) printf("cdt_A length = %d\n",cdt_A->np);
    printf("move_B = %d, ctr_lda_B = %ld, ctr_sub_lda_B = %ld\n",
            move_B, ctr_lda_B, ctr_sub_lda_B);
    if (move_B) printf("cdt_B length = %d\n",cdt_B->np);
    printf("move_C = %d, ctr_lda_C = %ld, ctr_sub_lda_C = %ld\n",
            move_C, ctr_lda_C, ctr_sub_lda_C);
    if (move_C) printf("cdt_C length = %d\n",cdt_C->np);
#if 0 //def OFFLOAD
    if (alloc_host_buf)
      printf("alloc_host_buf is true\n");
    else
      printf("alloc_host_buf is false\n");
#endif
    rec_ctr->print();
  }

  spctr * spctr_2d_general::clone() {
    return new spctr_2d_general(this);
  }

  void spctr_2d_general::find_bsizes(int64_t & b_A,
                                   int64_t & b_B,
                                   int64_t & b_C,
                                   int64_t & s_A,
                                   int64_t & s_B,
                                   int64_t & s_C,
                                   int64_t & aux_size){
    b_A = 0, b_B = 0, b_C = 0;
    s_A = ctr_sub_lda_A*ctr_lda_A;
    s_B = ctr_sub_lda_B*ctr_lda_B;
    s_C = ctr_lda_C*ctr_sub_lda_C;
    if (move_A){
      b_A = edge_len/cdt_A->np;
    } 
    if (move_B){
      b_B = edge_len/cdt_B->np;
    }
    if (move_C){
      b_C = edge_len/cdt_C->np;
    }

    aux_size = MAX(move_A*sr_A->el_size*s_A, MAX(move_B*sr_B->el_size*s_B, move_C*sr_C->el_size*s_C));
  }

  double spctr_2d_general::est_time_fp(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    double est_bcast_time = 0.0;
    if (move_A)
      est_bcast_time += cdt_A->estimate_bcast_time(sr_A->el_size*s_A*nnz_frac_A);
    if (move_B)
      est_bcast_time += cdt_B->estimate_bcast_time(sr_B->el_size*s_B*nnz_frac_B);
    if (move_C)
      est_bcast_time += cdt_C->estimate_red_time(sr_C->el_size*s_C*nnz_frac_C, sr_C->addmop());
    return (est_bcast_time*(double)edge_len)/MIN(nlyr,edge_len);
  }

  double spctr_2d_general::est_time_rec(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C) {
    return rec_ctr->est_time_rec(1, nnz_frac_A, nnz_frac_B, nnz_frac_C)*(double)edge_len/MIN(nlyr,edge_len) + est_time_fp(nlyr, nnz_frac_A, nnz_frac_B, nnz_frac_C);
  }

  int64_t spctr_2d_general::mem_fp() {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    return sr_A->el_size*s_A+sr_B->el_size*s_B+sr_C->el_size*s_C+aux_size;
  }

  int64_t spctr_2d_general::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void spctr_2d_general::run(char * A, int nblk_A, int64_t const * size_blk_A,
                             char * B, int nblk_B, int64_t const * size_blk_B,
                             char * C, int nblk_C, int64_t * size_blk_C,
                             char *& new_C){
    int owner_A, owner_B, owner_C, ret;
    int64_t ib;
    char * buf_A, * buf_B, * buf_C, * buf_aux; 
    char * op_A = NULL;
    char * op_B = NULL;
    char * op_C = NULL; 
    int rank_A, rank_B, rank_C;
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    if (move_A) rank_A = cdt_A->rank;
    else rank_A = -1;
    if (move_B) rank_B = cdt_B->rank;
    else rank_B = -1;
    if (move_C) rank_C = cdt_C->rank;
    else rank_C = -1;
    
    TAU_FSTART(spctr_2d_general);

    /* Must move at most two tensors */
    ASSERT(!(move_A && move_B && move_C));
    
    rec_ctr->beta         = this->beta;

    int iidx_lyr, inum_lyr;
    if (edge_len >= num_lyr && edge_len % num_lyr == 0){
      inum_lyr         = num_lyr;
      iidx_lyr         = idx_lyr;
      rec_ctr->num_lyr = 1;
      rec_ctr->idx_lyr = 0;
    } else if (edge_len < num_lyr && num_lyr % edge_len == 0){
      inum_lyr         = edge_len;
      iidx_lyr         = idx_lyr%edge_len;
      rec_ctr->num_lyr = num_lyr/edge_len;
      rec_ctr->idx_lyr = idx_lyr/edge_len;
    } else {
      rec_ctr->num_lyr = num_lyr;
      rec_ctr->idx_lyr = idx_lyr;
      inum_lyr         = 1;
      iidx_lyr         = 0;
    }

    
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    
#if 0 //def OFFLOAD
    if (alloc_host_buf){
      host_pinned_alloc((void**)&buf_A, s_A*sr_A->el_size);
      host_pinned_alloc((void**)&buf_B, s_B*sr_B->el_size);
      host_pinned_alloc((void**)&buf_C, s_C*sr_C->el_size);
#endif
    if (0){
    } else {
      if (!is_sparse_A){
        ret = CTF_int::mst_alloc_ptr(s_A*sr_A->el_size, (void**)&buf_A);
        ASSERT(ret==0);
      } else buf_A = NULL;
      ret = CTF_int::mst_alloc_ptr(s_B*sr_B->el_size, (void**)&buf_B);
      ASSERT(ret==0);
      ret = CTF_int::mst_alloc_ptr(s_C*sr_C->el_size, (void**)&buf_C);
      ASSERT(ret==0);
    }
    ret = CTF_int::mst_alloc_ptr(aux_size, (void**)&buf_aux);
    ASSERT(ret==0);

    int64_t * offsets_A;
    if (is_sparse_A){
      CTF_int::alloc_ptr(sizeof(int64_t)*nblk_A, (void**)&offsets_A);
      for (int i=0; i<nblk_A; i++){
        if (i==0) offsets_A[0] = 0;
        else offsets_A[i] = offsets_A[i-1]+size_blk_A[i-1];
      }
    }
    ASSERT(!is_sparse_B);
    ASSERT(!is_sparse_C);
    int64_t * new_size_blk_A = (int64_t*)size_blk_A;
    int new_nblk_A = nblk_A;
    //for (ib=this->idx_lyr; ib<edge_len; ib+=this->num_lyr){

    for (ib=iidx_lyr; ib<edge_len; ib+=inum_lyr){
      new_size_blk_A = (int64_t*)size_blk_A;
      if (move_A){
        new_nblk_A = nblk_A/b_A;
        owner_A   = ib % cdt_A->np;
        if (rank_A == owner_A){
          if (b_A == 1){
            op_A = A;
          } else {
            if (is_sparse_A){
              int64_t * new_offsets_A;
              socopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*b_A, ctr_sub_lda_A,
                     size_blk_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                     new_size_blk_A, new_offsets_A);

              int64_t bc_size_A = 0;
              for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];
              ret = CTF_int::mst_alloc_ptr(bc_size_A, (void**)&buf_A);
              ASSERT(ret==0);
              op_A = buf_A;
              spcopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*b_A, ctr_sub_lda_A,
                     size_blk_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                     offsets_A+(ib/cdt_A->np)*ctr_sub_lda_A, 
                     A,
                     new_size_blk_A, new_offsets_A, op_A);
              cdealloc(new_offsets_A);
            } else {
              op_A = buf_A;
              sr_A->copy(ctr_sub_lda_A, ctr_lda_A, 
                         A+sr_A->el_size*(ib/cdt_A->np)*ctr_sub_lda_A, ctr_sub_lda_A*b_A, 
                         op_A, ctr_sub_lda_A);
            }
          }
        } else {
          if (is_sparse_A)
            CTF_int::alloc_ptr(sizeof(int64_t)*nblk_A/b_A, (void**)&new_size_blk_A);
          else
            op_A = buf_A;
        }
        if (is_sparse_A){
          cdt_A->bcast(new_size_blk_A, new_nblk_A, MPI_INT64_T, owner_A);
          int64_t bc_size_A = 0;
          for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];

          if (cdt_A->rank != owner_A){ 
            ret = CTF_int::mst_alloc_ptr(bc_size_A, (void**)&buf_A);
            ASSERT(ret==0);
            op_A = buf_A;
          }
          cdt_A->bcast(op_A, bc_size_A, MPI_CHAR, owner_A);
          /*int rrank;
          MPI_Comm_rank(MPI_COMM_WORLD, &rrank);
          printf("rrank = %d new_nblk_A = %d rank = %d owner = %d new_nnz_A = %ld old_nnz_A = %ld\n",rrank,new_nblk_A,cdt_A->rank, owner_A, new_nnz_A, nnz_A);
          for (int rr=0; rr<new_nblk_A; rr++){
            printf("rrank = %d new_nblk_A = %d new_size_blk_A[%d] = %ld\n", rrank, new_nblk_A, rr, new_size_blk_A[rr]);
          }*/
        } else {
          cdt_A->bcast(op_A, s_A, sr_A->mdtype(), owner_A);
        }
      } else {
        if (ctr_sub_lda_A == 0)
          op_A = A;
        else {
          new_nblk_A = nblk_A/edge_len;
          if (ctr_lda_A == 1){
            if (is_sparse_A){
              op_A = A+offsets_A[ib*ctr_sub_lda_A];
              CTF_int::alloc_ptr(sizeof(int64_t)*new_nblk_A, (void**)&new_size_blk_A);
              memcpy(new_size_blk_A, size_blk_A+ib*ctr_sub_lda_A, sizeof(int64_t)*new_nblk_A);
/*              int rrank;
              MPI_Comm_rank(MPI_COMM_WORLD, &rrank);
              printf("rrank = %d ib = %ld new_nblk_A = %d, new_nnz_A = %ld offset = %ld\n", rrank, ib, new_nblk_A, new_nnz_A, offsets_A[ib*ctr_sub_lda_A]);*/
            } else {
              op_A = A+sr_A->el_size*ib*ctr_sub_lda_A;
            }
          } else {
            if (is_sparse_A){
              int64_t * new_offsets_A;
              socopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                     size_blk_A+ib*ctr_sub_lda_A,
                     new_size_blk_A, new_offsets_A);
              int64_t bc_size_A = 0;
              for (int z=0; z<new_nblk_A; z++) bc_size_A += new_size_blk_A[z];

              ret = CTF_int::mst_alloc_ptr(bc_size_A, (void**)&buf_A);
              ASSERT(ret==0);
              op_A = buf_A;
              spcopy(ctr_sub_lda_A, ctr_lda_A, ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                     size_blk_A+ib*ctr_sub_lda_A, offsets_A+ib*ctr_sub_lda_A, A,
                     new_size_blk_A, new_offsets_A, op_A);
              cdealloc(new_offsets_A);
            } else {
              op_A = buf_A;
              sr_A->copy(ctr_sub_lda_A, ctr_lda_A,
                         A+sr_A->el_size*ib*ctr_sub_lda_A, ctr_sub_lda_A*edge_len, 
                         buf_A, ctr_sub_lda_A);
            }
          }      
        }
      }
      if (move_B){
        owner_B   = ib % cdt_B->np;
        if (rank_B == owner_B){
          if (b_B == 1){
            op_B = B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       B+sr_B->el_size*(ib/cdt_B->np)*ctr_sub_lda_B, ctr_sub_lda_B*b_B, 
                       op_B, ctr_sub_lda_B);
          }
        } else 
          op_B = buf_B;
//        printf("c_B = %ld, s_B = %ld, d_B = %ld, b_B = %ld\n", c_B, s_B,db, b_B);
        cdt_B->bcast(op_B, s_B, sr_B->mdtype(), owner_B);
      } else {
        if (ctr_sub_lda_B == 0)
          op_B = B;
        else {
          if (ctr_lda_B == 1){
            op_B = B+sr_B->el_size*ib*ctr_sub_lda_B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       B+sr_B->el_size*ib*ctr_sub_lda_B, ctr_sub_lda_B*edge_len, 
                       buf_B, ctr_sub_lda_B);
          }      
        }
      }
      if (move_C){
        op_C = buf_C;
        rec_ctr->beta = sr_C->addid();
      } else {
        if (ctr_sub_lda_C == 0)
          op_C = C;
        else {
          if (ctr_lda_C == 1) 
            op_C = C+sr_C->el_size*ib*ctr_sub_lda_C;
          else {
            op_C = buf_C;
            rec_ctr->beta = sr_C->addid();
          }
        }
      } 


      TAU_FSTOP(spctr_2d_general);
      rec_ctr->run(op_A, new_nblk_A, new_size_blk_A,
                   op_B, nblk_B, size_blk_B,
                   op_C, nblk_C, size_blk_C,
                   op_C);

      TAU_FSTART(spctr_2d_general);
      if (new_size_blk_A != size_blk_A)
        cdealloc(new_size_blk_A);
      if (is_sparse_A && buf_A != NULL){
        cdealloc(buf_A);
        buf_A = NULL;
      }
      new_C = C;
      /*for (int i=0; i<ctr_sub_lda_C*ctr_lda_C; i++){
        printf("[%d] P%d op_C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)op_C)[i]);
      }*/
      if (move_C){
#ifdef PROFILE
        TAU_FSTART(spctr_2d_general_barrier);
        MPI_Barrier(cdt_C->cm);
        TAU_FSTOP(spctr_2d_general_barrier);
#endif
        owner_C   = ib % cdt_C->np;
        if (cdt_C->rank == owner_C)
          cdt_C->red(MPI_IN_PLACE, op_C, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        else
          cdt_C->red(op_C, NULL, s_C, sr_C->mdtype(), sr_C->addmop(), owner_C);
        if (rank_C == owner_C){
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     op_C, ctr_sub_lda_C, sr_C->mulid(),
                     C+sr_C->el_size*(ib/cdt_C->np)*ctr_sub_lda_C, 
                     ctr_sub_lda_C*b_C, this->beta);
        }
      } else {
        if (ctr_lda_C != 1 && ctr_sub_lda_C != 0)
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     buf_C, ctr_sub_lda_C, sr_C->mulid(), 
                     C+sr_C->el_size*ib*ctr_sub_lda_C, 
                     ctr_sub_lda_C*edge_len, this->beta);
        if (ctr_sub_lda_C == 0)
          rec_ctr->beta = sr_C->mulid();
      }
/*      for (int i=0; i<ctr_sub_lda_C*ctr_lda_C*edge_len; i++){
        printf("[%d] P%d C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)C)[i]);
      }*/
    }
    /* FIXME: reuse that */
#if 0 //def OFFLOAD
    if (alloc_host_buf){
      host_pinned_free(buf_A);
      host_pinned_free(buf_B);
      host_pinned_free(buf_C);
#endif
    if (0){
    } else {
      if (buf_A != NULL) CTF_int::cdealloc(buf_A);
      if (buf_B != NULL) CTF_int::cdealloc(buf_B);
      if (buf_C != NULL) CTF_int::cdealloc(buf_C);
      CTF_int::cdealloc(buf_aux);
    }
    if (is_sparse_A){
      cdealloc(offsets_A);
    }
    TAU_FSTOP(spctr_2d_general);
  }
}

