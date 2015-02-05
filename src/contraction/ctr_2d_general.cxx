/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "ctr_2d_general.h"
#include "../shared/util.h"
#include <climits>

namespace CTF_int {
  ctr_2d_general::~ctr_2d_general() {
    if (move_A) cdt_A.deactivate();
    if (move_B) cdt_B.deactivate();
    if (move_C) cdt_C.deactivate();
    if (rec_ctr != NULL)
      delete rec_ctr;
  }

  ctr_2d_general::ctr_2d_general(ctr * other) : ctr(other) {
    ctr_2d_general * o = (ctr_2d_general*)other;
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
#ifdef OFFLOAD
    alloc_host_buf = o->alloc_host_buf;
#endif
  }

  void ctr_2d_general::print() {
    printf("ctr_2d_general: edge_len = %d\n", edge_len);
    printf("move_A = %d, ctr_lda_A = " PRId64 ", ctr_sub_lda_A = " PRId64 "\n",
            move_A, ctr_lda_A, ctr_sub_lda_A);
    if (move_A) printf("cdt_A length = %d\n",cdt_A.np);
    printf("move_B = %d, ctr_lda_B = " PRId64 ", ctr_sub_lda_B = " PRId64 "\n",
            move_B, ctr_lda_B, ctr_sub_lda_B);
    if (move_B) printf("cdt_B length = %d\n",cdt_B.np);
    printf("move_C = %d, ctr_lda_C = " PRId64 ", ctr_sub_lda_C = " PRId64 "\n",
            move_C, ctr_lda_C, ctr_sub_lda_C);
    if (move_C) printf("cdt_C length = %d\n",cdt_C.np);
#ifdef OFFLOAD
    if (alloc_host_buf)
      printf("alloc_host_buf is true\n");
    else
      printf("alloc_host_buf is false\n");
#endif
    rec_ctr->print();
  }

  ctr * ctr_2d_general::clone() {
    return new ctr_2d_general(this);
  }

  void ctr_2d_general::find_bsizes(int64_t & b_A,
                                   int64_t & b_B,
                                   int64_t & b_C,
                                   int64_t & s_A,
                                   int64_t & s_B,
                                   int64_t & s_C,
                                   int64_t & db,
                                   int64_t & aux_size){
    db = int64_t_max;
    s_A = 0, s_B = 0, s_C = 0;
    b_A = 0, b_B = 0, b_C = 0;
    if (move_A){
      b_A         = edge_len/cdt_A.np;
      s_A         = cdt_A.estimate_bcast_time(sr_A.el_size*ctr_lda_A*ctr_sub_lda_A);
      db          = MIN(b_A, db);
    } 
    if (move_B){
      b_B         = edge_len/cdt_B.np;
      s_B         = cdt_B.estimate_bcast_time(sr_B.el_size*ctr_lda_B*ctr_sub_lda_B);
      db          = MIN(b_B, db);
    }
    if (move_C){
      b_C         = edge_len/cdt_C.np;
      s_C         = cdt_C.estimate_allred_time(sr_C.el_size*ctr_lda_C*ctr_sub_lda_C);
      db          = MIN(b_C, db);
    }

    aux_size = db*MAX(move_A*sr_A.el_size*s_A, MAX(move_B*sr_B.el_size*s_B, move_C*sr_C.el_size*s_C));
  }

  double ctr_2d_general::est_time_fp(int nlyr) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size);
    return ((s_A+s_B+s_C)*(double)db*edge_len/db)/MIN(nlyr,edge_len);
  }

  double ctr_2d_general::est_time_rec(int nlyr) {
    int64_t db;
    db = int64_t_max;
    if (move_A)
      db          = MIN(db,edge_len/cdt_A.np);
    if (move_B)
      db          = MIN(db,edge_len/cdt_B.np);
    if (move_C)
      db          = MIN(db,edge_len/cdt_C.np);
    return (edge_len/db)*rec_ctr->est_time_rec(1) + est_time_fp(nlyr);
  }

  int64_t ctr_2d_general::mem_fp() {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size);
    return sr_A.el_size*s_A*db+sr_B.el_size*s_B*db+sr_C.el_size*s_C*db+aux_size;
  }

  int64_t ctr_2d_general::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void ctr_2d_general::run() {
    int owner_A, owner_B, owner_C, ret;
    int64_t ib, c_A, c_B, c_C;
    char * buf_A, * buf_B, * buf_C, * buf_aux; 
    char * op_A, * op_B, * op_C; 
    int rank_A, rank_B, rank_C;
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size;
    rank_A = cdt_A.rank;
    rank_B = cdt_B.rank;
    rank_C = cdt_C.rank;
    
    TAU_FSTART(ctr_2d_general);

    /* Must move at most two tensors */
    ASSERT(!(move_A && move_B && move_C));
    
    rec_ctr->beta         = this->beta;
    rec_ctr->num_lyr      = 1;
    rec_ctr->idx_lyr      = 0;
    
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size);
    
#ifdef OFFLOAD
    if (alloc_host_buf){
      host_pinned_alloc((void**)&buf_A, s_A*db*sr_A.el_size);
      host_pinned_alloc((void**)&buf_B, s_B*db*sr_B.el_size);
      host_pinned_alloc((void**)&buf_C, s_C*db*sr_C.el_size);
#endif
    if (0){
    } else {
      ret = CTF_int::mst_alloc_ptr(s_A*db*sr_A.el_size, (void**)&buf_A);
      ASSERT(ret==0);
      ret = CTF_int::mst_alloc_ptr(s_B*db*sr_B.el_size, (void**)&buf_B);
      ASSERT(ret==0);
      ret = CTF_int::mst_alloc_ptr(s_C*db*sr_C.el_size, (void**)&buf_C);
      ASSERT(ret==0);
    }
    ret = CTF_int::mst_alloc_ptr(aux_size, (void**)&buf_aux);
    ASSERT(ret==0);

    for (ib=this->idx_lyr*db; ib<edge_len; ib+=db*this->num_lyr){
      if (move_A){
        owner_A   = ib / b_A;
        c_A       = MIN(((owner_A+1)*b_A-ib), db);
        if (rank_A == owner_A){
          if (c_A == b_A){
            op_A = this->A;
          } else {
            op_A = buf_A;
            sr_A.copy(ctr_sub_lda_A*c_A, ctr_lda_A, 
                      this->A+sr_A.el_size*(ib%b_A)*ctr_sub_lda_A, ctr_sub_lda_A*b_A, 
                      op_A, ctr_sub_lda_A*c_A);
          }
        } else
          op_A = buf_A;
        //POST_BCAST(op_A, c_A*s_A*sr_A.el_size, MPI_CHAR, owner_A, cdt_A);
        MPI_Bcast(op_A, c_A*s_A*sr_A.el_size, MPI_CHAR, owner_A, cdt_A.cm);
        if (c_A < db){ /* If the required A block is cut between 2 procs */
          if (rank_A == owner_A+1)
            sr_A.copy(ctr_sub_lda_A*(db-c_A), ctr_lda_A,
                      this->A, ctr_sub_lda_A*b_A, 
                      buf_aux, ctr_sub_lda_A*(db-c_A));
          MPI_Bcast(buf_aux, s_A*(db-c_A)*sr_A.el_size, MPI_CHAR, owner_A+1, cdt_A.cm);
          coalesce_bwd(sr_A.el_size,
                       buf_A, 
                       buf_aux, 
                       ctr_sub_lda_A*db, 
                       ctr_lda_A, 
                       ctr_sub_lda_A*c_A);
          op_A = buf_A;
        }
      } else {
        if (ctr_sub_lda_A == 0)
          op_A = this->A;
        else {
          if (false && ctr_lda_A == 1)
            op_A = this->A+sr_A.el_size*ib*ctr_sub_lda_A;
          else {
            op_A = buf_A;
            sr_A.copy(ctr_sub_lda_A, ctr_lda_A,
                      this->A+sr_A.el_size*ib*ctr_sub_lda_A, ctr_sub_lda_A*edge_len, 
                      buf_A, ctr_sub_lda_A);
          }      
        }
      }
      if (move_B){
        owner_B   = ib / b_B;
        c_B       = MIN(((owner_B+1)*b_B-ib), db);
        if (rank_B == owner_B){
          if (c_B == b_B){
            op_B = this->B;
          } else {
            op_B = buf_B;
            sr_B.copy(ctr_sub_lda_B*c_B, ctr_lda_B,
                      this->B+sr_B.el_size*(ib%b_B)*ctr_sub_lda_B, ctr_sub_lda_B*b_B, 
                      op_B, ctr_sub_lda_B*c_B);
          }
        } else 
          op_B = buf_B;
        MPI_Bcast(op_B, c_B*s_B*sr_B.el_size, MPI_CHAR, owner_B, cdt_B.cm);
        if (c_B < db){ /* If the required B block is cut between 2 procs */
          if (rank_B == owner_B+1)
            sr_B.copy(ctr_sub_lda_B*(db-c_B), ctr_lda_B,
                      this->B, ctr_sub_lda_B*b_B, 
                      buf_aux, ctr_sub_lda_B*(db-c_B)); 
          MPI_Bcast(buf_aux, s_B*(db-c_B)*sr_B.el_size, MPI_CHAR, owner_B+1, cdt_B.cm);
          coalesce_bwd(sr_B.el_size,
                       buf_B, 
                       buf_aux, 
                       ctr_sub_lda_B*db, 
                       ctr_lda_B, 
                       ctr_sub_lda_B*c_B);
          op_B = buf_B;
        }
      } else {
        if (ctr_sub_lda_B == 0)
          op_B = this->B;
        else {
          if (false && ctr_lda_B == 1){
            op_B = this->B+sr_B.el_size*ib*ctr_sub_lda_B;
          } else {
            op_B = buf_B;
            sr_B.copy(ctr_sub_lda_B, ctr_lda_B,
                      this->B+sr_B.el_size*ib*ctr_sub_lda_B, ctr_sub_lda_B*edge_len, 
                      buf_B, ctr_sub_lda_B);
          }      
        }
      }
      if (move_C){
        op_C = buf_C;
        rec_ctr->beta = sr_C.addid();
      } else {
        if (ctr_sub_lda_C == 0)
          op_C = this->C;
        else {
          if (false && ctr_lda_C == 1) 
            op_C = this->C+sr_A.el_size*ib*ctr_sub_lda_C;
          else {
            op_C = buf_C;
            rec_ctr->beta = sr_C.addid();
          }
        }
      } 


      rec_ctr->A = op_A;
      rec_ctr->B = op_B;
      rec_ctr->C = op_C;

      rec_ctr->run();

      if (move_C){
        /* FIXME: Wont work for single precsion */
        MPI_Allreduce(MPI_IN_PLACE, op_C, db*s_C, sr_C.mdtype(), sr_C.addmop(), cdt_C.cm);
        owner_C   = ib / b_C;
        c_C       = MIN(((owner_C+1)*b_C-ib), db);
        if (rank_C == owner_C){
          sr_C.copy(ctr_sub_lda_C*c_C, ctr_lda_C,
                    op_C, ctr_sub_lda_C*db, sr_C.mulid(),
                    this->C+sr_C.el_size*(ib%b_C)*ctr_sub_lda_C, 
                    ctr_sub_lda_C*b_C, this->beta);
        }
        if (c_C < db){ /* If the required B block is cut between 2 procs */
          if (rank_C == owner_C+1)
            sr_C.copy(ctr_sub_lda_C*(db-c_C), ctr_lda_C,
                      op_C+sr_C.el_size*ctr_sub_lda_C*c_C,
                      ctr_sub_lda_C*db, sr_C.mulid(),
                      this->C, ctr_sub_lda_C*b_C, this->beta);
        }
      } else {
        if (ctr_sub_lda_C != 0)
          sr_C.copy(ctr_sub_lda_C, ctr_lda_C,
                    buf_C, ctr_sub_lda_C, sr_C.mulid(), 
                    this->C+sr_C.el_size*ib*ctr_sub_lda_C, 
                    ctr_sub_lda_C*edge_len, this->beta);
      }
      rec_ctr->beta = sr_C.mulid();
    }
    /* FIXME: reuse that */
#ifdef OFFLOAD
    if (alloc_host_buf){
      host_pinned_free(buf_A);
      host_pinned_free(buf_B);
      host_pinned_free(buf_C);
#endif
    if (0){
    } else {
      CTF_int::cfree(buf_A);
      CTF_int::cfree(buf_B);
      CTF_int::cfree(buf_C);
    }
    TAU_FSTOP(ctr_2d_general);
  }
}

