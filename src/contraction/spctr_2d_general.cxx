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
#ifdef OFFLOAD
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
#ifdef OFFLOAD
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

  double spctr_2d_general::est_time_fp(int nlyr) {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    double est_bcast_time = 0.0;
    if (move_A)
      est_bcast_time += cdt_A->estimate_bcast_time(sr_A->el_size*s_A);
    if (move_B)
      est_bcast_time += cdt_B->estimate_bcast_time(sr_B->el_size*s_B);
    if (move_C)
      est_bcast_time += cdt_C->estimate_bcast_time(sr_C->el_size*s_C);
    return (est_bcast_time*(double)edge_len)/MIN(nlyr,edge_len);
  }

  double spctr_2d_general::est_time_rec(int nlyr) {
    return edge_len*rec_ctr->est_time_rec(1) + est_time_fp(nlyr);
  }

  int64_t spctr_2d_general::mem_fp() {
    int64_t b_A, b_B, b_C, s_A, s_B, s_C, aux_size;
    find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, aux_size);
    return sr_A->el_size*s_A+sr_B->el_size*s_B+sr_C->el_size*s_C+aux_size;
  }

  int64_t spctr_2d_general::mem_rec() {
    return rec_ctr->mem_rec() + mem_fp();
  }

  void spctr_2d_general::run() {
    int owner_A, owner_B, owner_C, ret;
    int64_t ib;
    char * buf_A, * buf_B, * buf_C, * buf_aux; 
    char * op_A, * op_B, * op_C; 
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
    
#ifdef OFFLOAD
    if (alloc_host_buf){
      host_pinned_alloc((void**)&buf_A, s_A*sr_A->el_size);
      host_pinned_alloc((void**)&buf_B, s_B*sr_B->el_size);
      host_pinned_alloc((void**)&buf_C, s_C*sr_C->el_size);
#endif
    if (0){
    } else {
      ret = CTF_int::mst_alloc_ptr(s_A*sr_A->el_size, (void**)&buf_A);
      ASSERT(ret==0);
      ret = CTF_int::mst_alloc_ptr(s_B*sr_B->el_size, (void**)&buf_B);
      ASSERT(ret==0);
      ret = CTF_int::mst_alloc_ptr(s_C*sr_C->el_size, (void**)&buf_C);
      ASSERT(ret==0);
    }
    ret = CTF_int::mst_alloc_ptr(aux_size, (void**)&buf_aux);
    ASSERT(ret==0);

    //for (ib=this->idx_lyr; ib<edge_len; ib+=this->num_lyr){
    for (ib=iidx_lyr; ib<edge_len; ib+=inum_lyr){
      if (move_A){
        owner_A   = ib % cdt_A->np;
        if (rank_A == owner_A){
          if (b_A == 1){
            op_A = this->A;
          } else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A, 
                       this->A+sr_A->el_size*(ib/cdt_A->np)*ctr_sub_lda_A, ctr_sub_lda_A*b_A, 
                       op_A, ctr_sub_lda_A);
          }
        } else
          op_A = buf_A;
        MPI_Bcast(op_A, s_A, sr_A->mdtype(), owner_A, cdt_A->cm);
      } else {
        if (ctr_sub_lda_A == 0)
          op_A = this->A;
        else {
          if (ctr_lda_A == 1)
            op_A = this->A+sr_A->el_size*ib*ctr_sub_lda_A;
          else {
            op_A = buf_A;
            sr_A->copy(ctr_sub_lda_A, ctr_lda_A,
                       this->A+sr_A->el_size*ib*ctr_sub_lda_A, ctr_sub_lda_A*edge_len, 
                       buf_A, ctr_sub_lda_A);
          }      
        }
      }
      if (move_B){
        owner_B   = ib % cdt_B->np;
        if (rank_B == owner_B){
          if (b_B == 1){
            op_B = this->B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       this->B+sr_B->el_size*(ib/cdt_B->np)*ctr_sub_lda_B, ctr_sub_lda_B*b_B, 
                       op_B, ctr_sub_lda_B);
          }
        } else 
          op_B = buf_B;
//        printf("c_B = %ld, s_B = %ld, d_B = %ld, b_B = %ld\n", c_B, s_B,db, b_B);
        MPI_Bcast(op_B, s_B, sr_B->mdtype(), owner_B, cdt_B->cm);
      } else {
        if (ctr_sub_lda_B == 0)
          op_B = this->B;
        else {
          if (ctr_lda_B == 1){
            op_B = this->B+sr_B->el_size*ib*ctr_sub_lda_B;
          } else {
            op_B = buf_B;
            sr_B->copy(ctr_sub_lda_B, ctr_lda_B,
                       this->B+sr_B->el_size*ib*ctr_sub_lda_B, ctr_sub_lda_B*edge_len, 
                       buf_B, ctr_sub_lda_B);
          }      
        }
      }
      if (move_C){
        op_C = buf_C;
        rec_ctr->beta = sr_C->addid();
      } else {
        if (ctr_sub_lda_C == 0)
          op_C = this->C;
        else {
          if (ctr_lda_C == 1) 
            op_C = this->C+sr_C->el_size*ib*ctr_sub_lda_C;
          else {
            op_C = buf_C;
            rec_ctr->beta = sr_C->addid();
          }
        }
      } 


      rec_ctr->A = op_A;
      rec_ctr->B = op_B;
      rec_ctr->C = op_C;

      rec_ctr->run();

      /*for (int i=0; i<ctr_sub_lda_C*ctr_lda_C; i++){
        printf("[%d] P%d op_C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)op_C)[i]);
      }*/
      if (move_C){
        /* FIXME: Wont work for single precsion */
        MPI_Allreduce(MPI_IN_PLACE, op_C, s_C, sr_C->mdtype(), sr_C->addmop(), cdt_C->cm);
        owner_C   = ib % cdt_C->np;
        if (rank_C == owner_C){
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     op_C, ctr_sub_lda_C, sr_C->mulid(),
                     this->C+sr_C->el_size*(ib/cdt_C->np)*ctr_sub_lda_C, 
                     ctr_sub_lda_C*b_C, this->beta);
        }
      } else {
        if (ctr_lda_C != 1 && ctr_sub_lda_C != 0)
          sr_C->copy(ctr_sub_lda_C, ctr_lda_C,
                     buf_C, ctr_sub_lda_C, sr_C->mulid(), 
                     this->C+sr_C->el_size*ib*ctr_sub_lda_C, 
                     ctr_sub_lda_C*edge_len, this->beta);
        if (ctr_sub_lda_C == 0)
          rec_ctr->beta = sr_C->mulid();
      }
/*      for (int i=0; i<ctr_sub_lda_C*ctr_lda_C*edge_len; i++){
        printf("[%d] P%d C[%d]  = %lf\n",ctr_lda_C,idx_lyr,i, ((double*)C)[i]);
      }*/
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
      CTF_int::cdealloc(buf_A);
      CTF_int::cdealloc(buf_B);
      CTF_int::cdealloc(buf_C);
      CTF_int::cdealloc(buf_aux);
    }
    TAU_FSTOP(spctr_2d_general);
  }
}

