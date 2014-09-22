/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "ctr_comm.h"
#include "../shared/util.h"
#include <climits>

/**
 * \brief deallocs ctr_2d_general object
 */
ctr_2d_general::~ctr_2d_general() {
  if (move_A) FREE_CDT(cdt_A);
  if (move_B) FREE_CDT(cdt_B);
  if (move_C) FREE_CDT(cdt_C);
  if (rec_ctr != NULL)
    delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
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

/**
 * \brief print ctr object
 */
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



/**
 * \brief copies ctr object
 */
ctr * ctr_2d_general::clone() {
  return new ctr_2d_general(this);
}

/**
 * \brief determines buffer and block sizes needed for ctr_2d_general
 *
 * \param[out] b_A block size of A if its communicated, 0 otherwise
 * \param[out] b_B block size of A if its communicated, 0 otherwise
 * \param[out] b_C block size of A if its communicated, 0 otherwise
 * \param[out] b_A total size of A if its communicated, 0 otherwise
 * \param[out] b_B total size of B if its communicated, 0 otherwise
 * \param[out] b_C total size of C if its communicated, 0 otherwise
 * \param[out] db contraction block size = min(b_A,b_B,b_C)
 * \param[out] aux_size size of auxillary buffer needed 
 */
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
    np_A        = cdt_A.np;
    b_A         = edge_len/np_A;
    s_A         = cdt_A.estimate_bcast_time(el_size_A*ctr_lda_A*ctr_sub_lda_A);
    db          = MIN(b_A, db);
  } 
  if (move_B){
    np_B        = cdt_B.np;
    b_B         = edge_len/np_B;
    s_B         = cdt_B.estimate_bcast_time(el_size_B*ctr_lda_B*ctr_sub_lda_B);
    db          = MIN(b_B, db);
  }
  if (move_C){
    np_C        = cdt_C.np;
    b_C         = edge_len/np_C;
    s_C         = cdt_C.estimate_allred_time(sr_C.el_size*ctr_lda_C*ctr_sub_lda_C);
    db          = MIN(b_C, db);
  }

  aux_size = db*MAX(move_A*el_size_A*s_A, MAX(move_B*el_size_B*s_B, move_C*sr_C.el_size*s_C));
}

/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes sent
 */
double ctr_2d_general::est_time_fp(int nlyr) {
  int64_t b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size;
  find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size);
  return ((s_A+s_B+s_C)*(double)db*edge_len/db)/MIN(nlyr,edge_len);
}
/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
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

/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
int64_t ctr_2d_general::mem_fp() {
  int64_t b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size;
  find_bsizes(b_A, b_B, b_C, s_A, s_B, s_C, db, aux_size);
  return el_size_A*s_A*db+el_size_B*s_B*db+sr_C.el_size*s_C*db+aux_size;
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
int64_t ctr_2d_general::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}

/**
 * \brief Basically doing SUMMA, except assumes equal block size on
 *  each processor. Performs rank-b updates 
 *  where b is the smallest blocking factor among A and B or A and C or B and C. 
 */
void ctr_2d_general::run() {
  int owner_A, owner_B, owner_C,  alloced, ret;
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
    host_pinned_alloc((void**)&buf_A, s_A*db*el_size_A);
    host_pinned_alloc((void**)&buf_B, s_B*db*el_size_B);
    host_pinned_alloc((void**)&buf_C, s_C*db*sr_C.el_size);
#endif
  if (0){
  } else {
    ret = CTF_mst_alloc_ptr(s_A*db*el_size_A, (void**)&buf_A);
    ASSERT(ret==0);
    ret = CTF_mst_alloc_ptr(s_B*db*el_size_B, (void**)&buf_B);
    LIBT_BSSERT(ret==0);
    ret = CTF_mst_alloc_ptr(s_C*db*sr_C.el_size, (void**)&buf_C);
    LIBT_CSSERT(ret==0);
  }
  ret = CTF_mst_alloc_ptr(aux_size, (void**)&buf_aux);
  LIBT_CSSERT(ret==0);

  for (ib=this->idx_lyr*db; ib<edge_len; ib+=db*this->num_lyr){
    if (move_A){
      owner_A   = ib / b_A;
      c_A       = MIN(((owner_A+1)*b_A-ib), db);
      if (rank_A == owner_A){
        if (c_A == b_A){
          op_A = this->A;
        } else {
          op_A = buf_A;
          lda_cpy(el_size_A,
                  ctr_sub_lda_A*c_A, ctr_lda_A,
                  ctr_sub_lda_A*b_A, ctr_sub_lda_A*c_A, 
                  this->A+el_size_A*(ib%b_A)*ctr_sub_lda_A, op_A);
        }
      } else
        op_A = buf_A;
      POST_BCAST(op_A, c_A*s_A*el_size_A, COMM_CHAR_T, owner_A, cdt_A, 0);
      if (c_A < db){ /* If the required A block is cut between 2 procs */
        if (rank_A == owner_A+1)
          lda_cpy(el_size_A,
                  ctr_sub_lda_A*(db-c_A), ctr_lda_A,
                  ctr_sub_lda_A*b_A, ctr_sub_lda_A*(db-c_A), 
                  this->A, buf_aux);
        POST_BCAST(buf_aux, s_A*(db-c_A)*el_size_A, COMM_CHAR_T, owner_A+1, cdt_A, 0);
        coalesce_bwd(el_size_A,
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
          op_A = this->A+el_size_A*ib*ctr_sub_lda_A;
        else {
          op_A = buf_A;
          lda_cpy(el_size_A,
                  ctr_sub_lda_A, ctr_lda_A,
                  ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                  this->A+el_size_A*ib*ctr_sub_lda_A, buf_A);
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
          lda_cpy(el_size_B,
                  ctr_sub_lda_B*c_B, ctr_lda_B,
                  ctr_sub_lda_B*b_B, ctr_sub_lda_B*c_B, 
                  this->B+el_size_B*(ib%b_B)*ctr_sub_lda_B, op_B);
        }
      } else 
        op_B = buf_B;
      POST_BCAST(op_B, c_B*s_B*el_size_B, COMM_CHAR_T, owner_B, cdt_B, 0);
      if (c_B < db){ /* If the required B block is cut between 2 procs */
        if (rank_B == owner_B+1)
          lda_cpy(el_size_B,
                  ctr_sub_lda_B*(db-c_B), ctr_lda_B,
                  ctr_sub_lda_B*b_B, ctr_sub_lda_B*(db-c_B), 
                  this->B, buf_aux);
        POST_BCAST(buf_aux, s_B*(db-c_B)*el_size_B, COMM_CHAR_T, owner_B+1, cdt_B, 0);
        coalesce_bwd(el_size_B,
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
          op_B = this->B+el_size_B*ib*ctr_sub_lda_B;
        } else {
          op_B = buf_B;
          lda_cpy(el_size_B,
                  ctr_sub_lda_B, ctr_lda_B,
                  ctr_sub_lda_B*edge_len, ctr_sub_lda_B,
                  this->B+el_size_B*ib*ctr_sub_lda_B, buf_B);
        }      
      }
    }
    if (move_C){
      op_C = buf_C;
      rec_ctr->beta = sr_C.addid;
    } else {
      if (ctr_sub_lda_C == 0)
        op_C = this->C;
      else {
        if (false && ctr_lda_C == 1) 
          op_C = this->C+el_size_A*ib*ctr_sub_lda_C;
        else {
          op_C = buf_C;
          rec_ctr->beta = sr_C.addid;
        }
      }
    } 


    rec_ctr->A = op_A;
    rec_ctr->B = op_B;
    rec_ctr->C = op_C;

    rec_ctr->run();

    if (move_C){
      /* FIXME: Wont work for single precsion */
      ALLREDUCE(MPI_IN_PLACE, op_C, db*s_C, sr_C.mdtype, sr_C.addmop, cdt_C);
      owner_C   = ib / b_C;
      c_C       = MIN(((owner_C+1)*b_C-ib), db);
      if (rank_C == owner_C){
        lda_cpy(sr_C.axpy,
                ctr_sub_lda_C*c_C, ctr_lda_C,
                ctr_sub_lda_C*db, ctr_sub_lda_C*b_C, 
                op_C, this->C+sr_C.el_size*(ib%b_C)*ctr_sub_lda_C, 
                sr_C.mulid, this->beta);
      }
      if (c_C < db){ /* If the required B block is cut between 2 procs */
        if (rank_C == owner_C+1)
          lda_cpy(sr_C.axpy,
                  ctr_sub_lda_C*(db-c_C), ctr_lda_C,
                  ctr_sub_lda_C*db, ctr_sub_lda_C*b_C, 
                  op_C+sr_C.el_size*ctr_sub_lda_C*c_C, this->C, 
                  sr_C.mulid, this->beta);
      }
    } else {
      if (ctr_sub_lda_C != 0)
        lda_cpy(sr_C.axpy,
                ctr_sub_lda_C, ctr_lda_C,
                ctr_sub_lda_C, ctr_sub_lda_C*edge_len, 
                buf_C, this->C+sr_C.el_size*ib*ctr_sub_lda_C,
                sr_C.mulid, this->beta);
    }
    rec_ctr->beta = sr_C.mulid;
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
    CTF_free(buf_A);
    CTF_free(buf_B);
    CTF_free(buf_C);
  }
  TAU_FSTOP(ctr_2d_general);
}


template class ctr_2d_general<double>;
template class ctr_2d_general< std::complex<double> >;
