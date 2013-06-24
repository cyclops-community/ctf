/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "ctr_comm.h"
#include "../shared/util.h"
#include <climits>

/**
 * \brief deallocates ctr_2d_general object
 */
template<typename dtype>
ctr_2d_general<dtype>::~ctr_2d_general() {
   delete rec_ctr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr_2d_general<dtype>::ctr_2d_general(ctr<dtype> * other) : ctr<dtype>(other) {
  ctr_2d_general<dtype> * o = (ctr_2d_general<dtype>*)other;
  rec_ctr = o->rec_ctr->clone();
  edge_len      = o->edge_len;
  ctr_lda_A     = o->ctr_lda_A;
  ctr_sub_lda_A = o->ctr_sub_lda_A;
  cdt_A         = o->cdt_A;
  ctr_lda_B     = o->ctr_lda_B;
  ctr_sub_lda_B = o->ctr_sub_lda_B;
  cdt_B         = o->cdt_B;
  ctr_lda_C     = o->ctr_lda_C;
  ctr_sub_lda_C = o->ctr_sub_lda_C;
  cdt_C         = o->cdt_C;
}

/**
 * \brief print ctr object
 */
template<typename dtype>
void ctr_2d_general<dtype>::print() {
  printf("ctr_2d_general: edge_len = %d\n", edge_len);
  printf("cdt_A = %p, ctr_lda_A = %lld, ctr_sub_lda_A = %lld\n",
          cdt_A, ctr_lda_A, ctr_sub_lda_A);
  if (cdt_A != NULL) printf("cdt_A length = %d\n",cdt_A->np);
  printf("cdt_B = %p, ctr_lda_B = %lld, ctr_sub_lda_B = %lld\n",
          cdt_B, ctr_lda_B, ctr_sub_lda_B);
  if (cdt_B != NULL) printf("cdt_B length = %d\n",cdt_B->np);
  printf("cdt_C = %p, ctr_lda_C = %lld, ctr_sub_lda_C = %lld\n",
          cdt_C, ctr_lda_C, ctr_sub_lda_C);
  if (cdt_C != NULL) printf("cdt_C length = %d\n",cdt_C->np);
  rec_ctr->print();
}



/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * ctr_2d_general<dtype>::clone() {
  return new ctr_2d_general<dtype>(this);
}
/**
 * \brief returns the number of bytes this kernel will send per processor
 * \return bytes sent
 */
template<typename dtype>
uint64_t ctr_2d_general<dtype>::comm_fp(int nlyr) {
  long_int db;
  int np_A,     np_B,   np_C;
  long_int b_A,         b_B,    b_C;
  long_int s_A,         s_B,    s_C;
  db = long_int_max;
  s_A = 0, s_B = 0, s_C = 0;
  if (cdt_A != NULL){
    np_A        = cdt_A->np;
    b_A         = edge_len/np_A;
    s_A         = ctr_lda_A*ctr_sub_lda_A*(long_int)log(cdt_A->np);
    db          = MIN(b_A, db);
  } 
  if (cdt_B != NULL){
    np_B        = cdt_B->np;
    b_B         = edge_len/np_B;
    s_B         = ctr_lda_B*ctr_sub_lda_B*(long_int)log(cdt_B->np);
    db          = MIN(b_B, db);
  }
  if (cdt_C != NULL){
    np_C        = cdt_C->np;
    b_C         = edge_len/np_C;
    s_C         = ctr_lda_C*ctr_sub_lda_C*(long_int)log(cdt_C->np);
    db          = MIN(b_C, db);
  }
  return ((s_A+s_B+s_C)*(uint64_t)db*sizeof(dtype)*edge_len/db)/MIN(nlyr,edge_len);
}
/**
 * \brief returns the number of bytes send by each proc recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
uint64_t ctr_2d_general<dtype>::comm_rec(int nlyr) {
  long_int db;
  db = long_int_max;
  if (cdt_A != NULL)
    db          = MIN(db,edge_len/cdt_A->np);
  if (cdt_B != NULL)
    db          = MIN(db,edge_len/cdt_B->np);
  if (cdt_C != NULL)
    db          = MIN(db,edge_len/cdt_C->np);
  return (edge_len/db)*rec_ctr->comm_rec(1) + comm_fp(nlyr);
}

/**
 * \brief returns the number of bytes of buffer space
   we need 
 * \return bytes needed
 */
template<typename dtype>
long_int ctr_2d_general<dtype>::mem_fp() {
  long_int db, aux_size;
  int np_A,     np_B,   np_C;
  long_int b_A,         b_B,    b_C;
  long_int s_A,         s_B,    s_C;
  db = long_int_max;
  s_A = 0, s_B = 0, s_C = 0;
  if (ctr_sub_lda_A != 0)
    s_A = ctr_sub_lda_A*ctr_lda_A;
  if (ctr_sub_lda_B != 0)
    s_B = ctr_sub_lda_B*ctr_lda_B;
  if (ctr_sub_lda_C != 0)
    s_C = ctr_sub_lda_C*ctr_lda_C;
  aux_size = 0;
  if (cdt_A != NULL){
    np_A        = cdt_A->np;
    LIBT_ASSERT(np_A!=0);
    b_A         = edge_len/np_A;
    s_A         = ctr_lda_A*ctr_sub_lda_A;
    db          = MIN(b_A, db);
  } 
  if (cdt_B != NULL){
    np_B        = cdt_B->np;
    LIBT_ASSERT(np_B!=0);
    b_B         = edge_len/np_B;
    s_B         = ctr_lda_B*ctr_sub_lda_B;
    if (db != long_int_max && b_B != db){
      aux_size  = MAX(s_A,s_B)*MIN(b_B,db);
    }
    db          = MIN(b_B, db);
  }
  if (cdt_C != NULL){
    np_C        = cdt_C->np;
    LIBT_ASSERT(np_C!=0);
    b_C         = edge_len/np_C;
    s_C         = ctr_lda_C*ctr_sub_lda_C;
    if (db != long_int_max && b_C != db){
      aux_size  = MAX(aux_size, MAX(s_A,s_B)*MIN(b_C,db));
    }
    db          = MIN(b_C, db);
  }
  return (s_A*db+s_B*db+s_C*db+aux_size)*sizeof(dtype);
}

/**
 * \brief returns the number of bytes of buffer space we need recursively 
 * \return bytes needed for recursive contraction
 */
template<typename dtype>
long_int ctr_2d_general<dtype>::mem_rec() {
  return rec_ctr->mem_rec() + mem_fp();
}

/**
 * \brief Basically doing SUMMA, except assumes equal block size on
 *  each processor. Performs rank-b updates 
 *  where b is the smallest blocking factor among A and B or A and C or B and C. 
 */
template<typename dtype>
void ctr_2d_general<dtype>::run() {
  int owner_A, owner_B, owner_C,  alloced, ret;
  long_int db, ib, c_A, c_B, c_C;
  dtype * buf_A, * buf_B, * buf_C, * buf_aux; 
  dtype * op_A, * op_B, * op_C; 
  int rank_A,   rank_B, rank_C;
  int np_A,     np_B,   np_C;
  long_int b_A,         b_B,    b_C;
  long_int s_A,         s_B,    s_C;
  
  TAU_FSTART(ctr_2d_general);

  /* Must move at least one tensor */
  LIBT_ASSERT(!(cdt_A == NULL && cdt_B == NULL && cdt_C == NULL));
  /* Must move at most two tensors */
  LIBT_ASSERT(!(cdt_A != NULL && cdt_B != NULL && cdt_C != NULL));
  
  rec_ctr->beta         = this->beta;
  rec_ctr->num_lyr      = 1;
  rec_ctr->idx_lyr      = 0;
  
  if (this->buffer != NULL){    
    alloced = 0;
  } else {
    alloced = 1;
    ret = CTF_mst_alloc_ptr(mem_fp(), (void**)&this->buffer);
    LIBT_ASSERT(ret==0);
  }

  db = long_int_max;
  buf_aux = this->buffer;
  std::fill(this->buffer, this->buffer + mem_fp()/sizeof(dtype), get_zero<dtype>());
  s_A = 0, s_B = 0, s_C = 0;
  b_A = 0, b_B = 0, b_C = 0;
  rank_A = 0, rank_B = 0, rank_C = 0;
  if (ctr_sub_lda_A != 0)
    s_A = ctr_sub_lda_A*ctr_lda_A;
  if (ctr_sub_lda_B != 0)
    s_B = ctr_sub_lda_B*ctr_lda_B;
  if (ctr_sub_lda_C != 0)
    s_C = ctr_sub_lda_C*ctr_lda_C;
  if (cdt_A != NULL){
    rank_A      = cdt_A->rank;
    np_A        = cdt_A->np;
    b_A         = edge_len/np_A;
    s_A         = ctr_lda_A*ctr_sub_lda_A;
    db          = MIN(b_A, db);
    LIBT_ASSERT(edge_len%np_A == 0);
  } 
  if (cdt_B != NULL){
    rank_B      = cdt_B->rank;
    np_B        = cdt_B->np;
    b_B         = edge_len/np_B;
    s_B         = ctr_lda_B*ctr_sub_lda_B;
    db          = MIN(b_B, db);
    LIBT_ASSERT(edge_len%np_B == 0);
  }
  if (cdt_C != NULL){
    rank_C      = cdt_C->rank;
    np_C        = cdt_C->np;
    b_C         = edge_len/np_C;
    s_C         = ctr_lda_C*ctr_sub_lda_C;
    db          = MIN(b_C, db);
    LIBT_ASSERT(edge_len%np_C == 0);
  }
  buf_A         = buf_aux;
  buf_aux       += s_A*db;
  buf_B         = buf_aux;
  buf_aux       += s_B*db;
  buf_C         = buf_aux;
  buf_aux       += s_C*db;


  for (ib=this->idx_lyr*db; ib<edge_len; ib+=db*this->num_lyr){
    if (cdt_A != NULL){
      owner_A   = ib / b_A;
      c_A       = MIN(((owner_A+1)*b_A-ib), db);
      if (rank_A == owner_A){
        if (c_A == b_A){
          op_A = this->A;
        } else {
          op_A = buf_A;
          lda_cpy<dtype>( ctr_sub_lda_A*c_A, ctr_lda_A,
                          ctr_sub_lda_A*b_A, ctr_sub_lda_A*c_A, 
                          this->A+(ib%b_A)*ctr_sub_lda_A, op_A);
        }
      } else
        op_A = buf_A;
      POST_BCAST(op_A, c_A*s_A*sizeof(dtype), COMM_CHAR_T, owner_A, cdt_A, 0);
      if (c_A < db){ /* If the required A block is cut between 2 procs */
        if (rank_A == owner_A+1)
          lda_cpy<dtype>( ctr_sub_lda_A*(db-c_A), ctr_lda_A,
                          ctr_sub_lda_A*b_A, ctr_sub_lda_A*(db-c_A), 
                          this->A, buf_aux);
        POST_BCAST(buf_aux, s_A*(db-c_A)*sizeof(dtype), COMM_CHAR_T, owner_A+1, cdt_A, 0);
        coalesce_bwd<dtype>( buf_A, 
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
          op_A = this->A+ib*ctr_sub_lda_A;
        else {
          op_A = buf_A;
          lda_cpy<dtype>( ctr_sub_lda_A, ctr_lda_A,
                          ctr_sub_lda_A*edge_len, ctr_sub_lda_A,
                          this->A+ib*ctr_sub_lda_A, buf_A);
        }      
      }
    }
    if (cdt_B != NULL){
      owner_B   = ib / b_B;
      c_B       = MIN(((owner_B+1)*b_B-ib), db);
      if (rank_B == owner_B){
        if (c_B == b_B){
          op_B = this->B;
        } else {
          op_B = buf_B;
          lda_cpy<dtype>( ctr_sub_lda_B*c_B, ctr_lda_B,
                          ctr_sub_lda_B*b_B, ctr_sub_lda_B*c_B, 
                          this->B+(ib%b_B)*ctr_sub_lda_B, op_B);
        }
      } else 
        op_B = buf_B;
      POST_BCAST(op_B, c_B*s_B*sizeof(dtype), COMM_CHAR_T, owner_B, cdt_B, 0);
      if (c_B < db){ /* If the required B block is cut between 2 procs */
        if (rank_B == owner_B+1)
          lda_cpy<dtype>( ctr_sub_lda_B*(db-c_B), ctr_lda_B,
                          ctr_sub_lda_B*b_B, ctr_sub_lda_B*(db-c_B), 
                          this->B, buf_aux);
        POST_BCAST(buf_aux, s_B*(db-c_B)*sizeof(dtype), COMM_CHAR_T, owner_B+1, cdt_B, 0);
        coalesce_bwd<dtype>( buf_B, 
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
          op_B = this->B+ib*ctr_sub_lda_B;
        } else {
          op_B = buf_B;
          lda_cpy<dtype>(ctr_sub_lda_B, ctr_lda_B,
                  ctr_sub_lda_B*edge_len, ctr_sub_lda_B,
                  this->B+ib*ctr_sub_lda_B, buf_B);
        }      
      }
    }
    if (cdt_C != NULL){
      op_C = buf_C;
      rec_ctr->beta = get_zero<dtype>();
    } else {
      if (ctr_sub_lda_C == 0)
        op_C = this->C;
      else {
        if (false && ctr_lda_C == 1) 
          op_C = this->C+ib*ctr_sub_lda_C;
        else {
          op_C = buf_C;
          rec_ctr->beta = get_zero<dtype>();
        }
      }
    } 


    rec_ctr->A = op_A;
    rec_ctr->B = op_B;
    rec_ctr->C = op_C;

    rec_ctr->run();

    if (cdt_C != NULL){
      /* FIXME: Wont work for single precsion */
      ALLREDUCE(MPI_IN_PLACE, op_C, db*s_C*(sizeof(dtype)/sizeof(double)), COMM_DOUBLE_T, COMM_OP_SUM, cdt_C);
      owner_C   = ib / b_C;
      c_C       = MIN(((owner_C+1)*b_C-ib), db);
      if (rank_C == owner_C){
        lda_cpy<dtype>(ctr_sub_lda_C*c_C, ctr_lda_C,
                ctr_sub_lda_C*db, ctr_sub_lda_C*b_C, 
                op_C, this->C+(ib%b_C)*ctr_sub_lda_C, 
                get_one<dtype>(), this->beta);
      }
      if (c_C < db){ /* If the required B block is cut between 2 procs */
        if (rank_C == owner_C+1)
          lda_cpy<dtype>(ctr_sub_lda_C*(db-c_C), ctr_lda_C,
                  ctr_sub_lda_C*db, ctr_sub_lda_C*b_C, 
                  op_C+ctr_sub_lda_C*c_C, this->C, 
                  get_one<dtype>(), this->beta);
      }
    } else {
      if (ctr_sub_lda_C != 0)
        lda_cpy<dtype>(ctr_sub_lda_C, ctr_lda_C,
                ctr_sub_lda_C, ctr_sub_lda_C*edge_len, 
                buf_C, this->C+ib*ctr_sub_lda_C,
                get_one<dtype>(), this->beta);
    }
    rec_ctr->beta = get_one<dtype>();
  }
  /* FIXME: reuse that */
  if (alloced){
    CTF_free(this->buffer);
    this->buffer = NULL;
  }
  TAU_FSTOP(ctr_2d_general);
}


