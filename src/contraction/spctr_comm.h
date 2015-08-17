/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SPCTR_COMM_H__
#define __SPCTR_COMM_H__

#include "ctr_comm.h"

namespace CTF_int{

  class spctr : public ctr {
    public: 
      bool      is_sparse_A;
      int64_t   nnz_A;
      int       nvirt_A;
      int64_t * nnz_blk_A;
      bool      is_sparse_B;
      int64_t   nnz_B;
      int       nvirt_B;
      int64_t * nnz_blk_B;
      bool      is_sparse_C;
      int64_t   nnz_C;
      int       nvirt_C;
      int64_t * nnz_blk_C;
      int64_t   new_nnz_C;
      char *    new_C;

      ~spctr();
      spctr(spctr * other);
      virtual spctr * clone() { return NULL; }
      spctr(contraction const * c);
      virtual void set_nnz_blk_A(int64_t const * nnbA){
        if (nnbA != NULL) memcpy(nnz_blk_A, nnbA, nvirt_A*sizeof(int64_t));
      }
      virtual void set_nnz_blk_B(int64_t const * nnbB){
        if (nnbB != NULL) memcpy(nnz_blk_B, nnbB, nvirt_B*sizeof(int64_t));
      }
  };

}

#endif
