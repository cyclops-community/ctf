/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SPCTR_COMM_H__
#define __SPCTR_COMM_H__

#include "spctr_tsr.h"

namespace CTF_int{
  class contraction;

  class spctr_replicate : public spctr {
    public: 
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */
      int ncdt_C; /* number of processor dimensions to replicate C along */
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int64_t size_C; /* size of C blocks */

      CommData ** cdt_A;
      CommData ** cdt_B;
      CommData ** cdt_C;
      /* Class to be called on sub-blocks */
      spctr * rec_ctr;
      void set_nnz_blk_A(int new_nvirt_A, int64_t const * nnbA){
        spctr::set_nnz_blk_A(new_nvirt_A, nnbA);
        rec_ctr->set_nnz_blk_A(new_nvirt_A, nnbA);
      }
      
      void run();
      /**
       * \brief returns the number of bytes of buffer space
       *  we need 
       * \return bytes needed
       */
      int64_t mem_fp();
      /**
       * \brief returns the number of bytes need by each processor in this kernel and its recursive calls
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec();
      /**
       * \brief returns the execution time the local part this kernel is estimated to take
       * \return time in sec
       */
      double est_time_fp(int nlyr);
      /**
       * \brief returns the execution time this kernel and its recursive calls are estimated to take
       * \return time in sec
       */
      double est_time_rec(int nlyr);
      void print();
      spctr * clone();

      spctr_replicate(spctr * other);
      ~spctr_replicate();
      spctr_replicate(contraction const * c,
                    int const *         phys_mapped,
                    int64_t             blk_sz_A,
                    int64_t             blk_sz_B,
                    int64_t             blk_sz_C);
  };

} 
#endif // __CTR_COMM_H__
