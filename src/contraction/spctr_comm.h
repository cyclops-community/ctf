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
    /*  void set_size_blk_A(int new_nblk_A, int64_t const * nnbA){
        spctr::set_size_blk_A(new_nblk_A, nnbA);
        rec_ctr->set_size_blk_A(new_nblk_A, nnbA);
      }*/
      
      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);
      /**
       * \brief returns the number of bytes of buffer space
       *  we need 
       * \return bytes needed
       */
      /**
       * \brief returns the number of bytes need by each processor in this kernel 
       * \return bytes needed for contraction
       */
      int64_t spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the number of bytes used temporarily (not needed as we recurse)
       * \return bytes needed
       */
      int64_t spmem_tmp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the number of bytes need by each processor in this kernel and its recursive calls
       * \return bytes needed for recursive contraction
       */
      int64_t spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      double est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
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
