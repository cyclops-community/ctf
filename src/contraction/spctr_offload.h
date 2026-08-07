/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SPCTR_OFFLOAD_H__
#define __SPCTR_OFFLOAD_H__

#include "../shared/offload.h"
#include "spctr_tsr.h"

namespace CTF_int {
  #ifdef OFFLOAD
  class spctr_offload : public spctr {
    public: 
      /* Class to be called on sub-blocks */
      spctr * rec_ctr;
      int iter_counter;
      int total_iter;
      int upload_phase_A;
      int upload_phase_B;
      int download_phase_C;
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int64_t size_C; /* size of C blocks */
      offload_arr * spr_A;
      offload_arr * spr_B;
      offload_arr * spr_C;
      
      /**
       * \brief print ctr object
       */
      void print();

      /**
       * \brief offloads and downloads local blocks of dense or CSR tensors
       */
      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);

      /**
       * \brief returns the number of bytes of buffer space
         we need 
       * \return bytes needed
       */
      int64_t spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the time this kernel will take excluding calls to rec_ctr
       * \return seconds needed
       */
      double est_time_fp(int nlyr, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the time this kernel will take including calls to rec_ctr
       * \return seconds needed for recursive contraction
       */
      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      spctr * clone();

      spctr_offload(spctr * other);

      /**
       * \brief deallocates ctr_offload object
       */
      ~spctr_offload();

      /**
       * \brief allocates ctr_offload object
       * \param[in] c contraction object
       * \param[in] size_A size of the A tensor
       * \param[in] size_B size of the B tensor
       * \param[in] size_C size of the C tensor
       * \param[in] total_iter number of gemms to be done
       * \param[in] upload_phase_A period in iterations with which to upload A
       * \param[in] upload_phase_B period in iterations with which to upload B
       * \param[in] download_phase_C period in iterations with which to download C
       */
      spctr_offload(contraction const * c,
                    int64_t size_A,
                    int64_t size_B,
                    int64_t size_C,
                    int total_iter,
                    int upload_phase_A,
                    int upload_phase_B,
                    int download_phase_C);

  };
  #endif

}
#endif
