/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "spctr_comm.h"

#ifndef __SPCTR_2D_GENERAL_H__
#define __SPCTR_2D_GENERAL_H__

namespace CTF_int{
  class tensor;

  class spctr_2d_general : public spctr {
    public: 
      int64_t edge_len;

      int64_t ctr_lda_A; /* local lda_A of contraction dimension 'k' */
      int64_t ctr_sub_lda_A; /* elements per local lda_A 
                            of contraction dimension 'k' */
      int64_t ctr_lda_B; /* local lda_B of contraction dimension 'k' */
      int64_t ctr_sub_lda_B; /* elements per local lda_B 
                            of contraction dimension 'k' */
      int64_t ctr_lda_C; /* local lda_C of contraction dimension 'k' */
      int64_t ctr_sub_lda_C; /* elements per local lda_C 
                            of contraction dimension 'k' */
      int64_t dns_vrt_sz_A; /* size of each block of A when stored as dense, needed for time estimation given nnz frac */
      int64_t dns_vrt_sz_B; /* size of each block of A when stored as dense, needed for time estimation given nnz frac */
      int64_t dns_vrt_sz_C; /* size of each block of A when stored as dense, needed for time estimation given nnz frac */
  #ifdef OFFLOAD
      bool alloc_host_buf;
  #endif

      bool move_A;
      bool move_B;
      bool move_C;

      CommData * cdt_A;
      CommData * cdt_B;
      CommData * cdt_C;
      /* Class to be called on sub-blocks */
      spctr * rec_ctr;
      
      /**
       * \brief print ctr object
       */
      void print();
      /**
       * \brief Basically doing SUMMA, except assumes equal block size on
       *  each processor. Performs rank-b updates 
       *  where b is the smallest blocking factor among A and B or A and C or B and C. 
       */
      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);

      /**
       * \brief returns the number of bytes of buffer space
       *  we need 
       * \return bytes needed
       */
      int64_t spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the number of bytes used temporarily (not needed as we recurse)
       * \return bytes needed
       */
      int64_t spmem_tmp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      /**
       * \brief returns the time this kernel will take including calls to rec_ctr
       * \return seconds needed for recursive contraction
       */
      double est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief returns the time this kernel will take including calls to rec_ctr
       * \return seconds needed for recursive contraction
       */
      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      spctr * clone();
/*      void set_size_blk_A(int new_nblk_A, int64_t const * nnbA){
        spctr::set_size_blk_A(new_nblk_A, nnbA);
        rec_ctr->set_size_blk_A(new_nblk_A, nnbA);
      }*/

      /**
       * \brief determines buffer and block sizes needed for spctr_2d_general
       *
       * \param[out] b_A block size of A if its communicated, 0 otherwise
       * \param[out] b_B block size of A if its communicated, 0 otherwise
       * \param[out] b_C block size of A if its communicated, 0 otherwise
       * \param[out] s_A total size of A if its communicated, 0 otherwise
       * \param[out] s_B total size of B if its communicated, 0 otherwise
       * \param[out] s_C total size of C if its communicated, 0 otherwise
       * \param[out] aux_size size of auxillary buffer needed 
       */
      void find_bsizes(int64_t & b_A,
                       int64_t & b_B,
                       int64_t & b_C,
                       int64_t & s_A,
                       int64_t & s_B,
                       int64_t & s_C,
                       int64_t & aux_size);
      /**
       * \brief copies spctr object
       */
      spctr_2d_general(spctr * other);
      /**
       * \brief deallocs spctr_2d_general object
       */
      ~spctr_2d_general();
      /**
       * \brief partial constructor, most of the logic is in the spctr_2d_gen_build function
       * \param[in] c contraction object to get info about spctr from
       */
      spctr_2d_general(contraction * c) : spctr(c){ move_A=0; move_B=0; move_C=0; }
  };
}

#endif
