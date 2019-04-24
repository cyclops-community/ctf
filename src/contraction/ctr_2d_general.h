/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include "ctr_comm.h"

#ifndef __CTR_2D_GENERAL_H__
#define __CTR_2D_GENERAL_H__

namespace CTF_int{
  class tensor;
  int  ctr_2d_gen_build(int                        is_used,
                        CommData                   global_comm,
                        int                        i,
                        int *                      virt_dim,
                        int64_t &                  cg_edge_len,
                        int &                      total_iter,
                        tensor *                   A,
                        int                        i_A,
                        CommData *&                cg_cdt_A,
                        int64_t &                  cg_ctr_lda_A,
                        int64_t &                  cg_ctr_sub_lda_A,
                        bool &                     cg_move_A,
                        int64_t *                  blk_len_A,
                        int64_t &                  blk_sz_A,
                        int64_t const *            virt_blk_len_A,
                        int &                      load_phase_A,
                        tensor *                   B,
                        int                        i_B,
                        CommData *&                cg_cdt_B,
                        int64_t &                  cg_ctr_lda_B,
                        int64_t &                  cg_ctr_sub_lda_B,
                        bool &                     cg_move_B,
                        int64_t *                  blk_len_B,
                        int64_t &                  blk_sz_B,
                        int64_t const *            virt_blk_len_B,
                        int &                      load_phase_B,
                        tensor *                   C,
                        int                        i_C,
                        CommData *&                cg_cdt_C,
                        int64_t &                  cg_ctr_lda_C,
                        int64_t &                  cg_ctr_sub_lda_C,
                        bool &                     cg_move_C,
                        int64_t *                  blk_len_C,
                        int64_t &                  blk_sz_C,
                        int64_t const *            virt_blk_len_C,
                        int &                      load_phase_C);


  class ctr_2d_general : public ctr {
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
      ctr * rec_ctr;
      
      /**
       * \brief print ctr object
       */
      void print();
      /**
       * \brief Basically doing SUMMA, except assumes equal block size on
       *  each processor. Performs rank-b updates 
       *  where b is the smallest blocking factor among A and B or A and C or B and C. 
       */
      void run(char * A, char * B, char * C);
      /**
       * \brief returns the number of bytes of buffer space
       *  we need 
       * \return bytes needed
       */
      int64_t mem_fp();
      /**
       * \brief returns the number of bytes of buffer space we need recursively 
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec();
      /**
       * \brief returns the number of bytes this kernel will send per processor
       * \return bytes sent
       */
      double est_time_fp(int nlyr);
      /**
       * \brief returns the number of bytes send by each proc recursively 
       * \return bytes needed for recursive contraction
       */
      double est_time_rec(int nlyr);
      ctr * clone();

      /**
       * \brief determines buffer and block sizes needed for ctr_2d_general
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
       * \brief copies ctr object
       */
      ctr_2d_general(ctr * other);
      /**
       * \brief deallocs ctr_2d_general object
       */
      ~ctr_2d_general();
      /**
       * \brief partial constructor, most of the logic is in the ctr_2d_gen_build function
       * \param[in] c contraction object to get info about ctr from
       */
      ctr_2d_general(contraction * c) : ctr(c){ move_A=0; move_B=0; move_C=0; }
  };
}
#endif
