/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SPCTR_TSR_H__
#define __SPCTR_TSR_H__

#include "ctr_tsr.h"

namespace CTF_int{

  class spctr : public ctr {
    public: 
      bool      is_sparse_A;
      bool      is_sparse_B;
      bool      is_sparse_C;
      bool      is_ccsr_C;
      char *    new_C;

      ~spctr();
      spctr(spctr * other);
      virtual spctr * clone() { return NULL; }

      /**
       * \brief returns the execution time the local part this kernel is estimated to take
       * \param[in] nlyr amount of replication
       * \param[in] nblk_A number of virtual blocks in A
       * \param[in] nblk_B number of virtual blocks in B
       * \param[in] nblk_C number of virtual blocks in C
       * \param[in] nnz_frac_A percentage of nonzeros in tensor A
       * \param[in] nnz_frac_B percentage of nonzeros in tensor B
       * \param[in] nnz_frac_C percentage of nonzeros in tensor C
       * \return time in sec
       */
      virtual double est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){ return 0.0; }
      double est_time_fp(int nlyr){ return est_time_fp(nlyr, 1, 1, 1, 1.0, 1.0, 1.0); }
      /**
       * \brief returns the execution time this kernel and its recursive calls are estimated to take
       * \param[in] nlyr amount of replication
       * \param[in] nblk_A number of virtual blocks in A
       * \param[in] nblk_B number of virtual blocks in B
       * \param[in] nblk_C number of virtual blocks in C
       * \param[in] nnz_frac_A percentage of nonzeros in tensor A
       * \param[in] nnz_frac_B percentage of nonzeros in tensor B
       * \param[in] nnz_frac_C percentage of nonzeros in tensor C
       * \return time in sec
       */
      virtual double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){ return 0.0; }
      double est_time_rec(int nlyr){ return est_time_rec(nlyr, 1, 1, 1, 1.0, 1.0, 1.0); }
      /**
       * \brief returns the number of bytes need by each processor in this kernel and its recursive calls
       * \return bytes needed for recursive contraction
       */
      virtual int64_t spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C){ return 0; };


      void run(char * A, char * B, char * C) { printf("CTF ERROR: PROVIDE SPARSITY ARGS TO RUN\n"); assert(0); };
      virtual void run(char * A, int nblk_A, int64_t const * size_blk_A,
                       char * B, int nblk_B, int64_t const * size_blk_B,
                       char * C, int nblk_C, int64_t * size_blk_C,
                       char *& new_C) { IASSERT(0); }
      spctr(contraction const * c);
  };

  class seq_tsr_spctr : public spctr {
    public:
      char const * alpha;
      int         order_A;
      int64_t *   edge_len_A;
      int const * idx_map_A;
      int *       sym_A;

      int         order_B;
      int64_t *   edge_len_B;
      int const * idx_map_B;
      int *       sym_B;
      
      int         order_C;
      int64_t *   edge_len_C;
      int const * idx_map_C;
      int *       sym_C;

      int krnl_type;
      iparam inner_params;
      
      int is_custom;
      bivar_function const * func; // custom_params;
      

      /**
       * \brief wraps user sequential function signature
       */
      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);
      void print();
      int64_t spmem_fp();
      spctr * clone();
      double est_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      int64_t est_spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      uint64_t est_membw(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      double est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      /**
       * \brief copies ctr object
       * \param[in] other object to copy
       */
      seq_tsr_spctr(spctr * other);
      ~seq_tsr_spctr(){ 
        CTF_int::cdealloc(edge_len_A), CTF_int::cdealloc(edge_len_B), CTF_int::cdealloc(edge_len_C), 
        CTF_int::cdealloc(sym_A), CTF_int::cdealloc(sym_B), CTF_int::cdealloc(sym_C); 
      }

      seq_tsr_spctr(contraction const * s,
                    int                 krnl_type,
                    iparam const *      inner_params,
                    int64_t *           virt_blk_len_A,
                    int64_t *           virt_blk_len_B,
                    int64_t *           virt_blk_len_C,
                    int64_t             vrt_sz_C);

  };

  class spctr_virt : public spctr {
    public: 
      spctr * rec_ctr;
      int num_dim;
      int * virt_dim;
      int order_A;
      int64_t blk_sz_A;
      int const * idx_map_A;
      int order_B;
      int64_t blk_sz_B;
      int const * idx_map_B;
      int order_C;
      int64_t blk_sz_C;
      int const * idx_map_C;
      
      void print();

      /**
       * \brief iterates over the dense virtualization block grid and contracts
       */
      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);
      int64_t spmem_fp();
      int64_t spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);

      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      spctr * clone();

      /**
       * \brief deallocates spctr_virt object
       */
      ~spctr_virt();

      /**
       * \brief copies spctr_virt object
       */
      spctr_virt(spctr *other);
      spctr_virt(contraction const * c,
                 int                 num_tot,
                 int *               virt_dim,
                 int64_t             vrt_sz_A,
                 int64_t             vrt_sz_B,
                 int64_t             vrt_sz_C);
  };

  class spctr_pin_keys : public spctr {
    public:
      spctr * rec_ctr;
      int AxBxC;
      int order;
      int64_t const * lens;
      int * divisor;
      int * virt_dim;
      int * phys_rank;
      int64_t dns_blk_sz;

      void run(char * A, int nblk_A, int64_t const * size_blk_A,
               char * B, int nblk_B, int64_t const * size_blk_B,
               char * C, int nblk_C, int64_t * size_blk_C,
               char *& new_C);
      void print();
      int64_t spmem_fp(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      int64_t spmem_rec(double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      spctr * clone();

      double est_time_fp(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      double est_time_rec(int nlyr, int nblk_A, int nblk_B, int nblk_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C);
      spctr_pin_keys(spctr * other);
      ~spctr_pin_keys();
      spctr_pin_keys(contraction const * s, int AxBxC);

  };
}

#endif
