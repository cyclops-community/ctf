#ifndef __SPSUM_TSR_H__
#define __SPSUM_TSR_H__

#include "sum_tsr.h"

namespace CTF_int {

  class tspsum : public tsum {
    public: 
      bool      is_sparse_A;
      int64_t   nnz_A;
      int       nvirt_A;
      int64_t * nnz_blk_A;
      bool      is_sparse_B;
      int64_t   nnz_B;
      int       nvirt_B;
      int64_t * nnz_blk_B;
      int64_t   new_nnz_B;
      char *    new_B;

      ~tspsum();
      tspsum(tspsum * other);
      virtual tspsum * clone() { return NULL; }
      tspsum(summation const * s);
      virtual void set_nnz_blk_A(int64_t const * nnbA){
        if (nnbA != NULL) memcpy(nnz_blk_A, nnbA, nvirt_A*sizeof(int64_t));
      }
  };

  class tspsum_virt : public tspsum {
    public: 
      /* Class to be called on sub-blocks */
      tspsum * rec_tsum;

      int         num_dim;
      int *       virt_dim;
      int         order_A;
      int64_t     blk_sz_A; //if dense
      int const * idx_map_A;
      int         order_B;
      int64_t     blk_sz_B; //if dense
      int const * idx_map_B;
      
      void run();
      void print();
      int64_t mem_fp();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
        rec_tsum->set_nnz_blk_A(nnbA);
      }
      tspsum * clone();

      /**
       * \brief iterates over the dense virtualization block grid and contracts
       */
      tspsum_virt(tspsum * other);
      ~tspsum_virt();
      tspsum_virt(summation const * s);
  };

  /**
   * \brief performs replication along a dimension, generates 2.5D algs
   */
  class tspsum_replicate : public tspsum {
    public: 
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */

      CommData ** cdt_A;
      CommData ** cdt_B;
      /* Class to be called on sub-blocks */
      tspsum * rec_tsum;
      
      void run();
      void print();
      int64_t mem_fp();
      tspsum * clone();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
        rec_tsum->set_nnz_blk_A(nnbA);
      }

      tspsum_replicate(tspsum * other);
      ~tspsum_replicate();
      tspsum_replicate(summation const * s,
                       int const *       phys_mapped,
                       int64_t           blk_sz_A,
                       int64_t           blk_sz_B);
  };

  class seq_tsr_spsum : public tspsum {
    public:
      int         order_A;
      int64_t *   edge_len_A;
      int const * idx_map_A;
      int *       sym_A;
      int         order_B;
      int64_t *   edge_len_B;
      int const * idx_map_B;
      int *       sym_B;
      //fseq_tsr_sum func_ptr;

      int is_inner;
      int inr_stride;

      int64_t map_pfx;

      int is_custom;
      univar_function const * func; //fseq_elm_sum custom_params;

      /**
       * \brief wraps user sequential function signature
       */
      void run();
      void print();
      int64_t mem_fp();
      tspsum * clone();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
      }

      /**
       * \brief copies sum object
       * \param[in] other object to copy
       */
      seq_tsr_spsum(tspsum * other);
      ~seq_tsr_spsum(){ CTF_int::cdealloc(edge_len_A), CTF_int::cdealloc(edge_len_B), 
                      CTF_int::cdealloc(sym_A), CTF_int::cdealloc(sym_B); };
      seq_tsr_spsum(summation const * s);

  };

  class tspsum_map : public tspsum {
    public:
      tspsum * rec_tsum;
      int nmap_idx;
      int64_t * map_idx_len;
      int64_t * map_idx_lda;

      void run();
      void print();
      int64_t mem_fp();
      tspsum * clone();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
        rec_tsum->set_nnz_blk_A(nnbA);
      }

      tspsum_map(tspsum * other);
      ~tspsum_map();
      tspsum_map(summation const * s);
  };

  class tspsum_permute : public tspsum {
    public:
      tspsum * rec_tsum;
      bool A_or_B; //if false perm_B
      int order;
      int64_t * lens_new;
      int64_t * lens_old; // FIXME = lens_new?
      int * p;
      bool skip;

      void run();
      void print();
      int64_t mem_fp();
      tspsum * clone();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
        rec_tsum->set_nnz_blk_A(nnbA);
      }

      tspsum_permute(tspsum * other);
      ~tspsum_permute();
      tspsum_permute(summation const * s, bool A_or_B, int64_t const * lens);
  };

  class tspsum_pin_keys : public tspsum {
    public:
      tspsum * rec_tsum;
      bool A_or_B;
      int order;
      int64_t const * lens;
      int * divisor;
      int * virt_dim;
      int * phys_rank;

      void run();
      void print();
      int64_t mem_fp();
      tspsum * clone();
      void set_nnz_blk_A(int64_t const * nnbA){
        tspsum::set_nnz_blk_A(nnbA);
        rec_tsum->set_nnz_blk_A(nnbA);
      }

      tspsum_pin_keys(tspsum * other);
      ~tspsum_pin_keys();
      tspsum_pin_keys(summation const * s, bool A_or_B);

  };

}

#endif
