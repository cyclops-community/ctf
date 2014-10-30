/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SUM_TSR_H__
#define __SUM_TSR_H__

#include "../tensor/untyped_semiring.h"

namespace CTF_int {

  /**
   * \brief untyped internal class for doubly-typed univariate function
   */
  class univar_function {
    public:
      /**
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to type by extending class
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      virtual void apply_f(char const * a, char * b) { assert(0); }
  };


  class tsum {
    public:
      char * A; 
      semiring sr_A;
      char const * alpha;
      char * B; 
      semiring sr_B;
      char const * beta;
      void * buffer;

      virtual void run() {};
      virtual int64_t mem_fp() { return 0; };
      virtual tsum * clone() { return NULL; };
      
      virtual ~tsum(){ if (buffer != NULL) CTF_free(buffer); }
      tsum(tsum * other);
      tsum(){ buffer = NULL; }
  };

  class tsum_virt : public tsum {
    public: 
      /* Class to be called on sub-blocks */
      tsum * rec_tsum;

      int num_dim;
      int * virt_dim;
      int order_A;
      int64_t blk_sz_A;
      int const * idx_map_A;
      int order_B;
      int64_t blk_sz_B;
      int const * idx_map_B;
      
      void run();
      int64_t mem_fp();
      tsum * clone();
      
      tsum_virt(tsum * other);
      ~tsum_virt();
      tsum_virt(){}
  };

  class tsum_replicate : public tsum {
    public: 
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */

      CommData *   cdt_A;
      CommData *   cdt_B;
      /* Class to be called on sub-blocks */
      tsum * rec_tsum;
      
      void run();
      int64_t mem_fp();
      tsum * clone();

      tsum_replicate(tsum * other);
      ~tsum_replicate();
      tsum_replicate(){}
  };

  class seq_tsr_sum : public tsum {
    public:
      int order_A;
      int * edge_len_A;
      int const * idx_map_A;
      int * sym_A;
      int order_B;
      int * edge_len_B;
      int const * idx_map_B;
      int * sym_B;
      //fseq_tsr_sum func_ptr;

      int is_inner;
      int inr_stride;
      
      int is_custom;
      univar_function func; //fseq_elm_sum custom_params;
      
      /**
       * \brief wraps user sequential function signature
       */
      void run();
      void print();
      int64_t mem_fp();
      tsum * clone();

      /**
       * \brief copies sum object
       * \param[in] other object to copy
       */
      seq_tsr_sum(tsum * other);
      ~seq_tsr_sum(){ CTF_free(edge_len_A), CTF_free(edge_len_B), 
                      CTF_free(sym_A), CTF_free(sym_B); };
      seq_tsr_sum(){}

  };

  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] order_B number of dimensions of B
   * \param[in] idx_B index map of B
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 2*order_tot index array
   */
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int const          order_B,
               int const *        idx_B,
               int *              order_tot,
               int **             idx_arr);

}

#endif // __SUM_TSR_H__
