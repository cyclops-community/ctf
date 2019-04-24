/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SUM_TSR_H__
#define __SUM_TSR_H__

#include "../tensor/algstrct.h"
#include "../interface/fun_term.h"

namespace CTF_int {
  class summation;
  /**
   * \brief untyped internal class for doubly-typed univariate function
   */
  class univar_function {
    public:
      void (*f)(char const *, char *);

      bool is_distributive = false;

      /**
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to type by extending class
       * \param[in,out] b result &f(*a) of applying f on value of (different type) on a
       */
      virtual void apply_f(char const * a, char * b)const { f(a,b); }
      
      /**
       * \brief compute b = b+f(a)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] b result &f(*a) of applying f on value of (different type) on a
       * \param[in] sr_B algebraic structure for b, needed to do add
       */
      virtual void acc_f(char const * a, char * b, CTF_int::algstrct const * sr_B) const { 

        char tb[sr_B->el_size];
        f(a,tb);
        sr_B->add(b, tb, b);
      }

      virtual bool is_transform() const { return false; };

      univar_function(void (*f_)(char const *, char *)) { f=f_; }
      univar_function(void (*f_)(char const *, char *), bool is_dist) { f=f_; is_distributive=is_dist; }
      univar_function() { }
      univar_function(univar_function const & other) { f = other.f; is_distributive=other.is_distributive; }

      /** 
       * \brief evaluate B=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * \return Unifun_Term that evaluates f(A)
      */
      Unifun_Term operator()(Term const & A) const;

      /** 
       * \brief evaluate B+=f(A)  or f(A,B) if transform
       * \param[in] A operand tensor with pre-defined indices 
       * \param[in] B output tensor with pre-defined indices 
      */
      void operator()(Term const & A, Term const & B) const;

      virtual ~univar_function(){}

      virtual bool is_accumulator() const { return false; }
  };


  class tsum {
    public:
      char *           A;
      algstrct const * sr_A;
      char const *     alpha;
      char *           B;
      algstrct const * sr_B;
      char const *     beta;
      void *           buffer;

      virtual void run() {};
      virtual void print() {};
//      virtual int64_t calc_new_nnz_B() { return nnz_B; } //if sparse
      /**
       * \brief returns the number of bytes of buffer space needed
       * \return bytes needed
       */
      virtual int64_t mem_fp() { return 0; };
      virtual tsum * clone() { return NULL; };
      virtual ~tsum();
      tsum(tsum * other);
      tsum(summation const * s);
  };

  class tsum_virt : public tsum {
    public: 
      /* Class to be called on sub-blocks */
      tsum * rec_tsum;

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
      tsum * clone();

      /**
       * \brief iterates over the dense virtualization block grid and contracts
       */
      tsum_virt(tsum * other);
      ~tsum_virt();
      tsum_virt(summation const * s);
  };


  /**
   * \brief performs replication along a dimension, generates 2.5D algs
   */
  class tsum_replicate : public tsum {
    public: 
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */

      CommData ** cdt_A;
      CommData ** cdt_B;
      /* Class to be called on sub-blocks */
      tsum * rec_tsum;
      
      void run();
      void print();
      int64_t mem_fp();
      tsum * clone();

      tsum_replicate(tsum * other);
      ~tsum_replicate();
      tsum_replicate(summation const * s,
                     int const *       phys_mapped,
                     int64_t           blk_sz_A,
                     int64_t           blk_sz_B);
  };

  class seq_tsr_sum : public tsum {
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
      tsum * clone();

      /**
       * \brief copies sum object
       * \param[in] other object to copy
       */
      seq_tsr_sum(tsum * other);
      ~seq_tsr_sum(){ CTF_int::cdealloc(edge_len_A), CTF_int::cdealloc(edge_len_B), 
                      CTF_int::cdealloc(sym_A), CTF_int::cdealloc(sym_B); };
      seq_tsr_sum(summation const * s);

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
  void inv_idx(int         order_A,
               int const * idx_A,
               int         order_B,
               int const * idx_B,
               int *       order_tot,
               int **      idx_arr);

}

#endif // __SUM_TSR_H__
