/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

#include "../tensor/untyped_semiring.h"
#include "ctr_offload.h"

namespace CTF_int{

  /**
   * \brief untyped internal class for triply-typed bivariate function
   */
  class bivar_function {
    public:
      /**
       * \brief apply function f to values stored at a and b
       * \param[in] a pointer to first operand that will be cast to type by extending class
       * \param[in] b pointer to second operand that will be cast to type by extending class
       * \param[in,out] result: c=&f(*a,*b) 
       */
      virtual void apply_f(char const * a, char const * b, char * c) { assert(0); }
  };


  /**
   * \addtogroup nest_dist Nested distributed contraction and summation routines
   * @{
   */

  class ctr {
    public: 
      char * A; /* m by k */
      char * B; /* k by n */
      char * C; /* m by n */
      semiring sr_A;
      semiring sr_B;
      semiring sr_C;
      char const * beta;
      int num_lyr; /* number of copies of this matrix being computed on */
      int idx_lyr; /* the index of this copy */

      virtual void run() { printf("SHOULD NOTR\n"); };
      virtual void print() { };
      virtual int64_t mem_fp() { return 0; };
      virtual int64_t mem_rec() { return mem_fp(); };
      virtual double est_time_fp(int nlyr) { return 0; };
      virtual double est_time_rec(int nlyr) { return est_time_fp(nlyr); };
      virtual ctr * clone() { return NULL; };
      
      /**
       * \brief deallocates generic ctr object
       */
      virtual ~ctr();
    
      /**
       * \brief copies generic ctr object
       */
      ctr(ctr * other);
      ctr(){ idx_lyr = 0; num_lyr = 1; }
  };

  class ctr_replicate : public ctr {
    public: 
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */
      int ncdt_C; /* number of processor dimensions to replicate C along */
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int64_t size_C; /* size of C blocks */

      CommData *   cdt_A;
      CommData *   cdt_B;
      CommData *   cdt_C;
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      
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
      ctr * clone();

      ctr_replicate(ctr * other);
      ~ctr_replicate();
      ctr_replicate(){}
  };

  /* Assume LDA equal to dim */
/*  class ctr_dgemm : public ctr {
    public: 
      char transp_A;
      char transp_B;
      char const * alpha;
      int n;
      int m;
      int k;
      
      void print() {};
      void run();
      int64_t mem_fp();
      double est_time_fp(int nlyr);
      double est_time_rec(int nlyr);
      ctr * clone();

      ctr_dgemm(ctr * other);
      ~ctr_dgemm();
      ctr_dgemm(){}
  };*/

  /**
   * \brief performs replication along a dimension, generates 2.5D algs
   */
  class ctr_lyr : public ctr {
    public: 
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      int k;
      CommData cdt;
      int64_t sz_C;
      
      void print() {};
      void run();
      int64_t mem_fp();
      int64_t mem_rec();
      ctr * clone();

      ctr_lyr(ctr * other);
      ~ctr_lyr();
      ctr_lyr(){}
  };
  struct iparam {
    int n;
    int m;
    int k;
    int64_t sz_C;
    char tA;
    char tB;
  };

  class seq_tsr_ctr : public ctr {
    public:
      char const * alpha;
      int order_A;
      int * edge_len_A;
      int const * idx_map_A;
      int * sym_A;
      int order_B;
      int * edge_len_B;
      int const * idx_map_B;
      int * sym_B;
      int order_C;
      int * edge_len_C;
      int const * idx_map_C;
      int * sym_C;
      //fseq_tsr_ctr func_ptr;

      int is_inner;
      iparam inner_params;
      
      int is_custom;
      bivar_function func; // custom_params;
      
      /**
       * \brief wraps user sequential function signature
       */
      void run();
      void print();
      int64_t mem_fp();
      double est_time_rec(int nlyr);
      double est_time_fp(int nlyr);
      ctr * clone();

      /**
       * \brief clones ctr object
       * \param[in] other object to clone
       */
      seq_tsr_ctr(ctr * other);
      ~seq_tsr_ctr(){ CTF_int::cfree(edge_len_A), CTF_int::cfree(edge_len_B), CTF_int::cfree(edge_len_C), 
                      CTF_int::cfree(sym_A), CTF_int::cfree(sym_B), CTF_int::cfree(sym_C); }
      seq_tsr_ctr(){}
  };

  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] order_B number of dimensions of B
   * \param[in] idx_B index map of B
   * \param[in] order_C number of dimensions of C
   * \param[in] idx_C index map of C
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 3*order_tot index array
   */
  void inv_idx(int                order_A,
               int const *        idx_A,
               int                order_B,
               int const *        idx_B,
               int                order_C,
               int const *        idx_C,
               int *              order_tot,
               int **             idx_arr);

  /**
   * @}
   */


}
#endif // __CTR_COMM_H__
