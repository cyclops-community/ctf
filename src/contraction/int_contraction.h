#ifndef __INT_CONTRACTION_H__
#define __INT_CONTRACTION_H__

#include "assert.h"

namespace CTF_int {

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
      char * alpha;
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
      ~seq_tsr_ctr(){ CTF_free(edge_len_A), CTF_free(edge_len_B), CTF_free(edge_len_C), 
                      CTF_free(sym_A), CTF_free(sym_B), CTF_free(sym_C); }
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
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int const          order_B,
               int const *        idx_B,
               int const          order_C,
               int const *        idx_C,
               int *              order_tot,
               int **             idx_arr);

}

#endif
