#ifndef __SYM_SEQ_SCL_H__
#define __SYM_SEQ_SCL_H__

#include "../tensor/algstrct.h"
#include "../interface/term.h"

namespace CTF_int {

  /**
   * \brief untyped internal class for singly-typed single variable function (Endomorphism)
   */
  class endomorphism {
    public:
  /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to type by extending class
       *                  return result of applying f on value at a
       */
      virtual void apply_f(char * a) const { assert(0); }

      /** 
       * \brief apply f to A
       * \param[in] A operand tensor with pre-defined indices 
      */
      void operator()(Term const & A) const;

      virtual ~endomorphism(){}
  };

  /**
   * \brief performs symmetric scaling using custom func
   */
  int sym_seq_scl_cust(char const *         alpha,
                       char *               A,
                       algstrct const *     sr_A,
                       int const            order_A,
                       int64_t const *      edge_len_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       endomorphism const * func);
  /**
   * \brief performs symmetric scaling using algstrct const * sr_A
   */
  int sym_seq_scl_ref(char const *     alpha,
                      char *           A,
                      algstrct const * sr_A,
                      int              order_A,
                      int64_t const *  edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A);
  /**
   * \brief invert index map
   * \param[in] order_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[out] order_tot number of total dimensions
   * \param[out] idx_arr 2*ndim_tot index array
   */
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int *              order_tot,
               int **             idx_arr);
}
#endif
