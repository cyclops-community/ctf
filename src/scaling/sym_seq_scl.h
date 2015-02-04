#ifndef __SYM_SEQ_SCL_H__
#define __SYM_SEQ_SCL_H__

#include "../tensor/algstrct.h"

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
      virtual void apply_f(char * a) { assert(0); }
  };

  /**
   * \brief performs symmetric scaling using custom func
   */
  int sym_seq_scl_cust(char *               A,
                       algstrct const &     sr_A,
                       int const            order_A,
                       int const *          edge_len_A,
                       int const *          _lda_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       endomorphism        func);
  /**
   * \brief performs symmetric scaling using algstrct sr_A
   */
  int sym_seq_scl_ref(char const * alpha,
                      char *       A,
                      algstrct const &  sr_A,
                      int          order_A,
                      int const *  edge_len_A,
                      int const *  _lda_A,
                      int const *  sym_A,
                      int const *  idx_map_A);
  /**
   * \brief invert index map
   * \param[in] ndim_A number of dimensions of A
   * \param[in] idx_A index map of A
   * \param[in] edge_map_B mapping of each dimension of A
   * \param[out] ndim_tot number of total dimensions
   * \param[out] idx_arr 2*ndim_tot index array
   */
  void inv_idx(int const          order_A,
               int const *        idx_A,
               int *              order_tot,
               int **             idx_arr);
}
#endif
