#ifndef __INT_SYM_SEQ_SUM_H__
#define __INT_SYM_SEQ_SUM_H__

#include "summation.h"

namespace CTF_int {
  /**
   * \brief performs symmetric contraction with unblocked reference kernel
   */
  int sym_seq_sum_ref( char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int64_t const *  edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     beta,
                       char *           B,
                       algstrct const * sr_B,
                       int              order_B,
                       int64_t const *  edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B);

  /**
   * \brief performs symmetric summation with custom elementwise function
   */
  int sym_seq_sum_cust(char const *            alpha,
                       char const *            A,
                       algstrct const *        sr_A,
                       int                     order_A,
                       int64_t const *         edge_len_A,
                       int const *             sym_A,
                       int const *             idx_map_A,
                       char const *            beta,
                       char *                  B,
                       algstrct const *        sr_B,
                       int                     order_B,
                       int64_t const *         edge_len_B,
                       int const *             sym_B,
                       int const *             idx_map_B,
                       univar_function const * func);

  /**
   * \brief performs symmetric summation with blocked daxpy
   */
  int sym_seq_sum_inr( char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int64_t const *  edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     beta,
                       char *           B,
                       algstrct const * sr_B,
                       int              order_B,
                       int64_t const *  edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       int              inr_stride);
}

#endif
