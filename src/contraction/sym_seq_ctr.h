#ifndef __SYM_SEQ_CTR_H__
#define __SYM_SEQ_CTR_H__

#include "sym_seq_ctr.h"

namespace CTF_int {
  
  /**
   * \brief performs symmetric contraction with reference (unblocked) kernel
   */
  int sym_seq_ctr_ref(char const *       alpha,
                      char const *       A,
                      int                order_A,
                      int const *        edge_len_A,
                      int const *        _lda_A,
                      int const *        sym_A,
                      int const *        idx_map_A,
                      char const *       B,
                      int                order_B,
                      int const *        edge_len_B,
                      int const *        _lda_B,
                      int const *        sym_B,
                      int const *        idx_map_B,
                      char const *       beta,
                      char *             C,
                      semiring           sr_C,
                      int                order_C,
                      int const *        edge_len_C,
                      int const *        _lda_C,
                      int const *        sym_C,
                      int const *        idx_map_C);

  /**
   * \brief performs symmetric contraction with custom elementwise function
   */
  int sym_seq_ctr_cust(char const *         A,
                       semiring             sr_A,
                       int                  order_A,
                       int const *          edge_len_A,
                       int const *          _lda_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       char const *         B,
                       semiring             sr_B,
                       int                  order_B,
                       int const *          edge_len_B,
                       int const *          _lda_B,
                       int const *          sym_B,
                       int const *          idx_map_B,
                       char *               C,
                       semiring             sr_C,
                       int                  order_C,
                       int const *          edge_len_C,
                       int const *          _lda_C,
                       int const *          sym_C,
                       int const *          idx_map_C,
                       bivar_function       func);


  /**
   * \brief performs symmetric contraction with blocked gemm
   */
  int sym_seq_ctr_inr(char const *       alpha,
                      char const *       A,
                      semiring           sr_A,
                      int                order_A,
                      int const *        edge_len_A,
                      int const *        _lda_A,
                      int const *        sym_A,
                      int const *        idx_map_A,
                      char const *       B,
                      semiring           sr_B,
                      int                order_B,
                      int const *        edge_len_B,
                      int const *        _lda_B,
                      int const *        sym_B,
                      int const *        idx_map_B,
                      char const *       beta,
                      char *             C,
                      semiring           sr_C,
                      int                order_C,
                      int const *        edge_len_C,
                      int const *        _lda_C,
                      int const *        sym_C,
                      int const *        idx_map_C,
                      iparam const *     prm);
}
#endif
