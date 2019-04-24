#ifndef __INT_SPR_SEQ_SUM_H__
#define __INT_SPR_SEQ_SUM_H__

#include "summation.h"

namespace CTF_int {
  /**
   * \brief performs summation between two sparse tensors
   *   assumes A contains key value pairs sorted by key,
   *   with index permutation preapplied and with no repeated indices
   * \param[in] alpha scaling factor of A
   * \param[in] A data of right operand
   * \param[in] size_A number of nonzero entries in right operand
   * \param[in] sr_A algebraic structure of right operand
   * \param[in] beta scaling factor of left operand
   * \param[in,out] B data of left operand
   * \param[in] sr_B algebraic structure of left operand and output
   * \param[in] order_B order of tensor B
   * \param[in] edge_len_B dimensions of tensor B
   * \param[in] sym_B symmetry relations of tensor B
   * \param[in] func function (or NULL) to apply to right operand
   */
  void spA_dnB_seq_sum(char const *            alpha,
                       char const *            A,
                       int64_t                 size_A,
                       algstrct const *        sr_A,
                       char const *            beta,
                       char *                  B,
                       algstrct const *        sr_B,
                       int                     order_B,
                       int64_t const *         edge_len_B,
                       int const *             sym_B,
                       univar_function const * func);


  /**
   * \brief performs summation between two sparse tensors
   *   assumes B contain key value pairs sorted by key,
   *   with index permutation preapplied and with no repeated indices
   * \param[in] alpha scaling factor of A
   * \param[in] A data of right operand
   * \param[in] sr_A algebraic structure of right operand
   * \param[in] order_A order of tensor A
   * \param[in] edge_len_A dimensions of tensor A
   * \param[in] sym_A symmetry relations of tensor A
   * \param[in] beta scaling factor of left operand
   * \param[in] B data of left operand
   * \param[in] size_B number of nonzero entries in left operand
   * \param[in,out] new_B new data of output
   * \param[in,out] new_size_B number of nonzero entries in output
   * \param[in] sr_B algebraic structure of left operand and output
   * \param[in] func function (or NULL) to apply to right operand
   */
  void dnA_spB_seq_sum(char const *            alpha,
                       char const *            A,
                       algstrct const *        sr_A,
                       int                     order_A,
                       int64_t const *         edge_len_A,
                       int const *             sym_A,
                       char const *            beta,
                       char const *            B,
                       int64_t                 size_B,
                       char *&                 new_B,
                       int64_t &               new_size_B,
                       algstrct const *        sr_B,
                       univar_function const * func);

  /**
   * \brief performs summation between two sparse tensors
   *   assumes A and B contain key value pairs sorted by key,
   *   with index permutation preapplied and with no repeated indices
   * \param[in] alpha scaling factor of A
   * \param[in] A data of right operand
   * \param[in] size_A number of nonzero entries in right operand
   * \param[in] sr_A algebraic structure of right operand
   * \param[in] beta scaling factor of left operand
   * \param[in] B data of left operand
   * \param[in] size_B number of nonzero entries in left operand
   * \param[in,out] new_B new data of output
   * \param[in,out] new_size_B number of nonzero entries in output
   * \param[in] sr_B algebraic structure of left operand and output
   * \param[in] func function (or NULL) to apply to right operand
   * \param[in] map_pfx how many times each element of A should be replicated
   */
  void spA_spB_seq_sum(char const *            alpha,
                       char const *            A,
                       int64_t                 size_A,
                       algstrct const *        sr_A,
                       char const *            beta,
                       char *                  B,
                       int64_t                 size_B,
                       char *&                 new_B,
                       int64_t &               new_size_B,
                       algstrct const *        sr_B,
                       univar_function const * func,
                       int64_t                 map_pfx);

}
#endif
