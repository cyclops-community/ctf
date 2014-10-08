#ifndef __SYM_SEQ_SCL_H__
#define __SYM_SEQ_SCL_H__

namespace CTF_int {
  /**
   * \brief performs symmetric contraction
   */
  int sym_seq_scl_cust(char *               A,
                       semiring             sr_A,
                       int const            order_A,
                       int const *          edge_len_A,
                       int const *          _lda_A,
                       int const *          sym_A,
                       int const *          idx_map_A,
                       endomorphism        func);
  /**
   * \brief performs symmetric contraction
   */
  int sym_seq_scl_ref(char const * alpha,
                      char *       A,
                      semiring     sr_A,
                      int          order_A,
                      int const *  edge_len_A,
                      int const *  _lda_A,
                      int const *  sym_A,
                      int const *  idx_map_A);

}
#endif
