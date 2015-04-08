#ifndef __SYM_SEQ_CTR_H__
#define __SYM_SEQ_CTR_H__

#include "../tensor/algstrct.h"
#include "../interface/functions.h"

namespace CTF_int {
  
  /**
   * \brief performs symmetric contraction with reference (unblocked) kernel
   */
  int sym_seq_ctr_ref(char const *     alpha,
                      char const *     A,
                      algstrct const * sr_A,
                      int              order_A,
                      int const *      edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int const *      edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int const *      edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C);

  /**
   * \brief performs symmetric contraction with custom elementwise function
   */
  int sym_seq_ctr_cust(char const *     alpha,
                       char const *     A,
                       algstrct const * sr_A,
                       int              order_A,
                       int const *      edge_len_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       char const *     B,
                       algstrct const * sr_B,
                       int              order_B,
                       int const *      edge_len_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       char const *     beta,
                       char *           C,
                       algstrct const * sr_C,
                       int              order_C,
                       int const *      edge_len_C,
                       int const *      sym_C,
                       int const *      idx_map_C,
                       bivar_function * func);


  /**
   * \brief performs symmetric contraction with blocked gemm
   */
  int sym_seq_ctr_inr(char const *     alpha,
                      char const *     A,
                      algstrct const * sr_A,
                      int              order_A,
                      int const *      edge_len_A,
                      int const *      sym_A,
                      int const *      idx_map_A,
                      char const *     B,
                      algstrct const * sr_B,
                      int              order_B,
                      int const *      edge_len_B,
                      int const *      sym_B,
                      int const *      idx_map_B,
                      char const *     beta,
                      char *           C,
                      algstrct const * sr_C,
                      int              order_C,
                      int const *      edge_len_C,
                      int const *      sym_C,
                      int const *      idx_map_C,
                      iparam const *   prm);
}
#endif
