#ifndef __SP_SEQ_CTR_H__
#define __SP_SEQ_CTR_H__

#include "contraction.h"
namespace CTF_int{
  void spA_dnB_dnC_seq_ctr(char const *            alpha,
                           char  const *           A,
                           int64_t                 size_A,
                           algstrct const *        sr_A,
                           int                     order_A,
                           int64_t const *         edge_len_A,
                           int const *             sym_A,
                           int const *             idx_map_A,
                           char const *            B,
                           algstrct const *        sr_B,
                           int                     order_B,
                           int64_t const *         edge_len_B,
                           int const *             sym_B,
                           int const *             idx_map_B,
                           char const *            beta,
                           char *                  C,
                           algstrct const *        sr_C,
                           int                     order_C,
                           int64_t const *         edge_len_C,
                           int const *             sym_C,
                           int const *             idx_map_C,
                           bivar_function const *  func);
}
#endif
