/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/iter_tsr.h"
#include <limits.h>
#include "sp_seq_ctr.h"
#include "../shared/offload.h"
#include "../shared/util.h"

namespace CTF_int{
  template<int idim>
  void spA_dnB_dnC_seq_ctr(char const *            alpha,
                           ConstPairIterator &     A,
                           int64_t &               size_A,
                           algstrct const *        sr_A,
                           char *&                 B,
                           algstrct const *        sr_B,
                           int                     order_B,
                           int64_t                 idx_B,
                           int const *             edge_len_B,
                           int64_t const *         lda_B,
                           int const *             sym_B,
                           char const *            beta,
                           int                     order_C,
                           int64_t                 idx_C,
                           int const *             edge_len_C,
                           int64_t const *         lda_C,
                           int const *             sym_C,
                           univar_function const * func){

  }
} 
