/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_REDIST_H__
#define __INT_REDIST_H__

#include "../tensor/int_tensor.h"

namespace CTF_int {

  int padded_reshuffle(int         tid,
                       int         order,
                       int         nval,
                       int const * old_edge_len,
                       int const * sym,
                       int const * old_phase,
                       int const * old_rank,
                       int         is_old_pad,
                       int const * old_padding,
                       int const * new_edge_len,
                       int const * new_phase,
                       int const * new_rank,
                       int const * new_pe_lda,
                       int         is_new_pad,
                       int const * new_padding,
                       int const * old_virt_dim,
                       int const * new_virt_dim,
                       char *      tsr_data,
                       char * *    tsr_cyclic_data,
                       CommData  ord_glb_comm);

  int cyclic_reshuffle(int         order,
                       int         nval,
                       int const * old_edge_len,
                       int const * sym,
                       int const * old_phase,
                       int const * old_rank,
                       int const * old_pe_lda,
                       int         is_old_pad,
                       int const * old_padding,
                       int const * new_edge_len,
                       int const * new_phase,
                       int const * new_rank,
                       int const * new_pe_lda,
                       int         is_new_pad,
                       int const * new_padding,
                       int const * old_virt_dim,
                       int const * new_virt_dim,
                       char **     tsr_data,
                       char **     tsr_cyclic_data,
                       CommData  ord_glb_comm,
                       int         was_cyclic = 0,
                       int         is_cyclic = 0);

  int remap_tensor(int const  tid,
                   tensor *tsr,
                   topology const * topo,
                   int64_t const old_size,
                   int const *  old_phase,
                   int const *  old_rank,
                   int const *  old_virt_dim,
                   int const *  old_pe_lda,
                   int const    was_cyclic,
                   int const *  old_padding,
                   int const *  old_edge_len,
                   CommData   global_comm);
                   /*int const *  old_offsets = NULL,
                   int * const * old_permutation = NULL,
                   int const *  new_offsets = NULL,
                   int * const * new_permutation = NULL);*/



}

#endif
