/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __PHASE_RESHUFFLE_H__
#define __PHASE_RESHUFFLE_H__

#include "../tensor/algstrct.h"
#include "../mapping/distribution.h"
#include "redist.h"

namespace CTF_int {
  void phase_reshuffle(int const *          sym,
                       int const *          edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char **              ptr_tsr_data,
                       char **              ptr_tsr_new_data,
                       algstrct const *     sr,
                       CommData             ord_glb_comm);

  /**
   * \brief computes the cardinality of the set of elements of a tensor of order idim+1 that are owned by processor index gidx_off in a distribution with dimensions sphase
   */
  template <int idim>
  int64_t calc_cnt(int const * sym,
                   int const * rep_phase,
                   int const * sphase,
                   int const * gidx_off,
                   int const * edge_len,
                   int const * loc_edge_len);

  /**
   * \brief computes the cardinality of the sets of elements of a tensor of order idim+1 for different values of the idim'th tensor dimension
   */
  template <int idim>
  int64_t * calc_sy_pfx(int const * sym,
                        int const * rep_phase,
                        int const * sphase,
                        int const * gidx_off,
                        int const * edge_len,
                        int const * loc_edge_len);

  template <int idim>
  void calc_drv_cnts(int         ndim,
                     int const * sym,
                     int64_t *   counts,
                     int const * rep_phase,
                     int const * rep_phase_lda,
                     int const * sphase,
                     int const * phys_phase,
                     int       * gidx_off,
                     int const * edge_len,
                     int const * loc_edge_len);

  template <int idim>
  void calc_cnt_from_rep_cnt(int const *     rep_phase,
                             int const *     rep_phase_lda,
                             int const *     rank,
                             int const *     new_pe_lda,
                             int const *     old_phys_phase,
                             int const *     new_phys_phase,
                             int64_t const * rep_counts,
                             int64_t *       counts,
                             int             coff=0,
                             int             roff=0);

  void calc_drv_displs(int const *          sym,
                       int const *          edge_len,
                       int const *          loc_edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int64_t *            counts,
                       CommData             ord_glb_comm,
                       int                  idx_lyr);

  template <int idim>
  void redist_bucket(int const *          sym,
                     int const *          phys_phase,
                     int const *          perank,
                     int const *          edge_len,
                     int const *          virt_edge_len,
                     int const *          virt_dim,
                     int const *          virt_lda,
                     int64_t              virt_nelem,
                     int * const *        bucket_offset,
                     int64_t * const *    data_offset,
                     int                  rep_phase0,
                     bool                 data_to_buckets,
                     char * __restrict__  data,
                     char ** __restrict__ buckets,
                     int64_t *            counts,
                     algstrct const *     sr,
                     int64_t              data_off=0,
                     int                  bucket_off=0,
                     int                  prev_idx=0);
}
#endif
