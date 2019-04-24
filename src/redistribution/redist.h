/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_REDIST_H__
#define __INT_REDIST_H__

#include "../tensor/algstrct.h"
#include "../mapping/distribution.h"

namespace CTF_int {
  /**
   * \brief Reshuffle elements using key-value pair read/write
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] tsr_data starting data buffer
   * \param[out] tsr_cyclic_data target data buffer
   * \param[in] sr algstrct defining data
   * \param[in] ord_glb_comm communicator on which to redistribute
   */
  void padded_reshuffle(int const *          sym,
                        distribution const & old_dist,
                        distribution const & new_dist,
                        char *               tsr_data,
                        char **              tsr_cyclic_data,
                        algstrct const *     sr,
                        CommData             ord_glb_comm);

  /**
   * \brief computes offsets for redistribution targets along each edge length
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] len unpadded edge lengths 
   * \param[in] old_phys_edge_len total edge lengths of old local tensor chunk
   * \param[in] old_virt_lda prefix sum of old_dist.virt_phase
   * \param[in] old_offsets old offsets of each tensor edge (corner 1 of slice)
   * \param[in] old_permutation permutation array for each edge length (no perm if NULL)
   * \param[in] new_phys_edge_len total edge lengths of new local tensor chunk
   * \param[in] new_virt_lda prefix sum of new_dist.virt_phase
   * \param[in] forward 1 for sending 0 for receiving
   * \param[in] old_virt_np number of blocks per processor in old_dist
   * \param[in] new_virt_np number of blocks per processor in new_dist
   * \param[in] old_virt_edge_len edge lengths of each block in old_dist
   * \return 2D array with dims [order][old_phys_edge_len[i]] with bucket offsets for each edge length 
   */
  int ** compute_bucket_offsets(distribution const & old_dist,
                                distribution const & new_dist,
                                int64_t const *      len,
                                int64_t const *      old_phys_edge_len,
                                int const *          old_virt_lda,
                                int64_t const *      old_offsets,
                                int * const *        old_permutation,
                                int64_t const *      new_phys_edge_len,
                                int const *          new_virt_lda,
                                int                  forward,
                                int                  old_virt_np,
                                int                  new_virt_np,
                                int64_t const *      old_virt_edge_len);

  /**
   * \brief assigns keys to an array of values
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] new_nvirt new total virtualization factor
   * \param[in] np number of processors
   * \param[in] old_virt_edge_len old edge lengths of blocks
   * \param[in] new_virt_lda prefix sum of new_dist.virt_phase
   * \param[out] send_counts outgoing counts of pairs by pe
   * \param[out] recv_counts incoming counts of pairs by pe
   * \param[out] send_displs outgoing displs of pairs by pe
   * \param[out] recv_displs incoming displs of pairs by pe
   * \param[in] ord_glb_comm the global communicator
   * \param[in] idx_lyr starting processor layer (2.5D)
   * \param[in] bucket_offset offsets for target index for each dimension
   */
  void calc_cnt_displs(int const *          sym,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int                  new_nvirt,
                       int                  np,
                       int64_t const *      old_virt_edge_len,
                       int const *          new_virt_lda,
                       int64_t *            send_counts,
                       int64_t *            recv_counts,
                       int64_t *            send_displs,
                       int64_t *            recv_displs,
                       CommData             ord_glb_comm,
                       int                  idx_lyr,
                       int * const *        bucket_offset);


  /**
   * \brief estimates execution time, given this processor sends a receives tot_sz across np procs
   * \param[in] tot_sz amount of data sent/recved
   * \param[in] nv0 starting number of blocks
   * \param[in] nv1 ending number of blocks
   */
  double blres_est_time(int64_t tot_sz, int nv0, int nv1);

  /**
   * \brief Reshuffle elements by block given the global phases stay the same
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] tsr_data starting data buffer
   * \param[out] tsr_cyclic_data target data buffer
   * \param[in] sr algstrct defining data
   * \param[in] glb_comm communicator on which to redistribute
   */
  void block_reshuffle(distribution const & old_dist,
                       distribution const & new_dist,
                       char *               tsr_data,
                       char *&              tsr_cyclic_data,
                       algstrct const *     sr,
                       CommData             glb_comm);

  /**
   * \brief determines if tensor can be permuted by block
   * \param[in] order dimension of tensor
   * \param[in] old_phase old cyclic phases in each dimension
   * \param[in] map new mapping for each edge length
   * \return 1 if block reshuffle allowed, 0 if not
   */
  int can_block_reshuffle(int             order,
                          int const *     old_phase,
                          mapping const * map);
}

#endif
