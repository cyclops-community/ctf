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
                                int const *          len,
                                int const *          old_phys_edge_len,
                                int const *          old_virt_lda,
                                int const *          old_offsets,
                                int * const *        old_permutation,
                                int const *          new_phys_edge_len,
                                int const *          new_virt_lda,
                                int                  forward,
                                int                  old_virt_np,
                                int                  new_virt_np,
                                int const *          old_virt_edge_len);

  /**
   * \brief assigns keys to an array of values
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] nbuf number of global virtual buckets
   * \param[in] new_nvirt new total virtualization factor
   * \param[in] np number of processors
   * \param[in] new_virt_lda prefix sum of new_dist.virt_phase
   * \param[in] buf_lda prefix sum of new_phase
   * \param[in] padding padding of tensor
   * \param[out] send_counts outgoing counts of pairs by pe
   * \param[out] recv_counts incoming counts of pairs by pe
   * \param[out] send_displs outgoing displs of pairs by pe
   * \param[out] recv_displs incoming displs of pairs by pe
   * \param[out] svirt_displs outgoing displs of pairs by virt bucket
   * \param[out] rvirt_displs incoming displs of pairs by virt bucket
   * \param[in] ord_glb_comm the global communicator
   * \param[in] idx_lyr starting processor layer (2.5D)
   * \param[in] bucket_offset offsets for target index for each dimension
   */
  void calc_cnt_displs(int const *          sym,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int                  nbuf,
                       int                  new_nvirt,
                       int                  np,
                       int const *          old_virt_edge_len,
                       int const *          new_virt_lda,
                       int const *          buf_lda,
                       int64_t *            send_counts,
                       int64_t *            recv_counts,
                       int64_t *            send_displs,
                       int64_t *            recv_displs,
                       int64_t *            svirt_displs,
                       int64_t *            rvirt_displs,
                       CommData             ord_glb_comm,
                       int                  idx_lyr,
                       int * const *        bucket_offset);
 
  /** 
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] len non-padded edge lengths of tensor
   * \param[in] old_phys_dim edge lengths of the old processor grid
   * \param[in] old_phys_edge_len the old tensor processor block lengths
   * \param[in] old_virt_edge_len the old tensor block lengths
   * \param[in] old_virt_nelem the old number of elements per block
   * \param[in] old_offsets old offsets of each tensor edge (corner 1 of slice)
   * \param[in] old_permutation permutation array for each edge length (no perm if NULL)
   * \param[in] total_np the total number of processors
   * \param[in] new_phys_dim edge lengths of the new processor grid
   * \param[in] new_phys_edge_len the new tensor processor block lengths
   * \param[in] new_virt_edge_len the new tensor block lengths
   * \param[in] new_virt_nelem the new number of elements per block
   * \param[in,out] old_data the previous set of values stored locally
   * \param[in,out] new_data buffers to fill with data to send to each process and virtual bucket
   * \param[in] forward is 0 on the receiving side and reverses the role of all the previous parameters
   * \param[in] bucket_offset offsets for target index for each dimension
   * \param[in] sr algstrct defining elements and ops
   */
  void pad_cyclic_pup_virt_buff(int const *          sym,
                                distribution const & old_dist,
                                distribution const & new_dist,
                                int const *          len,
                                int const *          old_phys_dim,
                                int const *          old_phys_edge_len,
                                int const *          old_virt_edge_len,
                                int64_t              old_virt_nelem,
                                int const *          old_offsets,
                                int * const *        old_permutation,
                                int                  total_np,
                                int const *          new_phys_dim,
                                int const *          new_phys_edge_len,
                                int const *          new_virt_edge_len,
                                int64_t              new_virt_nelem,
                                char *               old_data,
                                char **              new_data,
                                int                  forward,
                                int * const *        bucket_offset,
                                char const *         alpha,
                                char const *         beta,
                                algstrct const *     sr);

  /**
   * \brief reorder local buffer so that elements are in ordered according to where they
   *        are in the global tensor (interleave virtual blocks)
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] virt_edge_len dimensions of each block
   * \param[in] virt_phase prefix sum of virtual blocks
   * \param[in] vbs size of virtual blocks
   * \param[in] if 1 then go to global layout, if 0 than from
   * \param[in] tsr_data_in starting data buffer
   * \param[out] tsr_data_out target data buffer
   * \param[in] sr algstrct defining data
   */
  void order_globally(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      int                  dir,
                      char const *         tsr_data_in,
                      char *               tsr_data_out,
                      algstrct const *     sr);

  /**
   * \brief Goes from any set of phases to any new set of phases 
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] old_offsets old offsets of each tensor edge (corner 1 of slice)
   * \param[in] old_permutation permutation array for each edge length (no perm if NULL)
   * \param[in] new_dist target data distrubtion
   * \param[in] new_offsets old offsets of each tensor edge (corner 1 of slice)
   * \param[in] new_permutation permutation array for each edge length (no perm if NULL)
   * \param[in] tsr_data starting data buffer
   * \param[out] tsr_cyclic_data target data buffer
   * \param[in] sr algstrct defining data
   * \param[in] ord_glb_comm communicator on which to redistribute
   * \param[in] reuse_buffers if 1: ptr_tsr_cyclic_data is allocated dynamically and ptr_tsr_data 
   *                                 is overwritten with intermediate data
   *                          if 0: ptr_tsr_cyclic_data is preallocated and can be scaled by beta,
   *                                 however, more memory is used for temp buffers
   * \param[in] alpha scaling tensor for new data
   * \param[in] beta scaling tensor for original data
   */
  void cyclic_reshuffle(int const *          sym,
                        distribution const & old_dist,
                        int const *          old_offsets,
                        int * const *        old_permutation,
                        distribution const & new_dist,
                        int const *          new_offsets,
                        int * const *        new_permutation,
                        char **              tsr_data,
                        char **              tsr_cyclic_data,
                        algstrct const *     sr,
                        CommData             ord_glb_comm,
                        bool                 reuse_buffers,
                        char const *         alpha,
                        char const *         beta);

  void glb_cyclic_reshuffle(int const *          sym,
                            distribution const & old_dist,
                            int const *          old_offsets,
                            int * const *        old_permutation,
                            distribution const & new_dist,
                            int const *          new_offsets,
                            int * const *        new_permutation,
                            char **              ptr_tsr_data,
                            char **              ptr_tsr_cyclic_data,
                            algstrct const *     sr,
                            CommData             ord_glb_comm,
                            bool                 reuse_buffers,
                            char const *         alpha,
                            char const *         beta);

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
