/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CYCLIC_RESHUFFLE_H__
#define __CYCLIC_RESHUFFLE_H__

#include "../tensor/algstrct.h"
#include "../mapping/distribution.h"
#include "redist.h"

namespace CTF_int {
 
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
   * \param[in] alpha scaling factor for received data
   * \param[in] beta scaling factor for previous data
   * \param[in] sr algstrct defining elements and ops
   */
  void pad_cyclic_pup_virt_buff(int const *          sym,
                                distribution const & old_dist,
                                distribution const & new_dist,
                                int64_t const *      len,
                                int const *          old_phys_dim,
                                int64_t const *      old_phys_edge_len,
                                int64_t const *      old_virt_edge_len,
                                int64_t              old_virt_nelem,
                                int64_t const *      old_offsets,
                                int * const *        old_permutation,
                                int                  total_np,
                                int const *          new_phys_dim,
                                int64_t const *      new_phys_edge_len,
                                int64_t const *      new_virt_edge_len,
                                int64_t              new_virt_nelem,
                                char *               old_data,
                                char **              new_data,
                                int                  forward,
                                int64_t * const *    bucket_offset,
                                char const *         alpha,
                                char const *         beta,
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
                        int64_t const *      old_offsets,
                        int * const *        old_permutation,
                        distribution const & new_dist,
                        int64_t const *      new_offsets,
                        int * const *        new_permutation,
                        char **              tsr_data,
                        char **              tsr_cyclic_data,
                        algstrct const *     sr,
                        CommData             ord_glb_comm,
                        bool                 reuse_buffers,
                        char const *         alpha,
                        char const *         beta);
}
#endif
