/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_REDIST_H__
#define __INT_REDIST_H__

#include "../tensor/untyped_semiring.h"

namespace CTF_int {
  /**
   * \brief Reshuffle elements using key-value pair read/write
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] tsr_data starting data buffer
   * \param[out] tsr_cyclic_data target data buffer
   * \param[in] sr semiring defining data
   * \param[in] ord_glb_comm communicator on which to redistribute
   */
  void padded_reshuffle(int const * sym,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char *      tsr_data,
                       char **     tsr_cyclic_data,
                       semiring    sr,
                       CommData  ord_glb_comm);

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
   * \param[in] sr semiring defining data
   * \param[in] ord_glb_comm communicator on which to redistribute
   * \param[in] reuse_buffers if 1: ptr_tsr_cyclic_data is allocated dynamically and ptr_tsr_data 
   *                                 is overwritten with intermediate data
   *                          if 0: ptr_tsr_cyclic_data is preallocated and can be scaled by beta,
   *                                 however, more memory is used for temp buffers
   * \param[in] alpha scaling tensor for new data
   * \param[in] beta scaling tensor for original data
   */
  void cyclic_reshuffle(int const * sym,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char **     tsr_data,
                       char **     tsr_cyclic_data,
                       semiring    sr,
                       CommData    ord_glb_comm,
                       bool        reuse_buffers,
                       char const *       alpha,
                       char const *       beta);
  /**
   * \brief Reshuffle elements by block given the global phases stay the same
   * \param[in] old_dist starting data distrubtion
   * \param[in] new_dist target data distrubtion
   * \param[in] tsr_data starting data buffer
   * \param[out] tsr_cyclic_data target data buffer
   * \param[in] sr semiring defining data
   * \param[in] glb_comm communicator on which to redistribute
   */
  void block_reshuffle(distribution const & old_dist,
                       distribution const & new_dist,
                       char **     tsr_data,
                       char **     tsr_cyclic_data,
                       semiring    sr,
                       CommData   glb_comm);
}

#endif
