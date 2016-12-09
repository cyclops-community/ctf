/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __GLB_CYCLIC_RESHUFFLE_H__
#define __GLB_CYCLIC_RESHUFFLE_H__

#include "../tensor/algstrct.h"
#include "../mapping/distribution.h"
#include "redist.h"

namespace CTF_int {
  /**
   * \brief reorder local buffer so that elements are in ordered according to where they
   *        are in the global tensor (interleave virtual blocks)
   * \param[in] sym symmetry relations between tensor dimensions
   * \param[in] dist distribution of data
   * \param[in] virt_edge_len dimensions of each block
   * \param[in] virt_phase_lda prefix sum of virtual blocks
   * \param[in] vbs size of virtual blocks
   * \param[in] dir if 1 then go to global layout, if 0 than from
   * \param[in] tsr_data_in starting data buffer
   * \param[out] tsr_data_out target data buffer
   * \param[in] sr algstrct defining data
   */
  void order_globally(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      bool                 dir,
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
   * \param[in] ptr_tsr_data starting data buffer
   * \param[out] ptr_tsr_cyclic_data target data buffer
   * \param[in] sr algstrct defining data
   * \param[in] ord_glb_comm communicator on which to redistribute
   * \param[in] reuse_buffers if 1: ptr_tsr_cyclic_data is allocated dynamically and ptr_tsr_data 
   *                                 is overwritten with intermediate data
   *                          if 0: ptr_tsr_cyclic_data is preallocated and can be scaled by beta,
   *                                 however, more memory is used for temp buffers
   * \param[in] alpha scaling tensor for new data
   * \param[in] beta scaling tensor for original data
   */
//  void glb_cyclic_reshuffle(int const *          sym,
  char * glb_cyclic_reshuffle(int const *          sym,
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
}
#endif
