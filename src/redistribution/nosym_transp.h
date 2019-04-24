/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __NOSYM_TRANSP_H__
#define __NOSYM_TRANSP_H__

#include "../tensor/algstrct.h"

namespace CTF_int {
  class tensor;

  /**
   * \brief transposes a non-symmetric (folded) tensor
   *
   * \param[in] order dimension of tensor
   * \param[in] new_order new ordering of dimensions
   * \param[in] edge_len original edge lengths
   * \param[in,out] data data tp transpose
   * \param[in] dir which way are we going?
   * \param[in] sr algstrct defining element size
   */
  void nosym_transpose(int              order,
                       int const *      new_order,
                       int64_t const *  edge_len,
                       char *           data,
                       int              dir,
                       algstrct const * sr);
  /**
   * \brief estimates time needed to transposes a non-symmetric (folded) tensor based on performance models
   *
   * \param[in] order dimension of tensor
   * \param[in] new_order new ordering of dimensions
   * \param[in] edge_len original edge lengths
   * \param[in] dir which way are we going?
   * \param[in] sr algstrct defining element size
   * \return estimated time in seconds
   */
  double est_time_transp(int              order,
                         int const *      new_order,
                         int64_t const *  edge_len,
                         int              dir,
                         algstrct const * sr);

  void nosym_transpose(tensor *        A,
                       int             all_fdim_A,
                       int64_t const * all_flen_A,
                       int const *     new_order,
                       int             dir);
  /**
   * \brief transposes a non-symmetric (folded) tensor internal kernel
   *
   * \param[in] order dimension of tensor
   * \param[in] new_order new ordering of dimensions
   * \param[in] edge_len original edge lengths
   * \param[in] data data tp transpose
   * \param[in] dir which way are we going?
   * \param[in] max_ntd how many threads to use
   * \param[out] tswap_data tranposed data
   * \param[out] chunk_size chunk sizes of tranposed data
   * \param[in] sr algstrct defining element size
   */
  void nosym_transpose(int              order,
                       int const *      new_order,
                       int64_t const *  edge_len,
                       char const *     data,
                       int              dir,
                       int              max_ntd,
                       char **          tswap_data,
                       int64_t *        chunk_size,
                       algstrct const * sr);

  /**
   * \brief Checks if the HPTT library is applicable
   * \param[in] order dimension of tensor
   * \param[in] new_order new ordering of dimensions
   * \param[in] elementSize element size
   */
  bool hptt_is_applicable(int order, int const * new_order, int elementSize);

  /**
   * \brief High-performance implementation of nosym_transpose using HPTT
   *
   * \param[in] order dimension of tensor
   * \param[in] st_new_order new ordering of modes
   * \param[in] st_edge_len original edge lengths
   * \param[in] dir which way are we going?
   * \param[in] st_buffer initial buffer
   * \param[in,out] new_buffer destination buffer, preallocated
   * \param[in] sr algebraic structure
   */

  void nosym_transpose_hptt(int              order,
                            int const *      st_new_order,
                            int64_t const *  st_edge_len,
                            int              dir,
                            char const *     st_buffer,
                            char *           new_buffer,
                            algstrct const * sr);
}
#endif
