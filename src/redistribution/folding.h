/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_FOLDING_H__
#define __INT_FOLDING_H__

#include "../tensor/algstrct.h"

namespace CTF_int {
 
  /**
   * \brief permute an array
   *
   * \param order number of elements
   * \param perm permutation array
   * \param arr array to permute
   */
  void permute(int         order,
               int const * perm,
               int *       arr);

  /**
   * \brief permutes a permutation array 
   *
   * \param order number of elements in perm
   * \param order_perm number of elements in arr
   * \param perm permutation array
   * \param arr permutation array to permute
   */
  void permute_target(int         order,
                      int const * perm,
                      int *       arr);

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
                       int const *      edge_len,
                       char *           data,
                       int              dir,
                       algstrct const * sr);

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
                       int const *      edge_len,
                       char const *     data,
                       int              dir,
                       int              max_ntd,
                       char **          tswap_data,
                       int *            chunk_size,
                       algstrct const * sr);

}
#endif
