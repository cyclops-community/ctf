/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_SORT_H__
#define __INT_SORT_H__

#include "../tensor/int_tensor.h"

namespace CTF_int {

  /**
   * \brief permutes keys
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len old nonpadded tensor edge lengths
   * \param[in] new_edge_len new nonpadded tensor edge lengths
   * \param[in] permutation permutation to apply to keys of each pair
   * \param[in,out] pairs the keys and values as pairs
   * \param[out] new_num_pair number of new pairs, since pairs are ignored if perm[i][j] == -1
   */
  void permute_keys(int                         order,
                    int                         num_pair,
                    int const *                 edge_len,
                    int const *                 new_edge_len,
                    int * const *               permutation,
                    pair *                      pairs,
                    int64_t *                   new_num_pair,
                    semiring                    sr);

  /**
   * \brief depermutes keys (apply P^T)
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len old nonpadded tensor edge lengths
   * \param[in] new_edge_len new nonpadded tensor edge lengths
   * \param[in] permutation permutation to apply to keys of each pair
   * \param[in,out] pairs the keys and values as pairs
   */
  void depermute_keys(int                         order,
                      int                         num_pair,
                      int const *                 edge_len,
                      int const *                 new_edge_len,
                      int * const *               permutation,
                      pair *                      pairs,
                      semiring                    sr);
}
#endif
