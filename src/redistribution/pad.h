/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_PAD_H__
#define __INT_PAD_H__

#include "../tensor/algstrct.h"

namespace CTF_int {
 
  /**
   * \brief applies padding to keys
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in,out] pairs set of pairs which to pad
   * \param[in] algstrct defines sizeo of each pair
   * \param[in] offsets (default NULL, none applied), offsets keys
   */
  void pad_key(int              order,
               int64_t          num_pair,
               int const *      edge_len,
               int const *      padding,
               PairIterator     pairs,
               algstrct const * sr,
               int const *      offsets = NULL);

  /**
   * \brief retrieves the unpadded pairs
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetry types of tensor
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in] prepadding padding at start of tensor (included in edge_len)
   * \param[in] pairs padded array of pairs
   * \param[out] new_pairs unpadded pairs
   * \param[out] new_num_pair number of unpadded pairs
   * \param[in] algstrct defines sizeo of each pair
   */
  void depad_tsr(int              order,
                 int64_t          num_pair,
                 int const *      edge_len,
                 int const *      sym,
                 int const *      padding,
                 int const *      prepadding,
                 char const *     pairsb,
                 char *           new_pairsb,
                 int64_t *        new_num_pair,
                 algstrct const * sr);

  /**
   * \brief pads a tensor
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in] phys_phase phase of the tensor on virtualized processor grid
   * \param[in] virt_phase_rank physical phase rank multiplied by virtual phase
   * \param[in] virt_phase virtual phase in each dimension
   * \param[in] old_data array of input pairs
   * \param[out] new_pairs padded pairs
   * \param[out] new_size number of new padded pairs
   * \param[in] algstrct defines sizeo of each pair
   */
  void pad_tsr(int              order,
               int64_t          size,
               int const *      edge_len,
               int const *      sym,
               int const *      padding,
               int const *      phys_phase,
               int *            virt_phys_rank,
               int const *      virt_phase,
               char const *     old_data,
               char **          new_pairs,
               int64_t *        new_size,
               algstrct const * sr);
  
  /**
   * \brief sets to zero all values in padded region of tensor
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths with padding
   * \param[in] sym symmetries of tensor
   * \param[in] padding how much of the edge lengths is padding
   * \param[in] phase phase of the tensor on virtualized processor grid
   * \param[in] virt_dim virtual phase in each dimension
   * \param[in] phase_rank physical phase rank multiplied by virtual phase
   * \param[in,out] vdata array of all local data
   * \param[in] algstrct defines sizeo of each pair
   */
  void zero_padding( int              order,
                     int64_t          size,
                     int              nvirt,
                     int const *      edge_len,
                     int const *      sym,
                     int const *      padding,
                     int const *      phase,
                     int const *      virt_dim,
                     int const *      cphase_rank,
                     char *           vdata,
                     algstrct const * sr);
}

#endif
