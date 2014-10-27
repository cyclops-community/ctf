/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_PAD_H__
#define __INT_PAD_H__

#include "../tensor/untyped_semiring.h"

namespace CTF_int {
 
  /**
   * \brief applies padding to keys
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in,out] pairs set of pairs which to pad
   * \param[in] semiring defines sizeo of each pair
   * \param[in] offsets (default NULL, none applied), offsets keys
   */
  void pad_key(int           order,
               int64_t      num_pair,
               int const *        edge_len,
               int const *        padding,
               pair *  pairs,
               semiring const & sr,
               int const *        offsets = NULL);

  /**
   * \brief retrieves the unpadded pairs
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetry types of tensor
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in] prepadding padding at start of tensor (included in edge_len)
   * \param[in] pairs padded array of pairs
   * \param[in] semiring defines sizeo of each pair
   * \param[out] new_pairs unpadded pairs
   * \param[out] new_num_pair number of unpadded pairs
   */
  void depad_tsr(int                 order,
                 int64_t            num_pair,
                 int const *              edge_len,
                 int const *              sym,
                 int const *              padding,
                 int const *              prepadding,
                 pair const *  pairs,
                 semiring const & sr,
                 pair *        new_pairs,
               int64_t *                new_num_pair);

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
   * \param[in] semiring defines sizeo of each pair
   * \param[out] new_pairs padded pairs
   * \param[out] new_size number of new padded pairs
   */
  void pad_tsr(int                 order,
               int64_t            size,
               int const *              edge_len,
               int const *              sym,
               int const *              padding,
               int const *              phys_phase,
               int *                    virt_phys_rank,
               int const *              virt_phase,
               pair const *  old_data,
                 semiring const & sr,
               pair **       new_pairs,
               int64_t *                new_size);
  
  /**
   * \brief sets to zero all values in padded region of tensor
   * \param[in] ndim tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths with padding
   * \param[in] sym symmetries of tensor
   * \param[in] padding how much of the edge lengths is padding
   * \param[in] phase phase of the tensor on virtualized processor grid
   * \param[in] virt_dim virtual phase in each dimension
   * \param[in] phase_rank physical phase rank multiplied by virtual phase
   * \param[in,out] vdata array of all local data
   * \param[in] semiring defines sizeo of each pair
   */
  void zero_padding( int           ndim,
                     int64_t      size,
                     int           nvirt,
                     int const *        edge_len,
                     int const *        sym,
                     int const *        padding,
                     int const *        phase,
                     int const *        virt_dim,
                     int const *        cphase_rank,
                     char *            vdata,
                     semiring const & sr);
}

#endif
