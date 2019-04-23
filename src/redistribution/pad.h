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
   * \param[in] sr algstrct defines sizeo of each pair
   * \param[in] offsets (default NULL, none applied), offsets keys
   */
  void pad_key(int              order,
               int64_t          num_pair,
               int64_t const *  edge_len,
               int64_t const *  padding,
               PairIterator     pairs,
               algstrct const * sr,
               int64_t const *  offsets = NULL);

  /**
   * \brief retrieves the unpadded pairs
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetry types of tensor
   * \param[in] padding padding of tensor (included in edge_len)
   * \param[in] prepadding padding at start of tensor (included in edge_len)
   * \param[in] pairsb padded array of pairs
   * \param[out] new_pairsb unpadded pairs
   * \param[out] new_num_pair number of unpadded pairs
   * \param[in] sr algstrct defines sizeo of each pair
   */
  void depad_tsr(int              order,
                 int64_t          num_pair,
                 int64_t const *  edge_len,
                 int const *      sym,
                 int64_t const *  padding,
                 int64_t const *  prepadding,
                 char const *     pairsb,
                 char *           new_pairsb,
                 int64_t *        new_num_pair,
                 algstrct const * sr);

  /**
   * brief pads a tensor
   * param[in] order tensor dimension
   * param[in] num_pair number of pairs
   * param[in] edge_len tensor edge lengths
   * param[in] sym symmetries of tensor
   * param[in] padding padding of tensor (included in edge_len)
   * param[in] phys_phase phase of the tensor on virtualized processor grid
   * param[in] phase_rank physical phase rank multiplied by virtual phase
   * param[in] virt_phase virtual phase in each dimension
   * param[in] old_data array of input pairs
   * param[out] new_pairs padded pairs
   * param[out] new_size number of new padded pairs
   * param[in] algstrct defines sizeo of each pair
   */
/*  void pad_tsr(int              order,
               int64_t          size,
               int const *      edge_len,
               int const *      sym,
               int const *      padding,
               int const *      phys_phase,
               int *            phase_rank,
               int const *      virt_phase,
               char const *     old_data,
               char **          new_pairs,
               int64_t *        new_size,
               algstrct const * sr);
  */

  /**
   * \brief sets to zero all values in padded region of tensor
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths with padding
   * \param[in] sym symmetries of tensor
   * \param[in] padding how much of the edge lengths is padding
   * \param[in] phase phase of the tensor on virtualized processor grid
   * \param[in] phys_phase phase of the tensor on virtualized processor grid
   * \param[in] virt_phase virtual phase in each dimension
   * \param[in] cphase_rank physical phase rank 
   * \param[in,out] vdata array of all local data
   * \param[in] sr algstrct defines sizeo of each pair
   */
  void zero_padding( int              order,
                     int64_t          size,
                     int              nvirt,
                     int64_t const *  edge_len,
                     int const *      sym,
                     int64_t const *  padding,
                     int const *      phase,
                     int const *      phys_phase,
                     int const *      virt_phase,
                     int const *      cphase_rank,
                     char *           vdata,
                     algstrct const * sr);

  /**
   * \brief scales each element by 1/(number of entries equivalent to it after permutation of indices for which sym_mask is 1)
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths with padding
   * \param[in] sym symmetries of tensor
   * \param[in] padding how much of the edge lengths is padding
   * \param[in] phase phase of the tensor on virtualized processor grid
   * \param[in] phys_phase phase of the tensor on virtualized processor grid
   * \param[in] virt_phase virtual phase in each dimension
   * \param[in] cphase_rank physical phase rank 
   * \param[in,out] vdata array of all local data
   * \param[in] sr algstrct defines sizeo of each pair
   * \param[in] sym_mask identifies which tensor indices are part of the symmetric group which diagonals we want to scale (i.e. sym_mask [1,1] does A["ii"]= (1./2.)*A["ii"])
*/
  void scal_diag(int              order,
                 int64_t          size,
                 int              nvirt,
                 int64_t const *  edge_len,
                 int const *      sym,
                 int64_t const *  padding,
                 int const *      phase,
                 int const *      phys_phase,
                 int const *      virt_phase,
                 int const *      cphase_rank,
                 char *           vdata,
                 algstrct const * sr,
                 int const *      sym_mask);

  /**
   * \brief scales each element by 1/(number of entries equivalent to it after permutation of indices for which sym_mask is 1)
   * \param[in] order tensor dimension
   * \param[in] lens tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] nnz_loc number of local nonzeros
   * \param[in,out] vdata array of all local data pairs
   * \param[in] sr algstrct defines sizeo of each pair
   * \param[in] sym_mask identifies which tensor indices are part of the symmetric group which diagonals we want to scale (i.e. sym_mask [1,1] does A["ii"]= (1./2.)*A["ii"])
*/

  void sp_scal_diag(int              order,
                    int64_t const *  lens,
                    int const *      sym,
                    int64_t          nnz_loc,
                    char *           vdata,
                    algstrct const * sr,
                    int const *      sym_mask);
}

#endif
