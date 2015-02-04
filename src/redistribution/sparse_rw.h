/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_SORT_H__
#define __INT_SORT_H__

#include "../tensor/algstrct.h"

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
  void permute_keys(int              order,
                    int              num_pair,
                    int const *      edge_len,
                    int const *      new_edge_len,
                    int * const *    permutation,
                    char *           pairs,
                    int64_t *        new_num_pair,
                    algstrct const & sr);

  /**
   * \brief depermutes keys (apply P^T)
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len old nonpadded tensor edge lengths
   * \param[in] new_edge_len new nonpadded tensor edge lengths
   * \param[in] permutation permutation to apply to keys of each pair
   * \param[in,out] pairs the keys and values as pairs
   */
  void depermute_keys(int           order,
                      int           num_pair,
                      int const *   edge_len,
                      int const *   new_edge_len,
                      int * const * permutation,
                      char *        pairs,
                      algstrct      sr);

  /**
   * \brief assigns keys to an array of values
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] phase phase of the tensor on virtualized processor grid
   * \param[in] virt_dim virtual phase in each dimension
   * \param[in] phase_rank physical phase rank multiplied by virtual phase
   * \param[in] vdata array of input values
   * \param[out] vpairs pairs of keys and inputted values
   * \param[in] sr algstrct defining data type of array
   */
  void assign_keys(int              order,
                   int64_t          size,
                   int              nvirt,
                   int const *      edge_len,
                   int const *      sym,
                   int const *      phase,
                   int const *      virt_dim,
                   int *            phase_rank,
                   char const *     vdata,
                   char *           vpairs,
                   algstrct const & sr);


  /**
   * \brief buckets key-value pairs by processor according to distribution
   * \param[in] order number of tensor dims
   * \param[in] num_pairs numbers of values being written
   * \param[in] np number of processor buckets
   * \param[in] phase total distribution phase
   * \param[in] virt_phase factor of phase due to local blocking
   * \param[in] bucket_lda iterator hop along each bucket dim
   * \param[in] edge_len padded edge lengths of tensor
   * \param[in] mapped_data set of sparse key-value pairs
   * \param[out] bucket_counts how many keys belong to each processor
   * \param[out] bucket_offsets prefix sum of bucket_counts
   * \param[out] bucket_data mapped_data reordered by bucket
   * \param[in] sr algstrct context defining values
   */
  void bucket_by_pe( int               order,
                     int64_t           num_pair,
                     int64_t           np,
                     int const *       phase,
                     int const *       virt_phase,
                     int const *       bucket_lda,
                     int const *       edge_len,
                     ConstPairIterator mapped_data,
                     int64_t *         bucket_counts,
                     int64_t *         bucket_off,
                     PairIterator      bucket_data,
                     algstrct const &  sr);

  /**
   * \brief buckets key value pairs by block/virtual-processor
   * \param[in] order number of tensor dims
   * \param[in] num_virt number of local blocks
   * \param[in] num_pair numbers of values being written
   * \param[in] virt_phase factor of phase due to local blocking
   * \param[in] edge_len padded edge lengths of tensor
   * \param[in] mapped_data set of sparse key-value pairs
   * \param[out] bucket_data mapped_data reordered by bucket
   * \param[in] sr algstrct context defining values
   */
  void bucket_by_virt(int               order,
                      int               num_virt,
                      int64_t           num_pair,
                      int const *       virt_phase,
                      int const *       edge_len,
                      ConstPairIterator mapped_data,
                      PairIterator      bucket_data,
                      algstrct const &  sr);

  /**
   * \brief read or write pairs from / to tensor
   * \param[in] order tensor dimension
   * \param[in] size number of pairs
   * \param[in] alpha multiplier for new value
   * \param[in] beta multiplier for old value
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] phase total phase in each dimension
   * \param[in] virt_dim virtualization in each dimension
   * \param[in] phase_rank virtualized rank in total phase
   * \param[in,out] vdata data to read from or write to
   * \param[in,out] pairs pairs to read or write
   * \param[in] rw whether to read 'r' or write 'w'
   * \param[in] sr algstrct context defining values
   */
  void readwrite(int              order,
                 int64_t          size,
                 char const *     alpha,
                 char const *     beta,
                 int              nvirt,
                 int const *      edge_len,
                 int const *      sym,
                 int const *      phase,
                 int const *      virt_dim,
                 int *            phase_rank,
                 char *           vdata,
                 char             *pairs,
                 char             rw,
                 algstrct const & sr);


  /**
   * \brief read or write pairs from / to tensor
   * \param[in] order tensor dimension
   * \param[in] np number of processors
   * \param[in] inwrite number of pairs
   * \param[in] alpha multiplier for new value
   * \param[in] beta multiplier for old value
   * \param[in] rw whether to read 'r' or write 'w'
   * \param[in] num_virt new total virtualization factor
   * \param[in] sym symmetries of tensor
   * \param[in] edge_len tensor edge lengths
   * \param[in] padding padding of tensor
   * \param[in] phys_phase total phase in each dimension
   * \param[in] virt_phase virtualization in each dimension
   * \param[in] virt_phase_rank virtualized rank in total phase
   * \param[in] bucket_lda prefix sum of the processor grid
   * \param[in,out] wr_pairs pairs to read or write
   * \param[in,out] rw_data data to read from or write to
   * \param[in] glb_comm the global communicator
   * \param[in] sr algstrct context defining values
   */
  void wr_pairs_layout(int              order,
                       int              np,
                       int64_t          inwrite,
                       char const *     alpha,
                       char const *     beta,
                       char             rw,
                       int              num_virt,
                       int const *      sym,
                       int const *      edge_len,
                       int const *      padding,
                       int const *      phys_phase,
                       int const *      virt_phase,
                       int *            virt_phys_rank,
                       int const *      bucket_lda,
                       char *           wr_pairs,
                       char *           rw_data,
                       CommData         glb_comm,
                       algstrct const & sr);

  /**
   * \brief read tensor pairs local to processor
   * \param[in] order tensor dimension
   * \param[in] nval number of local values
   * \param[in] pad whether tensor is padded
   * \param[in] num_virt new total virtualization factor
   * \param[in] sym symmetries of tensor
   * \param[in] edge_len tensor edge lengths
   * \param[in] padding padding of tensor
   * \param[in] virt_dim virtualization in each dimension
   * \param[in] virt_phase total phase in each dimension
   * \param[in] virt_phase_rank virtualized rank in total phase
   * \param[in] bucket_lda prefix sum of the processor grid
   * \param[out] nread number of local pairs read
   * \param[in] tensor data data to read from
   * \param[out] pairs local pairs read
   * \param[in] sr algstrct context defining values
   */
  void read_loc_pairs(int              order,
                      int64_t          nval,
                      int              num_virt,
                      int const *      sym,
                      int const *      edge_len,
                      int const *      padding,
                      int const *      virt_dim,
                      int const *      virt_phase,
                      int *            virt_phase_rank,
                      int64_t *        nread,
                      char const *     data,
                      char **          pairs,
                      algstrct const & sr);


}
#endif
