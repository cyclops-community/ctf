/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_SORT_H__
#define __INT_SORT_H__

#include "../tensor/algstrct.h"
#include <functional>

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
   * \param[in] sr algstrct defining data type of array
   */
  void permute_keys(int              order,
                    int              num_pair,
                    int64_t const *  edge_len,
                    int64_t const *  new_edge_len,
                    int * const *    permutation,
                    char *           pairs,
                    int64_t *        new_num_pair,
                    algstrct const * sr);

  /**
   * \brief depermutes keys (apply P^T)
   * \param[in] order tensor dimension
   * \param[in] num_pair number of pairs
   * \param[in] edge_len old nonpadded tensor edge lengths
   * \param[in] new_edge_len new nonpadded tensor edge lengths
   * \param[in] permutation permutation to apply to keys of each pair
   * \param[in,out] pairs the keys and values as pairs
   * \param[in] sr algstrct defining data type of array
   */
  void depermute_keys(int              order,
                      int              num_pair,
                      int64_t const *  edge_len,
                      int64_t const *  new_edge_len,
                      int * const *    permutation,
                      char *           pairs,
                      algstrct const * sr);

  /**
   * \brief assigns keys to an array of values
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] phase total phase of the tensor on virtualized processor grid
   * \param[in] phys_phase physical phase of the tensor
   * \param[in] virt_dim virtual phase in each dimension
   * \param[in] phase_rank physical phase rank multiplied by virtual phase
   * \param[in] vdata array of input values
   * \param[out] vpairs pairs of keys and inputted values
   * \param[in] sr algstrct defining data type of array
   */
  void assign_keys(int              order,
                   int64_t          size,
                   int              nvirt,
                   int64_t const *  edge_len,
                   int const *      sym,
                   int const *      phase,
                   int const *      phys_phase,
                   int const *      virt_dim,
                   int *            phase_rank,
                   char const *     vdata,
                   char *           vpairs,
                   algstrct const * sr);

  /**
   * \brief extracts all tensor values (in pair format) that pass a sparsifier function (including padded zeros if they pass the fliter)
   * \param[in] order tensor dimension
   * \param[in] size number of values
   * \param[in] nvirt total virtualization factor
   * \param[in] edge_len tensor edge lengths (padded)
   * \param[in] sym symmetries of tensor
   * \param[in] phase total phase of the tensor on virtualized processor grid
   * \param[in] phys_phase physical phase of the tensor
   * \param[in] virt_dim virtual phase in each dimension
   * \param[in] phase_rank physical phase rank multiplied by virtual phase
   * \param[in] vdata array of input values
   * \param[out] vpairs pairs of keys and inputted values, allocated internally
   * \param[in,out] nnz_blk nonzero counts for each virtual block
   * \param[in] sr algstrct defining data type of array
   * \param[in] edge_lda lda of non-padded edge-lengths
   * \param[in] f sparsification filter
   */
  void spsfy_tsr(int              order,
                 int64_t          size,
                 int              nvirt,
                 int64_t const *  edge_len,
                 int const *      sym,
                 int const *      phase,
                 int const *      phys_phase,
                 int const *      virt_dim,
                 int *            phase_rank,
                 char const *     vdata,
                 char *&          vpairs,
                 int64_t *        nnz_blk,
                 algstrct const * sr,
                 int64_t const *  edge_lda,
                 std::function<bool(char const*)> f);

  /**
   * \brief buckets key-value pairs by processor according to distribution
   * \param[in] order number of tensor dims
   * \param[in] num_pair numbers of values being written
   * \param[in] np number of processor buckets
   * \param[in] phys_phase physical distribution phase
   * \param[in] virt_phase factor of phase due to local blocking
   * \param[in] bucket_lda iterator hop along each bucket dim
   * \param[in] edge_len padded edge lengths of tensor
   * \param[in] mapped_data set of sparse key-value pairs
   * \param[out] bucket_counts how many keys belong to each processor
   * \param[out] bucket_off prefix sum of bucket_counts
   * \param[out] bucket_data mapped_data reordered by bucket
   * \param[in] sr algstrct context defining values
   */
  void bucket_by_pe(int               order,
                    int64_t           num_pair,
                    int64_t           np,
                    int const *       phys_phase,
                    int const *       virt_phase,
                    int const *       bucket_lda,
                    int64_t const *   edge_len,
                    ConstPairIterator mapped_data,
                    int64_t *         bucket_counts,
                    int64_t *         bucket_off,
                    PairIterator      bucket_data,
                    algstrct const *  sr);

  /**
   * \brief buckets key value pairs by block/virtual-processor
   * \param[in] order number of tensor dims
   * \param[in] num_virt number of local blocks
   * \param[in] num_pair numbers of values being written
   * \param[in] phys_phase physical distribution phase
   * \param[in] virt_phase factor of phase due to local blocking
   * \param[in] edge_len padded edge lengths of tensor
   * \param[in] mapped_data set of sparse key-value pairs
   * \param[out] bucket_data mapped_data reordered by bucket
   * \param[in] sr algstrct context defining values
   */
  int64_t * bucket_by_virt(int               order,
                           int               num_virt,
                           int64_t           num_pair,
                           int const *       phys_phase,
                           int const *       virt_phase,
                           int64_t const *   edge_len,
                           ConstPairIterator mapped_data,
                           PairIterator      bucket_data,
                           algstrct const *  sr);

  /**
   * \brief read or write pairs from / to tensor
   * \param[in] order tensor dimension
   * \param[in] size number of pairs
   * \param[in] alpha multiplier for new value
   * \param[in] beta multiplier for old value
   * \param[in] nvirt num local blocks
   * \param[in] edge_len tensor edge lengths
   * \param[in] sym symmetries of tensor
   * \param[in] phase total phase in each dimension
   * \param[in] phys_phase physical distribution phase
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
                 int64_t const *  edge_len,
                 int const *      sym,
                 int const *      phase,
                 int const *      phys_phase,
                 int const *      virt_dim,
                 int *            phase_rank,
                 char *           vdata,
                 char             *pairs,
                 char             rw,
                 algstrct const * sr);


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
   * \param[in] phase total phase in each dimension
   * \param[in] phys_phase physical phase in each dimension
   * \param[in] virt_phase virtualization in each dimension
   * \param[in] virt_phys_rank virtualized rank in total phase
   * \param[in] bucket_lda prefix sum of the processor grid
   * \param[in,out] wr_pairs_buf pairs to read or write
   * \param[in,out] rw_data data to read from or write to
   * \param[in] glb_comm the global communicator
   * \param[in] sr algstrct context defining values
   * \param[in] is_sparse if true local data (vdata) is sparse, otherwise rest of params irrelevant
   * \param[in] nnz_loc number of local elements in sparse tensor
   * \param[in] nnz_blk number of local elements in each block
   * \param[out] pprs_new new elements of the local sparse tensor (if rw=='w')
   * \param[out] nnz_loc_new number of elements of the new local sparse tensor (if rw=='w')
   */
  void wr_pairs_layout(int              order,
                       int              np,
                       int64_t          inwrite,
                       char const *     alpha,
                       char const *     beta,
                       char             rw,
                       int              num_virt,
                       int const *      sym,
                       int64_t const *  edge_len,
                       int64_t const *  padding,
                       int const *      phase,
                       int const *      phys_phase,
                       int const *      virt_phase,
                       int *            virt_phys_rank,
                       int const *      bucket_lda,
                       char *           wr_pairs_buf,
                       char *           rw_data,
                       CommData         glb_comm,
                       algstrct const * sr,
                       bool             is_sparse,
                       int64_t          nnz_loc,
                       int64_t *        nnz_blk,
                       char *&          pprs_new,
                       int64_t &        nnz_loc_new);


  /**
   * \brief read tensor pairs local to processor
   * \param[in] order tensor dimension
   * \param[in] nval number of local values
   * \param[in] num_virt new total virtualization factor
   * \param[in] sym symmetries of tensor
   * \param[in] edge_len tensor edge lengths
   * \param[in] padding padding of tensor
   * \param[in] phase blocking factor in each dimension
   * \param[in] phys_phase number of procs in each dimension
   * \param[in] virt_phase virtualization in each dimension
   * \param[in] phase_rank virtualized rank in total phase
   * \param[out] nread number of local pairs read
   * \param[in] data tensor data to read from
   * \param[out] pairs local pairs read
   * \param[in] sr algstrct context defining values
   */
  void read_loc_pairs(int              order,
                      int64_t          nval,
                      int              num_virt,
                      int const *      sym,
                      int64_t const *  edge_len,
                      int64_t const *  padding,
                      int const *      phase,
                      int const *      phys_phase,
                      int const *      virt_phase,
                      int *            phase_rank,
                      int64_t *        nread,
                      char const *     data,
                      char **          pairs,
                      algstrct const * sr);


  /**
   * \brief reads elements of a sparse set defining the tensor, 
   *    into a sparse read set with potentially repeating keys
   * \param[in] sr algstrct defining data type of array
   * \param[in] ntsr number of elements in sparse tensor
   * \param[in] prs_tsr pairs of the sparse tensor
   * \param[in] alpha scaling factor for data of the sparse tensor
   * \param[in] nread number of elements in the read set
   * \param[in,out] prs_read pairs of the read set
   * \param[in] beta scaling factor for data of the read set
   */
  void sp_read(algstrct const *  sr, 
               int64_t           ntsr,
               ConstPairIterator prs_tsr,
               char const *      alpha,
               int64_t           nread,
               PairIterator      prs_read,
               char const *      beta);


  /**
   * \brief writes pairs in a sparse write set to the 
   *         sparse set of elements defining the tensor,
   *         resulting in a set of size between ntsr and ntsr+nwrite
   * \param[in] num_virt num local blocks
   * \param[in] sr algstrct defining data type of array
   * \param[in] vntsr number of elements in sparse tensor
   * \param[in] vprs_tsr pairs of the sparse tensor
   * \param[in] beta scaling factor for data of the sparse tensor
   * \param[in] vnwrite number of elements in the write set
   * \param[in] vprs_write pairs of the write set
   * \param[in] alpha scaling factor for data of the write set
   * \param[out] vnnew number of elements in resulting set
   * \param[out] pprs_new char array containing the pairs of the resulting set
   */
  void sp_write(int               num_virt,
                algstrct const *  sr,
                int64_t *         vntsr,
                ConstPairIterator vprs_tsr,
                char const *      beta,
                int64_t *         vnwrite,
                ConstPairIterator vprs_write,
                char const *      alpha,
                int64_t *         vnnew,
                char *&           pprs_new);
}
#endif
