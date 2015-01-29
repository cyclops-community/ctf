/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_DISTRIBUTION_H__
#define __INT_DISTRIBUTION_H__

#include "mapping.h"

namespace CTF_int {

  class tensor;

  inline
  int get_distribution_size(int order){
    return sizeof(int)*2 + sizeof(int64_t) + order*sizeof(int)*6;
  }

  // \brief data distribution object used for redistribution
  class distribution {
    public:
      int     order;
      int *   phase;
      int *   virt_phase;
      int *   pe_lda;
      int *   pad_edge_len;
      int *   padding;
      int *   perank;
      int     is_cyclic;
      int64_t size;

      distribution();
      ~distribution();

      /**
       * \brief create distribution object which defines a tensor's data decomposition
       * \param[in] tsr tensor whose distribution to record
       */
      distribution(tensor const * tsr);

      /**
       * \brief create distribution object by deserializing buffer
       * \param[in] buffer serialized distribution data 
       */
      distribution(char const * buffer);

      /**
       * \brief serialize object into contiguous data buffer
        \param[out] buffer unallocated array into which to serialize
       * \param[out] size length of serialized array
      */
      void serialize(char ** buffer, int * size);
    private:
      void free_data();
  };

  /**
   * \brief calculate the block-sizes of a tensor
   * \param[in] order number of dimensions of this tensor
   * \param[in] size is the size of the local tensor stored
   * \param[in] edge_len edge lengths of global tensor
   * \param[in] edge_map mapping of each dimension
   * \param[out] vrt_sz size of virtual block
   * \param[out] vrt_edge_len edge lengths of virtual block
   * \param[out] blk_edge_len edge lengths of local block
   */
  void calc_dim(int             order,
                int64_t         size,
                int const *     edge_len,
                mapping const * edge_map,
                int64_t *       vrt_sz,
                int *           vrt_edge_len,
                int *           blk_edge_len);
}

#endif
