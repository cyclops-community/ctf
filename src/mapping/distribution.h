/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_DISTRIBUTION_H__
#define __INT_DISTRIBUTION_H__



namespace CTF_int {

  class tensor;

  int get_distribution_size(int order){
    return sizeof(int)*2 + sizeof(int64_t) + order*sizeof(int)*6;
  }

  // \brief data distribution object used for redistribution
  class distribution {
    public:
      int order;
      int * phase;
      int * virt_phase;
      int * pe_lda;
      int * pad_edge_len;
      int * padding;
      int * perank;
      int is_cyclic;
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

}

#endif
