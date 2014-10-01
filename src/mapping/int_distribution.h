/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_DISTRIBUTION_H__
#define __INT_DISTRIBUTION_H__

#include "../world/int_mapping.h"


namespace CTF_int {

  inline int get_distribution_size(int order){
    return sizeof(int)*2 + sizeof(int64_t) + order*sizeof(int)*6;
  }

  class distribution {
    public:
      int order;
      int * phase;
      int * virt_phase;
      int * pe_lda;
      int * edge_len;
      int * padding;
      int * perank;
      int is_cyclic;
      int64_t size;

      mapping * edge_map;
      topology topo;

      distribution();
      ~distribution();

      distribution(mapping const * edge_map_,
                   topology topo_);

      void serialize(char ** buffer, int * size);
      void deserialize(char const * buffer);
    private:
      void free_data();
  };

}

#endif
