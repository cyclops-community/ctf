/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_MAPPING_H__
#define __INT_MAPPING_H__

#include "../world/int_topology.h"

namespace CTF_int {

struct CommData;

enum {
  NOT_MAPPED,
  PHYSICAL_MAP,
  VIRTUAL_MAP
};

struct mapping {
  int type;
  int np;
  int cdt;
  int has_child;
  mapping * child;
};

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

  distribution();
  ~distribution();

  void serialize(char ** buffer, int * size);
  void deserialize(char const * buffer);
  private:
  void free_data();
};

int comp_dim_map(mapping const *  map_A,
                 mapping const *  map_B);

}

#endif
