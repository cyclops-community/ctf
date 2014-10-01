/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_MAPPING_H__
#define __INT_MAPPING_H__

#include "../world/int_topology.h"

namespace CTF_int {

  struct CommData;

  enum map_type {
    NOT_MAPPED,
    PHYSICAL_MAP,
    VIRTUAL_MAP
  };

  class mapping {
    public:
      map_type type;
      int np;
      int cdt;
      int has_child;
      mapping * child;
  };
  int comp_dim_map(mapping const *  map_A,
                   mapping const *  map_B);

}

#endif
