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

  // \brief object keeping track of a mapping of a tensor dimension onto a torus dimension
  class mapping {
    public:
      map_type type;
      int np;
      int cdt;
      int has_child;
      mapping * child;

      /**
       * \brief compute the phase of a mapping
       *
       * \return int phase
       */
      int calc_phase();
      
      /**
       * \brief compute the physical phase of a mapping
       *
       * \return int physical phase
       */
      int calc_phys_phase();
      
      /**
       * \brief compute the physical rank of a mapping
       *
       * \param topo topology
       * \return int physical rank
       */
      int calc_phys_rank(topology const * topo);
  };
  
  /** \brief compares two mappings
   * \param map_A first map
   * \param map_B second map
   * return true if mapping is exactly the same, false otherwise 
   */
  int comp_dim_map(mapping const *  map_A,
                   mapping const *  map_B);

  /**
   * \brief copies mapping A to B
   * \param[in] order number of dimensions
   * \param[in] mapping_A mapping to copy from 
   * \param[in,out] mapping_B mapping to copy to
   */
  void copy_mapping(int const        order,
                    mapping const *  mapping_A,
                    mapping *        mapping_B);

}

#endif
