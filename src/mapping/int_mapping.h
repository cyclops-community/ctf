/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_MAPPING_H__
#define __INT_MAPPING_H__

#include "int_topology.h"

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

      /** \brief resets mapping to NOT_MAPPED */
      void clear();
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
  void copy_mapping(int              order,
                    mapping const *  mapping_A,
                    mapping *        mapping_B);

  /**
   * \brief copies mapping A to B
   * \param[in] order_A number of dimensions in A
   * \param[in] order_B number of dimensions in B
   * \param[in] idx_A index mapping of A
   * \param[in] mapping_A mapping to copy from 
   * \param[in] idx_B index mapping of B
   * \param[in,out] mapping_B mapping to copy to
   */
  int copy_mapping(int          order_A,
                   int          order_B,
                   int const *    idx_A,
                   mapping const *  mapping_A,
                   int const *    idx_B,
                   mapping *    mapping_B,
                   int          make_virt = 1);


  /**
   * \brief map a tensor
   * \param[in] num_phys_dims number of physical processor grid dimensions
   * \param[in] tsr_edge_len edge lengths of the tensor
   * \param[in] tsr_sym_table the symmetry table of a tensor
   * \param[in,out] restricted an array used to restricted the mapping of tensor dims
   * \param[in] phys_comm dimensional communicators
   * \param[in] comm_idx dimensional ordering
   * \param[in] fill if set does recursive mappings and uses all phys dims
   * \param[in,out] tsr_edge_map mapping of tensor
   * \return CTF_SUCCESS if mapping successful, CTF_NEGATIVE if not, 
   *     CTF_ERROR if err'ed out
   */
  int map_tensor(int            num_phys_dims,
                 int            tsr_order,
                 int const *    tsr_edge_len,
                 int const *    tsr_sym_table,
                 int *          restricted,
                 CommData  *  phys_comm,
                 int const *    comm_idx,
                 int            fill,
                 mapping *      tsr_edge_map);

}

#endif
