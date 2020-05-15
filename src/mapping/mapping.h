/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_MAPPING_H__
#define __INT_MAPPING_H__

#include "topology.h"

namespace CTF_int {

  class CommData;
  class tensor;

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

      /** \brief constructor */
      mapping();

      /** \brief destructor */
      ~mapping();

      /** \brief copy constructor */
      mapping(mapping const & other);
      mapping & operator=(mapping const & other);
  
      /**
       * \brief compute the phase of a mapping
       *
       * \return int phase
       */
      int calc_phase() const;
      
      /**
       * \brief compute the physical phase of a mapping
       *
       * \return int physical phase
       */
      int calc_phys_phase() const;
      
      /**
       * \brief compute the physical rank of a mapping
       *
       * \param topo topology
       * \return int physical rank
       */
      int calc_phys_rank(topology const * topo) const;

      /** \brief resets mapping to NOT_MAPPED */
      void clear();

      /**
       * \brief adds a physical mapping to this mapping
       * \param[in] topo topology to map to
       * \param[in] idim dimension of topology to map to
       */
      void aug_phys(topology const * topo, int idim);
      
      /**
       * \brief augments mapping to have sufficient virtualization so that the total phas is exactly tot_phase (assumes tot_phase is not current phase)
       * \param[in] tot_phase the desired total phase
       */
      void aug_virt(int tot_phase);
  };
   
  /** \brief compares two mappings
   * \param map_A first map
   * \param map_B second map
   * return 0 if mapping is exactly the same, 1 if map_A is ranked greater, -1 if map_B is ranked greater
   */
  int rank_dim_map(mapping const * map_A,
                   mapping const * map_B);


  /** \brief compares two mappings
   * \param map_A first map
   * \param map_B second map
   * return true if mapping is exactly the same, false otherwise 
   */
  int comp_dim_map(mapping const * map_A,
                   mapping const * map_B);

  /**
   * \brief copies mapping A to B
   * \param[in] order number of dimensions
   * \param[in] mapping_A mapping to copy from 
   * \param[in,out] mapping_B mapping to copy to
   */
  void copy_mapping(int             order,
                    mapping const * mapping_A,
                    mapping *       mapping_B);

  /**
   * \brief copies mapping A to B
   * \param[in] order_A number of dimensions in A
   * \param[in] order_B number of dimensions in B
   * \param[in] idx_A index mapping of A
   * \param[in] mapping_A mapping to copy from 
   * \param[in] idx_B index mapping of B
   * \param[in,out] mapping_B mapping to copy to
   * \param[in] make_virt makes virtual
   */
  int copy_mapping(int             order_A,
                   int             order_B,
                   int const *     idx_A,
                   mapping const * mapping_A,
                   int const *     idx_B,
                   mapping *       mapping_B,
                   int             make_virt = 1);


  /**
   * \brief map a tensor
   * \param[in] num_phys_dims number of physical processor grid dimensions
   * \param[in] tsr_order number dims
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
  int map_tensor(int             num_phys_dims,
                 int             tsr_order,
                 int64_t const * tsr_edge_len,
                 int const *     tsr_sym_table,
                 int *           restricted,
                 CommData  *     phys_comm,
                 int const *     comm_idx,
                 int             fill,
                 mapping *       tsr_edge_map);

  /**
   * \brief checks mapping in preparation for tensors scale, summ or contract
   * \param[in] tsr handle to tensor 
   * \param[in] idx_map is the mapping of tensor to global indices
   * \return whether the self mapping is consistent
  */
  int check_self_mapping(tensor const * tsr,
                         int const *    idx_map);

  /**
   * \brief create virtual mapping for idx_maps that have repeating indices
   * \param[in] tsr tensor handle
   * \param[in] idx_map mapping of tensor indices to contraction map
   */
  int map_self_indices(tensor const * tsr,
                       int const *    idx_map);


  /**
   * \brief adjust a mapping to maintan symmetry
   * \param[in] tsr_order is the number of dimensions of the tensor
   * \param[in] tsr_sym_table the symmetry table of a tensor
   * \param[in,out] tsr_edge_map is the mapping
   * \return CTF::SUCCESS if mapping successful, CTF::NEGATIVE if not, 
   *     CTF::ERROR if err'ed out
   */
  int map_symtsr(int         tsr_order,
                 int const * tsr_sym_table,
                 mapping *   tsr_edge_map);

  
  /**
   * \brief stretch virtualization by a factor
   * \param[in] order number of maps to stretch
   * \param[in] stretch_factor factor to strech by
   * \param[in] maps mappings along each dimension to stretch
   */
  int stretch_virt(int       order,
                   int       stretch_factor,
                   mapping * maps);


}

#endif
