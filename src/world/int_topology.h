/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TOPOLOGY_H__
#define __INT_TOPOLOGY_H__

#include "../interface/common.h"

namespace CTF_int {

  enum TOPOLOGY { TOPOLOGY_GENERIC, TOPOLOGY_BGP, TOPOLOGY_BGQ,
                  TOPOLOGY_8D, NO_TOPOLOGY };

  /* mesh/torus topology configuration */
  class topology {
    public:
    int order;
    int * lens;
    int * lda;
    bool is_activated;
    CommData * dim_comm;

    topology();
    ~topology();

    /** 
     * \brief copy constructor
     * \param[in] other topology to copy
     */
    topology(topology const & other);

    /**
     * \brief constructs torus topology 
     * \param[in] order_ number of torus dimensions
     * \param[in] lens_ lengths of torus dimensions
     * \param[in] cdt communicator for whole torus 
     * \param[in] activate whether to create MPI_Comms
     */
    topology(int order_, 
             int const * lens_, 
             CommData cdt,
             bool activate=false);
    
    void activate();
    void deactivate();
  };

  /**
   * \brief get dimension and torus lengths of specified topology
   *
   * \param[in] mach specified topology
   * \param[out] order dimension of torus
   * \param[out] dim_len torus lengths of topology
   */
  topology get_phys_topo(CommData glb_comm,
                         TOPOLOGY mach);

   /**
   * \brief folds specified topology into all configurations of lesser dimensionality
   * \param[in] topo topology to fold
   * \param[in] glb_comm  global communicator
   */
  std::vector<topology> peel_torus(topology const & topo, 
                                   CommData         glb_comm);

  /**
   * \brief searches for an equivalent topology in avector of topologies
   * \param[in] topo topology to match
   * \param[in] topovec vector of existing parameters
   * \return -1 if not found, otherwise index of first found topology
   */
  int find_topology(topology const &              topo, 
                    std::vector<topology>         topovec);

  
}

#endif
