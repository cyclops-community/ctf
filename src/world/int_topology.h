/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_TOPOLOGY_H__
#define __INT_TOPOLOGY_H__

#include "../world/int_world.h"
#include "../interface/common.h"
#include "../shared/util.h"

namespace CTF_int {

  enum TOPOLOGY { TOPOLOGY_GENERIC, TOPOLOGY_BGP, TOPOLOGY_BGQ,
                  TOPOLOGY_8D, NO_TOPOLOGY };

  /* mesh/torus topology configuration */
  class topology {
    int ndim;
    int * lens;
    int * lda;
    bool is_activated;
    CommData * dim_comm;

    topology();
    /**
     * \brief constructs torus topology 
     * \param[in] ndim_ number of torus dimensions
     * \param[in] lens_ lengths of torus dimensions
     * \param[in] cdt communicator for whole torus 
     * \param[in] activate whether to create MPI_Comms
     */
    topology(int ndim_, 
             int const * lens_, 
             CommData cdt,
             bool activate=false);
    ~topology();
    
    void activate();
    void deactivate();
  };

  /**
   * \brief get dimension and torus lengths of specified topology
   *
   * \param[in] mach specified topology
   * \param[out] ndim dimension of torus
   * \param[out] dim_len torus lengths of topology
   */
  topology get_phys_topo(CommData glb_comm,
                         TOPOLOGY mach);

   /**
   * \brief folds specified topology into all configurations of lesser dimensionality
   * \param[in] topo topology to fold
   * \param[in] glb_comm  global communicator
   */
  std::vector<topology> peel_torus(topology topo, 
                                   CommData glb_comm);

  /**
   * \brief folds a torus topology into all configurations of 1 less dimensionality
   * \param[in] topo topology to fold
   * \param[in] glb_comm  global communicator
   */
  void fold_torus(topology *           topo, 
                  CommData             glb_comm,
                  dist_tensor<dtype> * dt){
  /**
   * \brief searches for an equivalent topology in avector of topologies
   * \param[in] topo topology to match
   * \param[in] topovec vector of existing parameters
   * \return -1 if not found, otherwise index of first found topology
   */
  int find_topology(topology *                    topo, 
                    std::vector<topology>         topovec);

  
}

#endif
