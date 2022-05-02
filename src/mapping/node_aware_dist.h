/*Copyright (c) 2022, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_NODE_DISTRIBUTION_H__
#define __INT_NODE_DISTRIBUTION_H__

namespace CTF_int {
  /**
   * \brief returns all possible valid choices inter-node grids, given an overall processor grid and a number of nodes
   * \param[in] rGrid overall processor grid
   * \param[in] nodes number of nodes
   * \return vector of inter node processor grids of total size equal to the number of nodes and of same dimension as rGrid, where each dimension divides into the respective dimension of rGrid
   */
  std::vector<std::vector<int> > get_inter_node_grids(std::vector<int> rGrid, int nodes);
}

#endif
