/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#ifndef __DIST_TENSOR_TOPO_HXX__
#define __DIST_TENSOR_TOPO_HXX__

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"

/**
 * \brief searches for an equivalent topology in avector of topologies
 * \param[in] topo topology to match
 * \param[in] topovec vector of existing parameters
 * \return -1 if not found, otherwise index of first found topology
 */
inline
int find_topology(topology *                    topo, 
                  std::vector<topology>         topovec){
  int i, j, found;
  std::vector<topology>::iterator iter;
  
  found = -1;
  for (j=0, iter=topovec.begin(); iter<topovec.end(); iter++, j++){
    if (iter->ndim == topo->ndim){
      found = j;
      for (i=0; i<iter->ndim; i++) {
        if (iter->dim_comm[i]->np != topo->dim_comm[i]->np){
          found = -1;
        }
      }
    }
    if (found != -1) return found;
  }
  return -1;  
}

/**
 * \brief folds a torus topology into all configurations of 1 less dimensionality
 * \param[in] topo topology to fold
 * \param[in] glb_comm  global communicator
 */
template<typename dtype>
void fold_torus(topology *              topo, 
                CommData * const        glb_comm,
                dist_tensor<dtype> *    dt){
  int i, j, k, ndim, rank, color, np;
  //int ins;
  CommData_t * new_comm;
  CommData_t ** comm_arr;

  ndim = topo->ndim;
  
  if (ndim <= 1) return;

  for (i=0; i<ndim; i++){
    /* WARNING: need to deal with nasty stuff in transpose when j-i > 1 */
    for (j=i+1; j<MIN(i+2,ndim); j++){
      get_buffer_space((ndim-1)*sizeof(CommData_t*),    (void**)&comm_arr);
      get_buffer_space(sizeof(CommData_t),              (void**)&new_comm);
      if (glb_comm != NULL){
        rank = topo->dim_comm[j]->rank*topo->dim_comm[i]->np + topo->dim_comm[i]->rank;
        /* Reorder the lda, bring j lda to lower lda and adjust other ldas */
        color = glb_comm->rank - topo->dim_comm[i]->rank*topo->lda[i]
                               - topo->dim_comm[j]->rank*topo->lda[j];
      }
      np = topo->dim_comm[i]->np*topo->dim_comm[j]->np;

      if (glb_comm != NULL){
        SETUP_SUB_COMM(glb_comm, new_comm, rank, color, np, NREQ, NBCAST);
      } else {
        new_comm->np    = np;
        new_comm->rank  = 0;
      }

      for (k=0; k<ndim-1; k++){
        if (k<i) 
          comm_arr[k] = topo->dim_comm[k];
        else {
          if (k==i) 
            comm_arr[k] = new_comm;
          else {
            if (k>i && k<j) 
              comm_arr[k] = topo->dim_comm[k];
            else
              comm_arr[k] = topo->dim_comm[k+1];
          }
        }
      }
/*      ins = 0;
      for (k=0; k<ndim-1; k++){
        if (k<i) {
          if (ins == 0){
            if (topo->dim_comm[k]->np <= np){
              comm_arr[k] = new_comm;
              ins = 1;
            } else
              comm_arr[k] = topo->dim_comm[k];
          } else
            comm_arr[k] = topo->dim_comm[k-1];
        }
        else {
          if (k==i) {
            if (ins == 0) {
              comm_arr[k] = new_comm;
              ins = 1;
            } else comm_arr[k] = topo->dim_comm[k-1];
          }
          else {
            LIBT_ASSERT(ins == 1);
            if (k>i && k<j) comm_arr[k] = topo->dim_comm[k];
            else comm_arr[k] = topo->dim_comm[k+1];
          }
        }
      }*/
      if (glb_comm != NULL)
        dt->set_phys_comm(comm_arr, ndim-1);
      else
        dt->set_inner_comm(comm_arr, ndim-1);
    }
  }
}





#endif
