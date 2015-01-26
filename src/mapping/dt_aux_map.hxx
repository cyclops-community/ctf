/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTF_MAP_HXX__
#define __CTF_MAP_HXX__

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include <stdint.h>
#include <limits.h>

#define MAX_PHASE 2048


/** 
 * \brief clears mapping and frees children
 * \param map mapping
 */
inline
int clear_mapping(mapping * map){
}

/* \brief zeros out mapping
 * \param[in] tsr tensor to clear mapping of
 */
template<typename dtype>
int clear_mapping(tensor<dtype> * tsr){
}

/**
 * \brief copies mapping A to B
 * \param[in] order number of dimensions
 * \param[in] mapping_A mapping to copy from 
 * \param[in,out] mapping_B mapping to copy to
 */
inline
int copy_mapping(int const        order,
                 mapping const *  mapping_A,
                 mapping *        mapping_B){
}

/**
 * \brief saves the mapping of a tensor in auxillary arrays
 *        function allocates memory for these arrays, delete it later
 * \param[in] tsr tensor pointer
 * \param[out] old_phase phase of the tensor in each dimension
 * \param[out] old_rank rank of this processor along each dimension
 * \param[out] old_virt_dim virtualization of the mapping
 * \param[out] old_pe_lda processor lda of mapping
 * \param[out] was_cyclic whether the mapping was cyclic
 * \param[out] old_padding what the padding was
 * \param[out] old_edge_len what the edge lengths were
 * \param[in] topo topology of the processor grid mapped to
 */ 
template<typename dtype>
int save_mapping(tensor<dtype> *  tsr,
                 int **     old_phase,
                 int **     old_rank,
                 int **     old_virt_dim,
                 int **     old_pe_lda,
                 int64_t *    old_size,
                 int *      was_cyclic,
                 int **     old_padding,
                 int **     old_edge_len,
                 topology const * topo){
  int is_inner = 0;
  int j;
  mapping * map;
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_rank);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_virt_dim);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_pe_lda);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_edge_len);
  
  *old_size = tsr->size;
  
  for (j=0; j<tsr->order; j++){
    map     = tsr->edge_map + j;
    (*old_phase)[j]   = calc_phase(map);
    (*old_rank)[j]  = calc_phys_rank(map, topo);
    (*old_virt_dim)[j]  = (*old_phase)[j]/calc_phys_phase(map);
    if (!is_inner && map->type == PHYSICAL_MAP)
      (*old_pe_lda)[j]  = topo->lda[map->cdt];
    else
      (*old_pe_lda)[j]  = 0;
  }
  memcpy(*old_edge_len, tsr->edge_len, sizeof(int)*tsr->order);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)old_padding);
  memcpy(*old_padding, tsr->padding, sizeof(int)*tsr->order);
  *was_cyclic = tsr->is_cyclic;
  return CTF_SUCCESS;
}

/**
 * \brief saves the mapping of a tensor to a distribution data structure
 *        function allocates memory for the internal arrays, delete it later
 * \param[in] tsr tensor pointer
 * \param[in,out] dstrib object to save distribution data into
 * \param[out] old_rank rank of this processor along each dimension
 * \param[in] topo topology of the processor grid mapped to
 * \param[in] is_inner whether this is an inner mapping
 */ 
template<typename dtype>
int save_mapping(tensor<dtype> *  tsr,
                 distribution &   dstrib,
                 topology const * topo){
  dstrib.order = tsr->order;
  return save_mapping(tsr, &dstrib.phase, &dstrib.perank, &dstrib.virt_phase, 
                      &dstrib.pe_lda, &dstrib.size, &dstrib.is_cyclic, &dstrib.padding, &dstrib.edge_len, topo);
}

/**
 * \brief sets padding of a tensor
 * \param[in,out] tsr tensor in its new mapping
 * \param[in] is_inner whether this is an inner mapping
 */
template<typename dtype>
int set_padding(tensor<dtype> * tsr, int const is_inner=0){
  int j, pad, i;
  int * new_phase, * sub_edge_len;
  mapping * map;

  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&new_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&sub_edge_len);


  for (i=0; i<tsr->order; i++){
    tsr->edge_len[i] -= tsr->padding[i];
  }
/*  for (i=0; i<tsr->order; i++){
    printf("tensor edge len[%d] = %d\n",i,tsr->edge_len[i]);
    if (tsr->is_padded){
      printf("tensor padding was [%d] = %d\n",i,tsr->padding[i]);
    }
  }*/
  for (j=0; j<tsr->order; j++){
    map = tsr->edge_map + j;
    new_phase[j] = calc_phase(map);
    pad = tsr->edge_len[j]%new_phase[j];
    if (pad != 0) {
      pad = new_phase[j]-pad;
    }
    tsr->padding[j] = pad;
  }
  for (i=0; i<tsr->order; i++){
    tsr->edge_len[i] += tsr->padding[i];
    sub_edge_len[i] = tsr->edge_len[i]/new_phase[i];
  }
  tsr->size = calc_nvirt(tsr,is_inner)
    *sy_packed_size(tsr->order, sub_edge_len, tsr->sym);
  

  CTF_free(sub_edge_len);
  CTF_free(new_phase);
  return CTF_SUCCESS;
}


/**
 * \brief permutes the data of a tensor to its new layout
 * \param[in] tid tensor id
 * \param[in,out] tsr tensor in its new mapping
 * \param[in] old_size size of tensor before redistribution
 * \param[in] old_phase old distribution phase
 * \param[in] old_rank old distribution rank
 * \param[in] old_virt_dim old distribution virtualization
 * \param[in] old_pe_lda old distribution processor ldas
 * \param[in] old_padding what the padding was
 * \param[in] old_edge_len what the padded edge lengths were
 * \param[in] global_comm global communicator
 */
template<typename dtype>
int remap_tensor(int const  tid,
                 tensor<dtype> *tsr,
                 topology const * topo,
                 int64_t const old_size,
                 int const *  old_phase,
                 int const *  old_rank,
                 int const *  old_virt_dim,
                 int const *  old_pe_lda,
                 int const    was_cyclic,
                 int const *  old_padding,
                 int const *  old_edge_len,
                 CommData   global_comm){/*
                 int const *  old_offsets = NULL,
                 int * const * old_permutation = NULL,
                 int const *  new_offsets = NULL,
                 int * const * new_permutation = NULL){*/
}

/**
 * \brief does a permute add of data from one distribution to another
 * \param[in] sym symmetry of tensor
 * \param[in] cdt comm to redistribute on
 * \param[in] old_dist old distribution info
 * \param[in] old_data old data (data to add)
 * \param[in] alpha scaling factor of the data to add (old_data)
 * \param[in] new_dist new distribution info
 * \param[in] new_data new data (data to be accumulated to)
 * \param[in] beta scaling factor of the data to add (new_data)
 */
template<typename dtype>
int redistribute(int const *          sym,
                 CommData &         cdt,
                 distribution const & old_dist,
                 dtype *              old_data,
                 dtype                alpha,
                 distribution const & new_dist,
                 dtype *              new_data,
                 dtype                beta){

  return  cyclic_reshuffle(old_dist.order,
                           old_dist.size,
                           old_dist.edge_len,
                           sym,
                           old_dist.phase,
                           old_dist.perank,
                           old_dist.pe_lda,
                           old_dist.padding,
                           NULL,
                           NULL,
                           new_dist.edge_len,
                           new_dist.phase,
                           new_dist.perank,
                           new_dist.pe_lda,
                           new_dist.padding,
                           NULL,
                           NULL,
                           old_dist.virt_phase,
                           new_dist.virt_phase,
                           &old_data,
                           &new_data,
                           cdt,
                           old_dist.is_cyclic,
                           new_dist.is_cyclic,
                           false,
                           alpha,
                           beta);
}

#endif
