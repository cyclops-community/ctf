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
 * \brief sets up a ctr_2d_general (2D SUMMA) level where A is not communicated
 *        function will be called with A/B/C permuted depending on desired alg
 *
 * \param[in] is_used whether this ctr will actually be run
 * \param[in] global_comm comm for this CTF instance
 * \param[in] i index in the total index map currently worked on
 * \param[in,out] virt_dim virtual processor grid lengths
 * \param[out] cg_edge_len edge lengths of ctr_2d_gen object to set
 * \param[in,out] total_iter the total number of ctr_2d_gen iterations
 * \param[in] topovec vector of topologies
 * \param[in] tsr_A A tensor
 * \param[in] i_A the index in A to which index i corresponds
 * \param[out] cg_cdt_A the communicator for A to be set for ctr_2d_gen
 * \param[out] cg_ctr_lda_A parameter of ctr_2d_gen corresponding to upper lda for lda_cpy
 * \param[out] cg_ctr_sub_lda_A parameter of ctr_2d_gen corresponding to lower lda for lda_cpy
 * \param[out] cg_move_A tells ctr_2d_gen whether A should be communicated
 * \param[in,out] blk_len_A lengths of local A piece after this ctr_2d_gen level
 * \param[in,out] blk_sz_A size of local A piece after this ctr_2d_gen level
 * \param[in] virt_blk_edge_len_A edge lengths of virtual blocks of A
 * \param[in] load_phase_A tells the offloader how often A buffer changes for ctr_2d_gen
 *
 * ... the other parameters are specified the same as for _A but this time for _B and _C
 */
template<typename dtype>
int  ctr_2d_gen_build(int                     is_used,
                      CommData              global_comm,
                      int                     i,
                      int                   * virt_dim,
                      int                   & cg_edge_len,
                      int                   & total_iter,
                      std::vector<topology> & topovec,
                      tensor<dtype>         * tsr_A,
                      int                     i_A,
                      CommData            & cg_cdt_A,
                      int64_t               & cg_ctr_lda_A,
                      int64_t               & cg_ctr_sub_lda_A,
                      bool                  & cg_move_A,
                      int                   * blk_len_A,
                      int64_t               & blk_sz_A,
                      int const             * virt_blk_len_A,
                      int                   & load_phase_A,
                      tensor<dtype>         * tsr_B,
                      int                     i_B,
                      CommData            & cg_cdt_B,
                      int64_t               & cg_ctr_lda_B,
                      int64_t               & cg_ctr_sub_lda_B,
                      bool                  & cg_move_B,
                      int                   * blk_len_B,
                      int64_t               & blk_sz_B,
                      int const             * virt_blk_len_B,
                      int                   & load_phase_B,
                      tensor<dtype>         * tsr_C,
                      int                     i_C,
                      CommData            & cg_cdt_C,
                      int64_t               & cg_ctr_lda_C,
                      int64_t               & cg_ctr_sub_lda_C,
                      bool                  & cg_move_C,
                      int                   * blk_len_C,
                      int64_t               & blk_sz_C,
                      int const             * virt_blk_len_C,
                      int                   & load_phase_C){
  mapping * map;
  int j;
  int nstep = 1;
  if (comp_dim_map(&tsr_C->edge_map[i_C], &tsr_B->edge_map[i_B])){
    map = &tsr_B->edge_map[i_B];
    while (map->has_child) map = map->child;
    if (map->type == VIRTUAL_MAP){
      virt_dim[i] = map->np;
    }
    return 0;
  } else {
    if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP &&
      tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
      virt_dim[i] = tsr_B->edge_map[i_B].np;
      return 0;
    } else {
      cg_edge_len = 1;
      if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP){
        cg_edge_len = lcm(cg_edge_len, tsr_B->edge_map[i_B].np);
        cg_cdt_B = topovec[tsr_B->itopo].dim_comm[tsr_B->edge_map[i_B].cdt];
        if (is_used && cg_cdt_B.alive == 0)
          SHELL_SPLIT(global_comm, cg_cdt_B);
        nstep = tsr_B->edge_map[i_B].np;
        cg_move_B = 1;
      } else
        cg_move_B = 0;
      if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP){
        cg_edge_len = lcm(cg_edge_len, tsr_C->edge_map[i_C].np);
        cg_cdt_C = topovec[tsr_C->itopo].dim_comm[tsr_C->edge_map[i_C].cdt];
        if (is_used && cg_cdt_C.alive == 0)
          SHELL_SPLIT(global_comm, cg_cdt_C);
        nstep = MAX(nstep, tsr_C->edge_map[i_C].np);
        cg_move_C = 1;
      } else
        cg_move_C = 0;
      cg_ctr_lda_A = 1;
      cg_ctr_sub_lda_A = 0;
      cg_move_A = 0;


      /* Adjust the block lengths, since this algorithm will cut
         the block into smaller ones of the min block length */
      /* Determine the LDA of this dimension, based on virtualization */
      cg_ctr_lda_B  = 1;
      if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP)
        cg_ctr_sub_lda_B= blk_sz_B*tsr_B->edge_map[i_B].np/cg_edge_len;
      else
        cg_ctr_sub_lda_B= blk_sz_B/cg_edge_len;
      for (j=i_B+1; j<tsr_B->order; j++) {
        cg_ctr_sub_lda_B = (cg_ctr_sub_lda_B *
              virt_blk_len_B[j]) / blk_len_B[j];
        cg_ctr_lda_B = (cg_ctr_lda_B*blk_len_B[j])
              /virt_blk_len_B[j];
      }
      cg_ctr_lda_C  = 1;
      if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP)
        cg_ctr_sub_lda_C= blk_sz_C*tsr_C->edge_map[i_C].np/cg_edge_len;
      else
        cg_ctr_sub_lda_C= blk_sz_C/cg_edge_len;
      for (j=i_C+1; j<tsr_C->order; j++) {
        cg_ctr_sub_lda_C = (cg_ctr_sub_lda_C *
              virt_blk_len_C[j]) / blk_len_C[j];
        cg_ctr_lda_C = (cg_ctr_lda_C*blk_len_C[j])
              /virt_blk_len_C[j];
      }
      if (tsr_B->edge_map[i_B].type != PHYSICAL_MAP){
        blk_sz_B  = blk_sz_B / nstep;
        blk_len_B[i_B] = blk_len_B[i_B] / nstep;
      } else {
        blk_sz_B  = blk_sz_B * tsr_B->edge_map[i_B].np / nstep;
        blk_len_B[i_B] = blk_len_B[i_B] * tsr_B->edge_map[i_B].np / nstep;
      }
      if (tsr_C->edge_map[i_C].type != PHYSICAL_MAP){
        blk_sz_C  = blk_sz_C / nstep;
        blk_len_C[i_C] = blk_len_C[i_C] / nstep;
      } else {
        blk_sz_C  = blk_sz_C * tsr_C->edge_map[i_C].np / nstep;
        blk_len_C[i_C] = blk_len_C[i_C] * tsr_C->edge_map[i_C].np / nstep;
      }

      if (tsr_B->edge_map[i_B].has_child){
        ASSERT(tsr_B->edge_map[i_B].child->type == VIRTUAL_MAP);
        virt_dim[i] = tsr_B->edge_map[i_B].np*tsr_B->edge_map[i_B].child->np/nstep;
      }
      if (tsr_C->edge_map[i_C].has_child) {
        ASSERT(tsr_C->edge_map[i_C].child->type == VIRTUAL_MAP);
        virt_dim[i] = tsr_C->edge_map[i_C].np*tsr_C->edge_map[i_C].child->np/nstep;
      }
      if (tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
        virt_dim[i] = tsr_C->edge_map[i_C].np/nstep;
      }
      if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP)
        virt_dim[i] = tsr_B->edge_map[i_B].np/nstep;
#ifdef OFFLOAD
      total_iter *= nstep;
      if (cg_ctr_sub_lda_A == 0)
        load_phase_A *= nstep;
      else 
        load_phase_A  = 1;
      if (cg_ctr_sub_lda_B == 0)   
        load_phase_B *= nstep;
      else 
        load_phase_B  = 1;
      if (cg_ctr_sub_lda_C == 0) 
        load_phase_C *= nstep;
      else 
        load_phase_C  = 1;
#endif
    }
  } 
  return 1;
}


/**
 * \brief stretch virtualization by a factor
 * \param[in] order number of maps to stretch
 * \param[in] stretch_factor factor to strech by
 * \param[in] maps mappings along each dimension to stretch
 */
inline 
int stretch_virt(int const order,
     int const stretch_factor,
     mapping * maps){
  int i;
  mapping * map;
  for (i=0; i<order; i++){
    map = &maps[i];
    while (map->has_child) map = map->child;
    if (map->type == PHYSICAL_MAP){
      if (map->has_child){
        map->has_child    = 1;
        map->child    = (mapping*)CTF_alloc(sizeof(mapping));
        map->child->type  = VIRTUAL_MAP;
        map->child->np    = stretch_factor;
        map->child->has_child   = 0;
      }
    } else if (map->type == VIRTUAL_MAP){
      map->np = map->np * stretch_factor;
    } else {
      map->type = VIRTUAL_MAP;
      map->np   = stretch_factor;
    }
  }
  return CTF_SUCCESS;
}


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

/**
 * \brief extracts the set of physical dimensions still available for mapping
 * \param[in] topo topology
 * \param[in] order_A dimension of A
 * \param[in] edge_map_A mapping of A
 * \param[in] order_B dimension of B
 * \param[in] edge_map_B mapping of B
 * \param[out] num_sub_phys_dims number of free torus dimensions
 * \param[out] sub_phys_comm the torus dimensions
 * \param[out] comm_idx index of the free torus dimensions in the origin topology
 */
void extract_free_comms(topology const *  topo,
                        int               order_A,
                        mapping const *   edge_map_A,
                        int               order_B,
                        mapping const *   edge_map_B,
                        int &             num_sub_phys_dims,
                        CommData *  *     psub_phys_comm,
                        int **            pcomm_idx){
  int i;
  int phys_mapped[topo->order];
  CommData *   sub_phys_comm;
  int * comm_idx;
  mapping const * map;
  memset(phys_mapped, 0, topo->order*sizeof(int));  
  
  num_sub_phys_dims = 0;

  for (i=0; i<order_A; i++){
    map = &edge_map_A[i];
    while (map->type == PHYSICAL_MAP){
      phys_mapped[map->cdt] = 1;
      if (map->has_child) map = map->child;
      else break;
    } 
  }
  for (i=0; i<order_B; i++){
    map = &edge_map_B[i];
    while (map->type == PHYSICAL_MAP){
      phys_mapped[map->cdt] = 1;
      if (map->has_child) map = map->child;
      else break;
    } 
  }

  num_sub_phys_dims = 0;
  for (i=0; i<topo->order; i++){
    if (phys_mapped[i] == 0){
      num_sub_phys_dims++;
    }
  }
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
  num_sub_phys_dims = 0;
  for (i=0; i<topo->order; i++){
    if (phys_mapped[i] == 0){
      sub_phys_comm[num_sub_phys_dims] = topo->dim_comm[i];
      comm_idx[num_sub_phys_dims] = i;
      num_sub_phys_dims++;
    }
  }
  *pcomm_idx = comm_idx;
  *psub_phys_comm = sub_phys_comm;

}

/**
 * \brief determines if two topologies are compatible with each other
 * \param topo_keep topology to keep (larger dimension)
 * \param topo_change topology to change (smaller dimension)
 * \return true if its possible to change
 */
inline 
int can_morph(topology const * topo_keep, topology const * topo_change){
  int i, j, lda;
  lda = 1;
  j = 0;
  for (i=0; i<topo_keep->order; i++){
    lda *= topo_keep->dim_comm[i].np;
    if (lda == topo_change->dim_comm[j].np){
      j++;
      lda = 1;
    } else if (lda > topo_change->dim_comm[j].np){
      return 0;
    }
  }
  return 1;
}

/**
 * \brief morphs a tensor topology into another
 * \param[in] new_topo topology to change to
 * \param[in] old_topo topology we are changing from
 * \param[in] order number of tensor dimensions
 * \param[in,out] edge_map mapping whose topology mapping we are changing
 */
inline 
void morph_topo(topology const *  new_topo, 
    topology const *  old_topo, 
    int const     order,
    mapping *     edge_map){
  int i,j,old_lda,new_np;
  mapping * old_map, * new_map, * new_rec_map;

  for (i=0; i<order; i++){
    if (edge_map[i].type == PHYSICAL_MAP){
      old_map = &edge_map[i];
      CTF_alloc_ptr(sizeof(mapping), (void**)&new_map);
      new_rec_map = new_map;
      for (;;){
        old_lda = old_topo->lda[old_map->cdt];
        new_np = 1;
        do {
          for (j=0; j<new_topo->order; j++){
            if (new_topo->lda[j] == old_lda) break;
          } 
          ASSERT(j!=new_topo->order);
          new_rec_map->type   = PHYSICAL_MAP;
          new_rec_map->cdt    = j;
          new_rec_map->np     = new_topo->dim_comm[j].np;
          new_np    *= new_rec_map->np;
          if (new_np<old_map->np) {
            old_lda = old_lda * new_rec_map->np;
            new_rec_map->has_child = 1;
            CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
            new_rec_map = new_rec_map->child;
          }
        } while (new_np<old_map->np);

        if (old_map->has_child){
          if (old_map->child->type == VIRTUAL_MAP){
            new_rec_map->has_child = 1;
            CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
            new_rec_map->child->type  = VIRTUAL_MAP;
            new_rec_map->child->np    = old_map->child->np;
            new_rec_map->child->has_child   = 0;
            break;
          } else {
            new_rec_map->has_child = 1;
            CTF_alloc_ptr(sizeof(mapping), (void**)&new_rec_map->child);
            new_rec_map = new_rec_map->child;
            old_map = old_map->child;
            //continue
          }
        } else {
          new_rec_map->has_child = 0;
          break;
        }
      }
      clear_mapping(&edge_map[i]);      
      edge_map[i] = *new_map;
      CTF_free(new_map);
    }
  }
}
#endif
