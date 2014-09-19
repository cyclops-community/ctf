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
 * \brief compute the phase of a mapping
 *
 * \param map mapping
 * \return int phase
 */
inline
int calc_phase(mapping const * map){
  int phase;
  if (map->type == NOT_MAPPED){
    phase = 1;
  } else{
    phase = map->np;
#if DEBUG >1
    if (phase == 0)
      printf("phase should never be zero! map type = %d\n",map->type);
    LIBT_ASSERT(phase!=0);
#endif
    if (map->has_child){
      phase = phase*calc_phase(map->child);
    } 
  } 
  return phase;  
}

/**
 * \brief compute the physical phase of a mapping
 *
 * \param map mapping
 * \return int physical phase
 */
inline
int calc_phys_phase(mapping const * map){
  int phase;
  if (map->type == NOT_MAPPED){
    phase = 1;
  } else {
    if (map->type == PHYSICAL_MAP)
      phase = map->np;
    else
      phase = 1;
    if (map->has_child){
      phase = phase*calc_phys_phase(map->child);
    } 
  }
  return phase;
}


/**
 * \brief compute the physical rank of a mapping
 *
 * \param map mapping
 * \param topo topology
 * \return int physical rank
 */
inline
int calc_phys_rank(mapping const * map, topology const * topo){
  int rank, phase;
  if (map->type == NOT_MAPPED){
    rank = 0;
  } else {
    if (map->type == PHYSICAL_MAP) {
      rank = topo->dim_comm[map->cdt].rank;
      phase = map->np;
    } else {
      rank = 0;
      phase = 1;
    }
    if (map->has_child){
      /* WARNING: Assumes folding is ordered by lda */
      rank = rank + phase*calc_phys_rank(map->child, topo);
    } 
  }
  return rank;
}


/**
 * \brief compute the cyclic phase of each tensor dimension
 *
 * \param tsr tensor
 * \return int * of cyclic phases
 */
template<typename dtype>
int * calc_phase(tensor<dtype> const * tsr){
  mapping * map;
  int * phase;
  int i;
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&phase);
  for (i=0; i<tsr->ndim; i++){
    map = tsr->edge_map + i;
    phase[i] = calc_phase(map);
  }
  return phase;  
}

/**
 * \brief calculate the total number of blocks of the tensor
 *
 * \param tsr tensor
 * \return int total phase factor
 */
template<typename dtype>
int calc_tot_phase(tensor<dtype> const * tsr){
  int i, tot_phase;
  int * phase = calc_phase(tsr);
  tot_phase = 1;
  for (i=0 ; i<tsr->ndim; i++){
    tot_phase *= phase[i];
  }
  CTF_free(phase);
  return tot_phase;
}

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
      for (j=i_B+1; j<tsr_B->ndim; j++) {
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
      for (j=i_C+1; j<tsr_C->ndim; j++) {
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
        LIBT_ASSERT(tsr_B->edge_map[i_B].child->type == VIRTUAL_MAP);
        virt_dim[i] = tsr_B->edge_map[i_B].np*tsr_B->edge_map[i_B].child->np/nstep;
      }
      if (tsr_C->edge_map[i_C].has_child) {
        LIBT_ASSERT(tsr_C->edge_map[i_C].child->type == VIRTUAL_MAP);
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
 * \param[in] ndim number of maps to stretch
 * \param[in] stretch_factor factor to strech by
 * \param[in] maps mappings along each dimension to stretch
 */
inline 
int stretch_virt(int const ndim,
     int const stretch_factor,
     mapping * maps){
  int i;
  mapping * map;
  for (i=0; i<ndim; i++){
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
 * \brief get the best topologoes (least nvirt) over all procs
 * \param[in] nvirt best virtualization achieved by this proc
 * \param[in] topo topology index corresponding to best virtualization
 * \param[in] globla_comm is the global communicator
 * return virtualization factor
 */
inline
int get_best_topo(uint64_t const  nvirt,
		  int const       topo,
		  CommData      global_comm,
		  uint64_t const  bcomm_vol = 0,
		  uint64_t const  bmemuse = 0){

  uint64_t gnvirt, nv, gcomm_vol, gmemuse, bv;
  int btopo, gtopo;
  nv = nvirt;
  ALLREDUCE(&nv, &gnvirt, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, global_comm);

  LIBT_ASSERT(gnvirt <= nvirt);

  nv = bcomm_vol;
  bv = bmemuse;
  if (nvirt == gnvirt){
    btopo = topo;
  } else {
    btopo = INT_MAX;
    nv = UINT64_MAX;
    bv = UINT64_MAX;
  }
  ALLREDUCE(&nv, &gcomm_vol, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, global_comm);
  if (bcomm_vol != gcomm_vol){
    btopo = INT_MAX;
    bv = UINT64_MAX;
  }
  ALLREDUCE(&bv, &gmemuse, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, global_comm);
  if (bmemuse != gmemuse){
    btopo = INT_MAX;
  }
  ALLREDUCE(&btopo, &gtopo, 1, MPI_INT, MPI_MIN, global_comm);
  /*printf("nvirt = "PRIu64" bcomm_vol = "PRIu64" bmemuse = "PRIu64" topo = %d\n",
    nvirt, bcomm_vol, bmemuse, topo);
  printf("gnvirt = "PRIu64" gcomm_vol = "PRIu64" gmemuse = "PRIu64" bv = "PRIu64" nv = "PRIu64" gtopo = %d\n",
    gnvirt, gcomm_vol, gmemuse, bv, nv, gtopo);*/

  return gtopo;
}

/**
 * \brief calculate virtualization factor of tensor
 * \param[in] tsr tensor
 * return virtualization factor
 */
template<typename dtype>
uint64_t calc_nvirt(tensor<dtype> * tsr, int const is_inner=0){
  int j;
  uint64_t nvirt, tnvirt;
  mapping * map;
  nvirt = 1;
  if (is_inner) return calc_tot_phase(tsr);
  for (j=0; j<tsr->ndim; j++){
    map = &tsr->edge_map[j];
    while (map->has_child) map = map->child;
    if (map->type == VIRTUAL_MAP){
      tnvirt = nvirt*(uint64_t)map->np;
      if (tnvirt < nvirt) return UINT64_MAX;
      else nvirt = tnvirt;
    }
  }
  return nvirt;  
}

/** 
 * \brief clears mapping and frees children
 * \param map mapping
 */
inline
int clear_mapping(mapping * map){
  if (map->type != NOT_MAPPED && map->has_child) {
    clear_mapping(map->child);
    CTF_free(map->child);
  }
  map->type = NOT_MAPPED;
  map->np = 1;
  map->has_child = 0;
  return CTF_SUCCESS;
}

/* \brief zeros out mapping
 * \param[in] tsr tensor to clear mapping of
 */
template<typename dtype>
int clear_mapping(tensor<dtype> * tsr){
  int j;
  mapping * map;
  for (j=0; j<tsr->ndim; j++){
    map = tsr->edge_map + j;
    clear_mapping(map);
  }
  tsr->itopo = -1;
  tsr->is_mapped = 0;
  tsr->is_folded = 0;

  return CTF_SUCCESS;
}

/**
 * \brief copies mapping A to B
 * \param[in] ndim number of dimensions
 * \param[in] mapping_A mapping to copy from 
 * \param[in,out] mapping_B mapping to copy to
 */
inline
int copy_mapping(int const        ndim,
                 mapping const *  mapping_A,
                 mapping *        mapping_B){
  int i;
  for (i=0; i<ndim; i++){
    clear_mapping(&mapping_B[i]);
  }
  memcpy(mapping_B, mapping_A, sizeof(mapping)*ndim);
  for (i=0; i<ndim; i++){
    if (mapping_A[i].has_child){
      CTF_alloc_ptr(sizeof(mapping), (void**)&mapping_B[i].child);
      mapping_B[i].child->has_child   = 0;
      mapping_B[i].child->np    = 1;
      mapping_B[i].child->type    = NOT_MAPPED;
      copy_mapping(1, mapping_A[i].child, mapping_B[i].child);
    }
  }
  return CTF_SUCCESS;
}

/**
 * \brief copies mapping A to B
 * \param[in] ndim_A number of dimensions in A
 * \param[in] ndim_B number of dimensions in B
 * \param[in] idx_A index mapping of A
 * \param[in] mapping_A mapping to copy from 
 * \param[in] idx_B index mapping of B
 * \param[in,out] mapping_B mapping to copy to
 */
inline
int copy_mapping(int const    ndim_A,
                 int const    ndim_B,
                 int const *    idx_A,
                 mapping const *  mapping_A,
                 int const *    idx_B,
                 mapping *    mapping_B,
                 int const    make_virt = 1){
  int i, ndim_tot, iA, iB;
  int * idx_arr;


  inv_idx(ndim_A, idx_A, mapping_A,
          ndim_B, idx_B, mapping_B,
          &ndim_tot, &idx_arr);

  for (i=0; i<ndim_tot; i++){
    iA = idx_arr[2*i];
    iB = idx_arr[2*i+1];
    if (iA == -1){
      if (make_virt){
        LIBT_ASSERT(iB!=-1);
        mapping_B[iB].type = VIRTUAL_MAP;
        mapping_B[iB].np = 1;
        mapping_B[iB].has_child = 0;
      }
    } else if (iB != -1){
      clear_mapping(&mapping_B[iB]);
      copy_mapping(1, mapping_A+iA, mapping_B+iB);
    }
  }
  CTF_free(idx_arr);
  return CTF_SUCCESS;
}

/** \brief compares two mappings
 * \param map_A first map
 * \param map_B second map
 * return true if mapping is exactly the same, false otherwise 
 */
inline
int comp_dim_map(mapping const *  map_A,
                 mapping const *  map_B){
  if (map_A->type == NOT_MAPPED &&
      map_B->type == NOT_MAPPED) return 1;
/*  if (map_A->type == NOT_MAPPED ||
      map_B->type == NOT_MAPPED) return 0;*/
  if (map_A->type == NOT_MAPPED){
    if (map_B->type == VIRTUAL_MAP && map_B->np == 1) return 1;
    else return 0;
  }
  if (map_B->type == NOT_MAPPED){
    if (map_A->type == VIRTUAL_MAP && map_A->np == 1) return 1;
    else return 0;
  }

  if (map_A->type == PHYSICAL_MAP){
    if (map_B->type != PHYSICAL_MAP || 
      map_B->cdt != map_A->cdt ||
      map_B->np != map_A->np){
/*      DEBUG_PRINTF("failed confirmation here [%d %d %d] != [%d] [%d] [%d]\n",
       map_A->type, map_A->cdt, map_A->np,
       map_B->type, map_B->cdt, map_B->np);*/
      return 0;
    }
    if (map_A->has_child){
      if (map_B->has_child != 1 || 
          comp_dim_map(map_A->child, map_B->child) != 1){ 
        DEBUG_PRINTF("failed confirmation here\n");
        return 0;
      }
    } else {
      if (map_B->has_child){
        DEBUG_PRINTF("failed confirmation here\n");
        return 0;
      }
    }
  } else {
    LIBT_ASSERT(map_A->type == VIRTUAL_MAP);
    if (map_B->type != VIRTUAL_MAP ||
        map_B->np != map_A->np) {
      DEBUG_PRINTF("failed confirmation here\n");
      return 0;
    }
  }
  return 1;
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
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_rank);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_virt_dim);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_pe_lda);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_edge_len);
  
  *old_size = tsr->size;
  
  for (j=0; j<tsr->ndim; j++){
    map     = tsr->edge_map + j;
    (*old_phase)[j]   = calc_phase(map);
    (*old_rank)[j]  = calc_phys_rank(map, topo);
    (*old_virt_dim)[j]  = (*old_phase)[j]/calc_phys_phase(map);
    if (!is_inner && map->type == PHYSICAL_MAP)
      (*old_pe_lda)[j]  = topo->lda[map->cdt];
    else
      (*old_pe_lda)[j]  = 0;
  }
  memcpy(*old_edge_len, tsr->edge_len, sizeof(int)*tsr->ndim);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)old_padding);
  memcpy(*old_padding, tsr->padding, sizeof(int)*tsr->ndim);
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
  dstrib.ndim = tsr->ndim;
  return save_mapping(tsr, &dstrib.phase, &dstrib.perank, &dstrib.virt_phase, 
                      &dstrib.pe_lda, &dstrib.size, &dstrib.is_cyclic, &dstrib.padding, &dstrib.edge_len, topo);
}

/**
 * \brief adjust a mapping to maintan symmetry
 * \param[in] tsr_ndim is the number of dimensions of the tensor
 * \param[in] tsr_sym_table the symmetry table of a tensor
 * \param[in,out] tsr_edge_map is the mapping
 * \return CTF_SUCCESS if mapping successful, CTF_NEGATIVE if not, 
 *     CTF_ERROR if err'ed out
 */
inline
int map_symtsr(int const    tsr_ndim,
               int const *    tsr_sym_table,
               mapping *    tsr_edge_map){
  int i,j,phase,adj,loop,sym_phase,lcm_phase;
  mapping * sym_map, * map;

  /* Make sure the starting mappings are consistent among symmetries */
  adj = 1;
  loop = 0;
  while (adj){
    adj = 0;
#ifndef MAXLOOP
#define MAXLOOP 20
#endif
    if (loop >= MAXLOOP) return CTF_NEGATIVE;
    loop++;
    for (i=0; i<tsr_ndim; i++){
      if (tsr_edge_map[i].type != NOT_MAPPED){
        map   = &tsr_edge_map[i];
        phase   = calc_phase(map);
        for (j=0; j<tsr_ndim; j++){
          if (i!=j && tsr_sym_table[i*tsr_ndim+j] == 1){
            sym_map   = &(tsr_edge_map[j]);
            sym_phase   = calc_phase(sym_map);
            /* Check if symmetric phase inconsitent */
            if (sym_phase != phase) adj = 1;
            else continue;
            lcm_phase   = lcm(sym_phase, phase);
            if ((lcm_phase < sym_phase || lcm_phase < phase) || lcm_phase >= MAX_PHASE)
              return CTF_NEGATIVE;
            /* Adjust phase of symmetric (j) dimension */
            if (sym_map->type == NOT_MAPPED){
              sym_map->type     = VIRTUAL_MAP;
              sym_map->np   = lcm_phase;
              sym_map->has_child  = 0;
            } else if (sym_phase != lcm_phase) { 
              while (sym_map->has_child) sym_map = sym_map->child;
              if (sym_map->type == VIRTUAL_MAP){
                sym_map->np = sym_map->np*(lcm_phase/sym_phase);
              } else {
                LIBT_ASSERT(sym_map->type == PHYSICAL_MAP);
                if (!sym_map->has_child)
                  sym_map->child    = (mapping*)CTF_alloc(sizeof(mapping));
                sym_map->has_child  = 1;
                sym_map->child->type    = VIRTUAL_MAP;
                sym_map->child->np    = lcm_phase/sym_phase;

                sym_map->child->has_child = 0;
              }
            }
            /* Adjust phase of reference (i) dimension if also necessary */
            if (lcm_phase > phase){
              while (map->has_child) map = map->child;
              if (map->type == VIRTUAL_MAP){
                map->np = map->np*(lcm_phase/phase);
              } else {
                if (!map->has_child)
                  map->child    = (mapping*)CTF_alloc(sizeof(mapping));
                LIBT_ASSERT(map->type == PHYSICAL_MAP);
                map->has_child    = 1;
                map->child->type  = VIRTUAL_MAP;
                map->child->np    = lcm_phase/phase;
                map->child->has_child = 0;
              }
            }
          }
        }
      }
    }
  }
  return CTF_SUCCESS;
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

  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&new_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&sub_edge_len);


  for (i=0; i<tsr->ndim; i++){
    tsr->edge_len[i] -= tsr->padding[i];
  }
/*  for (i=0; i<tsr->ndim; i++){
    printf("tensor edge len[%d] = %d\n",i,tsr->edge_len[i]);
    if (tsr->is_padded){
      printf("tensor padding was [%d] = %d\n",i,tsr->padding[i]);
    }
  }*/
  for (j=0; j<tsr->ndim; j++){
    map = tsr->edge_map + j;
    new_phase[j] = calc_phase(map);
    pad = tsr->edge_len[j]%new_phase[j];
    if (pad != 0) {
      pad = new_phase[j]-pad;
    }
    tsr->padding[j] = pad;
  }
  for (i=0; i<tsr->ndim; i++){
    tsr->edge_len[i] += tsr->padding[i];
    sub_edge_len[i] = tsr->edge_len[i]/new_phase[i];
  }
  tsr->size = calc_nvirt(tsr,is_inner)
    *sy_packed_size(tsr->ndim, sub_edge_len, tsr->sym);
  

  CTF_free(sub_edge_len);
  CTF_free(new_phase);
  return CTF_SUCCESS;
}


/**
 * \brief determines if tensor can be permuted by block
 * \param[in] ndim dimension of tensor
 * \param[in] old_phase old cyclic phases in each dimension
 * \param[in] map new mapping for each edge length
 * \return 1 if block reshuffle allowed, 0 if not
 */
inline int can_block_reshuffle(int const        ndim,
                               int const *      old_phase,
                               mapping const *  map){
  int new_phase, j;
  int can_block_resh = 1;
  for (j=0; j<ndim; j++){
    new_phase  = calc_phase(map+j);
    if (new_phase != old_phase[j]) can_block_resh = 0;
  }
  return can_block_resh;
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
  int const *  old_offsets = NULL;
  int * const * old_permutation = NULL;
  int const *  new_offsets = NULL;
  int * const * new_permutation = NULL;
  int j, new_nvirt, can_block_shuffle;
  int * new_phase, * new_rank, * new_virt_dim, * new_pe_lda;
  mapping * map;
  dtype * shuffled_data;
#if VERIFY_REMAP
  dtype * shuffled_data_corr;
#endif


  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&new_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&new_rank);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&new_pe_lda);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&new_virt_dim);

  new_nvirt = 1;  
#ifdef USE_BLOCK_RESHUFFLE
  can_block_shuffle = can_block_reshuffle(tsr->ndim, old_phase, tsr->edge_map);
#else
  can_block_shuffle = 0;
#endif
  if (old_offsets != NULL || old_permutation != NULL ||
      new_offsets != NULL || new_permutation != NULL){
    can_block_shuffle = 0;
  }

  for (j=0; j<tsr->ndim; j++){
    map     = tsr->edge_map + j;
    new_phase[j]  = calc_phase(map);
    new_rank[j]   = calc_phys_rank(map, topo);
    new_virt_dim[j] = new_phase[j]/calc_phys_phase(map);
    if (map->type == PHYSICAL_MAP)
      new_pe_lda[j] = topo->lda[map->cdt];
    else
      new_pe_lda[j] = 0;
    new_nvirt = new_nvirt*new_virt_dim[j];
  }
#ifdef HOME_CONTRACT
  if (tsr->is_home){    
    if (global_comm.rank == 0)
      DPRINTF(2,"Tensor %d leaving home\n", tid);
    tsr->data = (dtype*)CTF_mst_alloc(old_size*sizeof(dtype));
    memcpy(tsr->data, tsr->home_buffer, old_size*sizeof(dtype));
    tsr->is_home = 0;
  }
#endif
  if (tsr->profile) {
    char spf[80];
    strcpy(spf,"redistribute_");
    strcat(spf,tsr->name);
    if (global_comm.rank == 0){
      if (can_block_shuffle) VPRINTF(1,"Remapping tensor %s via block_reshuffle\n",tsr->name);
      else VPRINTF(1,"Remapping tensor %s via cyclic_reshuffle\n",tsr->name);
#if VERBOSE >=1
      tsr->print_map(stdout);
#endif
    }
    CTF_Timer t_pf(spf);
    t_pf.start();
  }

#if VERIFY_REMAP
    padded_reshuffle(tid,
                     tsr->ndim,
                     old_size,
                     old_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     old_padding,
                     tsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     tsr->padding,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     &shuffled_data_corr,
                     global_comm);
#endif

  if (can_block_shuffle){
    block_reshuffle( tsr->ndim,
                     old_phase,
                     old_size,
                     old_virt_dim,
                     old_rank,
                     old_pe_lda,
                     tsr->size,
                     new_virt_dim,
                     new_rank,
                     new_pe_lda,
                     tsr->data,
                     shuffled_data,
                     global_comm);
  } else {
//    CTF_alloc_ptr(sizeof(dtype)*tsr->size, (void**)&shuffled_data);
    cyclic_reshuffle(tsr->ndim,
                     old_size,
                     old_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     old_padding,
                     old_offsets,
                     old_permutation,
                     tsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     tsr->padding,
                     new_offsets,
                     new_permutation,
                     old_virt_dim,
                     new_virt_dim,
                     &tsr->data,
                     &shuffled_data,
                     global_comm,
                     was_cyclic,
                     tsr->is_cyclic, 1, get_one<dtype>(), get_zero<dtype>());
  }

  CTF_free((void*)new_phase);
  CTF_free((void*)new_rank);
  CTF_free((void*)new_virt_dim);
  CTF_free((void*)new_pe_lda);
  CTF_free((void*)tsr->data);
  tsr->data = shuffled_data;

#if VERIFY_REMAP
  bool abortt = false;
  for (j=0; j<tsr->size; j++){
    if (tsr->data[j] != shuffled_data_corr[j]){
      printf("data element %d/"PRId64" not received correctly on process %d\n",
              j, tsr->size, global_comm.rank);
      printf("element received was %.3E, correct %.3E\n", 
              GET_REAL(tsr->data[j]), GET_REAL(shuffled_data_corr[j]));
      abortt = true;
    }
  }
  if (abortt) ABORT;
  CTF_free(shuffled_data_corr);

#endif
  if (tsr->profile) {
    char spf[80];
    strcpy(spf,"redistribute_");
    strcat(spf,tsr->name);
    CTF_Timer t_pf(spf);
    t_pf.stop();
  }


  return CTF_SUCCESS;
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

  return  cyclic_reshuffle(old_dist.ndim,
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
inline
int map_tensor(int const      num_phys_dims,
               int const      tsr_ndim,
               int const *    tsr_edge_len,
               int const *    tsr_sym_table,
               int *          restricted,
               CommData  *  phys_comm,
               int const *    comm_idx,
               int const      fill,
               mapping *      tsr_edge_map){
  int i,j,max_dim,max_len,phase,ret;
  mapping * map;

  /* Make sure the starting mappings are consistent among symmetries */
  ret = map_symtsr(tsr_ndim, tsr_sym_table, tsr_edge_map);
  if (ret!=CTF_SUCCESS) return ret;

  /* Assign physical dimensions */
  for (i=0; i<num_phys_dims; i++){
    max_len = -1;
    max_dim = -1;
    for (j=0; j<tsr_ndim; j++){
      if (tsr_edge_len[j]/calc_phys_phase(tsr_edge_map+j) > max_len) {
        /* if tsr dimension can be mapped */
        if (!restricted[j]){
          /* if tensor dimension not mapped ot physical dimension or
             mapped to a physical dimension that can be folded with
             this one */
          if (tsr_edge_map[j].type != PHYSICAL_MAP || 
              (fill && ((comm_idx == NULL && tsr_edge_map[j].cdt == i-1) ||
              (comm_idx != NULL && tsr_edge_map[j].cdt == comm_idx[i]-1)))){
            max_dim = j;  
            max_len = tsr_edge_len[j]/calc_phys_phase(tsr_edge_map+j);
          }
        } 
      }
    }
    if (max_dim == -1){
      if (fill){
        return CTF_NEGATIVE;
      }
      break;
    }
    map   = &(tsr_edge_map[max_dim]);
// FIXME: why?
  //  map->has_child  = 0;
    if (map->type != NOT_MAPPED){
      while (map->has_child) map = map->child;
      phase   = phys_comm[i].np;
      if (map->type == VIRTUAL_MAP){
        if (phys_comm[i].np != map->np){
          phase     = lcm(map->np, phys_comm[i].np);
          if ((phase < map->np || phase < phys_comm[i].np) || phase >= MAX_PHASE)
            return CTF_NEGATIVE;
          if (phase/phys_comm[i].np != 1){
            map->has_child  = 1;
            map->child    = (mapping*)CTF_alloc(sizeof(mapping));
            map->child->type  = VIRTUAL_MAP;
            map->child->np  = phase/phys_comm[i].np;
            map->child->has_child = 0;
          }
        }
      } else if (map->type == PHYSICAL_MAP){
        if (map->has_child != 1)
          map->child  = (mapping*)CTF_alloc(sizeof(mapping));
        map->has_child  = 1;
        map             = map->child;
        map->has_child  = 0;
      }
    }
    map->type     = PHYSICAL_MAP;
    map->np     = phys_comm[i].np;
    map->cdt    = (comm_idx == NULL) ? i : comm_idx[i];
    if (!fill)
      restricted[max_dim] = 1;
    ret = map_symtsr(tsr_ndim, tsr_sym_table, tsr_edge_map);
    if (ret!=CTF_SUCCESS) return ret;
  }
  for (i=0; i<tsr_ndim; i++){
    if (tsr_edge_map[i].type == NOT_MAPPED){
      tsr_edge_map[i].type        = VIRTUAL_MAP;
      tsr_edge_map[i].np          = 1;
      tsr_edge_map[i].has_child   = 0;
    }
  }
  return CTF_SUCCESS;
}

/**
* \brief map the remainder of a tensor 
* \param[in] num_phys_dims number of physical processor grid dimensions
* \param[in] phys_comm dimensional communicators
* \param[in,out] tsr pointer to tensor
*/
template<typename dtype>
int map_tensor_rem(int const    num_phys_dims,
                   CommData  *  phys_comm,
                   tensor<dtype> *  tsr,
                   int const    fill = 0){
  int i, num_sub_phys_dims, stat;
  int * restricted, * phys_mapped, * comm_idx;
  CommData  * sub_phys_comm;
  mapping * map;

  CTF_alloc_ptr(tsr->ndim*sizeof(int), (void**)&restricted);
  CTF_alloc_ptr(num_phys_dims*sizeof(int), (void**)&phys_mapped);

  memset(phys_mapped, 0, num_phys_dims*sizeof(int));  

  for (i=0; i<tsr->ndim; i++){
    restricted[i] = (tsr->edge_map[i].type != NOT_MAPPED);
    map = &tsr->edge_map[i];
    while (map->type == PHYSICAL_MAP){
      phys_mapped[map->cdt] = 1;
      if (map->has_child) map = map->child;
      else break;
    } 
  }

  num_sub_phys_dims = 0;
  for (i=0; i<num_phys_dims; i++){
    if (phys_mapped[i] == 0){
      num_sub_phys_dims++;
    }
  }
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
  num_sub_phys_dims = 0;
  for (i=0; i<num_phys_dims; i++){
    if (phys_mapped[i] == 0){
      sub_phys_comm[num_sub_phys_dims] = phys_comm[i];
      comm_idx[num_sub_phys_dims] = i;
      num_sub_phys_dims++;
    }
  }
  stat = map_tensor(num_sub_phys_dims,  tsr->ndim,
                    tsr->edge_len,  tsr->sym_table,
                    restricted,   sub_phys_comm,
                    comm_idx,     fill,
                    tsr->edge_map);
  CTF_free(restricted);
  CTF_free(phys_mapped);
  CTF_free(sub_phys_comm);
  CTF_free(comm_idx);
  return stat;
}

/**
 * \brief extracts the set of physical dimensions still available for mapping
 * \param[in] topo topology
 * \param[in] ndim_A dimension of A
 * \param[in] edge_map_A mapping of A
 * \param[in] ndim_B dimension of B
 * \param[in] edge_map_B mapping of B
 * \param[out] num_sub_phys_dims number of free torus dimensions
 * \param[out] sub_phys_comm the torus dimensions
 * \param[out] comm_idx index of the free torus dimensions in the origin topology
 */
void extract_free_comms(topology const *  topo,
                        int               ndim_A,
                        mapping const *   edge_map_A,
                        int               ndim_B,
                        mapping const *   edge_map_B,
                        int &             num_sub_phys_dims,
                        CommData *  *     psub_phys_comm,
                        int **            pcomm_idx){
  int i;
  int phys_mapped[topo->ndim];
  CommData *   sub_phys_comm;
  int * comm_idx;
  mapping const * map;
  memset(phys_mapped, 0, topo->ndim*sizeof(int));  
  
  num_sub_phys_dims = 0;

  for (i=0; i<ndim_A; i++){
    map = &edge_map_A[i];
    while (map->type == PHYSICAL_MAP){
      phys_mapped[map->cdt] = 1;
      if (map->has_child) map = map->child;
      else break;
    } 
  }
  for (i=0; i<ndim_B; i++){
    map = &edge_map_B[i];
    while (map->type == PHYSICAL_MAP){
      phys_mapped[map->cdt] = 1;
      if (map->has_child) map = map->child;
      else break;
    } 
  }

  num_sub_phys_dims = 0;
  for (i=0; i<topo->ndim; i++){
    if (phys_mapped[i] == 0){
      num_sub_phys_dims++;
    }
  }
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(CommData), (void**)&sub_phys_comm);
  CTF_alloc_ptr(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
  num_sub_phys_dims = 0;
  for (i=0; i<topo->ndim; i++){
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
  for (i=0; i<topo_keep->ndim; i++){
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
 * \param[in] ndim number of tensor dimensions
 * \param[in,out] edge_map mapping whose topology mapping we are changing
 */
inline 
void morph_topo(topology const *  new_topo, 
    topology const *  old_topo, 
    int const     ndim,
    mapping *     edge_map){
  int i,j,old_lda,new_np;
  mapping * old_map, * new_map, * new_rec_map;

  for (i=0; i<ndim; i++){
    if (edge_map[i].type == PHYSICAL_MAP){
      old_map = &edge_map[i];
      CTF_alloc_ptr(sizeof(mapping), (void**)&new_map);
      new_rec_map = new_map;
      for (;;){
        old_lda = old_topo->lda[old_map->cdt];
        new_np = 1;
        do {
          for (j=0; j<new_topo->ndim; j++){
            if (new_topo->lda[j] == old_lda) break;
          } 
          LIBT_ASSERT(j!=new_topo->ndim);
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

/**
 * \brief extracts the diagonal of a tensor if the index map specifies to do so
 * \param[in] tid id of tensor
 * \param[in] idx_map index map of tensor for this operation
 * \param[in] rw if 1 this writes to the diagonal, if 0 it reads the diagonal
 * \param[in,out] tid_new if rw=1 this will be output as new tid
                          if rw=0 this should be input as the tid of the extracted diagonal 
 * \param[out] idx_map_new if rw=1 this will be the new index map

 */
template<typename dtype>
int dist_tensor<dtype>::extract_diag(int const    tid,
                                     int const *  idx_map,
                                     int const    rw,
                                     int *        tid_new,
                                     int **       idx_map_new){
  int i, j, k, * edge_len, * sym, * ex_idx_map, * diag_idx_map;
  for (i=0; i<tensors[tid]->ndim; i++){
    for (j=i+1; j<tensors[tid]->ndim; j++){
      if (idx_map[i] == idx_map[j]){
        CTF_alloc_ptr(sizeof(int)*tensors[tid]->ndim-1, (void**)&edge_len);
        CTF_alloc_ptr(sizeof(int)*tensors[tid]->ndim-1, (void**)&sym);
        CTF_alloc_ptr(sizeof(int)*tensors[tid]->ndim,   (void**)idx_map_new);
        CTF_alloc_ptr(sizeof(int)*tensors[tid]->ndim,   (void**)&ex_idx_map);
        CTF_alloc_ptr(sizeof(int)*tensors[tid]->ndim-1, (void**)&diag_idx_map);
        for (k=0; k<tensors[tid]->ndim; k++){
          if (k<j){
            ex_idx_map[k]       = k;
            diag_idx_map[k]    = k;
            edge_len[k]        = tensors[tid]->edge_len[k]-tensors[tid]->padding[k];
            (*idx_map_new)[k]  = idx_map[k];
            if (k==j-1){
              sym[k] = NS;
            } else 
              sym[k] = tensors[tid]->sym[k];
          } else if (k>j) {
            ex_idx_map[k]       = k-1;
            diag_idx_map[k-1]   = k-1;
            edge_len[k-1]       = tensors[tid]->edge_len[k]-tensors[tid]->padding[k];
            sym[k-1]            = tensors[tid]->sym[k];
            (*idx_map_new)[k-1] = idx_map[k];
          } else {
            ex_idx_map[k] = i;
          }
        }
        fseq_tsr_sum<dtype> fs;
        fs.func_ptr=sym_seq_sum_ref<dtype>;
        fseq_elm_sum<dtype> felm;
        felm.func_ptr=NULL;
        if (rw){
          define_tensor(tensors[tid]->ndim-1, edge_len, sym, tid_new, 1);
#ifdef USE_SYM_SUM
          sym_sum_tsr(1.0, 0.0, tid, *tid_new, ex_idx_map, diag_idx_map, fs, felm, 1);
        } else {
          sym_sum_tsr(1.0, 0.0, *tid_new, tid, diag_idx_map, ex_idx_map, fs, felm, 1);
#else
          sum_tensors(1.0, 0.0, tid, *tid_new, ex_idx_map, diag_idx_map, fs, felm, 1);
        } else {
          sum_tensors(1.0, 0.0, *tid_new, tid, diag_idx_map, ex_idx_map, fs, felm, 1);
#endif
          CTF_free(*idx_map_new);
        }
        CTF_free(edge_len), CTF_free(sym), CTF_free(ex_idx_map), CTF_free(diag_idx_map);
        return CTF_SUCCESS;
      }
    }
  }
  return CTF_NEGATIVE;
}
                                    

/**
 * \brief build stack required for stripping out diagonals of tensor
 * \param[in] ndim number of dimensions of this tensor
 * \param[in] ndim_tot number of dimensions invovled in contraction/sum
 * \param[in] idx_map the index mapping for this contraction/sum
 * \param[in] vrt_sz size of virtual block
 * \param[in] edge_map mapping of each dimension
 * \param[in] topology the tensor is mapped to
 * \param[in,out] blk_edge_len edge lengths of local block after strip
 * \param[in,out] blk_sz size of local sub-block block after strip
 * \param[out] stpr class that recursively strips tensor
 * \return 1 if tensor needs to be stripped, 0 if not
 */
template<typename dtype>
int strip_diag(int const                ndim,
               int const                ndim_tot,
               int const *              idx_map,
               int64_t const           vrt_sz,
               mapping const *          edge_map,
               topology const *         topo,
               int *                    blk_edge_len,
               int64_t *                blk_sz,
               strp_tsr<dtype> **       stpr){
  int64_t i;
  int need_strip;
  int * pmap, * edge_len, * sdim, * sidx;
  strp_tsr<dtype> * stripper;

  CTF_alloc_ptr(ndim_tot*sizeof(int), (void**)&pmap);

  std::fill(pmap, pmap+ndim_tot, -1);

  need_strip = 0;

  for (i=0; i<ndim; i++){
    if (edge_map[i].type == PHYSICAL_MAP) {
      LIBT_ASSERT(pmap[idx_map[i]] == -1);
      pmap[idx_map[i]] = i;
    }
  }
  for (i=0; i<ndim; i++){
    if (edge_map[i].type == VIRTUAL_MAP && pmap[idx_map[i]] != -1)
      need_strip = 1;
  }
  if (need_strip == 0) {
    CTF_free(pmap);
    return 0;
  }

  CTF_alloc_ptr(ndim*sizeof(int), (void**)&edge_len);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&sdim);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&sidx);
  stripper = new strp_tsr<dtype>;

  std::fill(sdim, sdim+ndim, 1);
  std::fill(sidx, sidx+ndim, 0);

  for (i=0; i<ndim; i++){
    edge_len[i] = calc_phase(edge_map+i)/calc_phys_phase(edge_map+i);
    //if (edge_map[i].type == VIRTUAL_MAP) {
    //  edge_len[i] = edge_map[i].np;
    //}
    //if (edge_map[i].type == PHYSICAL_MAP && edge_map[i].has_child) {
      //dont allow recursive mappings for self indices
      // or things get weird here
      //LIBT_ASSERT(edge_map[i].child->type == VIRTUAL_MAP);
    //  edge_len[i] = edge_map[i].child->np;
   // }
    if (edge_map[i].type == VIRTUAL_MAP && pmap[idx_map[i]] != -1) {
      sdim[i] = edge_len[i];
      sidx[i] = calc_phys_rank(edge_map+pmap[idx_map[i]],topo);
      LIBT_ASSERT(edge_map[i].np == edge_map[pmap[idx_map[i]]].np);
    }
    blk_edge_len[i] = blk_edge_len[i] / sdim[i];
    *blk_sz = (*blk_sz) / sdim[i];
  }

  stripper->alloced     = 0;
  stripper->ndim        = ndim;
  stripper->edge_len    = edge_len;
  stripper->strip_dim   = sdim;
  stripper->strip_idx   = sidx;
  stripper->buffer      = NULL;
  stripper->blk_sz      = vrt_sz;

  *stpr = stripper;

  CTF_free(pmap);

  return 1;
}


#endif
