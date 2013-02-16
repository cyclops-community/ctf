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

#ifndef __DIST_TENSOR_MAP_HXX__
#define __DIST_TENSOR_MAP_HXX__

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
      rank = topo->dim_comm[map->cdt]->rank;
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
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&phase);
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
  free(phase);
  return tot_phase;
}

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
        map->child    = (mapping*)malloc(sizeof(mapping));
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
  return DIST_TENSOR_SUCCESS;
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
      int const   topo,
      CommData_t *    global_comm,
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
  /*printf("nvirt = %llu bcomm_vol = %llu bmemuse = %llu topo = %d\n",
    nvirt, bcomm_vol, bmemuse, topo);
  printf("gnvirt = %llu gcomm_vol = %llu gmemuse = %llu bv = %llu nv = %llu gtopo = %d\n",
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
    free(map->child);
  }
  map->type = NOT_MAPPED;
  map->np = 1;
  map->has_child = 0;
  return DIST_TENSOR_SUCCESS;
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
  tsr->is_inner_mapped = 0;
  tsr->is_folded = 0;

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief copies mapping A to B
 * \param[in] ndim number of dimensions
 * \param[in] mapping_A mapping to copy from 
 * \param[in,out] mapping_B mapping to copy to
 */
inline
int copy_mapping(int const    ndim,
     mapping const *  mapping_A,
     mapping *    mapping_B){
  int i;
  for (i=0; i<ndim; i++){
    clear_mapping(&mapping_B[i]);
  }
  memcpy(mapping_B, mapping_A, sizeof(mapping)*ndim);
  for (i=0; i<ndim; i++){
    if (mapping_A[i].has_child){
      get_buffer_space(sizeof(mapping), (void**)&mapping_B[i].child);
      mapping_B[i].child->has_child   = 0;
      mapping_B[i].child->np    = 1;
      mapping_B[i].child->type    = NOT_MAPPED;
      copy_mapping(1, mapping_A[i].child, mapping_B[i].child);
    }
  }
  return DIST_TENSOR_SUCCESS;
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
  free(idx_arr);
  return DIST_TENSOR_SUCCESS;
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
  if (map_A->type == NOT_MAPPED ||
      map_B->type == NOT_MAPPED) return 0;

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
 * \param[out] was_padded whether the tensor was padded
 * \param[out] was_cyclic whether the mapping was cyclic
 * \param[out] old_padding what the padding was
 * \param[out] old_edge_len what the edge lengths were
 * \param[in] topo topology of the processor grid mapped to
 * \param[in] is_inner whether this is an inner mapping
 */ 
template<typename dtype>
int save_mapping(tensor<dtype> *  tsr,
                 int **     old_phase,
                 int **     old_rank,
                 int **     old_virt_dim,
                 int **     old_pe_lda,
                 long_int *   old_size,
                 int *      was_padded,
                 int *      was_cyclic,
                 int **     old_padding,
                 int **     old_edge_len,
                 topology const * topo,
                 int const    is_inner = 0){
  int j;
  mapping * map;
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_phase);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_rank);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_virt_dim);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_pe_lda);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_edge_len);
  
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
  if (tsr->is_padded){
    *was_padded = 1;
    get_buffer_space(sizeof(int)*tsr->ndim, (void**)old_padding);
    memcpy(*old_padding, tsr->padding, sizeof(int)*tsr->ndim);
  } else {
    *was_padded = 0;
  }
  *was_cyclic = tsr->is_cyclic;
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief adjust a mapping to maintan symmetry
 * \param[in] tsr_ndim is the number of dimensions of the tensor
 * \param[in] tsr_sym_table the symmetry table of a tensor
 * \param[in,out] tsr_edge_map is the mapping
 * \return DIST_TENSOR_SUCCESS if mapping successful, DIST_TENSOR_NEGATIVE if not, 
 *     DIST_TENSOR_ERROR if err'ed out
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
    if (loop >= MAXLOOP) return DIST_TENSOR_NEGATIVE;
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
              return DIST_TENSOR_NEGATIVE;
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
                  sym_map->child    = (mapping*)malloc(sizeof(mapping));
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
                  map->child    = (mapping*)malloc(sizeof(mapping));
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
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief sets padding of a tensor
 * \param[in,out] tsr tensor in its new mapping
 * \param[in] is_inner whether this is an inner mapping
 */
template<typename dtype>
int set_padding(tensor<dtype> * tsr, int const is_inner=0){
  int is_pad, j, pad, i;
  int * new_phase, * sub_edge_len;
  mapping * map;

  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&new_phase);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&sub_edge_len);


  if (tsr->is_padded){
    for (i=0; i<tsr->ndim; i++){
      tsr->edge_len[i] -= tsr->padding[i];
    }
  }
/*  for (i=0; i<tsr->ndim; i++){
    printf("tensor edge len[%d] = %d\n",i,tsr->edge_len[i]);
    if (tsr->is_padded){
      printf("tensor padding was [%d] = %d\n",i,tsr->padding[i]);
    }
  }*/
  is_pad = 0; 
  for (j=0; j<tsr->ndim; j++){
    map = tsr->edge_map + j;
    new_phase[j] = calc_phase(map);
    if (tsr->is_padded){
      pad = tsr->edge_len[j]%new_phase[j];
      if (pad != 0) {
        pad = new_phase[j]-pad;
        is_pad = 1;
      }
      tsr->padding[j] = pad;
    } else {
      pad = tsr->edge_len[j]%new_phase[j];
      if (pad != 0) {
        if (is_pad == 0){
          get_buffer_space(sizeof(int)*tsr->ndim, (void**)&tsr->padding);
          memset(tsr->padding, 0, sizeof(int)*j);
        }
        pad = new_phase[j]-pad;
        tsr->padding[j] = pad;
        is_pad = 1;
      } else if (is_pad)
        tsr->padding[j] = 0;
    }
  }
  /* Set padding to 0 anyways... */
  if (is_pad == 0){
    if(!tsr->is_padded)
      get_buffer_space(sizeof(int)*tsr->ndim, (void**)&tsr->padding);
    memset(tsr->padding,0,sizeof(int)*tsr->ndim);
  }  
  for (i=0; i<tsr->ndim; i++){
    tsr->edge_len[i] += tsr->padding[i];
    sub_edge_len[i] = tsr->edge_len[i]/new_phase[i];
  }
  tsr->size = calc_nvirt(tsr,is_inner)
    *sy_packed_size(tsr->ndim, sub_edge_len, tsr->sym);
  
  tsr->is_padded = 1;

  free(sub_edge_len);
  free(new_phase);
  return DIST_TENSOR_SUCCESS;
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
 * \param[in] was_padded whether the tensor was padded
 * \param[in] old_padding what the padding was
 * \param[in] old_edge_len what the padded edge lengths were
 * \param[in] global_comm global communicator
 */
template<typename dtype>
int remap_tensor(int const  tid,
                 tensor<dtype> *tsr,
                 topology const * topo,
                 long_int const old_size,
                 int const *  old_phase,
                 int const *  old_rank,
                 int const *  old_virt_dim,
                 int const *  old_pe_lda,
                 int const    was_padded,
                 int const    was_cyclic,
                 int const *  old_padding,
                 int const *  old_edge_len,
                 CommData_t * global_comm){
  int j, new_nvirt;
  int * new_phase, * new_rank, * new_virt_dim, * new_pe_lda;
  mapping * map;
  dtype * shuffled_data;
#if VERIFY_REMAP
  dtype * shuffled_data_corr;
#endif

  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&new_phase);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&new_rank);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&new_pe_lda);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&new_virt_dim);

  new_nvirt = 1;  
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
#if DEBUG >= 1
  if (global_comm->rank == 0){
    printf("Remapping tensor %d with virtualization factor of %d\n",tid,new_nvirt);
  }
#endif

#if VERIFY_REMAP
    padded_reshuffle(tid,
                     tsr->ndim,
                     old_size,
                     old_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     was_padded,
                     old_padding,
                     tsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     tsr->is_padded,
                     tsr->padding,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     &shuffled_data_corr,
                     global_comm);
#endif

  if (false){
    LIBT_ASSERT(was_cyclic == 1);
    LIBT_ASSERT(tsr->is_cyclic == 1);
    padded_reshuffle(tid,
                     tsr->ndim,
                     old_size,
                     old_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     was_padded,
                     old_padding,
                     tsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     tsr->is_padded,
                     tsr->padding,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     &shuffled_data,
                     global_comm);
    /*if (tsr->is_padded)
      free_buffer_space(tsr->padding);
    tsr->is_padded = is_pad;
    tsr->padding = new_padding;*/
  } else {
    if (global_comm->rank == 0) {
      DEBUG_PRINTF("remapping with cyclic reshuffle (was padded = %d)\n",
        tsr->is_padded);
    }
//    get_buffer_space(sizeof(dtype)*tsr->size, (void**)&shuffled_data);
    cyclic_reshuffle(tsr->ndim,
                     old_size,
                     old_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     was_padded,
                     old_padding,
                     tsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     tsr->is_padded,
                     tsr->padding,
                     old_virt_dim,
                     new_virt_dim,
                     &tsr->data,
                     &shuffled_data,
                     global_comm,
                     was_cyclic,
                     tsr->is_cyclic);
  }

  free_buffer_space((void*)new_phase);
  free_buffer_space((void*)new_rank);
  free_buffer_space((void*)new_virt_dim);
  free_buffer_space((void*)new_pe_lda);
  free_buffer_space((void*)tsr->data);
  tsr->data = shuffled_data;

#if VERIFY_REMAP
  for (j=0; j<tsr->size; j++){
    if (tsr->data[j] != shuffled_data_corr[j]){
      printf("data element %d/%lld not received correctly on process %d\n",
              j, tsr->size, global_comm->rank);
      printf("element received was %.3E, correct %.3E\n", 
              tsr->data[j], shuffled_data_corr[j]);
      ABORT;
    }
  }

#endif

  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief map a tensor
 * \param[in] num_phys_dims number of physical processor grid dimensions
 * \param[in] tsr_edge_len edge lengths of the tensor
 * \param[in] tsr_sym_table the symmetry table of a tensor
 * \param[in,out] restricted an array used to restricted the mapping 
                  (used internally)
 * \param[in] phys_comm dimensional communicators
 * \param[in] comm_idx dimensional ordering
 * \param[in] fill if set does recursive mappings and uses all phys dims
 * \param[in,out] tsr_edge_map mapping of tensor
 * \return DIST_TENSOR_SUCCESS if mapping successful, DIST_TENSOR_NEGATIVE if not, 
 *     DIST_TENSOR_ERROR if err'ed out
 */
inline
int map_tensor(int const    num_phys_dims,
               int const    tsr_ndim,
               int const *    tsr_edge_len,
               int const *    tsr_sym_table,
               int *      restricted,
               CommData_t **    phys_comm,
               int const *    comm_idx,
               int const    fill,
               mapping *    tsr_edge_map){
  int i,j,max_dim,max_len,phase,ret;
  mapping * map;

  /* Make sure the starting mappings are consistent among symmetries */
  ret = map_symtsr(tsr_ndim, tsr_sym_table, tsr_edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;

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
      if (fill)
        return DIST_TENSOR_NEGATIVE;
      break;
    }
    map   = &(tsr_edge_map[max_dim]);
    map->has_child  = 0;
    if (map->type != NOT_MAPPED){
      while (map->has_child) map = map->child;
      phase   = phys_comm[i]->np;
      if (map->type == VIRTUAL_MAP){
        if (phys_comm[i]->np != map->np){
          phase     = lcm(map->np, phys_comm[i]->np);
          if ((phase < map->np || phase < phys_comm[i]->np) || phase >= MAX_PHASE)
            return DIST_TENSOR_NEGATIVE;
          if (phase/phys_comm[i]->np != 1){
            map->has_child  = 1;
            map->child    = (mapping*)malloc(sizeof(mapping));
            map->child->type  = VIRTUAL_MAP;
            map->child->np  = phase/phys_comm[i]->np;
            map->child->has_child = 0;
          }
        }
      } else if (map->type == PHYSICAL_MAP){
        if (map->has_child != 1)
          map->child  = (mapping*)malloc(sizeof(mapping));
        map->has_child  = 1;
        map             = map->child;
        map->has_child  = 0;
      }
    }
    map->type     = PHYSICAL_MAP;
    map->np     = phys_comm[i]->np;
    map->cdt    = (comm_idx == NULL) ? i : comm_idx[i];
    if (!fill)
      restricted[max_dim] = 1;
    ret = map_symtsr(tsr_ndim, tsr_sym_table, tsr_edge_map);
    if (ret!=DIST_TENSOR_SUCCESS) return ret;
  }
  for (i=0; i<tsr_ndim; i++){
    if (tsr_edge_map[i].type == NOT_MAPPED){
      tsr_edge_map[i].type        = VIRTUAL_MAP;
      tsr_edge_map[i].np          = 1;
      tsr_edge_map[i].has_child   = 0;
    }
  }
  return DIST_TENSOR_SUCCESS;
}

/**
* \brief map the remainder of a tensor 
* \param[in] num_phys_dims number of physical processor grid dimensions
* \param[in] phys_comm dimensional communicators
* \param[in,out] tsr pointer to tensor
*/
template<typename dtype>
int map_tensor_rem(int const    num_phys_dims,
                   CommData_t **  phys_comm,
                   tensor<dtype> *  tsr,
                   int const    fill = 0){
  int i, num_sub_phys_dims, stat;
  int * restricted, * phys_mapped, * comm_idx;
  CommData_t ** sub_phys_comm;
  mapping * map;

  get_buffer_space(tsr->ndim*sizeof(int), (void**)&restricted);
  get_buffer_space(num_phys_dims*sizeof(int), (void**)&phys_mapped);

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
  get_buffer_space(num_sub_phys_dims*sizeof(CommData_t*), (void**)&sub_phys_comm);
  get_buffer_space(num_sub_phys_dims*sizeof(int), (void**)&comm_idx);
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
  free_buffer_space(restricted);
  free_buffer_space(phys_mapped);
  free_buffer_space(sub_phys_comm);
  free_buffer_space(comm_idx);
  return stat;
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
    lda *= topo_keep->dim_comm[i]->np;
    if (lda == topo_change->dim_comm[j]->np){
      j++;
      lda = 1;
    } else if (lda > topo_change->dim_comm[j]->np){
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
      get_buffer_space(sizeof(mapping), (void**)&new_map);
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
          new_rec_map->np     = new_topo->dim_comm[j]->np;
          new_np    *= new_rec_map->np;
          if (new_np<old_map->np) {
            old_lda = old_lda * new_rec_map->np;
            new_rec_map->has_child = 1;
            get_buffer_space(sizeof(mapping), (void**)&new_rec_map->child);
            new_rec_map = new_rec_map->child;
          }
        } while (new_np<old_map->np);

        if (old_map->has_child){
          if (old_map->child->type == VIRTUAL_MAP){
            new_rec_map->has_child = 1;
            get_buffer_space(sizeof(mapping), (void**)&new_rec_map->child);
            new_rec_map->child->type  = VIRTUAL_MAP;
            new_rec_map->child->np    = old_map->child->np;
            new_rec_map->child->has_child   = 0;
            break;
          } else {
            new_rec_map->has_child = 1;
            get_buffer_space(sizeof(mapping), (void**)&new_rec_map->child);
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
      free(new_map);
    }
  }
}
#endif
