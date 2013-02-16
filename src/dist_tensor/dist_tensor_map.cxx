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

#define ALLOW_NVIRT 8

/**
 * \brief Checks whether A and B are mapped the same way 
 * \param[in] tid_A handle to A
 * \param[in] tid_B handle to B
 */
template<typename dtype>
int dist_tensor<dtype>::check_pair_mapping(const int tid_A, const int tid_B){
  int i, pass;
  tensor<dtype> * tsr_A, * tsr_B;
  mapping * map_A, * map_B;
    
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  if (!tsr_A->is_mapped) return 0;
  if (!tsr_B->is_mapped) return 0;
  if (tsr_A->is_inner_mapped) return 0;
  if (tsr_B->is_inner_mapped) return 0;
  if (tsr_A->is_folded) return 0;
  if (tsr_B->is_folded) return 0;
  if (tsr_A->need_remap) return 0;
  if (tsr_B->need_remap) return 0;

  LIBT_ASSERT(tsr_A->ndim == tsr_B->ndim);
//  LIBT_ASSERT(tsr_A->size == tsr_B->size);

  for (i=0; i<tsr_A->ndim; i++){
    map_A = &tsr_A->edge_map[i];
    map_B = &tsr_B->edge_map[i];

    pass = comp_dim_map(map_A, map_B);
    if (!pass) return 0;
  }

  return 1;
}

/**
 * \brief  Align the mapping of A and B 
 * \param[in] tid_A handle to tensor A
 * \param[in] tid_B handle to tensor B
 */
template<typename dtype>
int dist_tensor<dtype>::map_tensor_pair(const int tid_A, const int tid_B){
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda, * old_padding, * old_edge_len;
  int was_padded, was_cyclic;
  long_int old_size;
  tensor<dtype> * tsr_A, * tsr_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];

  unmap_inner(tsr_A);
  unmap_inner(tsr_B);
  set_padding(tsr_A);
  set_padding(tsr_B);

  save_mapping(tsr_B, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda, 
               &old_size, &was_padded, &was_cyclic, &old_padding, &old_edge_len, &topovec[tsr_B->itopo]);  
  tsr_B->itopo = tsr_A->itopo;
  tsr_B->is_cyclic = tsr_A->is_cyclic;
  copy_mapping(tsr_A->ndim, tsr_A->edge_map, tsr_B->edge_map);
  set_padding(tsr_B);
  remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size, 
               old_phase, old_rank, old_virt_dim, 
               old_pe_lda, was_padded, was_cyclic, 
               old_padding, old_edge_len, global_comm);   
  free(old_phase);
  free(old_rank);
  free(old_virt_dim);
  free(old_pe_lda);
  if (was_padded)
    free(old_padding);
  free(old_edge_len);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief  Align the mapping of A and B 
 * \param[in] tid_A handle to tensor A
 * \param[in] idx_map_A mapping of A indices
 * \param[in] tid_B handle to tensor B
 * \param[in] idx_map_B mapping of B indices
 */
template<typename dtype>
int dist_tensor<dtype>::map_tensor_pair( const int      tid_A, 
                                         const int *    idx_map_A,
                                         const int      tid_B,
                                         const int *    idx_map_B){
  int i, ret, num_sum, num_tot, was_padded_A, was_padded_B, need_remap;
  int was_cyclic_A, was_cyclic_B, need_remap_A, need_remap_B;
  int d, old_topo_A, old_topo_B;
  long_int old_size_A, old_size_B;
  int * idx_arr, * idx_sum;
  mapping * old_map_A, * old_map_B;
  int * old_phase_A, * old_rank_A, * old_virt_dim_A, * old_pe_lda_A, 
      * old_padding_A, * old_edge_len_A;
  int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B, 
      * old_padding_B, * old_edge_len_B;
//  uint64_t nvirt, tnvirt, bnvirt;
  int btopo;
  int gtopo;
  tensor<dtype> * tsr_A, * tsr_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  inv_idx(tsr_A->ndim, idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, idx_map_B, tsr_B->edge_map,
          &num_tot, &idx_arr);

  get_buffer_space(sizeof(int)*num_tot, (void**)&idx_sum);
  
  num_sum = 0;
  for (i=0; i<num_tot; i++){
    if (idx_arr[2*i] != -1 && idx_arr[2*i+1] != -1){
      idx_sum[num_sum] = i;
      num_sum++;
    }
  }
#if DEBUG >= 2
  if (global_comm->rank == 0)
    printf("Initial mappings:\n");
  print_map(stdout, tid_A);
  print_map(stdout, tid_B);
#endif

  unmap_inner(tsr_A);
  unmap_inner(tsr_B);
  set_padding(tsr_A);
  set_padding(tsr_B);
  save_mapping(tsr_A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A, 
               &old_size_A, &was_padded_A, &was_cyclic_A, &old_padding_A, &old_edge_len_A, &topovec[tsr_A->itopo]);  
  save_mapping(tsr_B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B, 
               &old_size_B, &was_padded_B, &was_cyclic_B, &old_padding_B, &old_edge_len_B, &topovec[tsr_B->itopo]);  
  old_topo_A = tsr_A->itopo;
  old_topo_B = tsr_B->itopo;
  get_buffer_space(sizeof(mapping)*tsr_A->ndim,         (void**)&old_map_A);
  get_buffer_space(sizeof(mapping)*tsr_B->ndim,         (void**)&old_map_B);
  for (i=0; i<tsr_A->ndim; i++){
    old_map_A[i].type         = NOT_MAPPED;
    old_map_A[i].has_child    = 0;
    old_map_A[i].np           = 1;
  }
  for (i=0; i<tsr_B->ndim; i++){
    old_map_B[i].type                 = NOT_MAPPED;
    old_map_B[i].has_child    = 0;
    old_map_B[i].np           = 1;
  }
  copy_mapping(tsr_A->ndim, tsr_A->edge_map, old_map_A);
  copy_mapping(tsr_B->ndim, tsr_B->edge_map, old_map_B);
  need_remap_A = tsr_A->need_remap;
  need_remap_B = tsr_B->need_remap;
  tsr_A->need_remap = 0;
  tsr_B->need_remap = 0;
//  bnvirt = 0;  
  btopo = -1;
  uint64_t size;
  uint64_t min_size = UINT64_MAX;
  /* Attempt to map to all possible permutations of processor topology */
  for (i=global_comm->rank; i<(int)topovec.size(); i+=global_comm->np){
//  for (i=global_comm->rank*topovec.size(); i<(int)topovec.size(); i++){
    clear_mapping(tsr_A);
    clear_mapping(tsr_B);
    set_padding(tsr_A);
    set_padding(tsr_B);

    tsr_A->itopo = i;
    tsr_B->itopo = i;
    tsr_A->is_mapped = 1;
    tsr_B->is_mapped = 1;

    ret = map_sum_indices(idx_arr, idx_sum, num_tot, num_sum, 
                          tid_A, tid_B, &topovec[tsr_A->itopo], 2);
    if (ret == DIST_TENSOR_NEGATIVE) continue;
    else if (ret != DIST_TENSOR_SUCCESS){
      tsr_A->need_remap = need_remap_A;
      tsr_B->need_remap = need_remap_B;
      return ret;
    }

    ret = map_self_indices(tid_A, idx_map_A);
    if (ret == DIST_TENSOR_NEGATIVE) continue;
    else if (ret != DIST_TENSOR_SUCCESS) {
      tsr_A->need_remap = need_remap_A;
      tsr_B->need_remap = need_remap_B;
      return ret;
    }
    ret = map_tensor_rem(topovec[tsr_A->itopo].ndim, 
                         topovec[tsr_A->itopo].dim_comm, tsr_A);

    if (ret == DIST_TENSOR_NEGATIVE) continue;
    else if (ret != DIST_TENSOR_SUCCESS) {
      tsr_A->need_remap = need_remap_A;
      tsr_B->need_remap = need_remap_B;
      return ret;
    }

    copy_mapping(tsr_A->ndim, tsr_B->ndim,
                 idx_map_A, tsr_A->edge_map, 
                 idx_map_B, tsr_B->edge_map,0);
    ret = map_self_indices(tid_B, idx_map_B);
    if (ret == DIST_TENSOR_NEGATIVE) continue;
    else if (ret != DIST_TENSOR_SUCCESS) {
      tsr_A->need_remap = need_remap_A;
      tsr_B->need_remap = need_remap_B;
      return ret;
    }
    ret = map_tensor_rem(topovec[tsr_B->itopo].ndim, 
                         topovec[tsr_B->itopo].dim_comm, tsr_B);
    if (ret == DIST_TENSOR_NEGATIVE) continue;
    else if (ret != DIST_TENSOR_SUCCESS) {
      tsr_A->need_remap = need_remap_A;
      tsr_B->need_remap = need_remap_B;
      return ret;
    }
    copy_mapping(tsr_B->ndim, tsr_A->ndim,
                 idx_map_B, tsr_B->edge_map,
                 idx_map_A, tsr_A->edge_map, 0);
/*    ret = map_symtsr(tsr_A->ndim, tsr_A->sym_table, tsr_A->edge_map);
    ret = map_symtsr(tsr_B->ndim, tsr_B->sym_table, tsr_B->edge_map);
    if (ret!=DIST_TENSOR_SUCCESS) return ret;
    return DIST_TENSOR_SUCCESS;*/

/*#if DEBUG >= 1  
    print_map(stdout, tid_A,0);
    print_map(stdout, tid_B,0);
#endif*/
    if (!check_sum_mapping(tid_A, idx_map_A, tid_B, idx_map_B)) continue;
    set_padding(tsr_A);
    set_padding(tsr_B);
    size = tsr_A->size + tsr_B->size;

    need_remap = need_remap_A;
    if (tsr_A->itopo == old_topo_A){
      for (d=0; d<tsr_A->ndim; d++){
        if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (tsr_B->itopo == old_topo_B){
      for (d=0; d<tsr_B->ndim; d++){
        if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (!need_remap) size = 0;

    /*nvirt = (uint64_t)calc_nvirt(tsr_A);
    tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_B);
    if (tnvirt < nvirt) nvirt = UINT64_MAX;
    else nvirt = tnvirt;
    if (btopo == -1 || nvirt < bnvirt ) {
      bnvirt = nvirt;
      btopo = i;      
    }*/
    if (btopo == -1 || size < min_size){
      min_size = size;
      btopo = i;      
    }
  }
  if (btopo == -1)
    min_size = UINT64_MAX;
  /* pick lower dimensional mappings, if equivalent */
  gtopo = get_best_topo(min_size, btopo, global_comm);
  if (gtopo == -1){
    printf("ERROR: Failed to map pair!\n");
    return DIST_TENSOR_ERROR;
  }
  
  clear_mapping(tsr_A);
  clear_mapping(tsr_B);
  set_padding(tsr_A);
  set_padding(tsr_B);

  tsr_A->itopo = gtopo;
  tsr_B->itopo = gtopo;
    
  ret = map_sum_indices(idx_arr, idx_sum, num_tot, num_sum, 
                          tid_A, tid_B, &topovec[tsr_A->itopo], 2);

  ret = map_self_indices(tid_A, idx_map_A);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  ret = map_tensor_rem(topovec[tsr_A->itopo].ndim, 
                       topovec[tsr_A->itopo].dim_comm, tsr_A);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  copy_mapping(tsr_A->ndim, tsr_B->ndim,
               idx_map_A, tsr_A->edge_map, 
               idx_map_B, tsr_B->edge_map,0);
  ret = map_self_indices(tid_B, idx_map_B);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  ret = map_tensor_rem(topovec[tsr_B->itopo].ndim, 
                       topovec[tsr_B->itopo].dim_comm, tsr_B);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  copy_mapping(tsr_B->ndim, tsr_A->ndim,
               idx_map_B, tsr_B->edge_map,
               idx_map_A, tsr_A->edge_map, 0);
/*  ret = map_symtsr(tsr_A->ndim, tsr_A->sym_table, tsr_A->edge_map);
  ret = map_symtsr(tsr_B->ndim, tsr_B->sym_table, tsr_B->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;
  return DIST_TENSOR_SUCCESS;*/
  tsr_A->is_mapped = 1;
  tsr_B->is_mapped = 1;


  set_padding(tsr_A);
  set_padding(tsr_B);
#if DEBUG >= 2
  if (global_comm->rank == 0)
    printf("New mappings:\n");
  print_map(stdout, tid_A);
  print_map(stdout, tid_B);
#endif
 
  tsr_A->is_cyclic = 1;
  tsr_B->is_cyclic = 1;
  need_remap = need_remap_A;
  if (tsr_A->itopo == old_topo_A){
    for (d=0; d<tsr_A->ndim; d++){
      if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
        need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_tensor(tid_A, tsr_A, &topovec[tsr_A->itopo], old_size_A, old_phase_A, old_rank_A, old_virt_dim_A, 
                 old_pe_lda_A, was_padded_A, was_cyclic_A, old_padding_A, old_edge_len_A, global_comm);   
  need_remap = 0;
  if (tsr_B->itopo == old_topo_B){
    for (d=0; d<tsr_B->ndim; d++){
      if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
        need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size_B, old_phase_B, old_rank_B, old_virt_dim_B, 
                 old_pe_lda_B, was_padded_B, was_cyclic_B, old_padding_B, old_edge_len_B, global_comm);   
  free(idx_sum);
  free(old_phase_A);
  free(old_rank_A);
  free(old_virt_dim_A);
  free(old_pe_lda_A);
  if (was_padded_A)
    free(old_padding_A);
  free(old_edge_len_A);
  free(old_phase_B);
  free(old_rank_B);
  free(old_virt_dim_B);
  free(old_pe_lda_B);
  if (was_padded_B)
    free(old_padding_B);
  free(old_edge_len_B);
  free(idx_arr);
  clear_mapping(old_map_A);
  clear_mapping(old_map_B);
  free_buffer_space(old_map_A);
  free_buffer_space(old_map_B);

  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief checks mapping in preparation for tensors sum or contract
 * \param[in] tid handle to tensor 
 * \param[in] max_idx maximum idx in idx_map
 * \param[in] idx_map is the mapping of tensor to global indices
 * \return whether the self mapping is consistent
*/
template<typename dtype>
int dist_tensor<dtype>::
    check_self_mapping(int const        tid, 
                       int const *      idx_map){
  int i, pass, iR, max_idx;
  int * idx_arr;
  tensor<dtype> * tsr;

  tsr = tensors[tid];
  if (tsr->need_remap) return 0;

  max_idx = -1;
  for (i=0; i<tsr->ndim; i++){
    if (idx_map[i] > max_idx) max_idx = idx_map[i];
  }
  max_idx++;

  get_buffer_space(sizeof(int)*max_idx, (void**)&idx_arr);
  std::fill(idx_arr, idx_arr+max_idx, -1);

  pass = 1;
/*  for (i=0; i<tsr->ndim; i++){
    if (tsr->edge_map[i].np > tsr->edge_len[i]){
      pass = 0;
      DPRINTF(1,"Failed confirmation here\n");
    }
  }*/
  /* Go in reverse, since the first index of the diagonal set will be mapped */
  for (i=0; i<tsr->ndim; i++){
    iR = idx_arr[idx_map[i]];
    if (iR != -1){
      if (tsr->edge_map[iR].type == PHYSICAL_MAP)
        pass = 0;
      if (tsr->edge_map[iR].type == VIRTUAL_MAP){
        if (calc_phase(&tsr->edge_map[i]) != tsr->edge_map[iR].np)
          pass = 0;
      }
    }
    idx_arr[idx_map[i]] = i;
  }
  free_buffer_space(idx_arr);
  return pass;
}

/**
 * \brief checks mapping in preparation for tensors sum
 * \param[in] tid_A handle to tensor A
 * \param[in] idx_A handle to tensor A
 * \param[in] tid_B handle to tensor B
 * \param[in] idx_B handle to tensor B
 * \return tsum summation class to run
*/
template<typename dtype>
int dist_tensor<dtype>::
    check_sum_mapping(int const         tid_A, 
                      int const *       idx_A,
                      int const         tid_B,
                      int const *       idx_B){
  int i, pass, ndim_tot, iA, iB;
  int * idx_arr, * phys_map;
  tensor<dtype> * tsr_A, * tsr_B;
  mapping * map;

  TAU_FSTART(check_sum_mapping);
  pass = 1;
  
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  if (tsr_A->is_mapped == 0) pass = 0;
  if (tsr_B->is_mapped == 0) pass = 0;
  
  if (tsr_A->is_inner_mapped == 1) pass = 0;
  if (tsr_B->is_inner_mapped == 1) pass = 0;
  
  if (tsr_A->is_folded == 1) pass = 0;
  if (tsr_B->is_folded == 1) pass = 0;
  if (tsr_A->need_remap) pass = 0;
  //if (tsr_B->need_remap) pass = 0;
  
  if (tsr_A->itopo != tsr_B->itopo) pass = 0;

  if (pass==0){
    TAU_FSTOP(check_sum_mapping);
    return 0;
  }
  
  get_buffer_space(sizeof(int)*topovec[tsr_A->itopo].ndim, (void**)&phys_map);
  memset(phys_map, 0, sizeof(int)*topovec[tsr_A->itopo].ndim);

  inv_idx(tsr_A->ndim, idx_A, tsr_A->edge_map,
          tsr_B->ndim, idx_B, tsr_B->edge_map,
          &ndim_tot, &idx_arr);

  if (!check_self_mapping(tid_A, idx_A))
    pass = 0;
  if (!check_self_mapping(tid_B, idx_B))
    pass = 0;
  if (pass == 0)
    DPRINTF(2,"failed confirmation here\n");

  for (i=0; i<ndim_tot; i++){
    iA = idx_arr[2*i];
    iB = idx_arr[2*i+1];
    if (iA != -1 && iB != -1) {
      if (!comp_dim_map(&tsr_A->edge_map[iA], &tsr_B->edge_map[iB])){
        pass = 0;
        DPRINTF(2,"failed confirmation here i=%d\n",i);
      }
    }
    if (iA != -1) {
      map = &tsr_A->edge_map[iA];
      if (map->type == PHYSICAL_MAP)
        phys_map[map->cdt] = 1;
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP)
          phys_map[map->cdt] = 1;
      }
    }
    if (iB != -1){
      map = &tsr_B->edge_map[iB];
      if (map->type == PHYSICAL_MAP)
        phys_map[map->cdt] = 1;
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP)
          phys_map[map->cdt] = 1;
      }
    }
  }
  /* Ensure that something is mapped to each dimension, since replciation
     does not make sense in sum for all tensors */
  for (i=0; i<topovec[tsr_A->itopo].ndim; i++){
    if (phys_map[i] == 0) {
      pass = 0;
      DPRINTF(2,"failed confirmation here i=%d\n",i);
    }
  }

  free(phys_map);
  free(idx_arr);

  TAU_FSTOP(check_sum_mapping);

  return pass;
}



/* \brief Check whether current tensor mapping can be contracted on 
 * \param type specification of contraction
 * \param is_inner whether its an inner contraction
 */
template<typename dtype>
int dist_tensor<dtype>::check_contraction_mapping(CTF_ctr_type_t const * type,
                                                  int const is_inner){
  int num_tot, i, ph_A, ph_B, iA, iB, iC, pass, order, topo_ndim;
  int * idx_arr;
  int * phys_mismatched, * phys_mapped;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  mapping * map;

  pass = 1;

  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];
    

  if (tsr_A->is_mapped == 0) pass = 0;
  if (tsr_B->is_mapped == 0) pass = 0;
  if (tsr_C->is_mapped == 0) pass = 0;
  LIBT_ASSERT(pass==1 || is_inner==0);
 
  if (!is_inner) {
    if (tsr_A->is_inner_mapped == 1) pass = 0;
    if (tsr_B->is_inner_mapped == 1) pass = 0;
    if (tsr_C->is_inner_mapped == 1) pass = 0;
  }
  
  if (tsr_A->is_folded == 1) pass = 0;
  if (tsr_B->is_folded == 1) pass = 0;
  if (tsr_C->is_folded == 1) pass = 0;
  if (tsr_A->need_remap) pass = 0;
  if (tsr_B->need_remap) pass = 0;
  //if (tsr_C->need_remap) pass = 0;
  
  if (pass==0){
    return 0;
  }

  if (tsr_A->itopo != tsr_B->itopo) pass = 0;
  if (tsr_A->itopo != tsr_C->itopo) pass = 0;

  if (pass==0){
    return 0;
  }

  if (is_inner)
    topo_ndim = inner_topovec[tsr_A->itopo].ndim;
  else
    topo_ndim = topovec[tsr_A->itopo].ndim;
  get_buffer_space(sizeof(int)*topo_ndim, (void**)&phys_mismatched);
  get_buffer_space(sizeof(int)*topo_ndim, (void**)&phys_mapped);
  memset(phys_mismatched, 0, sizeof(int)*topo_ndim);
  memset(phys_mapped, 0, sizeof(int)*topo_ndim);


  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);
  
  if (!check_self_mapping(type->tid_A, type->idx_map_A))
    pass = 0;
  if (!check_self_mapping(type->tid_B, type->idx_map_B))
    pass = 0;
  if (!check_self_mapping(type->tid_C, type->idx_map_C))
    pass = 0;


  for (i=0; i<num_tot; i++){
    if (idx_arr[3*i+0] != -1 &&
        idx_arr[3*i+1] != -1 &&
        idx_arr[3*i+2] != -1){
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      if (0 == comp_dim_map(&tsr_B->edge_map[iB], &tsr_A->edge_map[iA]) || 
          0 == comp_dim_map(&tsr_B->edge_map[iB], &tsr_C->edge_map[iC])){
        DPRINTF(2,"failed confirmation here %d\n",iA);
        pass = 0;
        break;
      } else {
        map = &tsr_A->edge_map[iA];
        for (;;){
          if (map->type == PHYSICAL_MAP){
            if (phys_mapped[map->cdt] == 1){
              DPRINTF(2,"failed confirmation here %d\n",iA);
              pass = 0;
              break;
            } else {
              phys_mapped[map->cdt] = 1;
              phys_mismatched[map->cdt] = 1;
            }
          } else break;
          if (map->has_child) map = map->child;
          else break;
        } 
      }
    } else {
      for (order=0; order<3; order++){
        switch (order){
          case 0:
            tsr_A = tensors[type->tid_A];
            tsr_B = tensors[type->tid_B];
            tsr_C = tensors[type->tid_C];
            iA = idx_arr[3*i+0];
            iB = idx_arr[3*i+1];
            iC = idx_arr[3*i+2];
            break;
          case 1:
            tsr_A = tensors[type->tid_A];
            tsr_B = tensors[type->tid_C];
            tsr_C = tensors[type->tid_B];
            iA = idx_arr[3*i+0];
            iB = idx_arr[3*i+2];
            iC = idx_arr[3*i+1];
            break;
          case 2:
            tsr_A = tensors[type->tid_C];
            tsr_B = tensors[type->tid_B];
            tsr_C = tensors[type->tid_A];
            iA = idx_arr[3*i+2];
            iB = idx_arr[3*i+1];
            iC = idx_arr[3*i+0];
            break;
        }
        if (iC == -1){
          if (iB == -1){
            if (iA != -1) {
              map = &tsr_A->edge_map[iA];
              for (;;){
          if (map->type == PHYSICAL_MAP){
            if (phys_mapped[map->cdt] == 1){
              DPRINTF(2,"failed confirmation here %d\n",iA);
              pass = 0;
              break;
            } else
              phys_mapped[map->cdt] = 1;
          } else break;
          if (map->has_child) map = map->child;
          else break;
              } 
            }
          } else if (iA == -1){
            map = &tsr_B->edge_map[iB];
            for (;;){
              if (map->type == PHYSICAL_MAP){
          if (phys_mapped[map->cdt] == 1){
            DPRINTF(2,"failed confirmation here %d\n",iA);
            pass = 0;
            break;
          } else
            phys_mapped[map->cdt] = 1;
        } else break;
        if (map->has_child) map = map->child;
        else break;
            } 
          } else { 
            /* Confirm that the phases of A and B 
               over which we are contracting are the same */
            ph_A = calc_phase(&tsr_A->edge_map[iA]);
            ph_B = calc_phase(&tsr_B->edge_map[iB]);

            if (ph_A != ph_B){
              //if (global_comm->rank == 0) 
                DPRINTF(2,"failed confirmation here\n");
              pass = 0;
              break;
            }
            /* If the mapping along this dimension is the same make sure
               its mapped to a onto a unique free dimension */
            if (comp_dim_map(&tsr_B->edge_map[iB], &tsr_A->edge_map[iA])){
              map = &tsr_B->edge_map[iB];
              for (;;){
                if (map->type == PHYSICAL_MAP){
                  if (phys_mapped[map->cdt] == 1){
                    DPRINTF(2,"failed confirmation here %d\n",iB);
                    pass = 0;
                    break;
                  } else
                    phys_mapped[map->cdt] = 1;
                } else break;
                if (map->has_child) map = map->child;
                else break;
              } 
            } else {
              /* If the mapping along this dimension is different, make sure
                 the mismatch is mapped onto unqiue physical dimensions */
              if (tsr_A->edge_map[iA].type == PHYSICAL_MAP){
                if (phys_mismatched[tsr_A->edge_map[iA].cdt] == 1){
                  //if (global_comm->rank == 0) 
                    DPRINTF(2,"failed confirmation here i=%d iA=%d iB=%d\n",i,iA,iB);
                  pass = 0;
                  break;
                } else
                  phys_mismatched[tsr_A->edge_map[iA].cdt] = 1;

              }
              if (tsr_B->edge_map[iB].type == PHYSICAL_MAP){
                if (phys_mismatched[tsr_B->edge_map[iB].cdt] == 1){
                  //if (global_comm->rank == 0) 
                    DPRINTF(2,"failed confirmation here i=%d iB=%d iB=%d\n",i,iB,iB);
                  pass = 0;
                  break;
                } else
                  phys_mismatched[tsr_B->edge_map[iB].cdt] = 1;
              }
            }
          }
        }
      }
    }
  }
  for (i=0; i<topo_ndim; i++){
    if (phys_mismatched[i] == 1 && phys_mapped[i] == 0){
      DPRINTF(2,"failed confirmation here i=%d\n",i);
      pass = 0;
      break;
    }
    if (phys_mismatched[i] == 0 && phys_mapped[i] == 0){
      DPRINTF(2,"failed confirmation here i=%d\n",i);
      pass = 0;
      break;
    }    
  }


  free_buffer_space(idx_arr);
  free_buffer_space(phys_mismatched);
  free_buffer_space(phys_mapped);
  return pass;
}

#ifndef BEST_VOL
#define BEST_VOL 0
#endif
#ifndef BEST_VIRT
#define BEST_VIRT 1
#endif
#if(!BEST_VOL)
#define BEST_COMM 1
#else
#define BEST_COMM 0
#endif
/* \brief map tensors so that they can be contracted on
 * \param type specification of contraction
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer 
 * \param[in] func_ptr sequential ctr func pointer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[out] ctrf contraction class to run
 */
template<typename dtype>
int dist_tensor<dtype>::map_tensors(CTF_ctr_type_t const *      type, 
                                    dtype *                     buffer, 
                                    int const                   buffer_len, 
                                    fseq_tsr_ctr<dtype>         func_ptr, 
                                    dtype const                 alpha,
                                    dtype const                 beta,
                                    ctr<dtype> **               ctrf,
                                    int const                   do_remap){
  int num_tot, i, ret, j, need_remap, d;
  int need_remap_A, need_remap_B, need_remap_C;
  uint64_t memuse, bmemuse;
#if BEST_COMM
  uint64_t comm_vol, bcomm_vol;
#endif
#if BEST_VOL
  int64_t n,m,k;
  uint64_t gnvirt;
#endif
#if BEST_VIRT
  uint64_t nvirt, tnvirt, bnvirt;
#endif
  int btopo, gtopo;
  int was_padded_A, was_padded_B, was_padded_C, old_nvirt_all;
  int was_cyclic_A, was_cyclic_B, was_cyclic_C, nvirt_all;
  long_int old_size_A, old_size_B, old_size_C;
  int * idx_arr, * idx_ctr, * idx_no_ctr, * idx_extra;
  int * old_phase_A, * old_rank_A, * old_virt_dim_A, * old_pe_lda_A;
  int * old_padding_A, * old_edge_len_A;
  int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B;
  int * old_padding_B, * old_edge_len_B;
  int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
  int * old_padding_C, * old_edge_len_C;
#if BEST_VOL
  int * virt_blk_len_A, * virt_blk_len_B, * virt_blk_len_C;
#endif
  mapping * old_map_A, * old_map_B, * old_map_C;
  int old_topo_A, old_topo_B, old_topo_C;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  ctr<dtype> * sctr;
  old_topo_A = -1;
  old_topo_B = -1;
  old_topo_C = -1;

  TAU_FSTART(map_tensors);

  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);

  get_buffer_space(sizeof(int)*num_tot,         (void**)&idx_no_ctr);
  get_buffer_space(sizeof(int)*num_tot,         (void**)&idx_extra);
  get_buffer_space(sizeof(int)*num_tot,         (void**)&idx_ctr);
#if BEST_VOL
  get_buffer_space(sizeof(int)*tsr_A->ndim,     (void**)&virt_blk_len_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim,     (void**)&virt_blk_len_B);
  get_buffer_space(sizeof(int)*tsr_C->ndim,     (void**)&virt_blk_len_C);
#endif
  old_map_A = NULL;
  old_map_B = NULL;
  old_map_C = NULL;

  if (do_remap){
    get_buffer_space(sizeof(mapping)*tsr_A->ndim,         (void**)&old_map_A);
    get_buffer_space(sizeof(mapping)*tsr_B->ndim,         (void**)&old_map_B);
    get_buffer_space(sizeof(mapping)*tsr_C->ndim,         (void**)&old_map_C);
    for (i=0; i<tsr_A->ndim; i++){
      old_map_A[i].type         = NOT_MAPPED;
      old_map_A[i].has_child    = 0;
      old_map_A[i].np           = 1;
    }
    for (i=0; i<tsr_B->ndim; i++){
      old_map_B[i].type                 = NOT_MAPPED;
      old_map_B[i].has_child    = 0;
      old_map_B[i].np           = 1;
    }
    for (i=0; i<tsr_C->ndim; i++){
      old_map_C[i].type                 = NOT_MAPPED;
      old_map_C[i].has_child    = 0;
      old_map_C[i].np           = 1;
    }
    copy_mapping(tsr_A->ndim, tsr_A->edge_map, old_map_A);
    copy_mapping(tsr_B->ndim, tsr_B->edge_map, old_map_B);
    copy_mapping(tsr_C->ndim, tsr_C->edge_map, old_map_C);
    old_topo_A = tsr_A->itopo;
    old_topo_B = tsr_B->itopo;
    old_topo_C = tsr_C->itopo;

    LIBT_ASSERT(tsr_A->is_mapped);
    LIBT_ASSERT(tsr_B->is_mapped);
    LIBT_ASSERT(tsr_C->is_mapped);
  #if DEBUG >= 2
    if (global_comm->rank == 0)
      printf("Initial mappings:\n");
    print_map(stdout, type->tid_A);
    print_map(stdout, type->tid_B);
    print_map(stdout, type->tid_C);
  #endif
    unmap_inner(tsr_A);
    unmap_inner(tsr_B);
    unmap_inner(tsr_C);
    set_padding(tsr_A);
    set_padding(tsr_B);
    set_padding(tsr_C);
    /* Save the current mappings of A, B, C */
    save_mapping(tsr_A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A, 
                 &old_size_A, &was_padded_A, &was_cyclic_A, &old_padding_A, 
                 &old_edge_len_A, &topovec[tsr_A->itopo]);
    save_mapping(tsr_B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B, 
                 &old_size_B, &was_padded_B, &was_cyclic_B, &old_padding_B, 
                 &old_edge_len_B, &topovec[tsr_B->itopo]);
    save_mapping(tsr_C, &old_phase_C, &old_rank_C, &old_virt_dim_C, &old_pe_lda_C, 
                 &old_size_C, &was_padded_C, &was_cyclic_C, &old_padding_C, 
                 &old_edge_len_C, &topovec[tsr_C->itopo]);
  }
  need_remap_A = tsr_A->need_remap;
  need_remap_B = tsr_B->need_remap;
  need_remap_C = tsr_C->need_remap;
  tsr_A->need_remap = 0;
  tsr_B->need_remap = 0;
  tsr_C->need_remap = 0;
  btopo = -1;
#if BEST_VIRT
  bnvirt = UINT64_MAX;
#endif  
#if BEST_COMM
  bcomm_vol = UINT64_MAX;
  bmemuse = UINT64_MAX;
#endif
  for (j=0; j<6; j++){
    /* Attempt to map to all possible permutations of processor topology */
    for (i=global_comm->rank; i<(int)topovec.size(); i+=global_comm->np){
//    for (i=global_comm->rank*topovec.size(); i<(int)topovec.size(); i++){
      clear_mapping(tsr_A);
      clear_mapping(tsr_B);
      clear_mapping(tsr_C);
      set_padding(tsr_A);
      set_padding(tsr_B);
      set_padding(tsr_C);

      ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                            type->idx_map_B, type->idx_map_C, i, j, 
                            idx_arr, idx_ctr, idx_extra, idx_no_ctr);
      /*print_map(stdout, type->tid_A, 0);
      print_map(stdout, type->tid_B, 0);
      print_map(stdout, type->tid_C, 0);*/
      

      if (ret == DIST_TENSOR_ERROR) {
        TAU_FSTOP(map_tensors);
        tsr_A->need_remap = need_remap_A;
        tsr_B->need_remap = need_remap_B;
        tsr_C->need_remap = need_remap_C;
        return DIST_TENSOR_ERROR;
      }
      if (ret == DIST_TENSOR_NEGATIVE) continue;
  
      tsr_A->is_mapped = 1;
      tsr_B->is_mapped = 1;
      tsr_C->is_mapped = 1;
      tsr_A->itopo = i;
      tsr_B->itopo = i;
      tsr_C->itopo = i;
      
      if (check_contraction_mapping(type) == 0) continue;
      
      nvirt_all = -1;
      old_nvirt_all = -2;
#if 0
      while (nvirt_all < MIN_NVIRT){
        old_nvirt_all = nvirt_all;
        set_padding(tsr_A);
        set_padding(tsr_B);
        set_padding(tsr_C);
        sctr = construct_contraction(type, buffer, buffer_len, func_ptr, 
                                      alpha, beta, 0, NULL, &nvirt_all);
        /* If this cannot be stretched */
        if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
          clear_mapping(tsr_A);
          clear_mapping(tsr_B);
          clear_mapping(tsr_C);
          set_padding(tsr_A);
          set_padding(tsr_B);
          set_padding(tsr_C);

          ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                                type->idx_map_B, type->idx_map_C, i, j, 
                                idx_arr, idx_ctr, idx_extra, idx_no_ctr);
          tsr_A->is_mapped = 1;
          tsr_B->is_mapped = 1;
          tsr_C->is_mapped = 1;
          tsr_A->itopo = i;
          tsr_B->itopo = i;
          tsr_C->itopo = i;
          break;

        }
        if (nvirt_all < MIN_NVIRT){
          stretch_virt(tsr_A->ndim, 2, tsr_A->edge_map);
          stretch_virt(tsr_B->ndim, 2, tsr_B->edge_map);
          stretch_virt(tsr_C->ndim, 2, tsr_C->edge_map);
        }
      }
#endif
      set_padding(tsr_A);
      set_padding(tsr_B);
      set_padding(tsr_C);
      sctr = construct_contraction(type, buffer, buffer_len, func_ptr, 
                                    alpha, beta, 0, NULL, &nvirt_all);
     
      comm_vol = sctr->comm_rec(sctr->num_lyr);
      memuse = 0;
      if (do_remap){
        need_remap = need_remap_A;
        if (i == old_topo_A){
          for (d=0; d<tsr_A->ndim; d++){
            if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
              need_remap = 1;
          }
        } else
          need_remap = 1;
        if (need_remap) {
          comm_vol += sizeof(dtype)*tsr_A->size*10.;//(int)(pow((double)global_comm->np,3./4.));
          memuse = (uint64_t)2*sizeof(dtype)*tsr_A->size;
        } else
          memuse = 0;
        need_remap = need_remap_B;
        if (i == old_topo_B){
          for (d=0; d<tsr_B->ndim; d++){
            if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
              need_remap = 1;
          }
        } else
          need_remap = 1;
        if (need_remap) {
          comm_vol += sizeof(dtype)*tsr_B->size*10.;//(int)(pow((double)global_comm->np,3./4.));
          memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_B->size*2);
        }
        need_remap = 0; //need_remap_C;
        if (i == old_topo_C){
          for (d=0; d<tsr_C->ndim; d++){
            if (!comp_dim_map(&tsr_C->edge_map[d],&old_map_C[d]))
              need_remap = 1;
          }
        } else
          need_remap = 1;
        if (need_remap) {
          comm_vol += sizeof(dtype)*tsr_C->size*10;//(int)(pow((double)global_comm->np,3./4.));
          memuse = MAX(memuse,(uint64_t)sizeof(dtype)*tsr_C->size*2);
        }
        memuse = MAX((uint64_t)sctr->mem_rec(), memuse);
      } else {
        memuse = (uint64_t)sctr->mem_rec();
      } 

      if ((uint64_t)memuse >= proc_bytes_available()){
        DPRINTF(1,"Not enough memory available for topo %d with order %d\n", i, j);
        delete sctr;
        continue;
      } 

#if BEST_VOL
      calc_dim(tsr_A->ndim, 0, tsr_A->edge_len, tsr_A->edge_map, 
               NULL, virt_blk_len_A, NULL);
      calc_dim(tsr_B->ndim, 0, tsr_B->edge_len, tsr_B->edge_map, 
               NULL, virt_blk_len_B, NULL);
      calc_dim(tsr_C->ndim, 0, tsr_C->edge_len, tsr_C->edge_map, 
               NULL, virt_blk_len_C, NULL);
      ggg_sym_nmk(tsr_A->ndim, virt_blk_len_A, type->idx_map_A, tsr_A->sym, 
                  tsr_B->ndim, virt_blk_len_B, type->idx_map_B, tsr_B->sym, 
                  tsr_C->ndim, &n, &m, &k);
      if (btopo == -1 || n*m*k > bnvirt ) {
        bnvirt = n*m*k;
        btopo = 6*i+j;      
      }
#endif
#if BEST_VIRT
      /* be careful about overflow */
      nvirt = (uint64_t)calc_nvirt(tsr_A);
      tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_B);
      if (tnvirt < nvirt) nvirt = UINT64_MAX;
      else {
        nvirt = tnvirt;
        tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_C);
        if (tnvirt < nvirt) nvirt = UINT64_MAX;
        else nvirt = tnvirt;
      }
      if (btopo == -1 || ((nvirt <= bnvirt  || nvirt <= ALLOW_NVIRT) && comm_vol < bcomm_vol)) {
        comm_vol = sctr->comm_rec(sctr->num_lyr);
        bcomm_vol = comm_vol;
        bmemuse = memuse;
        bnvirt = nvirt;
        btopo = 6*i+j;      
      }  else if (nvirt == bnvirt ) {
      }
      delete sctr;
#else
  #if BEST_COMM
      comm_vol = sctr->comm_rec(sctr->num_lyr);
      if (comm_vol < bcomm_vol){
        bcomm_vol = comm_vol;
        btopo = 6*i+j;
      }
  #endif
#endif
    }
  }
#if BEST_VOL
  ALLREDUCE(&bnvirt, &gnvirt, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, global_comm);
  if (bnvirt != gnvirt){
    btopo = INT_MAX;
  }
  ALLREDUCE(&btopo, &gtopo, 1, MPI_INT, MPI_MIN, global_comm);
#endif
#if BEST_VIRT
  if (btopo == -1){
    bnvirt = UINT64_MAX;
    btopo = INT_MAX;
  }
  DEBUG_PRINTF("bnvirt = %llu\n", (unsigned long long int)bnvirt);
  /* pick lower dimensional mappings, if equivalent */
#if BEST_COMM
  if (bnvirt >= ALLOW_NVIRT)
    gtopo = get_best_topo(bnvirt/ALLOW_NVIRT, btopo, global_comm, bcomm_vol, bmemuse);
  else
    gtopo = get_best_topo(1, btopo, global_comm, bcomm_vol, bmemuse);
#else
  gtopo = get_best_topo(bnvirt, btopo, global_comm);
#endif
#endif
  
  clear_mapping(tsr_A);
  clear_mapping(tsr_B);
  clear_mapping(tsr_C);
  set_padding(tsr_A);
  set_padding(tsr_B);
  set_padding(tsr_C);
  
  if (!do_remap || gtopo == INT_MAX || gtopo == -1){
    free_buffer_space((void*)idx_arr);
    free_buffer_space((void*)idx_no_ctr);
    free_buffer_space((void*)idx_ctr);
    free_buffer_space((void*)idx_extra);
    tsr_A->need_remap = need_remap_A;
    tsr_B->need_remap = need_remap_B;
    tsr_C->need_remap = need_remap_C;
    TAU_FSTOP(map_tensors);
    if (gtopo == INT_MAX || gtopo == -1){
      printf("ERROR: Failed to map contraction!\n");
      return DIST_TENSOR_ERROR;
    }
    return DIST_TENSOR_SUCCESS;
  }

  tsr_A->itopo = gtopo/6;
  tsr_B->itopo = gtopo/6;
  tsr_C->itopo = gtopo/6;
  
  ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                        type->idx_map_B, type->idx_map_C, gtopo/6, gtopo%6, 
                        idx_arr, idx_ctr, idx_extra, idx_no_ctr);


  if (ret == DIST_TENSOR_NEGATIVE || ret == DIST_TENSOR_ERROR) {
    tsr_A->need_remap = need_remap_A;
    tsr_B->need_remap = need_remap_B;
    tsr_C->need_remap = need_remap_C;
    printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
    TAU_FSTOP(map_tensors);
    return DIST_TENSOR_ERROR;
  }
  tsr_A->is_mapped = 1;
  tsr_B->is_mapped = 1;
  tsr_C->is_mapped = 1;
#if DEBUG > 2
  if (!check_contraction_mapping(type))
    printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
  else if (global_comm->rank == 0) printf("Mapping successful!\n");
#endif
  LIBT_ASSERT(check_contraction_mapping(type));


  nvirt_all = -1;
  old_nvirt_all = -2;
  while (nvirt_all < MIN_NVIRT){
    old_nvirt_all = nvirt_all;
    set_padding(tsr_A);
    set_padding(tsr_B);
    set_padding(tsr_C);
    *ctrf = construct_contraction(type, buffer, buffer_len, func_ptr, 
                                  alpha, beta, 0, NULL, &nvirt_all);
    delete *ctrf;
    /* If this cannot be stretched */
    if (old_nvirt_all == nvirt_all || nvirt_all > MAX_NVIRT){
      clear_mapping(tsr_A);
      clear_mapping(tsr_B);
      clear_mapping(tsr_C);
      set_padding(tsr_A);
      set_padding(tsr_B);
      set_padding(tsr_C);
      tsr_A->itopo = gtopo/6;
      tsr_B->itopo = gtopo/6;
      tsr_C->itopo = gtopo/6;

      ret = map_to_topology(type->tid_A, type->tid_B, type->tid_C, type->idx_map_A,
                            type->idx_map_B, type->idx_map_C, gtopo/6, gtopo%6, 
                            idx_arr, idx_ctr, idx_extra, idx_no_ctr);
      tsr_A->is_mapped = 1;
      tsr_B->is_mapped = 1;
      tsr_C->is_mapped = 1;
      break;
    }
    if (nvirt_all < MIN_NVIRT){
      stretch_virt(tsr_A->ndim, 2, tsr_A->edge_map);
      stretch_virt(tsr_B->ndim, 2, tsr_B->edge_map);
      stretch_virt(tsr_C->ndim, 2, tsr_C->edge_map);
    }
  }
  set_padding(tsr_A);
  set_padding(tsr_B);
  set_padding(tsr_C);
  *ctrf = construct_contraction(type, buffer, buffer_len, func_ptr, 
                                alpha, beta, 0, NULL, &nvirt_all);
#if DEBUG >= 2
  if (global_comm->rank == 0)
    printf("New mappings:\n");
  print_map(stdout, type->tid_A);
  print_map(stdout, type->tid_B);
  print_map(stdout, type->tid_C);
#endif
 
      
  memuse = MAX((uint64_t)(*ctrf)->mem_rec(), (uint64_t)(tsr_A->size+tsr_B->size+tsr_C->size)*sizeof(dtype)*2);
  if (global_comm->rank == 0)
    DPRINTF(1,"Contraction will use %E bytes per processor out of %E available memory\n",
            (double)memuse,(double)proc_bytes_available());
          
  if (tsr_A->is_cyclic == 0 &&
      tsr_B->is_cyclic == 0 &&
      tsr_C->is_cyclic == 0){
    tsr_A->is_cyclic = 0;
    tsr_B->is_cyclic = 0;
    tsr_C->is_cyclic = 0;
  } else {
    tsr_A->is_cyclic = 1;
    tsr_B->is_cyclic = 1;
    tsr_C->is_cyclic = 1;
  }
  /* redistribute tensor data */
  need_remap = need_remap_A;
  if (tsr_A->itopo == old_topo_A){
    for (d=0; d<tsr_A->ndim; d++){
      if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
        need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_tensor(type->tid_A, tsr_A, &topovec[tsr_A->itopo], old_size_A, 
                 old_phase_A, old_rank_A, old_virt_dim_A, 
                 old_pe_lda_A, was_padded_A, was_cyclic_A, 
                 old_padding_A, old_edge_len_A, global_comm);
  need_remap = need_remap_B;
  if (tsr_B->itopo == old_topo_B){
    for (d=0; d<tsr_B->ndim; d++){
      if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
        need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_tensor(type->tid_B, tsr_B, &topovec[tsr_A->itopo], old_size_B, 
                 old_phase_B, old_rank_B, old_virt_dim_B, 
                 old_pe_lda_B, was_padded_B, was_cyclic_B, 
                 old_padding_B, old_edge_len_B, global_comm);
  need_remap = need_remap_C;
  if (tsr_C->itopo == old_topo_C){
    for (d=0; d<tsr_C->ndim; d++){
      if (!comp_dim_map(&tsr_C->edge_map[d],&old_map_C[d]))
        need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_tensor(type->tid_C, tsr_C, &topovec[tsr_A->itopo], old_size_C, 
                 old_phase_C, old_rank_C, old_virt_dim_C, 
                 old_pe_lda_C, was_padded_C, was_cyclic_C, 
                 old_padding_C, old_edge_len_C, global_comm);
  
  (*ctrf)->A    = tsr_A->data;
  (*ctrf)->B    = tsr_B->data;
  (*ctrf)->C    = tsr_C->data;

  free( old_phase_A );          free( old_rank_A );
  free( old_virt_dim_A );       free( old_pe_lda_A );
  free( old_padding_A );        free( old_edge_len_A );
  free( old_phase_B );          free( old_rank_B );
  free( old_virt_dim_B );       free( old_pe_lda_B );
  free( old_padding_B );        free( old_edge_len_B );
  free( old_phase_C );          free( old_rank_C );
  free( old_virt_dim_C );       free( old_pe_lda_C );
  free( old_padding_C );        free( old_edge_len_C );
  
  for (i=0; i<tsr_A->ndim; i++)
    clear_mapping(old_map_A+i);
  for (i=0; i<tsr_B->ndim; i++)
    clear_mapping(old_map_B+i);
  for (i=0; i<tsr_C->ndim; i++)
    clear_mapping(old_map_C+i);
  free(old_map_A);
  free(old_map_B);
  free(old_map_C);

  free_buffer_space((void*)idx_arr);
  free_buffer_space((void*)idx_no_ctr);
  free_buffer_space((void*)idx_ctr);
  free_buffer_space((void*)idx_extra);
  
  TAU_FSTOP(map_tensors);


  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief map the indices which are common in a sum
 *
 * \param idx_arr array of index mappings of size ndim*idx_num that
 *        lists the indices (or -1) of A,B,... 
 *        corresponding to every global index
 * \param idx_sum specification of which indices are being contracted
 * \param num_tot total number of indices
 * \param num_sum number of indices being contracted over
 * \param tid_A id of A
 * \param tid_B id of B
 * \param topo topology to map to
 * \param idx_num is then number of tensors (2 for sum)
 */
template<typename dtype>
int dist_tensor<dtype>::
    map_sum_indices(int const *         idx_arr,
                    int const *         idx_sum,
                    int const           num_tot,
                    int const           num_sum,
                    int const           tid_A,
                    int const           tid_B,
                    topology const *    topo,
                    int const           idx_num){
  int tsr_ndim, isum, iA, iB, i, j, jsum, jX, stat;
  int * tsr_edge_len, * tsr_sym_table, * restricted;
  mapping * sum_map;

  tensor<dtype> * tsr_A, * tsr_B;
  TAU_FSTART(map_sum_indices);

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];

  tsr_ndim = num_sum;

  get_buffer_space(tsr_ndim*sizeof(int),                (void**)&restricted);
  get_buffer_space(tsr_ndim*sizeof(int),                (void**)&tsr_edge_len);
  get_buffer_space(tsr_ndim*tsr_ndim*sizeof(int),       (void**)&tsr_sym_table);
  get_buffer_space(tsr_ndim*sizeof(mapping),            (void**)&sum_map);

  memset(tsr_sym_table, 0, tsr_ndim*tsr_ndim*sizeof(int));
  memset(restricted, 0, tsr_ndim*sizeof(int));

  for (i=0; i<tsr_ndim; i++){ 
    sum_map[i].type             = NOT_MAPPED; 
    sum_map[i].has_child        = 0;
    sum_map[i].np               = 1;
  }

  /* Map a tensor of dimension.
   * Set the edge lengths and symmetries according to those in sum dims of A and B.
   * This gives us a mapping for the common mapped dimensions of tensors A and B. */
  for (i=0; i<num_sum; i++){
    isum = idx_sum[i];
    iA = idx_arr[isum*idx_num+0];
    iB = idx_arr[isum*idx_num+1];

    tsr_edge_len[i] = tsr_A->edge_len[iA];

    /* Check if A has symmetry among the dimensions being contracted over.
     * Ignore symmetry with non-contraction dimensions.
     * FIXME: this algorithm can be more efficient but should not be a bottleneck */
    if (tsr_A->sym[iA] != NS){
      for (j=0; j<num_sum; j++){
        jsum = idx_sum[j];
        jX = idx_arr[jsum*idx_num+0];
        if (jX == iA+1){
          /* FIXME: eeeewww... */
          tsr_sym_table[i*tsr_ndim+j] = 1;
          tsr_sym_table[j*tsr_ndim+i] = 1;
        }
      }
    }
    if (tsr_B->sym[iB] != NS){
      for (j=0; j<num_sum; j++){
        jsum = idx_sum[j];
        jX = idx_arr[jsum*idx_num+1];
        if (jX == iB+1){
          tsr_sym_table[i*tsr_ndim+j] = 1;
          tsr_sym_table[j*tsr_ndim+i] = 1;
        }
      }
    }
  }
  /* Run the mapping algorithm on this construct */
  stat = map_tensor(topo->ndim,         tsr_ndim, 
                    tsr_edge_len,       tsr_sym_table,
                    restricted,         topo->dim_comm,
                    NULL,               0,
                    sum_map);

  if (stat == DIST_TENSOR_ERROR){
    TAU_FSTOP(map_sum_indices);
    return DIST_TENSOR_ERROR;
  }
  
  /* define mapping of tensors A and B according to the mapping of sum dims */
  if (stat == DIST_TENSOR_SUCCESS){
    for (i=0; i<num_sum; i++){
      isum = idx_sum[i];
      iA = idx_arr[isum*idx_num+0];
      iB = idx_arr[isum*idx_num+1];

      copy_mapping(1, &sum_map[i], &tsr_A->edge_map[iA]);
      copy_mapping(1, &sum_map[i], &tsr_B->edge_map[iB]);
    }
  }
  free_buffer_space(restricted);
  free_buffer_space(tsr_edge_len);
  free_buffer_space(tsr_sym_table);
  for (i=0; i<num_sum; i++){
    clear_mapping(sum_map+i);
  }
  free_buffer_space(sum_map);

  TAU_FSTOP(map_sum_indices);
  return stat;
}


/**
 * \brief map the indices over which we will be contracting
 *
 * \param idx_arr array of index mappings of size ndim*3 that
 *        lists the indices (or -1) of A,B,C 
 *        corresponding to every global index
 * \param idx_ctr specification of which indices are being contracted
 * \param num_tot total number of indices
 * \param num_ctr number of indices being contracted over
 * \param tid_A id of A
 * \param tid_B id of B
 * \param topo topology to map to
 */

template<typename dtype>
int dist_tensor<dtype>::
    map_ctr_indices(int const *         idx_arr,
                    int const *         idx_ctr,
                    int const           num_tot,
                    int const           num_ctr,
                    int const           tid_A,
                    int const           tid_B,
                    topology const *    topo){
  int tsr_ndim, ictr, iA, iB, i, j, jctr, jX, stat;
  int * tsr_edge_len, * tsr_sym_table, * restricted;
  mapping * ctr_map;

  tensor<dtype> * tsr_A, * tsr_B;
  TAU_FSTART(map_ctr_indices);

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];

  tsr_ndim = num_ctr*2;

  get_buffer_space(tsr_ndim*sizeof(int),                (void**)&restricted);
  get_buffer_space(tsr_ndim*sizeof(int),                (void**)&tsr_edge_len);
  get_buffer_space(tsr_ndim*tsr_ndim*sizeof(int),       (void**)&tsr_sym_table);
  get_buffer_space(tsr_ndim*sizeof(mapping),            (void**)&ctr_map);

  memset(tsr_sym_table, 0, tsr_ndim*tsr_ndim*sizeof(int));
  memset(restricted, 0, tsr_ndim*sizeof(int));

  for (i=0; i<tsr_ndim; i++){ 
    ctr_map[i].type             = NOT_MAPPED; 
    ctr_map[i].has_child        = 0; 
    ctr_map[i].np               = 1; 
  }

  /* Map a tensor of dimension 2*num_ctr, with symmetries among each pair.
   * Set the edge lengths and symmetries according to those in ctr dims of A and B.
   * This gives us a mapping for the contraction dimensions of tensors A and B. */
  for (i=0; i<num_ctr; i++){
    ictr = idx_ctr[i];
    iA = idx_arr[ictr*3+0];
    iB = idx_arr[ictr*3+1];

    tsr_edge_len[2*i+0] = tsr_A->edge_len[iA];
    tsr_edge_len[2*i+1] = tsr_A->edge_len[iA];

    tsr_sym_table[2*i*tsr_ndim+2*i+1] = 1;
    tsr_sym_table[(2*i+1)*tsr_ndim+2*i] = 1;

    /* Check if A has symmetry among the dimensions being contracted over.
     * Ignore symmetry with non-contraction dimensions.
     * FIXME: this algorithm can be more efficient but should not be a bottleneck */
    if (tsr_A->sym[iA] != NS){
      for (j=0; j<num_ctr; j++){
        jctr = idx_ctr[j];
        jX = idx_arr[jctr*3+0];
        if (jX == iA+1){
          /* FIXME: eeeewww... */
          tsr_sym_table[2*i*tsr_ndim+2*j] = 1;
          tsr_sym_table[2*i*tsr_ndim+2*j+1] = 1;
          tsr_sym_table[2*j*tsr_ndim+2*i] = 1;
          tsr_sym_table[2*j*tsr_ndim+2*i+1] = 1;
          tsr_sym_table[(2*i+1)*tsr_ndim+2*j] = 1;
          tsr_sym_table[(2*i+1)*tsr_ndim+2*j+1] = 1;
          tsr_sym_table[(2*j+1)*tsr_ndim+2*i] = 1;
          tsr_sym_table[(2*j+1)*tsr_ndim+2*i+1] = 1;
        }
      }
    }
    if (tsr_B->sym[iB] != NS){
      for (j=0; j<num_ctr; j++){
        jctr = idx_ctr[j];
        jX = idx_arr[jctr*3+1];
        if (jX == iB+1){
          tsr_sym_table[2*i*tsr_ndim+2*j] = 1;
          tsr_sym_table[2*i*tsr_ndim+2*j+1] = 1;
          tsr_sym_table[2*j*tsr_ndim+2*i] = 1;
          tsr_sym_table[2*j*tsr_ndim+2*i+1] = 1;
          tsr_sym_table[(2*i+1)*tsr_ndim+2*j] = 1;
          tsr_sym_table[(2*i+1)*tsr_ndim+2*j+1] = 1;
          tsr_sym_table[(2*j+1)*tsr_ndim+2*i] = 1;
          tsr_sym_table[(2*j+1)*tsr_ndim+2*i+1] = 1;
        }
      }
    }
  }
  /* Run the mapping algorithm on this construct */
  stat = map_tensor(topo->ndim,         tsr_ndim, 
                    tsr_edge_len,       tsr_sym_table,
                    restricted,         topo->dim_comm,
                    NULL,               0,
                    ctr_map);

  if (stat == DIST_TENSOR_ERROR)
    return DIST_TENSOR_ERROR;
  
  /* define mapping of tensors A and B according to the mapping of ctr dims */
  if (stat == DIST_TENSOR_SUCCESS){
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

/*      tsr_A->edge_map[iA] = ctr_map[2*i+0];
      tsr_B->edge_map[iB] = ctr_map[2*i+1];*/
      copy_mapping(1, &ctr_map[2*i+0], &tsr_A->edge_map[iA]);
      copy_mapping(1, &ctr_map[2*i+1], &tsr_B->edge_map[iB]);
    }
  }
  free_buffer_space(restricted);
  free_buffer_space(tsr_edge_len);
  free_buffer_space(tsr_sym_table);
  for (i=0; i<2*num_ctr; i++){
    clear_mapping(ctr_map+i);
  }
  free_buffer_space(ctr_map);

  TAU_FSTOP(map_ctr_indices);
  return stat;
}

/**
 * \brief map the indices over which we will not be contracting
 *
 * \param idx_arr array of index mappings of size ndim*3 that
 *        lists the indices (or -1) of A,B,C 
 *        corresponding to every global index
 * \param idx_noctr specification of which indices are not being contracted
 * \param num_tot total number of indices
 * \param num_noctr number of indices not being contracted over
 * \param tid_A id of A
 * \param tid_B id of B
 * \param tid_C id of C
 * \param topo topology to map to
 */
template<typename dtype>
int dist_tensor<dtype>::
    map_no_ctr_indices(int const *              idx_arr,
                       int const *              idx_no_ctr,
                       int const                num_tot,
                       int const                num_no_ctr,
                       int const                tid_A,
                       int const                tid_B,
                       int const                tid_C,
                       topology const *         topo){
  int stat, i, inoctr, iA, iB, iC;

  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  
  TAU_FSTART(map_noctr_indices);

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  tsr_C = tensors[tid_C];

  /* Map remainders of A and B to remainders of phys grid */
  stat = map_tensor_rem(topo->ndim, topo->dim_comm, tsr_A, 1);
  if (stat != DIST_TENSOR_SUCCESS){
    TAU_FSTOP(map_noctr_indices);
    return stat;
  }
  /* The above 2 mappings implictly give us a mapping for C */
  for (i=0; i<num_no_ctr; i++){
    inoctr = idx_no_ctr[i];
    iA = idx_arr[3*inoctr+0];
    iC = idx_arr[3*inoctr+2];

    
    if (iA != -1 && iC != -1){
      copy_mapping(1, tsr_A->edge_map + iA, tsr_C->edge_map + iC); 
    } 
  }
  stat = map_tensor_rem(topo->ndim, topo->dim_comm, tsr_C, 0);
  if (stat != DIST_TENSOR_SUCCESS){
    TAU_FSTOP(map_noctr_indices);
    return stat;
  }
  /* The above 2 mappings implictly give us a mapping for C */
  for (i=0; i<num_no_ctr; i++){
    inoctr = idx_no_ctr[i];
    iB = idx_arr[3*inoctr+1];
    iC = idx_arr[3*inoctr+2];

    
    if (iB != -1 && iC != -1){
      copy_mapping(1, tsr_C->edge_map + iC, tsr_B->edge_map + iB); 
    }
//    printf("C mapping %d is type %d, np = %d\n",iC,tsr_C->edge_map[iC].type,tsr_c->edge_map[iC].np);
  }
  TAU_FSTOP(map_noctr_indices);

  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief create virtual mapping for idx_maps that have repeating indices
 * \param[in] tid tensor id
 * \param[in] idx_map mapping of tensor indices to contraction map
 */
template<typename dtype>
int dist_tensor<dtype>::
    map_self_indices(int const  tid,
                     int const* idx_map){
  int iR, max_idx, i, ret;
  int * idx_arr, * stable;
  tensor<dtype> * tsr;

  tsr = tensors[tid];
  
  max_idx = -1;
  for (i=0; i<tsr->ndim; i++){
    if (idx_map[i] > max_idx) max_idx = idx_map[i];
  }
  max_idx++;


  get_buffer_space(sizeof(int)*max_idx, (void**)&idx_arr);
  get_buffer_space(sizeof(int)*tsr->ndim*tsr->ndim, (void**)&stable);
  memcpy(stable, tsr->sym_table, sizeof(int)*tsr->ndim*tsr->ndim);

  std::fill(idx_arr, idx_arr+max_idx, -1);

  /* Go in reverse, since the first index of the diagonal set will be mapped */
  for (i=0; i<tsr->ndim; i++){
    iR = idx_arr[idx_map[i]];
    if (iR != -1){
      stable[iR*tsr->ndim+i] = 1;
      stable[i*tsr->ndim+iR] = 1;
      LIBT_ASSERT(tsr->edge_map[iR].type != PHYSICAL_MAP);
      if (tsr->edge_map[iR].type == NOT_MAPPED){
        tsr->edge_map[iR].type = VIRTUAL_MAP;
        tsr->edge_map[iR].np = 1;
        tsr->edge_map[iR].has_child = 0;
      }
    }
    idx_arr[idx_map[i]] = i;
  }

  ret = map_symtsr(tsr->ndim, stable, tsr->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;

  free_buffer_space(idx_arr);
  free_buffer_space(stable);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief map the indices which are indexed only for A or B or C
 *
 * \param idx_arr array of index mappings of size ndim*3 that
 *        lists the indices (or -1) of A,B,C 
 *        corresponding to every global index
 * \param idx_extra specification of which indices are not being contracted
 * \param num_extra number of indices not being contracted over
 * \param tid_A id of A
 * \param tid_B id of B
 * \param tid_B id of C
 */
template<typename dtype>
int dist_tensor<dtype>::
    map_extra_indices(int const *       idx_arr,
                      int const *       idx_extra,
                      int const         num_extra,
                      int const         tid_A,
                      int const         tid_B,
                      int const         tid_C){
  int i, iA, iB, iC, iextra;

  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  tsr_C = tensors[tid_C];

  for (i=0; i<num_extra; i++){
    iextra = idx_extra[i];
    iA = idx_arr[3*iextra+0];
    iB = idx_arr[3*iextra+1];
    iC = idx_arr[3*iextra+2];

    if (iA != -1){
      LIBT_ASSERT(tsr_A->edge_map[iA].type != PHYSICAL_MAP);
      if (tsr_A->edge_map[iA].type == NOT_MAPPED){
        tsr_A->edge_map[iA].type = VIRTUAL_MAP;
        tsr_A->edge_map[iA].np = 1;
        tsr_A->edge_map[iA].has_child = 0;
      }
    } else {
      if (iB != -1) {
        LIBT_ASSERT(tsr_B->edge_map[iB].type != PHYSICAL_MAP);
        if (tsr_B->edge_map[iB].type == NOT_MAPPED){
          tsr_B->edge_map[iB].type = VIRTUAL_MAP;
          tsr_B->edge_map[iB].np = 1;
          tsr_B->edge_map[iB].has_child = 0;
        }
      } else {
        LIBT_ASSERT(iC != -1);
        LIBT_ASSERT(tsr_C->edge_map[iC].type != PHYSICAL_MAP);
        if (tsr_C->edge_map[iC].type == NOT_MAPPED){
          tsr_C->edge_map[iC].type = VIRTUAL_MAP;
          tsr_C->edge_map[iC].np = 1;
          tsr_C->edge_map[iC].has_child = 0;
        }
      }
    }
  }
  return DIST_TENSOR_SUCCESS;
}



/**
 * \brief maps tensors to topology 
 *        with certain tensor ordering e.g. BCA
 *
 * \param tid_A id of tensor A
 * \param tid_B id of tensor B
 * \param tid_C id of tensor C
 * \param idx_map_A contraction index mapping of A
 * \param idx_map_B contraction index mapping of B
 * \param idx_map_C contraction index mapping of C
 * \param itopo topology index
 * \param order order of tensors (BCA, ACB, ABC, etc.)
 * \param idx_ctr buffer for contraction index storage
 * \param idx_extra buffer for extra index storage
 * \param idx_no_ctr buffer for non-contracted index storage
 */
template<typename dtype>
int dist_tensor<dtype>::
    map_to_topology(int const           tid_A,
                    int const           tid_B,
                    int const           tid_C,
                    int const *         idx_map_A,
                    int const *         idx_map_B,
                    int const *         idx_map_C,
                    int const           itopo,
                    int const           order,
                    int *               idx_arr,
                    int *               idx_ctr,
                    int *               idx_extra,
                    int *               idx_no_ctr){
  int tA, tB, tC, num_tot, num_ctr, num_no_ctr, num_extra, i, ret;
  int const * map_A, * map_B, * map_C;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  switch (order){
    case 0:
      tA = tid_A;
      tB = tid_B;
      tC = tid_C;
      map_A = idx_map_A;
      map_B = idx_map_B;
      map_C = idx_map_C;
      break;
    case 1:
      tA = tid_A;
      tB = tid_C;
      tC = tid_B;
      map_A = idx_map_A;
      map_B = idx_map_C;
      map_C = idx_map_B;
      break;
    case 2:
      tA = tid_B;
      tB = tid_A;
      tC = tid_C;
      map_A = idx_map_B;
      map_B = idx_map_A;
      map_C = idx_map_C;
      break;
    case 3:
      tA = tid_B;
      tB = tid_C;
      tC = tid_A;
      map_A = idx_map_B;
      map_B = idx_map_C;
      map_C = idx_map_A;
      break;
    case 4:
      tA = tid_C;
      tB = tid_A;
      tC = tid_B;
      map_A = idx_map_C;
      map_B = idx_map_A;
      map_C = idx_map_B;
      break;
    case 5:
      tA = tid_C;
      tB = tid_B;
      tC = tid_A;
      map_A = idx_map_C;
      map_B = idx_map_B;
      map_C = idx_map_A;
      break;
    default:
      return DIST_TENSOR_ERROR;
      break;
  }
  
  tsr_A = tensors[tA];
  tsr_B = tensors[tB];
  tsr_C = tensors[tC];

  inv_idx(tsr_A->ndim, map_A, tsr_A->edge_map,
          tsr_B->ndim, map_B, tsr_B->edge_map,
          tsr_C->ndim, map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);
  num_ctr = 0, num_no_ctr = 0, num_extra = 0;
  for (i=0; i<num_tot; i++){
    if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1 && idx_arr[3*i+2] != -1){
      idx_no_ctr[num_no_ctr] = i;
      num_no_ctr++;
    } else if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
      idx_ctr[num_ctr] = i;
      num_ctr++;
    } else if (idx_arr[3*i+2] != -1 &&  
                ((idx_arr[3*i+0] != -1) || (idx_arr[3*i+1] != -1))){
      idx_no_ctr[num_no_ctr] = i;
      num_no_ctr++;
    } else {
      idx_extra[num_extra] = i;
      num_extra++;
    }
  }
  tsr_A->itopo = itopo;
  tsr_B->itopo = itopo;
  tsr_C->itopo = itopo;
  
  /* Map the contraction indices of A and B */
  ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, 
                            tA, tB, &topovec[itopo]);
  if (ret == DIST_TENSOR_NEGATIVE) {
    free(idx_arr);
    return DIST_TENSOR_NEGATIVE;
  }
  if (ret == DIST_TENSOR_ERROR) {
    free(idx_arr);
    return DIST_TENSOR_ERROR;
  }


  ret = map_self_indices(tA, map_A);
  if (ret == DIST_TENSOR_NEGATIVE) {
    free(idx_arr);
    return DIST_TENSOR_NEGATIVE;
  }
  if (ret == DIST_TENSOR_ERROR) {
    free(idx_arr);
    return DIST_TENSOR_ERROR;
  }
  ret = map_self_indices(tB, map_B);
  if (ret == DIST_TENSOR_NEGATIVE) {
    free(idx_arr);
    return DIST_TENSOR_NEGATIVE;
  }
  if (ret == DIST_TENSOR_ERROR) {
    free(idx_arr);
    return DIST_TENSOR_ERROR;
  }
  ret = map_self_indices(tC, map_C);
  if (ret == DIST_TENSOR_NEGATIVE) {
    free(idx_arr);
    return DIST_TENSOR_NEGATIVE;
  }
  if (ret == DIST_TENSOR_ERROR) {
    free(idx_arr);
    return DIST_TENSOR_ERROR;
  }
  ret = map_extra_indices(idx_arr, idx_extra, num_extra,
                              tA, tB, tC);
  if (ret == DIST_TENSOR_NEGATIVE) {
    free(idx_arr);
    return DIST_TENSOR_NEGATIVE;
  }
  if (ret == DIST_TENSOR_ERROR) {
    free(idx_arr);
    return DIST_TENSOR_ERROR;
  }


  /* Map C or equivalently, the non-contraction indices of A and B */
  ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, 
                                tA, tB, tC, &topovec[itopo]);
  free(idx_arr);
  if (ret == DIST_TENSOR_NEGATIVE) return DIST_TENSOR_NEGATIVE;
  if (ret == DIST_TENSOR_ERROR) {
    return DIST_TENSOR_ERROR;
  }


  ret = map_symtsr(tsr_A->ndim, tsr_A->sym_table, tsr_A->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;
  ret = map_symtsr(tsr_B->ndim, tsr_B->sym_table, tsr_B->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;
  ret = map_symtsr(tsr_C->ndim, tsr_C->sym_table, tsr_C->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;
  return DIST_TENSOR_SUCCESS;

}
/**
 * \brief attempts to remap 3 tensors to the same topology if possible
 * \param[in,out] tid_A a tensor
 * \param[in,out] tid_B a tensor
 * \param[in,out] tid_C a tensor
 */
template<typename dtype>
int dist_tensor<dtype>::try_topo_morph(int const tid_A,
                                       int const tid_B,
                                       int const tid_C){
  int itA, itB, itC, ret;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tensor<dtype> * tsr_keep, * tsr_change_A, * tsr_change_B;
  
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  tsr_C = tensors[tid_C];

  itA = tsr_A->itopo;
  itB = tsr_B->itopo;
  itC = tsr_C->itopo;

  if (itA == itB && itB == itC){
    return DIST_TENSOR_SUCCESS;
  }

  if (topovec[itA].ndim >= topovec[itB].ndim){
    if (topovec[itA].ndim >= topovec[itC].ndim){
      tsr_keep = tsr_A;
      tsr_change_A = tsr_B;
      tsr_change_B = tsr_C;
    } else {
      tsr_keep = tsr_C;
      tsr_change_A = tsr_A;
      tsr_change_B = tsr_B;
    } 
  } else {
    if (topovec[itB].ndim >= topovec[itC].ndim){
      tsr_keep = tsr_B;
      tsr_change_A = tsr_A;
      tsr_change_B = tsr_C;
    } else {
      tsr_keep = tsr_C;
      tsr_change_A = tsr_A;
      tsr_change_B = tsr_B;
    }
  }
  
  itA = tsr_change_A->itopo;
  itB = tsr_change_B->itopo;
  itC = tsr_keep->itopo;

  if (itA != itC){
    ret = can_morph(&topovec[itC], &topovec[itA]);
    if (!ret)
      return DIST_TENSOR_NEGATIVE;
  }
  if (itB != itC){
    ret = can_morph(&topovec[itC], &topovec[itB]);
    if (!ret)
      return DIST_TENSOR_NEGATIVE;
  }
  
  if (itA != itC){
    morph_topo(&topovec[itC], &topovec[itA], 
               tsr_change_A->ndim, tsr_change_A->edge_map);
    tsr_change_A->itopo = itC;
  }
  if (itB != itC){
    morph_topo(&topovec[itC], &topovec[itB], 
               tsr_change_B->ndim, tsr_change_B->edge_map);
    tsr_change_B->itopo = itC;
  }
  return DIST_TENSOR_SUCCESS;

}


