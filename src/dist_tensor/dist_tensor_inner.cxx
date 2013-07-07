/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#define MAX_GROWTH 8

/**
 * \brief undo the inner blocking of a tensor
 *
 * \param tsr the tensor on hand
 */
template<typename dtype>
void dist_tensor<dtype>::unmap_inner(tensor<dtype> * tsr){
  if (tsr->is_inner_mapped){
    int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda;
    int * old_padding, * old_edge_len;
    int was_padded, was_cyclic;
    long_int old_size;
    tensor<dtype> * itsr = tensors[tsr->rec_tid];
    save_mapping(itsr, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda, 
                 &old_size, &was_padded, &was_cyclic, &old_padding, 
                 &old_edge_len, &inner_topovec[itsr->itopo], 1);  
    clear_mapping(itsr);
    set_padding(itsr,1);
    remap_inr_tsr(tsr, itsr, old_size, old_phase, old_rank, old_virt_dim,
                  old_pe_lda, was_padded, was_cyclic, old_padding,
                  old_edge_len, NULL);
    CTF_free(old_phase);
    CTF_free(old_rank);
    CTF_free(old_virt_dim);
    CTF_free(old_pe_lda);
    if (was_padded)
      CTF_free(old_padding);
    CTF_free(old_edge_len);
    del_tsr(tsr->rec_tid);
    tsr->is_inner_mapped = 0;
    set_padding(tsr);
  }
  if (tsr->is_folded){
    unfold_tsr(tsr);
    set_padding(tsr);
  }
}


/**
 * \brief determines whether symmetries are preserved in this contraction
 *
 * \param[in] type contraction specification
 * \param[in] tsr_A tensor A
 * \param[in] tsr_B tensor B
 * \param[in] tsr_C tensor C
 * \return true if symmetry is preserved
 */
template<typename dtype>
int is_sym_preserved( CTF_ctr_type_t const *    type, 
                      tensor<dtype> const *     tsr_A, 
                      tensor<dtype> const *     tsr_B,
                      tensor<dtype> const *     tsr_C){
  int i;

  for (i=0; i<tsr_A->ndim; i++){
    
  }
  return 0;
}

/**
 * \brief calculate the dimensions of the matrix 
 *        the contraction gets reduced to
 *
 * \param[in] type contraction specification
 * \param[in] ordering_A the dimensional-ordering of the inner mapping of A
 * \param[in] ordering_B the dimensional-ordering of the inner mapping of B
 * \param[in] tsr_A tensor A
 * \param[in] tsr_B tensor B
 * \param[in] tsr_C tensor C
 * \param[out] inner_prm parameters includng n,m,k
 */
template<typename dtype>
void calc_nmk(CTF_ctr_type_t const *    type, 
              int const *               ordering_A, 
              int const *               ordering_B, 
              tensor<dtype> const *     tsr_A, 
              tensor<dtype> const *     tsr_B,
              tensor<dtype> const *     tsr_C,
              iparam *                  inner_prm) {
  int i, num_ctr, num_tot;
  int * idx_arr;
  int * phase_A, * phase_B;
  iparam prm;

  phase_A = calc_phase<dtype>(tsr_A);
  phase_B = calc_phase<dtype>(tsr_B);

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);
  num_ctr = 0;
  for (i=0; i<num_tot; i++){
    if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
      num_ctr++;
    } 
  }
  prm.m = 1;
  prm.n = 1;
  prm.k = 1;
  for (i=0; i<tsr_A->ndim; i++){
    if (i >= num_ctr)
      prm.m = prm.m * phase_A[ordering_A[i]];
    else 
      prm.k = prm.k * phase_A[ordering_A[i]];
  }
  for (i=0; i<tsr_B->ndim; i++){
    if (i >= num_ctr)
      prm.n = prm.n * phase_B[ordering_B[i]];
  }
  /* This gets set later */
  prm.sz_C = 0;
  CTF_free(idx_arr);
  CTF_free(phase_A);
  CTF_free(phase_B);
  *inner_prm = prm;  
}


/**
 * \brief calculate the edge lengths of the sub-blocks
 *
 * \param[in] tsr tensor on hand
 * \param[out] psub_edge_len pointer to int array which will be the edge lengths
 */
template<typename dtype>
void calc_sub_edge_len(tensor<dtype> *  tsr,
                       int **           psub_edge_len){
  int i;
  int * sub_edge_len, * phase;
  mapping * map;

  CTF_alloc_ptr(tsr->ndim*sizeof(int), (void**)&sub_edge_len);
  CTF_alloc_ptr(tsr->ndim*sizeof(int), (void**)&phase);

  /* Pad the tensor */
  for (i=0; i<tsr->ndim; i++){
    if (tsr->edge_map[i].type == NOT_MAPPED)
      phase[i] = 1;
    else if (tsr->edge_map[i].type == PHYSICAL_MAP){
      phase[i] = tsr->edge_map[i].np;
      map = tsr->edge_map+i;
      while (map->has_child){
        map = map->child;
        phase[i] = phase[i]*map->np;
      } 
    } else {
      LIBT_ASSERT(tsr->edge_map[i].type == VIRTUAL_MAP);
      phase[i] = tsr->edge_map[i].np;
    }
  }

  for (i=0; i<tsr->ndim; i++){
    sub_edge_len[i] = tsr->edge_len[i]/phase[i];
  }
  CTF_free(phase);
  *psub_edge_len = sub_edge_len;
}

/**
 * \brief transposes hierarchical data ordering and reorders 
 *        so as to reduce to matmul
 *
 * \param[in] ndim dimension of tensor
 * \param[in] nb number of sub-tensors
 * \param[in] map mapping from old ordering of sub-blocks 
 *                to new ordering within inner blocks
 * \param[in] blk_size size of symmetric sub-tensor
 * \param[in] dir which way are we going?
 * \param[out] data values to transpose
 */
template<typename dtype>
void inner_transpose(int const          ndim,
                     int const          nb,
                     int const *        map,
                     long_int const     blk_sz,
                     int const          dir,
                     dtype *            data){
  int i, j, mi;
  dtype * swap_data;

  TAU_FSTART(inner_transpose);
  CTF_alloc_ptr(blk_sz*nb*sizeof(dtype), (void**)&swap_data);

  for (i=0; i<nb; i++){
    mi = map[i];
    for (j=0; j<blk_sz; j++){
      if (dir){
        swap_data[i*blk_sz+j] = data[j*nb+mi];
      } else {
        swap_data[j*nb+mi] = data[i*blk_sz+j];
      }
    }
  }
  memcpy(data,swap_data,sizeof(dtype)*blk_sz*nb);
  CTF_free(swap_data);
  TAU_FSTOP(inner_transpose);
}

/**
 * \brief find ordering of indices of tensor to reduce to DGEMM
 *
 * \param[in] type contraction specification
 * \param[out] new_ordering_A the new ordering for indices of A
 * \param[out] new_ordering_B the new ordering for indices of B
 * \param[out] new_ordering_C the new ordering for indices of C
 */
template<typename dtype>
void dist_tensor<dtype>::get_new_ordering(
                            CTF_ctr_type_t const *      type,
                            int **                      new_ordering_A,
                            int **                      new_ordering_B,
                            int **                      new_ordering_C){
  int i, num_tot, num_ctr, idx_ctr, num_no_ctr_A;
  int idx_no_ctr_A, idx_no_ctr_B;
  int * ordering_A, * ordering_B, * ordering_C, * idx_arr;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];
  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&ordering_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&ordering_B);
  CTF_alloc_ptr(sizeof(int)*tsr_C->ndim, (void**)&ordering_C);

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);
  num_ctr = 0, num_no_ctr_A = 0;
  for (i=0; i<num_tot; i++){
    if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
      num_ctr++;
    } else if (idx_arr[3*i] != -1){
      num_no_ctr_A++;
    }
  }
  /* Put all contraction indices up front, put A indices in front for C */
  idx_ctr = 0, idx_no_ctr_A = 0, idx_no_ctr_B = 0;
  for (i=0; i<num_tot; i++){
    if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
      ordering_A[idx_ctr] = idx_arr[3*i];
      ordering_B[idx_ctr] = idx_arr[3*i+1];
      idx_ctr++;
    } else {
      if (idx_arr[3*i] != -1){
        ordering_A[num_ctr+idx_no_ctr_A] = idx_arr[3*i];
        ordering_C[idx_no_ctr_A] = idx_arr[3*i+2];
        idx_no_ctr_A++;
      }
      if (idx_arr[3*i+1] != -1){
        ordering_B[num_ctr+idx_no_ctr_B] = idx_arr[3*i+1];
        ordering_C[num_no_ctr_A+idx_no_ctr_B] = idx_arr[3*i+2];
        idx_no_ctr_B++;
      }
    }
  }
  CTF_free(idx_arr);
  *new_ordering_A = ordering_A;
  *new_ordering_B = ordering_B;
  *new_ordering_C = ordering_C;
}

/**
 * \brief creates mapping from upper-level blocking to inner block
 *
 * \param[in] ndim dimension of tensor
 * \param[in] new_nvirt number of blocks
 * \param[in] phase number of blocks along each dimension
 * \param[in] ordering reordering of indices in tensor
 * \param[out] inner_map mapping that does as specified in brief
 */
inline
void create_inner_map(int const ndim,
                      int const new_nvirt,
                      int const * phase,
                      int const * ordering,
                      int       ** inner_map){
  int i, offset_map, idx_map;
  int * idx, * oidx, * lda, * map;

  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&lda);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&idx);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&oidx);
  CTF_alloc_ptr(sizeof(int)*new_nvirt, (void**)&map);

  if (ndim > 0)
    lda[0] = 1;
  for (i=1; i<ndim; i++){
    lda[i] = lda[i-1]*phase[i-1];
  }

  memset(idx, 0, sizeof(int)*ndim);
  memset(oidx, 0, sizeof(int)*ndim);
  idx_map = 0, offset_map =0;
  for (;;){
    map[offset_map] = idx_map;
    for (i=0; i<ndim; i++){
      idx_map -= idx[i]*lda[i];
      idx[i] = (idx[i]+1)%phase[i];
      idx_map += idx[i]*lda[i];
      if (idx[i] != 0) break;
    }
    for (i=0; i<ndim; i++){
      offset_map -= oidx[i]*lda[ordering[i]];
      oidx[i] = (oidx[i]+1)%phase[ordering[i]];
      offset_map += oidx[i]*lda[ordering[i]];
      if (oidx[i] != 0) break;
    }
    if (i==ndim) break;
  }
  *inner_map = map;
  CTF_free(lda);
  CTF_free(idx);
  CTF_free(oidx);
}

/**
 * \brief permutes the data of a tensor to its new inner layout
 * \param[in,out] otsr outer tensor 
 * \param[in,out] itsr inner tensor 
 * \param[in] old_size size of tensor before redistribution
 * \param[in] old_phase old distribution phase
 * \param[in] old_rank old distribution rank
 * \param[in] old_virt_dim old distribution virtualization
 * \param[in] old_pe_lda old distribution processor ldas
 * \param[in] was_cyclic whether the tensor was mapping cyclically
 * \param[in] was_padded whether the tensor was padded
 * \param[in] old_padding what the padding was
 * \param[in] old_edge_len what the padded edge lengths were
 * \param[in] ordering index reordering 
 */
template<typename dtype>
int dist_tensor<dtype>::
    remap_inr_tsr( tensor<dtype> *      otsr,
                   tensor<dtype> *      itsr,
                   long_int const       old_size,
                   int const *          old_phase,
                   int const *          old_rank,
                   int const *          old_virt_dim,
                   int const *          old_pe_lda,
                   int const            was_padded,
                   int const            was_cyclic,
                   int const *          old_padding,
                   int const *          old_edge_len,
                   int const *          ordering){
  int j, old_nvirt, new_nvirt, outer_nvirt;
  int * new_phase, * new_rank, * new_pe_lda;
  int * new_inner_ordering = NULL;
  dtype * shuffled_data, * start_vdata, * end_vdata;

  CTF_alloc_ptr(sizeof(int)*itsr->ndim,      (void**)&new_rank);
  CTF_alloc_ptr(sizeof(int)*itsr->ndim,      (void**)&new_pe_lda);
  CTF_alloc_ptr(sizeof(dtype)*old_size,      (void**)&start_vdata);

  new_phase = calc_phase<dtype>(itsr);
  
  old_nvirt = 1, new_nvirt = 1; 
  for (j=0; j<itsr->ndim; j++){
    new_rank[j] = 0;
    new_pe_lda[j] = 0;
    new_nvirt = new_nvirt*new_phase[j];
    old_nvirt = old_nvirt*old_phase[j];
  }
  outer_nvirt = (int)calc_nvirt(otsr);

  if (ordering != NULL)
    create_inner_map(itsr->ndim, new_nvirt, new_phase, 
                     ordering, &new_inner_ordering);

  CTF_alloc_ptr(sizeof(dtype)*outer_nvirt*itsr->size, 
                   (void**)&shuffled_data);

  for (j=0; j<outer_nvirt; j++){
    memcpy(start_vdata, otsr->data+j*old_size, old_size*sizeof(dtype)); 
    if (otsr->is_inner_mapped)
      inner_transpose(itsr->ndim, old_nvirt, otsr->inner_ordering,
                      old_size/old_nvirt, 1, start_vdata);
    cyclic_reshuffle(itsr->ndim,
                     old_size,
                     old_edge_len,
                     itsr->sym,
                     old_phase,
                     old_rank,
                     new_pe_lda,
                     was_padded,
                     old_padding,
                     itsr->edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     itsr->is_padded,
                     itsr->padding,
                     old_phase,
                     new_phase,
                     &start_vdata,
                     &end_vdata,
                     NULL,
                     1,
                     1);
    if (new_nvirt > 1 && ordering != NULL)
      inner_transpose(itsr->ndim, new_nvirt, new_inner_ordering,
                      itsr->size/new_nvirt, 0, end_vdata);
    memcpy(shuffled_data+j*itsr->size, end_vdata, itsr->size*sizeof(dtype)); 
    CTF_free(end_vdata);
  }

  CTF_free((void*)new_phase);
  CTF_free((void*)new_rank);
  CTF_free((void*)new_pe_lda);
  CTF_free((void*)otsr->data);
  CTF_free((void*)start_vdata);
    
  if (otsr->is_inner_mapped)
    CTF_free(otsr->inner_ordering);
  if (ordering != NULL){
    otsr->is_inner_mapped = 1;
    otsr->inner_ordering = new_inner_ordering;
  } else
    otsr->is_inner_mapped = 0;
  otsr->size = outer_nvirt*itsr->size;
  otsr->data = shuffled_data;

  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief defines inner topology, mappings to which will produce a 
 *        block-cylic layout
 *
 * \param inner_sz total volume of inner block
 */
template<typename dtype>
int dist_tensor<dtype>::init_inner_topology(int const inner_sz){
  int ndim;
  int * dim_len;
  int i;
  int * srt_dim_len;
  inner_size = inner_sz;
  
  factorize(inner_size, &ndim, &dim_len);

  CTF_alloc_ptr(ndim*sizeof(int), (void**)&srt_dim_len);
  memcpy(srt_dim_len, dim_len, ndim*sizeof(int));

  /* setup dimensional communicators */
  CommData_t ** inner_comm = (CommData_t**)CTF_alloc(ndim*sizeof(CommData_t*));

  std::sort(srt_dim_len, srt_dim_len + ndim);

  for (i=0; i<ndim; i++){
    inner_comm[i] = (CommData_t*)CTF_alloc(sizeof(CommData_t));
    inner_comm[i]->np = srt_dim_len[ndim-i-1];
    inner_comm[i]->rank = 0;
  }
  set_inner_comm(inner_comm, ndim);
  CTF_free(srt_dim_len);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief sets the physical torus topology
 * \param[in] cdt grid communicator
 * \param[in] ndim number of dimensions
 */
template<typename dtype>
void dist_tensor<dtype>::set_inner_comm(CommData_t ** cdt, int const ndim){ 
  topology new_topo;


  new_topo.ndim = ndim;
  new_topo.dim_comm = cdt; 
 
  /* do not duplicate topologies */ 
  if (find_topology(&new_topo, inner_topovec) != -1){
    CTF_free(cdt);
    return;
  }
  inner_topovec.push_back(new_topo);
 
  if (ndim > 1) 
    fold_torus(&new_topo, NULL, this);
}

/**
 * \brief map tensors so that they can be contracted on
 *
 * \param type specification of contraction
 * \param[out] inner_params n,m,k for sequential kernel
 */
template<typename dtype>
int dist_tensor<dtype>::map_inner(CTF_ctr_type_t const * type,
                                  iparam * inner_params){
  int num_tot, i, ret, j, need_remap, d;
//  uint64_t nvirt, tnvirt, bnvirt;
  uint64_t min_size, size;
  int btopo, gtopo;
  int was_padded_A, was_padded_B, was_padded_C;
  int was_cyclic_A, was_cyclic_B, was_cyclic_C;
  long_int old_size_A, old_size_B, old_size_C;
  int * idx_arr, * idx_ctr, * idx_no_ctr, * idx_extra;
  int * old_phase_A, * old_rank_A, * old_virt_dim_A, * old_pe_lda_A;
  int * old_padding_A, * old_edge_len_A;
  int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B;
  int * old_padding_B, * old_edge_len_B;
  int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
  int * old_padding_C, * old_edge_len_C;
  int * ordering_A, * ordering_B, * ordering_C;
  
  mapping * old_map_A, * old_map_B, * old_map_C;
  int old_topo_A, old_topo_B, old_topo_C;
  int tid_A, tid_B, tid_C;
  tensor<dtype> * otsr_A, * otsr_B, * otsr_C;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  CTF_ctr_type_t inner_type = *type;

  otsr_A = tensors[type->tid_A];
  otsr_B = tensors[type->tid_B];
  otsr_C = tensors[type->tid_C];
  if (otsr_A->is_inner_mapped)
    tid_A = otsr_A->rec_tid;
  else {
    int * sub_edge_len;
    calc_sub_edge_len(otsr_A, &sub_edge_len);
    define_tensor(otsr_A->ndim, sub_edge_len, otsr_A->sym, &tid_A, 0);
    otsr_A->rec_tid = tid_A;
    CTF_free(sub_edge_len);
  }
  if (otsr_B->is_inner_mapped)
    tid_B = otsr_B->rec_tid;
  else {
    int * sub_edge_len;
    calc_sub_edge_len(otsr_B, &sub_edge_len);
    define_tensor(otsr_B->ndim, sub_edge_len, otsr_B->sym, &tid_B, 0);
    otsr_B->rec_tid = tid_B;
    CTF_free(sub_edge_len);
  }
  if (otsr_C->is_inner_mapped)
    tid_C = otsr_C->rec_tid;
  else {
    int * sub_edge_len;
    calc_sub_edge_len(otsr_C, &sub_edge_len);
    define_tensor(otsr_C->ndim, sub_edge_len, otsr_C->sym, &tid_C, 0);
    otsr_C->rec_tid = tid_C;
    CTF_free(sub_edge_len);
  }
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  tsr_C = tensors[tid_C];

  inner_type.tid_A = tid_A;
  inner_type.tid_B = tid_B;
  inner_type.tid_C = tid_C;

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);
  CTF_free(idx_arr);
  CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_no_ctr);
  CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_extra);
  CTF_alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_ctr);
  CTF_alloc_ptr(sizeof(mapping)*tsr_A->ndim, (void**)&old_map_A);
  CTF_alloc_ptr(sizeof(mapping)*tsr_B->ndim, (void**)&old_map_B);
  CTF_alloc_ptr(sizeof(mapping)*tsr_C->ndim, (void**)&old_map_C);

  for (i=0; i<tsr_A->ndim; i++){
    old_map_A[i].type           = NOT_MAPPED;
    old_map_A[i].has_child      = 0;
    old_map_A[i].np             = 1;
  }
  for (i=0; i<tsr_B->ndim; i++){
    old_map_B[i].type           = NOT_MAPPED;
    old_map_B[i].has_child      = 0;
    old_map_B[i].np             = 1;
  }
  for (i=0; i<tsr_C->ndim; i++){
    old_map_C[i].type           = NOT_MAPPED;
    old_map_C[i].has_child      = 0;
    old_map_C[i].np             = 1;
  }
  copy_mapping(tsr_A->ndim, tsr_A->edge_map, old_map_A);
  copy_mapping(tsr_B->ndim, tsr_B->edge_map, old_map_B);
  copy_mapping(tsr_C->ndim, tsr_C->edge_map, old_map_C);
  old_topo_A = tsr_A->itopo;
  old_topo_B = tsr_B->itopo;
  old_topo_C = tsr_C->itopo;

#if DEBUG >= 1
  if (global_comm->rank == 0)
    printf("Initial inner mappings:\n");
  print_map(stdout, tid_A, 1, 1);
  print_map(stdout, tid_B, 1, 1);
  print_map(stdout, tid_C, 1, 1);
#endif
  clear_mapping(tsr_A);
  clear_mapping(tsr_B);
  clear_mapping(tsr_C);
  set_padding(tsr_A,1);
  set_padding(tsr_B,1);
  set_padding(tsr_C,1);
  /* Save the current mappings of A, B, C */
  save_mapping(tsr_A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A, 
               &old_size_A, &was_padded_A, &was_cyclic_A, &old_padding_A, 
               &old_edge_len_A, &inner_topovec[tsr_A->itopo], 1);
  save_mapping(tsr_B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B, 
               &old_size_B, &was_padded_B, &was_cyclic_B, &old_padding_B, 
               &old_edge_len_B, &inner_topovec[tsr_B->itopo], 1);
  save_mapping(tsr_C, &old_phase_C, &old_rank_C, &old_virt_dim_C, &old_pe_lda_C, 
               &old_size_C, &was_padded_C, &was_cyclic_C, &old_padding_C, 
               &old_edge_len_C, &inner_topovec[tsr_C->itopo], 1);

  btopo = -1;
  //bnvirt = UINT64_MAX;
  min_size = UINT64_MAX;
  for (j=0; j<6; j++){
    /* Attempt to map to all possible permutations of processor topology */
    for (i=global_comm->rank; i<(int)inner_topovec.size(); i+=global_comm->np){
      clear_mapping(tsr_A);
      clear_mapping(tsr_B);
      clear_mapping(tsr_C);
      set_padding(tsr_A,1);
      set_padding(tsr_B,1);
      set_padding(tsr_C,1);

      ret = map_to_inr_topo(tid_A, tid_B, tid_C, type->idx_map_A,
                            type->idx_map_B, type->idx_map_C, i, j,
                            idx_ctr, idx_extra, idx_no_ctr);

      if (ret == DIST_TENSOR_ERROR) return DIST_TENSOR_ERROR;
      if (ret == DIST_TENSOR_NEGATIVE) continue;
      tsr_A->is_mapped = 1;
      tsr_B->is_mapped = 1;
      tsr_C->is_mapped = 1;

      if (check_contraction_mapping(&inner_type, 1) == 0) continue;
      
      set_padding(tsr_A,1);
      set_padding(tsr_B,1);
      set_padding(tsr_C,1);
      size = tsr_A->size + tsr_B->size + tsr_C->size;
      if ((btopo == -1 || size < min_size) && size < 
          (uint64_t)(old_size_A+old_size_B+old_size_C)*MAX_GROWTH){
        min_size = size;
        btopo = 6*i+j;
      }

      /*nvirt = (uint64_t)calc_nvirt(tsr_A);
      tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_B);
      if (tnvirt < nvirt) nvirt = UINT64_MAX;
      else {
        nvirt = tnvirt;
        tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_C);
        if (tnvirt < nvirt) nvirt = UINT64_MAX;
        else nvirt = tnvirt;
      }
      if (btopo == -1 || nvirt < bnvirt ) {
        bnvirt = nvirt;
        btopo = 6*i+j;      
      }*/
    }
  }
  gtopo = get_best_topo(min_size, btopo, global_comm);
  if (gtopo == INT_MAX || gtopo == -1){
//#if DEBUG >=1
    if (global_comm->rank == 0)
      DPRINTF(1,"Could not map inner contraction\n");
//#endif
    if (otsr_A->is_inner_mapped)
      del_tsr(otsr_A->rec_tid);
    if (otsr_B->is_inner_mapped)
      del_tsr(otsr_B->rec_tid);
    if (otsr_C->is_inner_mapped)
      del_tsr(otsr_C->rec_tid);
    otsr_A->is_inner_mapped = 0;
    otsr_B->is_inner_mapped = 0;
    otsr_C->is_inner_mapped = 0;
    set_padding(otsr_A);
    set_padding(otsr_B);
    set_padding(otsr_C);
    return DIST_TENSOR_NEGATIVE;
  }
  
  clear_mapping(tsr_A);
  clear_mapping(tsr_B);
  clear_mapping(tsr_C);

  tsr_A->itopo = gtopo/6;
  tsr_B->itopo = gtopo/6;
  tsr_C->itopo = gtopo/6;
  
  ret = map_to_inr_topo(tid_A, tid_B, tid_C, type->idx_map_A,
                        type->idx_map_B, type->idx_map_C, gtopo/6, gtopo%6, 
                        idx_ctr, idx_extra, idx_no_ctr);
  tsr_A->is_mapped = 1;
  tsr_B->is_mapped = 1;
  tsr_C->is_mapped = 1;


  if (ret == DIST_TENSOR_NEGATIVE || ret == DIST_TENSOR_ERROR) {
    return DIST_TENSOR_ERROR;
  }
 
  LIBT_ASSERT(check_contraction_mapping(&inner_type, 1));

  set_padding(tsr_A,1);
  set_padding(tsr_B,1);
  set_padding(tsr_C,1);

  otsr_A->size = tsr_A->size;
  otsr_B->size = tsr_B->size;
  otsr_C->size = tsr_C->size;

#if DEBUG >= 1
  if (global_comm->rank == 0)
    printf("New inner mappings:\n");
  print_map(stdout, tid_A, 1, 1);
  print_map(stdout, tid_B, 1, 1);
  print_map(stdout, tid_C, 1, 1);
#endif

  get_new_ordering(&inner_type, &ordering_A, &ordering_B, &ordering_C);
#if DEBUG>=1
  if (global_comm->rank == 0){
    for (i=0; i<tsr_A->ndim; i++){
      printf("ordering_A[%d] = %d\n", i, ordering_A[i]);
    }
    for (i=0; i<tsr_B->ndim; i++){
      printf("ordering_B[%d] = %d\n", i, ordering_B[i]);
    }
    for (i=0; i<tsr_C->ndim; i++){
      printf("ordering_C[%d] = %d\n", i, ordering_C[i]);
    }
  }
#endif

  calc_nmk(&inner_type, ordering_A, ordering_B, 
           tsr_A, tsr_B, tsr_C, inner_params);

  /* redistribute tensor data */
  need_remap = 0;
  if (tsr_A->itopo == old_topo_A){
    for (d=0; d<tsr_A->ndim; d++){
      if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
              need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_inr_tsr(otsr_A, tsr_A, 
                  old_size_A, old_phase_A, old_rank_A, old_virt_dim_A, 
                  old_pe_lda_A, was_padded_A, was_cyclic_A, 
                  old_padding_A, old_edge_len_A, ordering_A);
  need_remap = 0;
  if (tsr_B->itopo == old_topo_B){
    for (d=0; d<tsr_B->ndim; d++){
      if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
              need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_inr_tsr(otsr_B, tsr_B, 
                  old_size_B, old_phase_B, old_rank_B, old_virt_dim_B, 
                  old_pe_lda_B, was_padded_B, was_cyclic_B, 
                  old_padding_B, old_edge_len_B, ordering_B);
  need_remap = 0;
  if (tsr_C->itopo == old_topo_C){
    for (d=0; d<tsr_C->ndim; d++){
      if (!comp_dim_map(&tsr_C->edge_map[d],&old_map_C[d]))
              need_remap = 1;
    }
  } else
    need_remap = 1;
  if (need_remap)
    remap_inr_tsr(otsr_C, tsr_C, 
                  old_size_C, old_phase_C, old_rank_C, old_virt_dim_C, 
                  old_pe_lda_C, was_padded_C, was_cyclic_C, 
                  old_padding_C, old_edge_len_C, ordering_C);
  

  CTF_free( old_phase_A );          CTF_free( old_rank_A );
  CTF_free( old_virt_dim_A );       CTF_free( old_pe_lda_A );
  CTF_free( old_padding_A );        CTF_free( old_edge_len_A );
  CTF_free( old_phase_B );          CTF_free( old_rank_B );
  CTF_free( old_virt_dim_B );       CTF_free( old_pe_lda_B );
  CTF_free( old_padding_B );        CTF_free( old_edge_len_B );
  CTF_free( old_phase_C );          CTF_free( old_rank_C );
  CTF_free( old_virt_dim_C );       CTF_free( old_pe_lda_C );
  CTF_free( old_padding_C );        CTF_free( old_edge_len_C );
  
  CTF_free((void*)idx_no_ctr);
  CTF_free((void*)idx_ctr);
  CTF_free((void*)idx_extra);
  CTF_free(old_map_A);
  CTF_free(old_map_B);
  CTF_free(old_map_C);
  CTF_free(ordering_A);
  CTF_free(ordering_B);
  CTF_free(ordering_C);

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
    map_to_inr_topo(int const           tid_A,
                    int const           tid_B,
                    int const           tid_C,
                    int const *         idx_map_A,
                    int const *         idx_map_B,
                    int const *         idx_map_C,
                    int const           itopo,
                    int const           order,
                    int *               idx_ctr,
                    int *               idx_extra,
                    int *               idx_no_ctr){
  int tA, tB, tC, num_tot, num_ctr, num_no_ctr, num_extra, i, ret, j;
  int * idx_arr;
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
      return DIST_TENSOR_NEGATIVE;
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
      return DIST_TENSOR_NEGATIVE;
    }
  }
  tsr_A->itopo = itopo;
  tsr_B->itopo = itopo;
  tsr_C->itopo = itopo;
 
  /* No weird crap like that for inner mappings */
  if (num_no_ctr == 0 || num_ctr == 0) return DIST_TENSOR_NEGATIVE;
 
  /* Map the contraction indices of A and B */
  ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, 
                        tA, tB, &inner_topovec[itopo]);
  /* We can only do an inner mapping if there are no self 
      or extra indices */
  for (i=0; i<tsr_A->ndim; i++){
    for (j=0; j<tsr_A->ndim; j++){
      if (i!=j && map_A[i] == map_A[j]){
        CTF_free(idx_arr);
        return DIST_TENSOR_NEGATIVE;
      }
    }
  }
  for (i=0; i<tsr_B->ndim; i++){
    for (j=0; j<tsr_B->ndim; j++){
      if (i!=j && map_B[i] == map_B[j]){
        CTF_free(idx_arr);
        return DIST_TENSOR_NEGATIVE;
      }
    }
  }
  for (i=0; i<tsr_C->ndim; i++){
    for (j=0; j<tsr_C->ndim; j++){
      if (i!=j && map_C[i] == map_C[j]){
        CTF_free(idx_arr);
        return DIST_TENSOR_NEGATIVE;
      }
    }
  }
  /* Map C or equivalently, the non-contraction indices of A and B */
  ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, 
                           tA, tB, tC, &inner_topovec[itopo]);
  CTF_free(idx_arr);
  if (ret == DIST_TENSOR_NEGATIVE) return DIST_TENSOR_NEGATIVE;
  if (ret == DIST_TENSOR_ERROR) {
    return DIST_TENSOR_ERROR;
  }
  ret = map_symtsr(tsr_A->ndim, tsr_A->sym_table, tsr_A->edge_map);
  ret = map_symtsr(tsr_B->ndim, tsr_B->sym_table, tsr_B->edge_map);
  ret = map_symtsr(tsr_C->ndim, tsr_C->sym_table, tsr_C->edge_map);
  if (ret!=DIST_TENSOR_SUCCESS) return ret;
  
  return DIST_TENSOR_SUCCESS;

}



