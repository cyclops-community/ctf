/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/


/**
 * \brief undo the inner blocking of a tensor
 *
 * \param tsr the tensor on hand
 */
template<typename dtype>
void dist_tensor<dtype>::unmap_inner(tensor<dtype> * tsr){
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
      ASSERT(tsr->edge_map[i].type == VIRTUAL_MAP);
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




