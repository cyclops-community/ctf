/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/


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
  if (tsr_A->is_folded) return 0;
  if (tsr_B->is_folded) return 0;

  ASSERT(tsr_A->order == tsr_B->order);
//  ASSERT(tsr_A->size == tsr_B->size);

  for (i=0; i<tsr_A->order; i++){
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
  int was_cyclic;
  int64_t old_size;
  tensor<dtype> * tsr_A, * tsr_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];

  if (tid_A == tid_B) return CTF_SUCCESS;

  unmap_inner(tsr_A);
  unmap_inner(tsr_B);
  set_padding(tsr_A);
  set_padding(tsr_B);

  save_mapping(tsr_B, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda, 
               &old_size, &was_cyclic, &old_padding, &old_edge_len, &topovec[tsr_B->itopo]);  
  tsr_B->itopo = tsr_A->itopo;
  tsr_B->is_cyclic = tsr_A->is_cyclic;
  copy_mapping(tsr_A->order, tsr_A->edge_map, tsr_B->edge_map);
  set_padding(tsr_B);
  remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size, 
               old_phase, old_rank, old_virt_dim, 
               old_pe_lda, was_cyclic, 
               old_padding, old_edge_len, global_comm);   
  CTF_free(old_phase);
  CTF_free(old_rank);
  CTF_free(old_virt_dim);
  CTF_free(old_pe_lda);
  CTF_free(old_padding);
  CTF_free(old_edge_len);
  return CTF_SUCCESS;
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



/* \brief Check whether current tensor mapping can be contracted on 
 * \param type specification of contraction
 */
template<typename dtype>
int dist_tensor<dtype>::check_contraction_mapping(CTF_ctr_type_t const * type){
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
 * \param[in] ftsr sequential ctr func pointer 
 * \param[in] felm elementwise ctr func pointer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::map_tensors(CTF_ctr_type_t const *      type, 
                                    fseq_tsr_ctr<dtype>         ftsr, 
                                    fseq_elm_ctr<dtype>         felm, 
                                    dtype const                 alpha,
                                    dtype const                 beta,
                                    ctr<dtype> **               ctrf,
                                    int const                   do_remap){

}

/**
 * \brief map the indices which are common in a sum
 *
 * \param idx_arr array of index mappings of size order*idx_num that
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
}



