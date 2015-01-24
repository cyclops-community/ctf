/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/**
 * \brief convert index maps from arbitrary indices to the smallest possible
 *
 * \param[in] order dimension of tensor 
 * \param[in] cidx old index map of tensor 
 * \param[out] iidx new index map of tensor 
 */
inline 
int conv_idx(int const    order,
             int const *  cidx,
             int **       iidx){
  int i, j, n;
  int c;

  *iidx = (int*)CTF_alloc(sizeof(int)*order);

  n = 0;
  for (i=0; i<order; i++){
    c = cidx[i];
    for (j=0; j<i; j++){
      if (c == cidx[j]){
  (*iidx)[i] = (*iidx)[j];
  break;
      }
    }
    if (j==i){
      (*iidx)[i] = n;
      n++;
    }
  }
  return n;
}
/**
 * \brief convert index maps from arbitrary indices to the smallest possible
 *
 * \param[in] order_A dimension of tensor A
 * \param[in] cidx_A old index map of tensor A
 * \param[out] iidx_A new index map of tensor A
 * \param[in] order_B dimension of tensor B
 * \param[in] cidx_B old index map of tensor B
 * \param[out] iidx_B new index map of tensor B
 * \param[in] order_C dimension of tensor C
 * \param[in] cidx_C old index map of tensor C
 * \param[out] iidx_C new index map of tensor C
 */
inline 
int  conv_idx(int const   order_A,
              int const * cidx_A,
              int **      iidx_A,
              int const   order_B,
              int const * cidx_B,
              int **      iidx_B,
              int const   order_C,
              int const * cidx_C,
              int **      iidx_C){
  int i, j, n;
  int c;

  *iidx_C = (int*)CTF_alloc(sizeof(int)*order_C);

  n = conv_idx(order_A, cidx_A, iidx_A,
               order_B, cidx_B, iidx_B);

  for (i=0; i<order_C; i++){
    c = cidx_C[i];
    for (j=0; j<order_B; j++){
      if (c == cidx_B[j]){
        (*iidx_C)[i] = (*iidx_B)[j];
        break;
      }
    }
    if (j==order_B){
      for (j=0; j<order_A; j++){
        if (c == cidx_A[j]){
          (*iidx_C)[i] = (*iidx_A)[j];
          break;
        }
      }
      if (j==order_A){
        for (j=0; j<i; j++){
          if (c == cidx_C[j]){
            (*iidx_C)[i] = (*iidx_C)[j];
            break;
          }
        }
        if (j==i){
          (*iidx_C)[i] = n;
          n++;
        }
      }
    }
  }
  return n;
}


/**
 * \brief fold a tensor by putting the symmetry-preserved portion
 *    in the leading dimensions of the tensor
 *
 * \param tsr the tensor on hand
 * \param nfold number of global indices we are folding
 * \param fold_idx which global indices we are folding
 * \param idx_map how this tensor indices map to the global indices
 */
template<typename dtype>
void dist_tensor<dtype>::fold_tsr(tensor<dtype> * tsr,
                                  int const       nfold,
                                  int const *     fold_idx,
                                  int const *     idx_map,
                                  int *           all_fdim,
                                  int **          all_flen){
  int i, j, k, fdim, allfold_dim, is_fold, fold_dim;
  int * sub_edge_len, * fold_edge_len, * all_edge_len, * dim_order;
  int * fold_sym;
  int fold_tid;
  
  if (tsr->is_folded != 0) unfold_tsr(tsr);
  
  CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&sub_edge_len);

  allfold_dim = 0, fold_dim = 0;
  for (j=0; j<tsr->order; j++){
    if (tsr->sym[j] == NS){
      allfold_dim++;
      for (i=0; i<nfold; i++){
        if (fold_idx[i] == idx_map[j])
          fold_dim++;
      }
    }
  }
  CTF_alloc_ptr(allfold_dim*sizeof(int), (void**)&all_edge_len);
  CTF_alloc_ptr(allfold_dim*sizeof(int), (void**)&dim_order);
  CTF_alloc_ptr(fold_dim*sizeof(int), (void**)&fold_edge_len);
  CTF_alloc_ptr(fold_dim*sizeof(int), (void**)&fold_sym);

  calc_dim(tsr->order, tsr->size, tsr->edge_len, tsr->edge_map,
     NULL, sub_edge_len, NULL);

  allfold_dim = 0, fdim = 0;
  for (j=0; j<tsr->order; j++){
    if (tsr->sym[j] == NS){
      k=1;
      while (j-k >= 0 && tsr->sym[j-k] != NS) k++;
      all_edge_len[allfold_dim] = sy_packed_size(k, sub_edge_len+j-k+1,
                                                  tsr->sym+j-k+1);
      is_fold = 0;
      for (i=0; i<nfold; i++){
        if (fold_idx[i] == idx_map[j]){
          k=1;
          while (j-k >= 0 && tsr->sym[j-k] != NS) k++;
          fold_edge_len[fdim] = sy_packed_size(k, sub_edge_len+j-k+1,
                                               tsr->sym+j-k+1);
          is_fold = 1;
        }
      }
      if (is_fold) {
        dim_order[fdim] = allfold_dim;
        fdim++;
      } else
        dim_order[fold_dim+allfold_dim-fdim] = allfold_dim;
      allfold_dim++;
    }
  }
  std::fill(fold_sym, fold_sym+fold_dim, NS);
  define_tensor(fold_dim, fold_edge_len, fold_sym, &fold_tid, 0);

  tsr->is_folded      = 1;
  tsr->rec_tid        = fold_tid;
  tsr->inner_ordering = dim_order;

  *all_fdim = allfold_dim;
  *all_flen = all_edge_len;

  CTF_free(fold_edge_len);
  CTF_free(fold_sym);
  
  CTF_free(sub_edge_len);
}

/**
 * \brief undo the folding of a local tensor block
 *
 * \param tsr the tensor on hand
 */
template<typename dtype>
void dist_tensor<dtype>::unfold_tsr(tensor<dtype> * tsr){
  int i, j, nvirt, allfold_dim;
  int * all_edge_len, * sub_edge_len;
  if (tsr->is_folded){
    CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&all_edge_len);
    CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&sub_edge_len);
    calc_dim(tsr->order, tsr->size, tsr->edge_len, tsr->edge_map,
             NULL, sub_edge_len, NULL);
    allfold_dim = 0;
    for (i=0; i<tsr->order; i++){
      if (tsr->sym[i] == NS){
        j=1;
        while (i-j >= 0 && tsr->sym[i-j] != NS) j++;
        all_edge_len[allfold_dim] = sy_packed_size(j, sub_edge_len+i-j+1,
                                                   tsr->sym+i-j+1);
        allfold_dim++;
      }
    }
    nvirt = calc_nvirt(tsr);
    for (i=0; i<nvirt; i++){
      nosym_transpose<dtype>(allfold_dim, tsr->inner_ordering, all_edge_len, 
                             tsr->data + i*(tsr->size/nvirt), 0);
    }
    del_tsr(tsr->rec_tid);
    CTF_free(tsr->inner_ordering);
    CTF_free(all_edge_len);
    CTF_free(sub_edge_len);

  }  
  tsr->is_folded = 0;
}

/**
 * \brief folds tensors for summation
 * \param[in] type contraction specification
 * \param[out] inner_stride inner stride (daxpy size)
 */
template<typename dtype>
int dist_tensor<dtype>::map_fold(CTF_sum_type_t const * type,
                                 int *                  inner_stride) {

 }


/**
 * \brief folds tensors for contraction
 * \param[in] type contraction specification
 * \param[out] inner_prm parameters includng n,m,k
 */
template<typename dtype>
int dist_tensor<dtype>::map_fold(CTF_ctr_type_t const * type,
                                  iparam *              inner_prm) {

  int i, j, nfold, nf, all_fdim_A, all_fdim_B, all_fdim_C;
  int nvirt_A, nvirt_B, nvirt_C;
  int * fold_idx, * fidx_map_A, * fidx_map_B, * fidx_map_C;
  int * fnew_ord_A, * fnew_ord_B, * fnew_ord_C;
  int * all_flen_A, * all_flen_B, * all_flen_C;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tensor<dtype> * ftsr_A, * ftsr_B, * ftsr_C;
  CTF_ctr_type_t fold_type;
  iparam iprm;

  get_fold_indices(type, &nfold, &fold_idx);
  if (nfold == 0) {
    CTF_free(fold_idx);
    return CTF_ERROR;
  }

  /* overestimate this space to not bother with it later */
  CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_A);
  CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_B);
  CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_C);

  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];

  fold_tsr(tsr_A, nfold, fold_idx, type->idx_map_A, 
           &all_fdim_A, &all_flen_A);
  fold_tsr(tsr_B, nfold, fold_idx, type->idx_map_B, 
           &all_fdim_B, &all_flen_B);
  fold_tsr(tsr_C, nfold, fold_idx, type->idx_map_C,
           &all_fdim_C, &all_flen_C);

  nf = 0;
  for (i=0; i<tsr_A->order; i++){
    for (j=0; j<nfold; j++){
      if (tsr_A->sym[i] == NS && type->idx_map_A[i] == fold_idx[j]){
        fidx_map_A[nf] = j;
        nf++;
      }
    }
  }
  nf = 0;
  for (i=0; i<tsr_B->order; i++){
    for (j=0; j<nfold; j++){
      if (tsr_B->sym[i] == NS && type->idx_map_B[i] == fold_idx[j]){
        fidx_map_B[nf] = j;
        nf++;
      }
    }
  }
  nf = 0;
  for (i=0; i<tsr_C->order; i++){
    for (j=0; j<nfold; j++){
      if (tsr_C->sym[i] == NS && type->idx_map_C[i] == fold_idx[j]){
        fidx_map_C[nf] = j;
        nf++;
      }
    }
  }

  ftsr_A = tensors[tsr_A->rec_tid];
  ftsr_B = tensors[tsr_B->rec_tid];
  ftsr_C = tensors[tsr_C->rec_tid];

  fold_type.tid_A = tsr_A->rec_tid;
  fold_type.tid_B = tsr_B->rec_tid;
  fold_type.tid_C = tsr_C->rec_tid;

  conv_idx(ftsr_A->order, fidx_map_A, &fold_type.idx_map_A,
           ftsr_B->order, fidx_map_B, &fold_type.idx_map_B,
           ftsr_C->order, fidx_map_C, &fold_type.idx_map_C);

#if DEBUG>=2
  if (global_comm.rank == 0){
    printf("Folded contraction type:\n");
  }
  print_ctr(&fold_type,0.0,0.0);
#endif
  
  //for type order 1 to 3 
  get_len_ordering(&fold_type, &fnew_ord_A, &fnew_ord_B, &fnew_ord_C); 
  
  //permute_target(ftsr_A->order, fnew_ord_A, cpy_tsr_A_inner_ordering);
  //permute_target(ftsr_B->order, fnew_ord_B, cpy_tsr_B_inner_ordering);

  //get nosym_transpose_estimate cost estimate an save best

  permute_target(ftsr_A->order, fnew_ord_A, tsr_A->inner_ordering);
  permute_target(ftsr_B->order, fnew_ord_B, tsr_B->inner_ordering);
  permute_target(ftsr_C->order, fnew_ord_C, tsr_C->inner_ordering);
  

  nvirt_A = calc_nvirt(tsr_A);
  for (i=0; i<nvirt_A; i++){
    nosym_transpose<dtype>(all_fdim_A, tsr_A->inner_ordering, all_flen_A, 
                           tsr_A->data + i*(tsr_A->size/nvirt_A), 1);
  }
  nvirt_B = calc_nvirt(tsr_B);
  for (i=0; i<nvirt_B; i++){
    nosym_transpose<dtype>(all_fdim_B, tsr_B->inner_ordering, all_flen_B, 
                           tsr_B->data + i*(tsr_B->size/nvirt_B), 1);
  }
  nvirt_C = calc_nvirt(tsr_C);
  for (i=0; i<nvirt_C; i++){
    nosym_transpose<dtype>(all_fdim_C, tsr_C->inner_ordering, all_flen_C, 
                           tsr_C->data + i*(tsr_C->size/nvirt_C), 1);
  }

  calc_fold_nmk<dtype>(&fold_type, fnew_ord_A, fnew_ord_B, 
                       ftsr_A, ftsr_B, ftsr_C, &iprm);

  //FIXME: try all possibilities
  iprm.tA = 'T';
  iprm.tB = 'N';

  CTF_free(fidx_map_A);
  CTF_free(fidx_map_B);
  CTF_free(fidx_map_C);
  CTF_free(fold_type.idx_map_A);
  CTF_free(fold_type.idx_map_B);
  CTF_free(fold_type.idx_map_C);
  CTF_free(fnew_ord_A);
  CTF_free(fnew_ord_B);
  CTF_free(fnew_ord_C);
  CTF_free(all_flen_A);
  CTF_free(all_flen_B);
  CTF_free(all_flen_C);
  CTF_free(fold_idx);

  *inner_prm = iprm;
  return CTF_SUCCESS;
}


/**
 * \brief unfolds a broken symmetry in a summation by defining new tensors
 *
 * \param[in] type contraction specification
 * \param[out] new_type new contraction specification (new tids)
 * \return 3*idx+tsr_type if finds broken sym, -1 otherwise
 */
template<typename dtype>

/**
 * \brief unfolds a broken symmetry in a contraction by defining new tensors
 *
 * \param[in] type contraction specification
 * \param[out] new_type new contraction specification (new tids)
 * \return 3*idx+tsr_type if finds broken sym, -1 otherwise
 */
template<typename dtype>
int dist_tensor<dtype>::unfold_broken_sym(CTF_ctr_type_t const *  type,
                                          CTF_ctr_type_t *        new_type){
  int i, num_tot, iA, iB, iC, iA2, iB2;
  int * idx_arr;
  
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  
  
  if (new_type == NULL){
    tsr_A = tensors[type->tid_A];
    tsr_B = tensors[type->tid_B];
    tsr_C = tensors[type->tid_C];
  } else {
    clone_tensor(type->tid_A, 0, &new_type->tid_A, 0);
    clone_tensor(type->tid_B, 0, &new_type->tid_B, 0);
    clone_tensor(type->tid_C, 0, &new_type->tid_C, 0);

    tsr_A = tensors[new_type->tid_A];
    tsr_B = tensors[new_type->tid_B];
    tsr_C = tensors[new_type->tid_C];
    
    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)&new_type->idx_map_A);
    CTF_alloc_ptr(tsr_B->order*sizeof(int), (void**)&new_type->idx_map_B);
    CTF_alloc_ptr(tsr_C->order*sizeof(int), (void**)&new_type->idx_map_C);

    memcpy(new_type->idx_map_A, type->idx_map_A, tsr_A->order*sizeof(int));
    memcpy(new_type->idx_map_B, type->idx_map_B, tsr_B->order*sizeof(int));
    memcpy(new_type->idx_map_C, type->idx_map_C, tsr_C->order*sizeof(int));
  }

  inv_idx(tsr_A->order, type->idx_map_A, tsr_A->edge_map,
          tsr_B->order, type->idx_map_B, tsr_B->edge_map,
          tsr_C->order, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);

  for (i=0; i<tsr_A->order; i++){
    if (tsr_A->sym[i] != NS){
      iA = type->idx_map_A[i];
      if (idx_arr[3*iA+1] != -1){
        if (tsr_B->sym[idx_arr[3*iA+1]] == NS ||
            type->idx_map_A[i+1] != type->idx_map_B[idx_arr[3*iA+1]+1]){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i;
        }
      } else {
        if (idx_arr[3*type->idx_map_A[i+1]+1] != -1){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i;
        }       
      }
      if (idx_arr[3*iA+2] != -1){
        if (tsr_C->sym[idx_arr[3*iA+2]] == NS ||
            type->idx_map_A[i+1] != type->idx_map_C[idx_arr[3*iA+2]+1]){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i;
        }
      } else {
        if (idx_arr[3*type->idx_map_A[i+1]+2] != -1){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i;
        }       
      }
    }
  }

 
  for (i=0; i<tsr_B->order; i++){
    if (tsr_B->sym[i] != NS){
      iB = type->idx_map_B[i];
      if (idx_arr[3*iB+0] != -1){
        if (tsr_A->sym[idx_arr[3*iB+0]] == NS ||
            type->idx_map_B[i+1] != type->idx_map_A[idx_arr[3*iB+0]+1]){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+1;
        }
      } else {
        if (idx_arr[3*type->idx_map_B[i+1]+0] != -1){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+1;
        }       
      }
      if (idx_arr[3*iB+2] != -1){
        if (tsr_C->sym[idx_arr[3*iB+2]] == NS || 
            type->idx_map_B[i+1] != type->idx_map_C[idx_arr[3*iB+2]+1]){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+1;
        }
      } else {
        if (idx_arr[3*type->idx_map_B[i+1]+2] != -1){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+1;
        }       
      }
    }
  } 
  //if A=B, output symmetry may still be preserved, so long as all indices in A and B are proper
  bool is_preserv = true;
  if (type->tid_A != type->tid_B) is_preserv = false; 
  else {
    for (int j=0; j<tsr_A->order; j++){
      if (type->idx_map_A[j] != type->idx_map_B[j]){
        iA = type->idx_map_A[j];
        iB = type->idx_map_B[j];
        if (idx_arr[3*iA+2] == -1 || idx_arr[3*iB+2] == -1) is_preserv = false;
        else {
          for (int k=MIN(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                   k<MAX(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                   k++){
             if (tsr_C->sym[k] != SY) is_preserv = false;
          }
        }
      }
    }
  }
  if (!is_preserv){
    for (i=0; i<tsr_C->order; i++){
      if (tsr_C->sym[i] != NS){
        iC = type->idx_map_C[i];
        if (idx_arr[3*iC+1] != -1){
          if (tsr_B->sym[idx_arr[3*iC+1]] == NS ||
              type->idx_map_C[i+1] != type->idx_map_B[idx_arr[3*iC+1]+1]){
            if (new_type != NULL)
              tsr_C->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+2;
          }
        } else if (idx_arr[3*type->idx_map_C[i+1]+1] != -1){
          if (new_type != NULL)
            tsr_C->sym[i] = NS;
          CTF_free(idx_arr); 
          return 3*i+2;
        }       
        if (idx_arr[3*iC+0] != -1){
          if (tsr_A->sym[idx_arr[3*iC+0]] == NS ||
              type->idx_map_C[i+1] != type->idx_map_A[idx_arr[3*iC+0]+1]){
            if (new_type != NULL)
              tsr_C->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+2;
          }
        } else if (idx_arr[3*iC+0] == -1){
          if (idx_arr[3*type->idx_map_C[i+1]] != -1){
            if (new_type != NULL)
              tsr_C->sym[i] = NS;
            CTF_free(idx_arr); 
            return 3*i+2;
          }       
        }
      }
    }
  }
  for (i=0; i<tsr_A->order; i++){
    if (tsr_A->sym[i] == SY){
      iA = type->idx_map_A[i];
      iA2 = type->idx_map_A[i+1];
      if (idx_arr[3*iA+2] == -1 &&
          idx_arr[3*iA2+2] == -1){
        if (new_type != NULL)
          tsr_A->sym[i] = NS;
        CTF_free(idx_arr); 
        return 3*i;
      }
    }
  }
  for (i=0; i<tsr_B->order; i++){
    if (tsr_B->sym[i] == SY){
      iB = type->idx_map_B[i];
      iB2 = type->idx_map_B[i+1];
      if (idx_arr[3*iB+2] == -1 &&
          idx_arr[3*iB2+2] == -1){
        if (new_type != NULL)
          tsr_B->sym[i] = NS;
        CTF_free(idx_arr); 
        return 3*i+1;
      }
    }
  }

  CTF_free(idx_arr);
  return -1;
}


/**
 * \brief returns alias data
 *
 * \param[in] sym_tid id of starting symmetric tensor (where data starts)
 * \param[in] nonsym_tid id of new tensor with a potentially unfolded symmetry
 */
template<typename dtype>
void dist_tensor<dtype>::dealias(int const sym_tid, int const nonsym_tid){
  tensor<dtype> * tsr_sym, * tsr_nonsym;
  tsr_sym = tensors[sym_tid];
  tsr_nonsym = tensors[nonsym_tid];

  if (tsr_nonsym->is_data_aliased){
    tsr_sym->itopo = tsr_nonsym->itopo;
    copy_mapping(tsr_nonsym->order, tsr_nonsym->edge_map, 
                 tsr_sym->edge_map);
    tsr_sym->data = tsr_nonsym->data;
    tsr_sym->is_home = tsr_nonsym->is_home;
    set_padding(tsr_sym);
  }
}



/**
 * \brief finds all permutations of a tensor according to a symmetry
 *
 * \param[in] order dimension of tensor
 * \param[in] sym symmetry specification of tensor
 * \param[out] nperm number of symmeitrc permutations to do
 * \param[out] perm the permutation
 * \param[out] sign sign of each permutation
 */
inline
void cmp_sym_perms(int const    order,
                   int const *  sym,
                   int *        nperm,
                   int **       perm,
                   double *     sign){
  int i, np;
  int * pm;
  double sgn;

  ASSERT(sym[0] != NS);
  CTF_alloc_ptr(sizeof(int)*order, (void**)&pm);

  np=0;
  sgn=1.0;
  for (i=0; i<order; i++){
    if (sym[i]==AS){
      sgn=-1.0;
    }
    if (sym[i]!=NS){
      np++;
    } else {
      np++;
      break;
    }    
  }
  /* a circular shift of n indices requires n-1 swaps */
  if (np % 2 == 1) sgn = 1.0;

  for (i=0; i<np; i++){
    pm[i] = (i+1)%np;
  }
  for (i=np; i<order; i++){
    pm[i] = i;
  }

  *nperm = np;
  *perm = pm;
  *sign = sgn;
}

template<typename dtype>
void dist_tensor<dtype>::copy_type(CTF_ctr_type_t const * old_type,
                                   CTF_ctr_type_t *       new_type){
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;

  new_type->tid_A = old_type->tid_A;
  new_type->tid_B = old_type->tid_B;
  new_type->tid_C = old_type->tid_C;
  
  tsr_A = tensors[old_type->tid_A];
  tsr_B = tensors[old_type->tid_B];
  tsr_C = tensors[old_type->tid_C];

  CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&new_type->idx_map_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&new_type->idx_map_B);
  CTF_alloc_ptr(sizeof(int)*tsr_C->order, (void**)&new_type->idx_map_C);

  memcpy(new_type->idx_map_A, old_type->idx_map_A, sizeof(int)*tsr_A->order);
  memcpy(new_type->idx_map_B, old_type->idx_map_B, sizeof(int)*tsr_B->order);
  memcpy(new_type->idx_map_C, old_type->idx_map_C, sizeof(int)*tsr_C->order);

}

template<typename dtype>
void dist_tensor<dtype>::free_type(CTF_ctr_type_t * type){
  CTF_free(type->idx_map_A);
  CTF_free(type->idx_map_B);
  CTF_free(type->idx_map_C);
}

template<typename dtype>
int dist_tensor<dtype>::is_equal_type(CTF_ctr_type_t const * type_A,
                                      CTF_ctr_type_t const * type_B){
  int i;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;

  if (type_A->tid_A != type_B->tid_A) return 0;
  if (type_A->tid_B != type_B->tid_B) return 0;
  if (type_A->tid_C != type_B->tid_C) return 0;
  
  tsr_A = tensors[type_A->tid_A];
  tsr_B = tensors[type_A->tid_B];
  tsr_C = tensors[type_A->tid_C];

  for (i=0; i<tsr_A->order; i++){
    if (type_A->idx_map_A[i] != type_B->idx_map_A[i]) return 0;
  }
  for (i=0; i<tsr_B->order; i++){
    if (type_A->idx_map_B[i] != type_B->idx_map_B[i]) return 0;
  }
  for (i=0; i<tsr_C->order; i++){
    if (type_A->idx_map_C[i] != type_B->idx_map_C[i]) return 0;
  }
  return 1;
}

/**
 * \brief orders the contraction indices of one tensor 
 *        that don't break contraction symmetries
 *
 * \param[in] tsr_A
 * \param[in] tsr_B
 * \param[in] tsr_C
 * \param[in] idx_arr inverted contraction index map
 * \param[in] off_A offset of A in inverted index map
 * \param[in] off_B offset of B in inverted index map
 * \param[in] off_C offset of C in inverted index map
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] idx_map_C index map of C
 * \param[in,out] add_sign sign of contraction
 * \param[in,out] mod 1 if permutation done
 */
template<typename dtype>
void dist_tensor<dtype>::order_perm(tensor<dtype> const * tsr_A,
                                    tensor<dtype> const * tsr_B,
                                    tensor<dtype> const * tsr_C,
                                    int *                 idx_arr,
                                    int const             off_A,
                                    int const             off_B,
                                    int const             off_C,
                                    int *                 idx_map_A,
                                    int *                 idx_map_B,
                                    int *                 idx_map_C,
                                    dtype &               add_sign,
                                    int &                 mod){
  int  iA, jA, iB, iC, jB, jC, iiB, iiC, broken, tmp;

  //find all symmetries in A
  for (iA=0; iA<tsr_A->order; iA++){
    if (tsr_A->sym[iA] != NS){
      jA=iA;
      iB = idx_arr[3*idx_map_A[iA]+off_B];
      iC = idx_arr[3*idx_map_A[iA]+off_C];
      while (tsr_A->sym[jA] != NS){
        broken = 0;
        jA++;
        jB = idx_arr[3*idx_map_A[jA]+off_B];
        //if (iB == jB) broken = 1;
        if (iB != -1 && jB != -1){
          for (iiB=MIN(iB,jB); iiB<MAX(iB,jB); iiB++){
            if (tsr_B->sym[iiB] ==  NS) broken = 1;
          }
        } 
        if ((iB == -1) ^ (jB == -1)) broken = 1;
        jC = idx_arr[3*idx_map_A[jA]+off_C];
        //if (iC == jC) broken = 1;
        if (iC != -1 && jC != -1){
          for (iiC=MIN(iC,jC); iiC<MAX(iC,jC); iiC++){
            if (tsr_C->sym[iiC] == NS) broken = 1;
          }
        } 
        if ((iC == -1) ^ (jC == -1)) broken = 1;
        //if the symmetry is preserved, make sure index map is ordered
        if (!broken){
          if (idx_map_A[iA] > idx_map_A[jA]){
            idx_arr[3*idx_map_A[iA]+off_A] = jA;
            idx_arr[3*idx_map_A[jA]+off_A] = iA;
            tmp                          = idx_map_A[iA];
            idx_map_A[iA] = idx_map_A[jA];
            idx_map_A[jA] = tmp;
            if (tsr_A->sym[iA] == AS) add_sign *= -1.0;
            mod = 1;
          } 
        }
      }
    }
  }
}

/**
 * \brief puts a contraction map into a nice ordering according to preserved
 *        symmetries, and adds it if it is distinct
 *
 * \param[in,out] perms the permuted contraction specifications
 * \param[in,out] signs sign of each contraction
 * \param[in] new_perm contraction signature
 * \param[in] new_sign alpha
 */
template<typename dtype>
void dist_tensor<dtype>::add_sym_perm(std::vector<CTF_ctr_type_t>&    perms,
                                      std::vector<dtype>&             signs, 
                                      CTF_ctr_type_t const *          new_perm,
                                      dtype const                     new_sign){
  int mod, num_tot, i;
  int * idx_arr;
  dtype add_sign;
  CTF_ctr_type_t norm_ord_perm;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  

  add_sign = new_sign;
  copy_type(new_perm, &norm_ord_perm);
  
  tsr_A = tensors[norm_ord_perm.tid_A];
  tsr_B = tensors[norm_ord_perm.tid_B];
  tsr_C = tensors[norm_ord_perm.tid_C];
  
  inv_idx(tsr_A->order, norm_ord_perm.idx_map_A, NULL,
          tsr_B->order, norm_ord_perm.idx_map_B, NULL,
          tsr_C->order, norm_ord_perm.idx_map_C, NULL,
          &num_tot, &idx_arr);
  //keep permuting until we get to normal order (no permutations left)
  do {
    mod = 0;
    order_perm(tsr_A, tsr_B, tsr_C, idx_arr, 0, 1, 2, 
               norm_ord_perm.idx_map_A, norm_ord_perm.idx_map_B,
               norm_ord_perm.idx_map_C, add_sign, mod);
    order_perm(tsr_B, tsr_A, tsr_C, idx_arr, 1, 0, 2, 
               norm_ord_perm.idx_map_B, norm_ord_perm.idx_map_A,
               norm_ord_perm.idx_map_C, add_sign, mod);
    order_perm(tsr_C, tsr_B, tsr_A, idx_arr, 2, 1, 0, 
               norm_ord_perm.idx_map_C, norm_ord_perm.idx_map_B,
               norm_ord_perm.idx_map_A, add_sign, mod);
  } while (mod);
  add_sign *= align_symmetric_indices(tsr_A->order,
                                      norm_ord_perm.idx_map_A,
                                      tsr_A->sym,
                                      tsr_B->order,
                                      norm_ord_perm.idx_map_B,
                                      tsr_B->sym,
                                      tsr_C->order,
                                      norm_ord_perm.idx_map_C,
                                      tsr_C->sym);

  for (i=0; i<(int)perms.size(); i++){
    if (is_equal_type(&perms[i], &norm_ord_perm)){
      free_type(&norm_ord_perm);
      return;
    }
  }
  perms.push_back(norm_ord_perm);
  signs.push_back(add_sign);
}

/**
 * \brief finds all permutations of acontraction 
 *        that must be done for a broken symmetry
 *
 * \param[in] type contraction specification
 * \param[in] alpha sign of contraction specification
// * \param[out] nperm number of permuted contractions to do
 * \param[out] perms the permuted contraction specifications
 * \param[out] signs sign of each contraction
// * \param[out] nscl number of times we need to rescale C diagonal
// * \param[out] scl_idx_maps_C C diagonals to rescale
// * \param[out] scl_alpha_C factor to rescale C diagonal by
 */
template<typename dtype>
void dist_tensor<dtype>::get_sym_perms(CTF_ctr_type_t const *           type,
                                       dtype const                      alpha,
                                       std::vector<CTF_ctr_type_t>&     perms,
                                       std::vector<dtype>&              signs){
        /*                               int **                 nscl,
                                       int ***                scl_idx_maps,
                                       dtype **               scl_alpha){*/
//  dtype * scl_alpha_C;
//  int ** scl_idx_maps_C;
//  nscl_C = 0;
//  CTF_alloc_ptr(sizeof(dtype)*order_C, (void**)&scl_alpha_C);
//  CTF_alloc_ptr(sizeof(int*)*order_C, (void**)&scl_idx_maps_C);

  int i, j, k, tmp;
  CTF_ctr_type_t new_type;
  dtype sign;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];

  copy_type(type, &new_type);
  add_sym_perm(perms, signs, &new_type, alpha);

  for (i=0; i<tsr_A->order; i++){
    j=i;
    while (tsr_A->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
        free_type(&new_type);
        copy_type(&perms[k], &new_type);
        sign = signs[k];
        if (tsr_A->sym[j-1] == AS) sign *= -1.0;
        tmp                    = new_type.idx_map_A[i];
        new_type.idx_map_A[i]  = new_type.idx_map_A[j];
        new_type.idx_map_A[j]  = tmp;
        add_sym_perm(perms, signs, &new_type, sign);
      }
    }
  }
  for (i=0; i<tsr_B->order; i++){
    j=i;
    while (tsr_B->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
        free_type(&new_type);
        copy_type(&perms[k], &new_type);
        sign = signs[k];
        if (tsr_B->sym[j-1] == AS) sign *= -1.0;
        tmp                    = new_type.idx_map_B[i];
        new_type.idx_map_B[i]  = new_type.idx_map_B[j];
        new_type.idx_map_B[j]  = tmp;
        add_sym_perm(perms, signs, &new_type, sign);
      }
    }
  }
  
  for (i=0; i<tsr_C->order; i++){
    j=i;
    while (tsr_C->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
        free_type(&new_type);
        copy_type(&perms[k], &new_type);
        sign = signs[k];
        if (tsr_C->sym[j-1] == AS) sign *= -1.0;
        tmp                    = new_type.idx_map_C[i];
        new_type.idx_map_C[i]  = new_type.idx_map_C[j];
        new_type.idx_map_C[j]  = tmp;
        add_sym_perm(perms, signs, &new_type, sign);
      }
    }
  }
}




template<typename dtype>
void dist_tensor<dtype>::copy_type(CTF_sum_type_t const * old_type,
                                   CTF_sum_type_t *       new_type){
  tensor<dtype> * tsr_A, * tsr_B;

  new_type->tid_A = old_type->tid_A;
  new_type->tid_B = old_type->tid_B;
  
  tsr_A = tensors[old_type->tid_A];
  tsr_B = tensors[old_type->tid_B];

  CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&new_type->idx_map_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&new_type->idx_map_B);

  memcpy(new_type->idx_map_A, old_type->idx_map_A, sizeof(int)*tsr_A->order);
  memcpy(new_type->idx_map_B, old_type->idx_map_B, sizeof(int)*tsr_B->order);

}

template<typename dtype>
void dist_tensor<dtype>::free_type(CTF_sum_type_t * type){
  CTF_free(type->idx_map_A);
  CTF_free(type->idx_map_B);
}

template<typename dtype>
int dist_tensor<dtype>::is_equal_type(CTF_sum_type_t const * type_A,
                                      CTF_sum_type_t const * type_B){
  int i;
  tensor<dtype> * tsr_A, * tsr_B;

  if (type_A->tid_A != type_B->tid_A) return 0;
  if (type_A->tid_B != type_B->tid_B) return 0;
  
  tsr_A = tensors[type_A->tid_A];
  tsr_B = tensors[type_A->tid_B];

  for (i=0; i<tsr_A->order; i++){
    if (type_A->idx_map_A[i] != type_B->idx_map_A[i]) return 0;
  }
  for (i=0; i<tsr_B->order; i++){
    if (type_A->idx_map_B[i] != type_B->idx_map_B[i]) return 0;
  }
  return 1;
}

/**
 * \brief orders the summation indices of one tensor 
 *        that don't break summation symmetries
 *
 * \param[in] tsr_A
 * \param[in] tsr_B
 * \param[in] idx_arr inverted summation index map
 * \param[in] off_A offset of A in inverted index map
 * \param[in] off_B offset of B in inverted index map
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in,out] add_sign sign of contraction
 * \param[in,out] mod 1 if permutation done
 */
template<typename dtype>
void dist_tensor<dtype>::order_perm(tensor<dtype> const * tsr_A,
                                    tensor<dtype> const * tsr_B,
                                    int *                 idx_arr,
                                    int const             off_A,
                                    int const             off_B,
                                    int *                 idx_map_A,
                                    int *                 idx_map_B,
                                    dtype &               add_sign,
                                    int &                 mod){
  int  iA, jA, iB, jB, iiB, broken, tmp;

  //find all symmetries in A
  for (iA=0; iA<tsr_A->order; iA++){
    if (tsr_A->sym[iA] != NS){
      jA=iA;
      iB = idx_arr[2*idx_map_A[iA]+off_B];
      while (tsr_A->sym[jA] != NS){
        broken = 0;
        jA++;
        jB = idx_arr[2*idx_map_A[jA]+off_B];
        if ((iB == -1) ^ (jB == -1)) broken = 1;
       /* if (iB != -1 && jB != -1) {
          if (tsr_B->sym[iB] != tsr_A->sym[iA]) broken = 1;
        }*/
        if (iB != -1 && jB != -1) {
        /* Do this because iB,jB can be in reversed order */
        for (iiB=MIN(iB,jB); iiB<MAX(iB,jB); iiB++){
          ASSERT(iiB >= 0 && iiB <= tsr_B->order);
          if (tsr_B->sym[iiB] == NS) broken = 1;
        }
        }
        

        /*//if (iB == jB) broken = 1;
        } */
        //if the symmetry is preserved, make sure index map is ordered
        if (!broken){
          if (idx_map_A[iA] > idx_map_A[jA]){
            idx_arr[2*idx_map_A[iA]+off_A] = jA;
            idx_arr[2*idx_map_A[jA]+off_A] = iA;
            tmp                          = idx_map_A[iA];
            idx_map_A[iA] = idx_map_A[jA];
            idx_map_A[jA] = tmp;
            if (tsr_A->sym[iA] == AS) add_sign *= -1.0;
            mod = 1;
          } 
        }
      }
    }
  }
}

/**
 * \brief puts a summation map into a nice ordering according to preserved
 *        symmetries, and adds it if it is distinct
 *
 * \param[in,out] perms the permuted summation specifications
 * \param[in,out] signs sign of each summation
 * \param[in] new_perm summation signature
 * \param[in] new_sign alpha
 */
template<typename dtype>
void dist_tensor<dtype>::add_sym_perm(std::vector<CTF_sum_type_t>&    perms,
                                      std::vector<dtype>&             signs, 
                                      CTF_sum_type_t const *          new_perm,
                                      dtype const                     new_sign){
  int mod, num_tot, i;
  int * idx_arr;
  dtype add_sign;
  CTF_sum_type_t norm_ord_perm;
  tensor<dtype> * tsr_A, * tsr_B;
  

  add_sign = new_sign;
  copy_type(new_perm, &norm_ord_perm);
  
  tsr_A = tensors[norm_ord_perm.tid_A];
  tsr_B = tensors[norm_ord_perm.tid_B];
  
  inv_idx(tsr_A->order, norm_ord_perm.idx_map_A, NULL,
          tsr_B->order, norm_ord_perm.idx_map_B, NULL,
          &num_tot, &idx_arr);
  //keep permuting until we get to normal order (no permutations left)
  do {
    mod = 0;
    order_perm(tsr_A, tsr_B, idx_arr, 0, 1,
               norm_ord_perm.idx_map_A, norm_ord_perm.idx_map_B,
               add_sign, mod);
    order_perm(tsr_B, tsr_A, idx_arr, 1, 0,
               norm_ord_perm.idx_map_B, norm_ord_perm.idx_map_A,
               add_sign, mod);
  } while (mod);
  add_sign = add_sign*align_symmetric_indices(tsr_A->order, norm_ord_perm.idx_map_A, tsr_A->sym,
                                              tsr_B->order, norm_ord_perm.idx_map_B, tsr_B->sym);

  for (i=0; i<(int)perms.size(); i++){
    if (is_equal_type(&perms[i], &norm_ord_perm)){
      free_type(&norm_ord_perm);
      CTF_free(idx_arr);
      return;
    }
  }
  perms.push_back(norm_ord_perm);
  signs.push_back(add_sign);
  CTF_free(idx_arr);
}

/**
 * \brief finds all permutations of asummation 
 *        that must be done for a broken symmetry
 *
 * \param[in] type summation specification
 * \param[in] alpha sign of summation specification
 * \param[out] perms the permuted summation specifications
 * \param[out] signs sign of each summation
 */
template<typename dtype>
void dist_tensor<dtype>::get_sym_perms(CTF_sum_type_t const *           type,
                                       dtype const                      alpha,
                                       std::vector<CTF_sum_type_t>&     perms,
                                       std::vector<dtype>&              signs){
  int i, j, k, tmp;
  CTF_sum_type_t new_type;
  dtype sign;
  tensor<dtype> * tsr_A, * tsr_B;
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];

  copy_type(type, &new_type);
  add_sym_perm(perms, signs, &new_type, alpha);

  for (i=0; i<tsr_A->order; i++){
    j=i;
    while (tsr_A->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
        free_type(&new_type);
        copy_type(&perms[k], &new_type);
        sign = signs[k];
        if (tsr_A->sym[j-1] == AS) sign *= -1.0;
        tmp                    = new_type.idx_map_A[i];
        new_type.idx_map_A[i]  = new_type.idx_map_A[j];
        new_type.idx_map_A[j]  = tmp;
        add_sym_perm(perms, signs, &new_type, sign);
      }
    }
  }
  for (i=0; i<tsr_B->order; i++){
    j=i;
    while (tsr_B->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
        free_type(&new_type);
        copy_type(&perms[k], &new_type);
        sign = signs[k];
        if (tsr_B->sym[j-1] == AS) sign *= -1.0;
        tmp                    = new_type.idx_map_B[i];
        new_type.idx_map_B[i]  = new_type.idx_map_B[j];
        new_type.idx_map_B[j]  = tmp;
        add_sym_perm(perms, signs, &new_type, sign);
      }
    }
  }
  free_type(&new_type);
}



