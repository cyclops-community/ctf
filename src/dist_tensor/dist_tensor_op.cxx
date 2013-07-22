/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "sym_indices.hxx"

/**
 * \brief Scale each tensor element by alpha
 * \param[in] alpha scaling factor
 * \param[in] tid handle to tensor
 */
template<typename dtype>
int dist_tensor<dtype>::scale_tsr(dtype const alpha, int const tid){
  if (global_comm->rank == 0)
    printf("FAILURE: scale_tsr currently only supported for tensors of type double\n");
  return DIST_TENSOR_ERROR;
}

template<> inline
int dist_tensor<double>::scale_tsr(double const alpha, int const tid){
  int i;
  tensor<double> * tsr;

  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }

  if (tsr->is_mapped){
    cdscal(tsr->size, alpha, tsr->data, 1);
  } else {
    for (i=0; i<tsr->size; i++){
      tsr->pairs[i].d = tsr->pairs[i].d*alpha;
    }
  }

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief  Compute dot product of two tensors. The tensors
    must have the same mapping.
 * \param[in] tid_A handle to tensor A
 * \param[in] tid_B handle to tensor B
 * \param[out] product dot product A dot B
 */
template<typename dtype>
int dist_tensor<dtype>::dot_loc_tsr(int const tid_A, int const tid_B, dtype *product){
  if (global_comm->rank == 0)
    printf("FAILURE: dot_loc_tsr currently only supported for tensors of type double\n");
  return DIST_TENSOR_ERROR;
}

template<> inline
int dist_tensor<double>::dot_loc_tsr(int const tid_A, int const tid_B, double *product){
  double dprod;
  tensor<double> * tsr_A, * tsr_B;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  if (tsr_A->has_zero_edge_len || tsr_B->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }

  LIBT_ASSERT(tsr_A->is_mapped && tsr_B->is_mapped);
  LIBT_ASSERT(tsr_A->size == tsr_B->size);

  dprod = cddot(tsr_A->size, tsr_A->data, 1, tsr_B->data, 1);

  /* FIXME: Wont work for single precision */
  ALLREDUCE(&dprod, product, 1, COMM_DOUBLE_T, COMM_OP_SUM, global_comm);

  return DIST_TENSOR_SUCCESS;
}

/* Perform an elementwise reduction on a tensor. All processors
   end up with the final answer. */
template<typename dtype>
int dist_tensor<dtype>::red_tsr(int const tid, CTF_OP op, dtype * result){
  if (global_comm->rank == 0)
    printf("FAILURE: reductions currently only supported for tensors of type double\n");
  return DIST_TENSOR_ERROR;
}

/* Perform an elementwise reduction on a tensor. All processors
   end up with the final answer. */
template<> inline
int dist_tensor<double>::red_tsr(int const tid, CTF_OP op, double * result){
  long_int i;
  double acc;
  tensor<double> * tsr;
  mapping * map;
  int idx_lyr = 0;


  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);

  if (tsr->is_mapped){
    idx_lyr = global_comm->rank;
    for (i=0; i<tsr->ndim; i++){
      map = &tsr->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        idx_lyr -= topovec[tsr->itopo].dim_comm[map->cdt]->rank
        *topovec[tsr->itopo].lda[map->cdt];
      }
      while (map->has_child){
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          idx_lyr -= topovec[tsr->itopo].dim_comm[map->cdt]->rank
               *topovec[tsr->itopo].lda[map->cdt];
        }
      }
    }
  }

  switch (op){
    case CTF_OP_SUM:
      acc = 0.0;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          for (i=0; i<tsr->size; i++){
            acc += tsr->data[i];
          }
        }
      } else {
        for (i=0; i<tsr->size; i++){
          acc += tsr->pairs[i].d;
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);
      break;

    case CTF_OP_SUMABS:
      acc = 0.0;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          for (i=0; i<tsr->size; i++){
            acc += fabs(tsr->data[i]);
          }
        }
      } else {
        for (i=0; i<tsr->size; i++){
          acc += fabs(tsr->pairs[i].d);
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);
      break;

    case CTF_OP_SQNRM2:
        acc = 0.0;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          for (i=0; i<tsr->size; i++){
            acc += tsr->data[i]*tsr->data[i];
          }
        }
      } else {
        for (i=0; i<tsr->size; i++){
          acc += tsr->pairs[i].d*tsr->pairs[i].d;
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);
      break;

    case CTF_OP_MAX:
      acc = -DBL_MAX;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          acc = tsr->data[0];
          for (i=1; i<tsr->size; i++){
            acc = MAX(acc, tsr->data[i]);
          }
        }
      } else {
        acc = tsr->pairs[0].d;
        for (i=1; i<tsr->size; i++){
          acc = MAX(acc, tsr->pairs[i].d);
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_MAX, global_comm);
      break;

    case CTF_OP_MIN:
      acc = DBL_MAX;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          acc = tsr->data[0];
          for (i=1; i<tsr->size; i++){
            acc = MIN(acc, tsr->data[i]);
          }
        }
      } else {
        acc = tsr->pairs[0].d;
        for (i=1; i<tsr->size; i++){
          acc = MIN(acc, tsr->pairs[i].d);
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_MIN, global_comm);
      break;

    case CTF_OP_MAXABS:
      acc = 0.0;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          acc = fabs(tsr->data[0]);
          for (i=1; i<tsr->size; i++){
            acc = MAX(fabs(acc), fabs(tsr->data[i]));
          }
        }
      } else {
        acc = fabs(tsr->pairs[0].d);
        for (i=1; i<tsr->size; i++){
          acc = MAX(fabs(acc), fabs(tsr->pairs[i].d));
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_MAX, global_comm);
      break;

    case CTF_OP_MINABS:
      acc = DBL_MAX;
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          acc = fabs(tsr->data[0]);
          for (i=1; i<tsr->size; i++){
            acc = MIN(fabs(acc), fabs(tsr->data[i]));
          }
        }
      } else {
        acc = fabs(tsr->pairs[0].d);
        for (i=1; i<tsr->size; i++){
          acc = MIN(fabs(acc), fabs(tsr->pairs[i].d));
        }
      }
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_MIN, global_comm);
      break;

    default:
      return DIST_TENSOR_ERROR;
      break;
  }
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief apply a function to each element to transform tensor
 * \param[in] tid handle to tensor
 * \param[in] map_func map function to apply to each element
 */
template<typename dtype>
int dist_tensor<dtype>::map_tsr(int const tid,
                                dtype (*map_func)(int const ndim,
                                                  int const * indices,
                                                  dtype const elem)){
  long_int i, j, np, stat;
  int * idx;
  tensor<dtype> * tsr;
  key k;
  tkv_pair<dtype> * prs;

  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);

  CTF_alloc_ptr(tsr->ndim*sizeof(int), (void**)&idx);

  /* Extract key-value pair representation */
  if (tsr->is_mapped){
    stat = read_local_pairs(tid, &np, &prs);
    if (stat != DIST_TENSOR_SUCCESS) return stat;
  } else {
    np = tsr->size;
    prs = tsr->pairs;
  }
  /* Extract location from key and map */
  for (i=0; i<np; i++){
    k = prs[i].k;
    for (j=0; j<tsr->ndim; j++){
      idx[j] = k%tsr->edge_len[j];
      k = k/tsr->edge_len[j];
    }
    prs[i].d = map_func(tsr->ndim, idx, prs[i].d);
  }
  /* Rewrite pairs to packed layout */
  if (tsr->is_mapped){
    stat = write_pairs(tid, np, 1.0, 0.0, prs,'w');
    CTF_free(prs);
    if (stat != DIST_TENSOR_SUCCESS) return stat;
  }
  CTF_free(idx);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief daxpy tensors A and B, B = B+alpha*A
 * \param[in] alpha scaling factor
 * \param[in] tid_A handle to tensor A
 * \param[in] tid_B handle to tensor B
 */
template<typename dtype>
int dist_tensor<dtype>::
    daxpy_local_tensor_pair(dtype alpha, const int tid_A, const int tid_B){
  if (global_comm->rank == 0)
    printf("FAILURE: daxpy currently only supported for tensors of type double\n");
  return DIST_TENSOR_ERROR;
}

template<> inline
int dist_tensor<double>::
    daxpy_local_tensor_pair(double alpha, const int tid_A, const int tid_B){
  tensor<double> * tsr_A, * tsr_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  if (tsr_A->has_zero_edge_len || tsr_B->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  LIBT_ASSERT(tsr_A->size == tsr_B->size);
  cdaxpy(tsr_A->size, alpha, tsr_A->data, 1, tsr_B->data, 1);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 * \param[in] ftsr pointer to sequential block scale function
 * \param[in] felm pointer to sequential element-wise scale function
 */
template<typename dtype>
int dist_tensor<dtype>::
     scale_tsr(dtype const                alpha,
               int const                  tid,
               int const *                idx_map,
               fseq_tsr_scl<dtype> const  ftsr,
               fseq_elm_scl<dtype> const  felm){
  int st, is_top, ndim_tot, iA, nvirt, i, ret, was_padded, was_cyclic, itopo;
  long_int blk_sz, vrt_sz, old_size;
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda,
      * old_padding, * old_edge_len;
  int * virt_dim, * idx_arr;
  int * virt_blk_len, * blk_len;
  mapping * map;
  tensor<dtype> * tsr;
  strp_tsr<dtype> * str;
  scl<dtype> * hscl = NULL, ** rec_scl = NULL;

  is_top = 1;
  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }

#if DEBUG>=1
  if (global_comm->rank == 0){
    printf("Scaling tensor %d by %lf.\n", tid, GET_REAL(alpha));
    printf("The index mapping is");
    for (i=0; i<tsr->ndim; i++){
      printf(" %d",idx_map[i]);
    }
    printf("\n");
  }
#endif

  unmap_inner(tsr);
  set_padding(tsr);
  inv_idx(tsr->ndim, idx_map, tsr->edge_map,
          &ndim_tot, &idx_arr);

  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&blk_len);
  CTF_alloc_ptr(sizeof(int)*tsr->ndim, (void**)&virt_blk_len);
  CTF_alloc_ptr(sizeof(int)*ndim_tot, (void**)&virt_dim);

  if (!check_self_mapping(tid, idx_map)){
    save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda,
                 &old_size, &was_padded, &was_cyclic, &old_padding, &old_edge_len, &topovec[tsr->itopo]);
    for (itopo=0; itopo<(int)topovec.size(); itopo++){
      clear_mapping(tsr);
      tsr->itopo = itopo;
      
      ret = map_self_indices(tid, idx_map);
      LIBT_ASSERT(ret==DIST_TENSOR_SUCCESS);
      ret = map_tensor_rem(topovec[tsr->itopo].ndim,
                           topovec[tsr->itopo].dim_comm, tsr, 1);
      LIBT_ASSERT(ret==DIST_TENSOR_SUCCESS);
      ret = map_self_indices(tid, idx_map);
      LIBT_ASSERT(ret==DIST_TENSOR_SUCCESS);
      if (check_self_mapping(tid, idx_map)) break;
    }
    if (itopo == (int)topovec.size()) return DIST_TENSOR_ERROR;
    tsr->is_mapped = 1;
    set_padding(tsr);
    tsr->is_cyclic = 1;
    remap_tensor(tid, tsr, &topovec[tsr->itopo], old_size, old_phase,
                 old_rank, old_virt_dim, old_pe_lda,
                 was_padded, was_cyclic, old_padding, old_edge_len,
                 global_comm);
    CTF_free(old_phase);
    CTF_free(old_rank);
    CTF_free(old_virt_dim);
    CTF_free(old_pe_lda);
    if (was_padded)
      CTF_free(old_padding);
    CTF_free(old_edge_len);
#if DEBUG >=1
    if (global_comm->rank == 0)
      printf("New mapping for tensor %d\n",tid);
    print_map(stdout,tid);
#endif
  }

  blk_sz = tsr->size;
  calc_dim(tsr->ndim, blk_sz, tsr->edge_len, tsr->edge_map,
           &vrt_sz, virt_blk_len, blk_len);

  st = strip_diag<dtype>(tsr->ndim, ndim_tot, idx_map, vrt_sz,
                         tsr->edge_map, &topovec[tsr->itopo],
                         blk_len, &blk_sz, &str);
  if (st){
    if (global_comm->rank == 0)
      DPRINTF(1,"Stripping tensor\n");
    strp_scl<dtype> * sscl = new strp_scl<dtype>;
    hscl = sscl;
    is_top = 0;
    rec_scl = &sscl->rec_scl;

    sscl->rec_strp = str;
  }

  nvirt = 1;
  for (i=0; i<ndim_tot; i++){
    iA = idx_arr[i];
    if (idx_map[i] != -1){
      map = &tsr->edge_map[iA];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
        if (st) virt_dim[i] = virt_dim[i]/str->strip_dim[iA];
      }
      else virt_dim[i] = 1;
    }
    nvirt *= virt_dim[i];
  }

  /* Multiply over virtual sub-blocks */
  if (nvirt > 1){
    scl_virt<dtype> * sclv = new scl_virt<dtype>;
    if (is_top) {
      hscl = sclv;
      is_top = 0;
    } else {
      *rec_scl = sclv;
    }
    rec_scl = &sclv->rec_scl;

    sclv->num_dim   = ndim_tot;
    sclv->virt_dim  = virt_dim;
    sclv->ndim_A  = tsr->ndim;
    sclv->blk_sz_A  = vrt_sz;
    sclv->idx_map_A = idx_map;
    sclv->buffer  = NULL;
  }

  seq_tsr_scl<dtype> * sclseq = new seq_tsr_scl<dtype>;
  if (is_top) {
    hscl = sclseq;
    is_top = 0;
  } else {
    *rec_scl = sclseq;
  }
  sclseq->alpha         = alpha;
  sclseq->ndim          = tsr->ndim;
  sclseq->idx_map       = idx_map;
  sclseq->edge_len      = virt_blk_len;
  sclseq->sym           = tsr->sym;
  sclseq->func_ptr      = ftsr;
  sclseq->custom_params = felm;
  sclseq->is_custom     = (felm.func_ptr != NULL);

  hscl->A   = tsr->data;
  hscl->alpha   = alpha;

  CTF_free(idx_arr);
  CTF_free(blk_len);

  hscl->run();

  delete hscl;

#if DEBUG>=1
  if (global_comm->rank == 0)
    printf("Done scaling tensor %d.\n", tid);
#endif

  return DIST_TENSOR_SUCCESS;

}


/**
 * \brief sums tensors A and B, B = B*beta+alpha*A
 * \param[in] alpha scaling factor of A
 * \param[in] beta scaling factor of A
 * \param[in] tid_A handle to tensor A
 * \param[in] idx_A handle to tensor A
 * \param[in] tid_B handle to tensor B
 * \param[in] idx_B handle to tensor B
 * \param[in] ftsr pointer to sequential block sum function
 * \param[in] felm pointer to sequential element-wise sum function
 * \return tsum summation class to run
*/
template<typename dtype>
tsum<dtype> * dist_tensor<dtype>::
    construct_sum(dtype const                 alpha,
                  dtype const                 beta,
                  int const                   tid_A,
                  int const *                 idx_A,
                  int const                   tid_B,
                  int const *                 idx_B,
                  fseq_tsr_sum<dtype> const   ftsr,
                  fseq_elm_sum<dtype> const   felm,
                  int const                   inner_stride){
  int nvirt, i, iA, iB, ndim_tot, is_top, sA, sB, need_rep, i_A, i_B, j, k;
  long_int blk_sz_A, blk_sz_B, vrt_sz_A, vrt_sz_B;
  int nphys_dim;
  int * idx_arr, * virt_dim, * phys_mapped;
  int * virt_blk_len_A, * virt_blk_len_B;
  int * blk_len_A, * blk_len_B;
  tensor<dtype> * tsr_A, * tsr_B;
  tsum<dtype> * htsum = NULL , ** rec_tsum = NULL;
  mapping * map;
  strp_tsr<dtype> * str_A, * str_B;

  is_top = 1;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];

  inv_idx(tsr_A->ndim, idx_A, tsr_A->edge_map,
          tsr_B->ndim, idx_B, tsr_B->edge_map,
          &ndim_tot, &idx_arr);

  nphys_dim = topovec[tsr_A->itopo].ndim;

  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&blk_len_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&blk_len_B);
  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&virt_blk_len_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&virt_blk_len_B);
  CTF_alloc_ptr(sizeof(int)*ndim_tot, (void**)&virt_dim);
  CTF_alloc_ptr(sizeof(int)*nphys_dim*2, (void**)&phys_mapped);
  memset(phys_mapped, 0, sizeof(int)*nphys_dim*2);


  /* Determine the block dimensions of each local subtensor */
  blk_sz_A = tsr_A->size;
  blk_sz_B = tsr_B->size;
  calc_dim(tsr_A->ndim, blk_sz_A, tsr_A->edge_len, tsr_A->edge_map,
           &vrt_sz_A, virt_blk_len_A, blk_len_A);
  calc_dim(tsr_B->ndim, blk_sz_B, tsr_B->edge_len, tsr_B->edge_map,
           &vrt_sz_B, virt_blk_len_B, blk_len_B);

  is_top = 1;
  /* Strip out the relevant part of the tensor if we are contracting over diagonal */
  sA = strip_diag<dtype>(tsr_A->ndim, ndim_tot, idx_A, vrt_sz_A,
                         tsr_A->edge_map, &topovec[tsr_A->itopo],
                         blk_len_A, &blk_sz_A, &str_A);
  sB = strip_diag<dtype>(tsr_B->ndim, ndim_tot, idx_B, vrt_sz_B,
                         tsr_B->edge_map, &topovec[tsr_B->itopo],
                         blk_len_B, &blk_sz_B, &str_B);
  if (sA || sB){
    if (global_comm->rank == 0)
      DPRINTF(1,"Stripping tensor\n");
    strp_sum<dtype> * ssum = new strp_sum<dtype>;
    htsum = ssum;
    is_top = 0;
    rec_tsum = &ssum->rec_tsum;

    ssum->rec_strp_A = str_A;
    ssum->rec_strp_B = str_B;
    ssum->strip_A = sA;
    ssum->strip_B = sB;
  }

  nvirt = 1;
  for (i=0; i<ndim_tot; i++){
    iA = idx_arr[2*i];
    iB = idx_arr[2*i+1];
    if (iA != -1){
      map = &tsr_A->edge_map[iA];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
        if (sA) virt_dim[i] = virt_dim[i]/str_A->strip_dim[iA];
      }
      else virt_dim[i] = 1;
    } else {
      LIBT_ASSERT(iB!=-1);
      map = &tsr_B->edge_map[iB];
      while (map->has_child) map = map->child;
      if (map->type == VIRTUAL_MAP){
        virt_dim[i] = map->np;
        if (sB) virt_dim[i] = virt_dim[i]/str_B->strip_dim[iA];
      }
      else virt_dim[i] = 1;
    }
    nvirt *= virt_dim[i];
  }

  for (i=0; i<tsr_A->ndim; i++){
    map = &tsr_A->edge_map[i];
    if (map->type == PHYSICAL_MAP){
      phys_mapped[2*map->cdt+0] = 1;
    }
    while (map->has_child) {
      map = map->child;
      if (map->type == PHYSICAL_MAP){
        phys_mapped[2*map->cdt+0] = 1;
      }
    }
  }
  for (i=0; i<tsr_B->ndim; i++){
    map = &tsr_B->edge_map[i];
    if (map->type == PHYSICAL_MAP){
      phys_mapped[2*map->cdt+1] = 1;
    }
    while (map->has_child) {
      map = map->child;
      if (map->type == PHYSICAL_MAP){
        phys_mapped[2*map->cdt+1] = 1;
      }
    }
  }
  need_rep = 0;
  for (i=0; i<nphys_dim; i++){
    if (phys_mapped[2*i+0] == 0 ||
        phys_mapped[2*i+1] == 0){
      need_rep = 1;
      break;
    }
  }
  if (need_rep){
    if (global_comm->rank == 0)
      DPRINTF(1,"Replicating tensor\n");

    tsum_replicate<dtype> * rtsum = new tsum_replicate<dtype>;
    if (is_top){
      htsum = rtsum;
      is_top = 0;
    } else {
      *rec_tsum = rtsum;
    }
    rec_tsum = &rtsum->rec_tsum;
    rtsum->ncdt_A = 0;
    rtsum->ncdt_B = 0;
    rtsum->size_A = blk_sz_A;
    rtsum->size_B = blk_sz_B;
    rtsum->cdt_A = NULL;
    rtsum->cdt_B = NULL;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
        rtsum->ncdt_A++;
      }
      if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
        rtsum->ncdt_B++;
      }
    }
    if (rtsum->ncdt_A > 0)
      CTF_alloc_ptr(sizeof(CommData_t*)*rtsum->ncdt_A, (void**)&rtsum->cdt_A);
    if (rtsum->ncdt_B > 0)
      CTF_alloc_ptr(sizeof(CommData_t*)*rtsum->ncdt_B, (void**)&rtsum->cdt_B);
    rtsum->ncdt_A = 0;
    rtsum->ncdt_B = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
        rtsum->cdt_A[rtsum->ncdt_A] = topovec[tsr_A->itopo].dim_comm[i];
        rtsum->ncdt_A++;
      }
      if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
        rtsum->cdt_B[rtsum->ncdt_B] = topovec[tsr_B->itopo].dim_comm[i];
        rtsum->ncdt_B++;
      }
    }
    LIBT_ASSERT(rtsum->ncdt_A == 0 || rtsum->cdt_B == 0);
  }

  int * new_sym_A, * new_sym_B;
  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&new_sym_A);
  memcpy(new_sym_A, tsr_A->sym, sizeof(int)*tsr_A->ndim);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&new_sym_B);
  memcpy(new_sym_B, tsr_B->sym, sizeof(int)*tsr_B->ndim);

  /* Multiply over virtual sub-blocks */
  if (nvirt > 1){
    tsum_virt<dtype> * tsumv = new tsum_virt<dtype>;
    if (is_top) {
      htsum = tsumv;
      is_top = 0;
    } else {
      *rec_tsum = tsumv;
    }
    rec_tsum = &tsumv->rec_tsum;

    tsumv->num_dim  = ndim_tot;
    tsumv->virt_dim   = virt_dim;
    tsumv->ndim_A = tsr_A->ndim;
    tsumv->blk_sz_A = vrt_sz_A;
    tsumv->idx_map_A  = idx_A;
    tsumv->ndim_B = tsr_B->ndim;
    tsumv->blk_sz_B = vrt_sz_B;
    tsumv->idx_map_B  = idx_B;
    tsumv->buffer = NULL;
  } else CTF_free(virt_dim);

  seq_tsr_sum<dtype> * tsumseq = new seq_tsr_sum<dtype>;
  if (inner_stride == -1){
    tsumseq->is_inner = 0;
  } else {
    tsumseq->is_inner = 1;
    tsumseq->inr_stride = inner_stride;
    tensor<dtype> * itsr;
    itsr = tensors[tsr_A->rec_tid];
    i_A = 0;
    for (i=0; i<tsr_A->ndim; i++){
      if (tsr_A->sym[i] == NS){
        for (j=0; j<itsr->ndim; j++){
          if (tsr_A->inner_ordering[j] == i_A){
            j=i;
            do {
              j--;
            } while (j>=0 && tsr_A->sym[j] != NS);
            for (k=j+1; k<=i; k++){
              virt_blk_len_A[k] = 1;
              new_sym_A[k] = NS;
            }
            break;
          }
        }
        i_A++;
      }
    }
    itsr = tensors[tsr_B->rec_tid];
    i_B = 0;
    for (i=0; i<tsr_B->ndim; i++){
      if (tsr_B->sym[i] == NS){
        for (j=0; j<itsr->ndim; j++){
          if (tsr_B->inner_ordering[j] == i_B){
            j=i;
            do {
              j--;
            } while (j>=0 && tsr_B->sym[j] != NS);
            for (k=j+1; k<=i; k++){
              virt_blk_len_B[k] = 1;
              new_sym_B[k] = NS;
            }
            break;
          }
        }
        i_B++;
      }
    }
  }
  if (is_top) {
    htsum = tsumseq;
    is_top = 0;
  } else {
    *rec_tsum = tsumseq;
  }
  tsumseq->ndim_A         = tsr_A->ndim;
  tsumseq->idx_map_A      = idx_A;
  tsumseq->edge_len_A     = virt_blk_len_A;
  tsumseq->sym_A          = new_sym_A;
  tsumseq->ndim_B         = tsr_B->ndim;
  tsumseq->idx_map_B      = idx_B;
  tsumseq->edge_len_B     = virt_blk_len_B;
  tsumseq->sym_B          = new_sym_B;
  tsumseq->func_ptr       = ftsr;
  tsumseq->custom_params  = felm;
  tsumseq->is_custom      = (felm.func_ptr != NULL);

  htsum->A      = tsr_A->data;
  htsum->B      = tsr_B->data;
  htsum->alpha  = alpha;
  htsum->beta   = beta;

  CTF_free(idx_arr);
  CTF_free(blk_len_A);
  CTF_free(blk_len_B);
  CTF_free(phys_mapped);

  return htsum;
}


/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
 *  seq_func needed to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] ftsr pointer to sequential block contract function
 * \param[in] felm pointer to sequential element-wise contract function
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] is_inner whether the tensors have two levels of blocking
 *                     0->no blocking 1->inner_blocking 2->folding
 * \param[in] inner_params parameters for inner contraction
 * \param[out] nvirt_all total virtualization factor
 * \return ctr contraction class to run
 */
template<typename dtype>
ctr<dtype> * dist_tensor<dtype>::
    construct_contraction(CTF_ctr_type_t const *      type,
                          fseq_tsr_ctr<dtype> const   ftsr,
                          fseq_elm_ctr<dtype> const   felm,
                          dtype const                 alpha,
                          dtype const                 beta,
                          int const                   is_inner,
                          iparam const *              inner_params,
                          int *                       nvirt_all){
  int num_tot, i, i_A, i_B, i_C, is_top, j, nphys_dim, nstep, k;
  long_int nvirt;
  long_int blk_sz_A, blk_sz_B, blk_sz_C;
  long_int vrt_sz_A, vrt_sz_B, vrt_sz_C;
  int sA, sB, sC, need_rep;
  int * blk_len_A, * virt_blk_len_A, * blk_len_B;
  int * virt_blk_len_B, * blk_len_C, * virt_blk_len_C;
  int * idx_arr, * virt_dim, * phys_mapped;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  strp_tsr<dtype> * str_A, * str_B, * str_C;
  mapping * map;
  ctr<dtype> * hctr = NULL;
  ctr<dtype> ** rec_ctr = NULL;

  TAU_FSTART(construct_contraction);

  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);

  nphys_dim = topovec[tsr_A->itopo].ndim;

  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&virt_blk_len_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&virt_blk_len_B);
  CTF_alloc_ptr(sizeof(int)*tsr_C->ndim, (void**)&virt_blk_len_C);

  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&blk_len_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&blk_len_B);
  CTF_alloc_ptr(sizeof(int)*tsr_C->ndim, (void**)&blk_len_C);
  CTF_alloc_ptr(sizeof(int)*num_tot, (void**)&virt_dim);
  CTF_alloc_ptr(sizeof(int)*nphys_dim*3, (void**)&phys_mapped);
  memset(phys_mapped, 0, sizeof(int)*nphys_dim*3);


  /* Determine the block dimensions of each local subtensor */
  blk_sz_A = tsr_A->size;
  blk_sz_B = tsr_B->size;
  blk_sz_C = tsr_C->size;
  calc_dim(tsr_A->ndim, blk_sz_A, tsr_A->edge_len, tsr_A->edge_map,
           &vrt_sz_A, virt_blk_len_A, blk_len_A);
  calc_dim(tsr_B->ndim, blk_sz_B, tsr_B->edge_len, tsr_B->edge_map,
           &vrt_sz_B, virt_blk_len_B, blk_len_B);
  calc_dim(tsr_C->ndim, blk_sz_C, tsr_C->edge_len, tsr_C->edge_map,
           &vrt_sz_C, virt_blk_len_C, blk_len_C);

  /* Strip out the relevant part of the tensor if we are contracting over diagonal */
  sA = strip_diag<dtype>( tsr_A->ndim, num_tot, type->idx_map_A, vrt_sz_A,
                          tsr_A->edge_map, &topovec[tsr_A->itopo],
                          blk_len_A, &blk_sz_A, &str_A);
  sB = strip_diag<dtype>( tsr_B->ndim, num_tot, type->idx_map_B, vrt_sz_B,
                          tsr_B->edge_map, &topovec[tsr_B->itopo],
                          blk_len_B, &blk_sz_B, &str_B);
  sC = strip_diag<dtype>( tsr_C->ndim, num_tot, type->idx_map_C, vrt_sz_C,
                          tsr_C->edge_map, &topovec[tsr_C->itopo],
                          blk_len_C, &blk_sz_C, &str_C);

  is_top = 1;
  if (sA || sB || sC){
    if (global_comm->rank == 0)
      DPRINTF(1,"Stripping tensor\n");
    strp_ctr<dtype> * sctr = new strp_ctr<dtype>;
    hctr = sctr;
    hctr->num_lyr = 1;
    hctr->idx_lyr = 0;
    is_top = 0;
    rec_ctr = &sctr->rec_ctr;

    sctr->rec_strp_A = str_A;
    sctr->rec_strp_B = str_B;
    sctr->rec_strp_C = str_C;
    sctr->strip_A = sA;
    sctr->strip_B = sB;
    sctr->strip_C = sC;
  }

  for (i=0; i<tsr_A->ndim; i++){
    map = &tsr_A->edge_map[i];
    if (map->type == PHYSICAL_MAP){
      phys_mapped[3*map->cdt+0] = 1;
    }
    while (map->has_child) {
      map = map->child;
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+0] = 1;
      }
    }
  }
  for (i=0; i<tsr_B->ndim; i++){
    map = &tsr_B->edge_map[i];
    if (map->type == PHYSICAL_MAP){
      phys_mapped[3*map->cdt+1] = 1;
    }
    while (map->has_child) {
      map = map->child;
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+1] = 1;
      }
    }
  }
  for (i=0; i<tsr_C->ndim; i++){
    map = &tsr_C->edge_map[i];
    if (map->type == PHYSICAL_MAP){
      phys_mapped[3*map->cdt+2] = 1;
    }
    while (map->has_child) {
      map = map->child;
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+2] = 1;
      }
    }
  }
  need_rep = 0;
  for (i=0; i<nphys_dim; i++){
    if (phys_mapped[3*i+0] == 0 ||
      phys_mapped[3*i+1] == 0 ||
      phys_mapped[3*i+2] == 0){
      /*LIBT_ASSERT((phys_mapped[3*i+0] == 0 && phys_mapped[3*i+1] == 0) ||
      (phys_mapped[3*i+0] == 0 && phys_mapped[3*i+2] == 0) ||
      (phys_mapped[3*i+1] == 0 && phys_mapped[3*i+2] == 0));*/
      need_rep = 1;
      break;
    }
  }
  if (need_rep){
    if (global_comm->rank == 0)
      DPRINTF(1,"Replicating tensor\n");

    ctr_replicate<dtype> * rctr = new ctr_replicate<dtype>;
    if (is_top){
      hctr = rctr;
      is_top = 0;
    } else {
      *rec_ctr = rctr;
    }
    rec_ctr = &rctr->rec_ctr;
    hctr->idx_lyr = 0;
    hctr->num_lyr = 1;
    rctr->idx_lyr = 0;
    rctr->num_lyr = 1;
    rctr->ncdt_A = 0;
    rctr->ncdt_B = 0;
    rctr->ncdt_C = 0;
    rctr->size_A = blk_sz_A;
    rctr->size_B = blk_sz_B;
    rctr->size_C = blk_sz_C;
    rctr->cdt_A = NULL;
    rctr->cdt_B = NULL;
    rctr->cdt_C = NULL;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[3*i+0] == 0 &&
          phys_mapped[3*i+1] == 0 &&
          phys_mapped[3*i+2] == 0){
/*        printf("ERROR: ALL-TENSOR REPLICATION NO LONGER DONE\n");
        ABORT;
        LIBT_ASSERT(rctr->num_lyr == 1);
        hctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i]->rank;
        hctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;
        rctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i]->rank;
        rctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;*/
      } else {
        if (phys_mapped[3*i+0] == 0){
          rctr->ncdt_A++;
        }
        if (phys_mapped[3*i+1] == 0){
          rctr->ncdt_B++;
        }
        if (phys_mapped[3*i+2] == 0){
          rctr->ncdt_C++;
        }
      }
    }
    if (rctr->ncdt_A > 0)
      CTF_alloc_ptr(sizeof(CommData_t*)*rctr->ncdt_A, (void**)&rctr->cdt_A);
    if (rctr->ncdt_B > 0)
      CTF_alloc_ptr(sizeof(CommData_t*)*rctr->ncdt_B, (void**)&rctr->cdt_B);
    if (rctr->ncdt_C > 0)
      CTF_alloc_ptr(sizeof(CommData_t*)*rctr->ncdt_C, (void**)&rctr->cdt_C);
    rctr->ncdt_A = 0;
    rctr->ncdt_B = 0;
    rctr->ncdt_C = 0;
    for (i=0; i<nphys_dim; i++){
      if (!(phys_mapped[3*i+0] == 0 &&
            phys_mapped[3*i+1] == 0 &&
            phys_mapped[3*i+2] == 0)){
        if (phys_mapped[3*i+0] == 0){
          rctr->cdt_A[rctr->ncdt_A] = topovec[tsr_A->itopo].dim_comm[i];
          rctr->ncdt_A++;
        }
        if (phys_mapped[3*i+1] == 0){
          rctr->cdt_B[rctr->ncdt_B] = topovec[tsr_B->itopo].dim_comm[i];
          rctr->ncdt_B++;
        }
        if (phys_mapped[3*i+2] == 0){
          rctr->cdt_C[rctr->ncdt_C] = topovec[tsr_C->itopo].dim_comm[i];
          rctr->ncdt_C++;
        }
      }
    }
  }

  nvirt = 1;
/*  if (nvirt_all != NULL)
    *nvirt_all = 1;*/
  for (i=0; i<num_tot; i++){
    virt_dim[i] = 1;
    i_A = idx_arr[3*i+0];
    i_B = idx_arr[3*i+1];
    i_C = idx_arr[3*i+2];
    nstep = 1;
    /* If this index belongs to exactly two tensors */
    if ((i_A != -1 && i_B != -1 && i_C == -1) ||
        (i_A != -1 && i_B == -1 && i_C != -1) ||
        (i_A == -1 && i_B != -1 && i_C != -1)) {
      if (i_A == -1){
  if (comp_dim_map(&tsr_C->edge_map[i_C], &tsr_B->edge_map[i_B])){
    map = &tsr_B->edge_map[i_B];
    while (map->has_child) map = map->child;
    if (map->type == VIRTUAL_MAP)
      virt_dim[i] = map->np;
  } else {
    if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP &&
        tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
      //LIBT_ASSERT(0); //why the hell would this happen?
      virt_dim[i] = tsr_B->edge_map[i_B].np;
    } else {
      ctr_2d_general<dtype> * ctr_gen = new ctr_2d_general<dtype>;
      if (is_top) {
        hctr = ctr_gen;
        hctr->idx_lyr = 0;
        hctr->num_lyr = 1;
        is_top = 0;
      } else {
        *rec_ctr = ctr_gen;
      }
      rec_ctr = &ctr_gen->rec_ctr;

      ctr_gen->buffer = NULL; /* FIXME: learn to use buffer space */
      ctr_gen->edge_len = 1;
      if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP){
        ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_B->edge_map[i_B].np);
        ctr_gen->cdt_B = topovec[tsr_B->itopo].dim_comm[tsr_B->edge_map[i_B].cdt];
        nstep = tsr_B->edge_map[i_B].np;
      } else
        ctr_gen->cdt_B = NULL;
      if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP){
        ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_C->edge_map[i_C].np);
        ctr_gen->cdt_C = topovec[tsr_C->itopo].dim_comm[tsr_C->edge_map[i_C].cdt];
        nstep = MAX(nstep, tsr_C->edge_map[i_C].np);
      } else
        ctr_gen->cdt_C = NULL;
      ctr_gen->cdt_A = NULL;

      ctr_gen->ctr_lda_A = 1;
      ctr_gen->ctr_sub_lda_A = 0;

      /* Adjust the block lengths, since this algorithm will cut
         the block into smaller ones of the min block length */
      /* Determine the LDA of this dimension, based on virtualization */
      ctr_gen->ctr_lda_B  = 1;
      if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP)
        ctr_gen->ctr_sub_lda_B= blk_sz_B*tsr_B->edge_map[i_B].np/ctr_gen->edge_len;
      else
        ctr_gen->ctr_sub_lda_B= blk_sz_B/ctr_gen->edge_len;
      for (j=i_B+1; j<tsr_B->ndim; j++) {
        ctr_gen->ctr_sub_lda_B = (ctr_gen->ctr_sub_lda_B *
              virt_blk_len_B[j]) / blk_len_B[j];
        ctr_gen->ctr_lda_B = (ctr_gen->ctr_lda_B*blk_len_B[j])
              /virt_blk_len_B[j];
      }
      ctr_gen->ctr_lda_C  = 1;
      if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP)
        ctr_gen->ctr_sub_lda_C= blk_sz_C*tsr_C->edge_map[i_C].np/ctr_gen->edge_len;
      else
        ctr_gen->ctr_sub_lda_C= blk_sz_C/ctr_gen->edge_len;
      for (j=i_C+1; j<tsr_C->ndim; j++) {
        ctr_gen->ctr_sub_lda_C = (ctr_gen->ctr_sub_lda_C *
              virt_blk_len_C[j]) / blk_len_C[j];
        ctr_gen->ctr_lda_C = (ctr_gen->ctr_lda_C*blk_len_C[j])
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
    }
  }
      }
      if (i_B == -1){
        if (comp_dim_map(&tsr_A->edge_map[i_A], &tsr_C->edge_map[i_C])){
          map = &tsr_C->edge_map[i_C];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else {
          if (tsr_C->edge_map[i_C].type == VIRTUAL_MAP &&
              tsr_A->edge_map[i_A].type == VIRTUAL_MAP){
      //      LIBT_ASSERT(0); //why the hell would this happen?
            virt_dim[i] = tsr_C->edge_map[i_C].np;
            /*if (nvirt_all != NULL)
              *nvirt_all = (*nvirt_all)*tsr_C->edge_map[i_C].np;*/
          } else {
            ctr_2d_general<dtype> * ctr_gen = new ctr_2d_general<dtype>;
            if (is_top) {
              hctr = ctr_gen;
              hctr->idx_lyr = 0;
              hctr->num_lyr = 1;
              is_top = 0;
            } else {
              *rec_ctr = ctr_gen;
            }
            rec_ctr = &ctr_gen->rec_ctr;

            ctr_gen->buffer = NULL; /* FIXME: learn to use buffer space */
            ctr_gen->edge_len = 1;
            if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP){
              ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_C->edge_map[i_C].np);
              ctr_gen->cdt_C = topovec[tsr_C->itopo].dim_comm[tsr_C->edge_map[i_C].cdt];
              nstep = tsr_C->edge_map[i_C].np;
            } else {
              ctr_gen->cdt_C = NULL;
            }
            if (tsr_A->edge_map[i_A].type == PHYSICAL_MAP){
              ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_A->edge_map[i_A].np);
              ctr_gen->cdt_A = topovec[tsr_A->itopo].dim_comm[tsr_A->edge_map[i_A].cdt];
              nstep = MAX(nstep, tsr_A->edge_map[i_A].np);
            } else
              ctr_gen->cdt_A = NULL;
            ctr_gen->cdt_B = NULL;

            ctr_gen->ctr_lda_B = 1;
            ctr_gen->ctr_sub_lda_B = 0;

            /* Adjust the block lengths, since this algorithm will cut
               the block into smaller ones of the min block length */
            /* Determine the LDA of this dimension, based on virtualization */
            ctr_gen->ctr_lda_C  = 1;
            if (tsr_C->edge_map[i_C].type == PHYSICAL_MAP)
              ctr_gen->ctr_sub_lda_C= blk_sz_C*tsr_C->edge_map[i_C].np/ctr_gen->edge_len;
            else
              ctr_gen->ctr_sub_lda_C= blk_sz_C/ctr_gen->edge_len;
            for (j=i_C+1; j<tsr_C->ndim; j++) {
              ctr_gen->ctr_sub_lda_C = (ctr_gen->ctr_sub_lda_C *
                    virt_blk_len_C[j]) / blk_len_C[j];
              ctr_gen->ctr_lda_C = (ctr_gen->ctr_lda_C*blk_len_C[j])
                    /virt_blk_len_C[j];
            }
            ctr_gen->ctr_lda_A  = 1;
            if (tsr_A->edge_map[i_A].type == PHYSICAL_MAP)
              ctr_gen->ctr_sub_lda_A= blk_sz_A*tsr_A->edge_map[i_A].np/ctr_gen->edge_len;
            else
              ctr_gen->ctr_sub_lda_A= blk_sz_A/ctr_gen->edge_len;
            for (j=i_A+1; j<tsr_A->ndim; j++) {
              ctr_gen->ctr_sub_lda_A = (ctr_gen->ctr_sub_lda_A *
                    virt_blk_len_A[j]) / blk_len_A[j];
              ctr_gen->ctr_lda_A = (ctr_gen->ctr_lda_A*blk_len_A[j])
                    /virt_blk_len_A[j];
            }

            if (tsr_A->edge_map[i_A].type != PHYSICAL_MAP){
              blk_sz_A  = blk_sz_A / nstep;
              blk_len_A[i_A] = blk_len_A[i_A] / nstep;
            } else {
              blk_sz_A  = blk_sz_A * tsr_A->edge_map[i_A].np / nstep;
              blk_len_A[i_A] = blk_len_A[i_A] * tsr_A->edge_map[i_A].np / nstep;

            }
            if (tsr_C->edge_map[i_C].type != PHYSICAL_MAP){
              blk_sz_C  = blk_sz_C / nstep;
              blk_len_C[i_C] = blk_len_C[i_C] / nstep;
            } else {
              blk_sz_C  = blk_sz_C * tsr_C->edge_map[i_C].np / nstep;
              blk_len_C[i_C] = blk_len_C[i_C] * tsr_C->edge_map[i_C].np / nstep;

            }

            if (tsr_C->edge_map[i_C].has_child) {
              LIBT_ASSERT(tsr_C->edge_map[i_C].child->type == VIRTUAL_MAP);
              virt_dim[i] = tsr_C->edge_map[i_C].np*tsr_C->edge_map[i_C].child->np/nstep;
              /*if (nvirt_all != NULL)
          *nvirt_all = (*nvirt_all)*tsr_C->edge_map[i_C].np
                   *tsr_C->edge_map[i_C].child->np
                   / nstep;*/
            }
            if (tsr_A->edge_map[i_A].has_child){
              LIBT_ASSERT(tsr_A->edge_map[i_A].child->type == VIRTUAL_MAP);
              virt_dim[i] = tsr_A->edge_map[i_A].np*tsr_A->edge_map[i_A].child->np/nstep;
            }
            if (tsr_C->edge_map[i_C].type == VIRTUAL_MAP){
              virt_dim[i] = tsr_C->edge_map[i_C].np/nstep;
            }
            if (tsr_A->edge_map[i_A].type == VIRTUAL_MAP)
              virt_dim[i] = tsr_A->edge_map[i_A].np/nstep;
          }
        }
      }
      if (i_C == -1){
        if (comp_dim_map(&tsr_B->edge_map[i_B], &tsr_A->edge_map[i_A])){
          map = &tsr_A->edge_map[i_A];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else {
          if (tsr_A->edge_map[i_A].type == VIRTUAL_MAP &&
              tsr_B->edge_map[i_B].type == VIRTUAL_MAP){
      //      LIBT_ASSERT(0); //why the hell would this happen?
            virt_dim[i] = tsr_A->edge_map[i_A].np;
          } else {
            ctr_2d_general<dtype> * ctr_gen = new ctr_2d_general<dtype>;
            if (is_top) {
              hctr = ctr_gen;
              hctr->idx_lyr = 0;
              hctr->num_lyr = 1;
              is_top = 0;
            } else {
              *rec_ctr = ctr_gen;
            }
            rec_ctr = &ctr_gen->rec_ctr;

            ctr_gen->buffer = NULL; /* FIXME: learn to use buffer space */
            ctr_gen->edge_len = 1;
            if (tsr_A->edge_map[i_A].type == PHYSICAL_MAP){
              ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_A->edge_map[i_A].np);
              ctr_gen->cdt_A = topovec[tsr_A->itopo].dim_comm[tsr_A->edge_map[i_A].cdt];
              nstep = tsr_A->edge_map[i_A].np;
            } else
              ctr_gen->cdt_A = NULL;
            if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP){
              ctr_gen->edge_len = lcm(ctr_gen->edge_len, tsr_B->edge_map[i_B].np);
              ctr_gen->cdt_B = topovec[tsr_B->itopo].dim_comm[tsr_B->edge_map[i_B].cdt];
              nstep = MAX(nstep, tsr_B->edge_map[i_B].np);
            } else
              ctr_gen->cdt_B = NULL;
            ctr_gen->cdt_C = NULL;

            ctr_gen->ctr_lda_C = 1;
            ctr_gen->ctr_sub_lda_C = 0;

            /* Adjust the block lengths, since this algorithm will cut
               the block into smaller ones of the min block length */
            /* Determine the LDA of this dimension, based on virtualization */
            ctr_gen->ctr_lda_A  = 1;
            if (tsr_A->edge_map[i_A].type == PHYSICAL_MAP)
              ctr_gen->ctr_sub_lda_A= blk_sz_A*tsr_A->edge_map[i_A].np/ctr_gen->edge_len;
            else
              ctr_gen->ctr_sub_lda_A= blk_sz_A/ctr_gen->edge_len;
            for (j=i_A+1; j<tsr_A->ndim; j++) {
              ctr_gen->ctr_sub_lda_A = (ctr_gen->ctr_sub_lda_A *
                    virt_blk_len_A[j]) / blk_len_A[j];
              ctr_gen->ctr_lda_A = (ctr_gen->ctr_lda_A*blk_len_A[j])
                    /virt_blk_len_A[j];
            }
            ctr_gen->ctr_lda_B  = 1;
            if (tsr_B->edge_map[i_B].type == PHYSICAL_MAP)
              ctr_gen->ctr_sub_lda_B= blk_sz_B*tsr_B->edge_map[i_B].np/ctr_gen->edge_len;
            else
              ctr_gen->ctr_sub_lda_B= blk_sz_B/ctr_gen->edge_len;
            for (j=i_B+1; j<tsr_B->ndim; j++) {
              ctr_gen->ctr_sub_lda_B = (ctr_gen->ctr_sub_lda_B *
                    virt_blk_len_B[j]) / blk_len_B[j];
              ctr_gen->ctr_lda_B = (ctr_gen->ctr_lda_B*blk_len_B[j])
                    /virt_blk_len_B[j];
            }

            if (tsr_A->edge_map[i_A].type != PHYSICAL_MAP){
              blk_sz_A  = blk_sz_A / nstep;
              blk_len_A[i_A] = blk_len_A[i_A] / nstep;
            } else {
              blk_sz_A  = blk_sz_A * tsr_A->edge_map[i_A].np / nstep;
              blk_len_A[i_A] = blk_len_A[i_A] * tsr_A->edge_map[i_A].np / nstep;
            }
            if (tsr_B->edge_map[i_B].type != PHYSICAL_MAP){
              blk_sz_B  = blk_sz_B / nstep;
              blk_len_B[i_B] = blk_len_B[i_B] / nstep;
            } else {
              blk_sz_B  = blk_sz_B * tsr_B->edge_map[i_B].np / nstep;
              blk_len_B[i_B] = blk_len_B[i_B] * tsr_B->edge_map[i_B].np / nstep;
            }

            if (tsr_A->edge_map[i_A].has_child){
              LIBT_ASSERT(tsr_A->edge_map[i_A].child->type == VIRTUAL_MAP);
              virt_dim[i] = tsr_A->edge_map[i_A].np*tsr_A->edge_map[i_A].child->np/nstep;
            }
            if (tsr_B->edge_map[i_B].has_child){
              LIBT_ASSERT(tsr_B->edge_map[i_B].child->type == VIRTUAL_MAP);
              virt_dim[i] = tsr_B->edge_map[i_B].np*tsr_B->edge_map[i_B].child->np/nstep;
            }
            if (tsr_B->edge_map[i_B].type == VIRTUAL_MAP)
              virt_dim[i] = tsr_B->edge_map[i_B].np/nstep;
            if (tsr_A->edge_map[i_A].type == VIRTUAL_MAP)
              virt_dim[i] = tsr_A->edge_map[i_A].np/nstep;
          }
        }
      }
    } else {
      if (i_A != -1){
        map = &tsr_A->edge_map[i_A];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP)
          virt_dim[i] = map->np;
      } else if (i_B != -1){
        map = &tsr_B->edge_map[i_B];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP)
          virt_dim[i] = map->np;
      } else if (i_C != -1){
        map = &tsr_C->edge_map[i_C];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP)
          virt_dim[i] = map->np;
      }
    }
    if (sA && i_A != -1){
      nvirt = virt_dim[i]/str_A->strip_dim[i_A];
    } else if (sB && i_B != -1){
      nvirt = virt_dim[i]/str_B->strip_dim[i_B];
    } else if (sC && i_C != -1){
      nvirt = virt_dim[i]/str_C->strip_dim[i_C];
    }
    
    nvirt = nvirt * virt_dim[i];
  }
  if (nvirt_all != NULL)
    *nvirt_all = nvirt;

  LIBT_ASSERT(blk_sz_A >= vrt_sz_A);
  LIBT_ASSERT(blk_sz_B >= vrt_sz_B);
  LIBT_ASSERT(blk_sz_C >= vrt_sz_C);
    
  int * new_sym_A, * new_sym_B, * new_sym_C;
  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&new_sym_A);
  memcpy(new_sym_A, tsr_A->sym, sizeof(int)*tsr_A->ndim);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&new_sym_B);
  memcpy(new_sym_B, tsr_B->sym, sizeof(int)*tsr_B->ndim);
  CTF_alloc_ptr(sizeof(int)*tsr_C->ndim, (void**)&new_sym_C);
  memcpy(new_sym_C, tsr_C->sym, sizeof(int)*tsr_C->ndim);

  /* Multiply over virtual sub-blocks */
  if (nvirt > 1){
#ifdef USE_VIRT_25D
    ctr_virt_25d<dtype> * ctrv = new ctr_virt_25d<dtype>;
#else
    ctr_virt<dtype> * ctrv = new ctr_virt<dtype>;
#endif
    if (is_top) {
      hctr = ctrv;
      hctr->idx_lyr = 0;
      hctr->num_lyr = 1;
      is_top = 0;
    } else {
      *rec_ctr = ctrv;
    }
    rec_ctr = &ctrv->rec_ctr;

    ctrv->num_dim   = num_tot;
    ctrv->virt_dim  = virt_dim;
    ctrv->ndim_A  = tsr_A->ndim;
    ctrv->blk_sz_A  = vrt_sz_A;
    ctrv->idx_map_A = type->idx_map_A;
    ctrv->ndim_B  = tsr_B->ndim;
    ctrv->blk_sz_B  = vrt_sz_B;
    ctrv->idx_map_B = type->idx_map_B;
    ctrv->ndim_C  = tsr_C->ndim;
    ctrv->blk_sz_C  = vrt_sz_C;
    ctrv->idx_map_C = type->idx_map_C;
    ctrv->buffer  = NULL;
  } else
    CTF_free(virt_dim);

  seq_tsr_ctr<dtype> * ctrseq = new seq_tsr_ctr<dtype>;
  if (is_top) {
    hctr = ctrseq;
    hctr->idx_lyr = 0;
    hctr->num_lyr = 1;
    is_top = 0;
  } else {
    *rec_ctr = ctrseq;
  }
  if (!is_inner){
    ctrseq->is_inner  = 0;
    ctrseq->func_ptr  = ftsr;
  } else if (is_inner == 1) {
    ctrseq->is_inner    = 1;
    ctrseq->inner_params  = *inner_params;
    ctrseq->inner_params.sz_C = vrt_sz_C;
    tensor<dtype> * itsr;
    int * iphase;
    itsr = tensors[tsr_A->rec_tid];
    iphase = calc_phase<dtype>(itsr);
    for (i=0; i<tsr_A->ndim; i++){
      if (virt_blk_len_A[i]%iphase[i] > 0)
        virt_blk_len_A[i] = virt_blk_len_A[i]/iphase[i]+1;
      else
        virt_blk_len_A[i] = virt_blk_len_A[i]/iphase[i];

    }
    CTF_free(iphase);
    itsr = tensors[tsr_B->rec_tid];
    iphase = calc_phase<dtype>(itsr);
    for (i=0; i<tsr_B->ndim; i++){
      if (virt_blk_len_B[i]%iphase[i] > 0)
        virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i]+1;
      else
        virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i];
    }
    CTF_free(iphase);
    itsr = tensors[tsr_C->rec_tid];
    iphase = calc_phase<dtype>(itsr);
    for (i=0; i<tsr_C->ndim; i++){
      if (virt_blk_len_C[i]%iphase[i] > 0)
        virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i]+1;
      else
        virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i];
    }
    CTF_free(iphase);
  } else if (is_inner == 2) {
    if (global_comm->rank == 0){
      DPRINTF(1,"Folded tensor n=%d m=%d k=%d\n", inner_params->n,
        inner_params->m, inner_params->k);
    }

    ctrseq->is_inner    = 1;
    ctrseq->inner_params  = *inner_params;
    ctrseq->inner_params.sz_C = vrt_sz_C;
    tensor<dtype> * itsr;
    itsr = tensors[tsr_A->rec_tid];
    for (i=0; i<itsr->ndim; i++){
      j = tsr_A->inner_ordering[i];
      for (k=0; k<tsr_A->ndim; k++){
        if (tsr_A->sym[k] == NS) j--;
        if (j<0) break;
      }
      j = k;
      while (k>0 && tsr_A->sym[k-1] != NS){
        k--;
      }
      for (; k<=j; k++){
/*        printf("inner_ordering[%d]=%d setting dim %d of A, to len %d from len %d\n",
                i, tsr_A->inner_ordering[i], k, 1, virt_blk_len_A[k]);*/
        virt_blk_len_A[k] = 1;
        new_sym_A[k] = NS;
      }
    }
    itsr = tensors[tsr_B->rec_tid];
    for (i=0; i<itsr->ndim; i++){
      j = tsr_B->inner_ordering[i];
      for (k=0; k<tsr_B->ndim; k++){
        if (tsr_B->sym[k] == NS) j--;
        if (j<0) break;
      }
      j = k;
      while (k>0 && tsr_B->sym[k-1] != NS){
        k--;
      }
      for (; k<=j; k++){
      /*  printf("inner_ordering[%d]=%d setting dim %d of B, to len %d from len %d\n",
                i, tsr_B->inner_ordering[i], k, 1, virt_blk_len_B[k]);*/
        virt_blk_len_B[k] = 1;
        new_sym_B[k] = NS;
      }
    }
    itsr = tensors[tsr_C->rec_tid];
    for (i=0; i<itsr->ndim; i++){
      j = tsr_C->inner_ordering[i];
      for (k=0; k<tsr_C->ndim; k++){
        if (tsr_C->sym[k] == NS) j--;
        if (j<0) break;
      }
      j = k;
      while (k>0 && tsr_C->sym[k-1] != NS){
        k--;
      }
      for (; k<=j; k++){
      /*  printf("inner_ordering[%d]=%d setting dim %d of C, to len %d from len %d\n",
                i, tsr_C->inner_ordering[i], k, 1, virt_blk_len_C[k]);*/
        virt_blk_len_C[k] = 1;
        new_sym_C[k] = NS;
      }
    }
  }
  ctrseq->alpha         = alpha;
  ctrseq->ndim_A        = tsr_A->ndim;
  ctrseq->idx_map_A     = type->idx_map_A;
  ctrseq->edge_len_A    = virt_blk_len_A;
  ctrseq->sym_A         = new_sym_A;
  ctrseq->ndim_B        = tsr_B->ndim;
  ctrseq->idx_map_B     = type->idx_map_B;
  ctrseq->edge_len_B    = virt_blk_len_B;
  ctrseq->sym_B         = new_sym_B;
  ctrseq->ndim_C        = tsr_C->ndim;
  ctrseq->idx_map_C     = type->idx_map_C;
  ctrseq->edge_len_C    = virt_blk_len_C;
  ctrseq->sym_C         = new_sym_C;
  ctrseq->custom_params = felm;
  ctrseq->is_custom     = (felm.func_ptr != NULL);

  hctr->A   = tsr_A->data;
  hctr->B   = tsr_B->data;
  hctr->C   = tsr_C->data;
  hctr->beta  = beta;
/*  if (global_comm->rank == 0){
    long_int n,m,k;
    dtype old_flops;
    dtype new_flops;
    ggg_sym_nmk(tsr_A->ndim, tsr_A->edge_len, type->idx_map_A, tsr_A->sym,
    tsr_B->ndim, tsr_B->edge_len, type->idx_map_B, tsr_B->sym,
    tsr_C->ndim, &n, &m, &k);
    old_flops = 2.0*(dtype)n*(dtype)m*(dtype)k;
    new_flops = calc_nvirt(tsr_A);
    new_flops *= calc_nvirt(tsr_B);
    new_flops *= calc_nvirt(tsr_C);
    new_flops *= global_comm->np;
    new_flops = sqrt(new_flops);
    new_flops *= global_comm->np;
    ggg_sym_nmk(tsr_A->ndim, virt_blk_len_A, type->idx_map_A, tsr_A->sym,
    tsr_B->ndim, virt_blk_len_B, type->idx_map_B, tsr_B->sym,
    tsr_C->ndim, &n, &m, &k);
    printf("Each subcontraction is a %lld by %lld by %lld DGEMM performing %E flops\n",n,m,k,
      2.0*(dtype)n*(dtype)m*(dtype)k);
    new_flops *= 2.0*(dtype)n*(dtype)m*(dtype)k;
    printf("Contraction performing %E flops rather than %E, a factor of %lf more flops due to padding\n",
      new_flops, old_flops, new_flops/old_flops);

  }*/

  CTF_free(idx_arr);
  CTF_free(blk_len_A);
  CTF_free(blk_len_B);
  CTF_free(blk_len_C);
  CTF_free(phys_mapped);
  TAU_FSTOP(construct_contraction);
  return hctr;
}

/**
 * \brief a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
 *        performs all necessary symmetric permutations
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] ftsr pointer to sequential block sum function
 * \param[in] felm pointer to sequential element-wise sum function
 * \param[in] run_diag if 1 run diagonal sum
 */
template<typename dtype>
int dist_tensor<dtype>::home_sum_tsr(dtype const                alpha_,
                                     dtype const                beta,
                                     int const                  tid_A,
                                     int const                  tid_B,
                                     int const *                idx_map_A,
                                     int const *                idx_map_B,
                                     fseq_tsr_sum<dtype> const  ftsr,
                                     fseq_elm_sum<dtype> const  felm,
                                     int const                  run_diag){
  int ret, was_home_A, was_home_B;
  tensor<dtype> * tsr_A, * tsr_B, * ntsr_A, * ntsr_B;
  int was_padded_B, was_cyclic_B;
  long_int old_size_B;
  int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B;
  int * old_padding_B, * old_edge_len_B;
  CTF_sum_type_t type;
  type.tid_A = tid_A;
  type.tid_B = tid_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  contract_mst();

  CTF_alloc_ptr(sizeof(int)*tsr_A->ndim, (void**)&type.idx_map_A);
  CTF_alloc_ptr(sizeof(int)*tsr_B->ndim, (void**)&type.idx_map_B);

  memcpy(type.idx_map_A, idx_map_A, sizeof(int)*tsr_A->ndim);
  memcpy(type.idx_map_B, idx_map_B, sizeof(int)*tsr_B->ndim);
#ifndef HOME_CONTRACT
  #ifdef USE_SYM_SUM
    ret = sym_sum_tsr(alpha_, beta, &type, ftsr, felm, run_diag);
    free_type(&type);
    return ret;
  #else
    ret = sum_tensors(alpha_, beta, tid_A, tid_B, idx_map_A, idx_map_B, ftsr, felm, run_diag);
    free_type(&type);
    return ret;
  #endif
#else
  int new_tid;
  if (tsr_A->has_zero_edge_len || 
      tsr_B->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  if (tid_A == tid_B){
    clone_tensor(tid_A, 1, &new_tid);
    ret = home_sum_tsr(alpha_, beta, new_tid, tid_B, 
                        idx_map_A, idx_map_B, ftsr, felm, run_diag);
    del_tsr(new_tid);
    return ret;
  }
  was_home_A = tsr_A->is_home;
  was_home_B = tsr_B->is_home;
  if (was_home_A){
    clone_tensor(tid_A, 0, &type.tid_A, 0);
    ntsr_A = tensors[type.tid_A];
    ntsr_A->data = tsr_A->data;
    ntsr_A->home_buffer = tsr_A->home_buffer;
    ntsr_A->is_home = 1;
    ntsr_A->is_mapped = 1;
    ntsr_A->itopo = tsr_A->itopo;
    copy_mapping(tsr_A->ndim, tsr_A->edge_map, ntsr_A->edge_map);
    set_padding(ntsr_A);
  }     
  if (was_home_B){
    clone_tensor(tid_B, 0, &type.tid_B, 0);
    ntsr_B = tensors[type.tid_B];
    ntsr_B->data = tsr_B->data;
    ntsr_B->home_buffer = tsr_B->home_buffer;
    ntsr_B->is_home = 1;
    ntsr_B->is_mapped = 1;
    ntsr_B->itopo = tsr_B->itopo;
    copy_mapping(tsr_B->ndim, tsr_B->edge_map, ntsr_B->edge_map);
    set_padding(ntsr_B);
  }
  
  #ifdef USE_SYM_SUM
  ret = sym_sum_tsr(alpha_, beta, &type, ftsr, felm, run_diag);
  #else
  ret = sum_tensors(alpha_, beta, type.tid_A, type.tid_B, idx_map_A, idx_map_B, ftsr, felm, run_diag);
  #endif

  if (ret!= DIST_TENSOR_SUCCESS) return ret;
  if (was_home_A) unmap_inner(ntsr_A);
  if (was_home_B) unmap_inner(ntsr_B);

  if (was_home_B && !ntsr_B->is_home){
    if (global_comm->rank == 0)
      DPRINTF(2,"Migrating tensor %d back to home\n", tid_B);
    save_mapping(ntsr_B,
                 &old_phase_B, &old_rank_B, 
                 &old_virt_dim_B, &old_pe_lda_B, 
                 &old_size_B, &was_padded_B, 
                 &was_cyclic_B, &old_padding_B, 
                 &old_edge_len_B, &topovec[ntsr_B->itopo]);
    tsr_B->data = ntsr_B->data;
    tsr_B->is_home = 0;
    remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size_B, 
                 old_phase_B, old_rank_B, old_virt_dim_B, 
                 old_pe_lda_B, was_padded_B, was_cyclic_B, 
                 old_padding_B, old_edge_len_B, global_comm);
    memcpy(tsr_B->home_buffer, tsr_B->data, tsr_B->size*sizeof(dtype));
    CTF_free(tsr_B->data);
    tsr_B->data = tsr_B->home_buffer;
    tsr_B->is_home = 1;
    ntsr_B->is_data_aliased = 1;
    del_tsr(type.tid_B);
    CTF_free(old_phase_B);
    CTF_free(old_rank_B);
    CTF_free(old_virt_dim_B);
    CTF_free(old_pe_lda_B);
    CTF_free(old_padding_B);
    CTF_free(old_edge_len_B);
  } else if (was_home_B){
    if (ntsr_B->data != tsr_B->data){
      printf("Tensor %d is a copy of %d and did not leave home but buffer is %p was %p\n", type.tid_B, tid_B, ntsr_B->data, tsr_B->data);
      ABORT;

    }
    ntsr_B->has_home = 0;
    ntsr_B->is_data_aliased = 1;
    del_tsr(type.tid_B);
  }
  if (was_home_A && !ntsr_A->is_home){
    ntsr_A->has_home = 0;
    del_tsr(type.tid_A);
  } else if (was_home_A) {
    ntsr_A->has_home = 0;
    ntsr_A->is_data_aliased = 1;
    del_tsr(type.tid_A);
  }
  free_type(&type);
  return ret;
#endif
}


/**
 * \brief a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
 *        performs all necessary symmetric permutations
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] type contains tensors ids and maps
 * \param[in] ftsr pointer to sequential block sum function
 * \param[in] felm pointer to sequential element-wise sum function
 * \param[in] run_diag if 1 run diagonal sum
 */
template<typename dtype>
int dist_tensor<dtype>::sym_sum_tsr( dtype const                alpha_,
                                     dtype const                beta,
                                     CTF_sum_type_t const *     type,
                                     fseq_tsr_sum<dtype> const  ftsr,
                                     fseq_elm_sum<dtype> const  felm,
                                     int const                  run_diag){
  int stat, i, new_tid, * new_idx_map;
  int * map_A, * map_B, * dstack_tid_B;
  int ** dstack_map_B;
  int ntid_A, ntid_B, nst_B;
  std::vector<CTF_sum_type_t> perm_types;
  std::vector<dtype> signs;
  dtype dbeta;
  tsum<dtype> * sumf;
  CTF_sum_type_t unfold_type;
  check_sum(type);
  if (tensors[type->tid_A]->has_zero_edge_len || 
      tensors[type->tid_B]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  ntid_A = type->tid_A;
  ntid_B = type->tid_B;
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_A]->ndim,   (void**)&map_A);
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->ndim,   (void**)&map_B);
  CTF_alloc_ptr(sizeof(int*)*tensors[ntid_B]->ndim,   (void**)&dstack_map_B);
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->ndim,   (void**)&dstack_tid_B);
  memcpy(map_A, type->idx_map_A, tensors[ntid_A]->ndim*sizeof(int));
  memcpy(map_B, type->idx_map_B, tensors[ntid_B]->ndim*sizeof(int));
  while (extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    if (ntid_A != type->tid_A) del_tsr(ntid_A);
    CTF_free(map_A);
    ntid_A = new_tid;
    map_A = new_idx_map;
  }
  nst_B = 0;
  while (extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    dstack_map_B[nst_B] = map_B;
    dstack_tid_B[nst_B] = ntid_B;
    nst_B++;
    ntid_B = new_tid;
    map_B = new_idx_map;
  }

  if (ntid_A == ntid_B){
    clone_tensor(ntid_A, 1, &new_tid);
    CTF_sum_type_t new_type = *type;
    new_type.tid_A = new_tid;
    stat = sym_sum_tsr(alpha_, beta, &new_type, ftsr, felm, run_diag);
    del_tsr(new_tid);
    return stat;
  }

  dtype alpha = alpha_*align_symmetric_indices(tensors[ntid_A]->ndim,
                                               (int*)map_A,
                                               tensors[ntid_A]->sym,
                                               tensors[ntid_B]->ndim,
                                               (int*)map_B,
                                               tensors[ntid_B]->sym);


  if (unfold_broken_sym(type, NULL) != -1){
    if (global_comm->rank == 0)
      DPRINTF(1,"Contraction index is broken\n");

    unfold_broken_sym(type, &unfold_type);
    int * sym, dim, sy;
    sy = 0;
    sym = get_sym(ntid_A);
    dim = get_dim(ntid_A);
    for (i=0; i<dim; i++){
      if (sym[i] == SY) sy = 1;
    }
    sym = get_sym(ntid_B);
    dim = get_dim(ntid_B);
    for (i=0; i<dim; i++){
      if (sym[i] == SY) sy = 1;
    }
    if (sy){/* && map_tensors(&unfold_type,
                          ftsr, felm, alpha, beta, &ctrf, 0) == DIST_TENSOR_SUCCESS){*/
      desymmetrize(ntid_A, unfold_type.tid_A, 0);
      desymmetrize(ntid_B, unfold_type.tid_B, 0);
      if (global_comm->rank == 0)
        DPRINTF(1,"Performing index desymmetrization\n");
      sym_sum_tsr(alpha, beta, &unfold_type, ftsr, felm, run_diag);
      symmetrize(ntid_B, unfold_type.tid_B);
      unmap_inner(tensors[unfold_type.tid_A]);
      unmap_inner(tensors[unfold_type.tid_B]);
      dealias(ntid_A, unfold_type.tid_A);
      dealias(ntid_B, unfold_type.tid_B);
      del_tsr(unfold_type.tid_A);
      del_tsr(unfold_type.tid_B);
      CTF_free(unfold_type.idx_map_A);
      CTF_free(unfold_type.idx_map_B);
    } else {
      get_sym_perms(type, alpha, perm_types, signs);
      dbeta = beta;
      for (i=0; i<(int)perm_types.size(); i++){
        sum_tensors(signs[i], dbeta, perm_types[i].tid_A, perm_types[i].tid_B,
                    perm_types[i].idx_map_A, perm_types[i].idx_map_B, ftsr, felm, run_diag);
        free_type(&perm_types[i]);
        dbeta = 1.0;
      }
      perm_types.clear();
      signs.clear();
    }
  } else {
    sum_tensors(alpha, beta, type->tid_A, type->tid_B, type->idx_map_A, 
                type->idx_map_B, ftsr, felm, run_diag);
  }
  if (ntid_A != type->tid_A) del_tsr(ntid_A);
  for (i=nst_B-1; i>=0; i--){
    extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
    del_tsr(ntid_B);
    ntid_B = dstack_tid_B[i];
  }
  LIBT_ASSERT(ntid_B == type->tid_B);

  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] ftsr pointer to sequential block sum function
 * \param[in] felm pointer to sequential element-wise sum function
 * \param[in] run_diag if 1 run diagonal sum
 */
template<typename dtype>
int dist_tensor<dtype>::sum_tensors( dtype const                alpha_,
                                     dtype const                beta,
                                     int const                  tid_A,
                                     int const                  tid_B,
                                     int const *                idx_map_A,
                                     int const *                idx_map_B,
                                     fseq_tsr_sum<dtype> const  ftsr,
                                     fseq_elm_sum<dtype> const  felm,
                                     int const                  run_diag){
  int stat, new_tid, * new_idx_map;
  int * map_A, * map_B, * dstack_tid_B;
  int ** dstack_map_B;
  int ntid_A, ntid_B, nst_B;
  tsum<dtype> * sumf;
  check_sum(tid_A, tid_B, idx_map_A, idx_map_B);
  if (tensors[tid_A]->has_zero_edge_len || tensors[tid_B]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }


  CTF_alloc_ptr(sizeof(int)*tensors[tid_A]->ndim,   (void**)&map_A);
  CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->ndim,   (void**)&map_B);
  CTF_alloc_ptr(sizeof(int*)*tensors[tid_B]->ndim,   (void**)&dstack_map_B);
  CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->ndim,   (void**)&dstack_tid_B);
  memcpy(map_A, idx_map_A, tensors[tid_A]->ndim*sizeof(int));
  memcpy(map_B, idx_map_B, tensors[tid_B]->ndim*sizeof(int));
  ntid_A = tid_A;
  ntid_B = tid_B;
  while (!run_diag && extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    if (ntid_A != tid_A) del_tsr(ntid_A);
    CTF_free(map_A);
    ntid_A = new_tid;
    map_A = new_idx_map;
  }
  nst_B = 0;
  while (!run_diag && extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    dstack_map_B[nst_B] = map_B;
    dstack_tid_B[nst_B] = ntid_B;
    nst_B++;
    ntid_B = new_tid;
    map_B = new_idx_map;
  }
  if (ntid_A == ntid_B){
    clone_tensor(ntid_A, 1, &new_tid);
    stat = sum_tensors(alpha_, beta, new_tid, ntid_B, map_A, map_B, ftsr, felm);
    del_tsr(new_tid);
    LIBT_ASSERT(stat == DIST_TENSOR_SUCCESS);
  } else{ 

    dtype alpha = alpha_*align_symmetric_indices(tensors[ntid_A]->ndim,
                                                 (int*)map_A,
                                                 tensors[ntid_A]->sym,
                                                 tensors[ntid_B]->ndim,
                                                 (int*)map_B,
                                                 tensors[ntid_B]->sym);

    CTF_sum_type_t type = {(int)ntid_A, (int)ntid_B,
                           (int*)map_A, (int*)map_B};
#if DEBUG >= 1
    print_sum(&type,alpha,beta);
#endif

#if VERIFY
    long_int nsA, nsB;
    long_int nA, nB;
    dtype * sA, * sB;
    dtype * uA, * uB;
    int ndim_A, ndim_B,  i;
    int * edge_len_A, * edge_len_B;
    int * sym_A, * sym_B;
    stat = allread_tsr(ntid_A, &nsA, &sA);
    assert(stat == DIST_TENSOR_SUCCESS);

    stat = allread_tsr(ntid_B, &nsB, &sB);
    assert(stat == DIST_TENSOR_SUCCESS);
#endif

    TAU_FSTART(sum_tensors);

    /* Check if the current tensor mappings can be summed on */
#if REDIST
    if (1) {
#else
    if (check_sum_mapping(ntid_A, map_A, ntid_B, map_B) == 0) {
#endif
      /* remap if necessary */
      stat = map_tensor_pair(ntid_A, map_A, ntid_B, map_B);
      if (stat == DIST_TENSOR_ERROR) {
        printf("Failed to map tensors to physical grid\n");
        return DIST_TENSOR_ERROR;
      }
    } else {
#if DEBUG >= 2
      if (get_global_comm()->rank == 0){
        printf("Keeping mappings:\n");
      }
      print_map(stdout, ntid_A);
      print_map(stdout, ntid_B);
#endif
    }
    /* Construct the tensor algorithm we would like to use */
    LIBT_ASSERT(check_sum_mapping(ntid_A, map_A, ntid_B, map_B));
#if FOLD_TSR
    if (felm.func_ptr == NULL && can_fold(&type)){
      int inner_stride;
      TAU_FSTART(map_fold);
      stat = map_fold(&type, &inner_stride);
      TAU_FSTOP(map_fold);
      if (stat == DIST_TENSOR_SUCCESS){
        sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                              ftsr, felm, inner_stride);
      } else
        return DIST_TENSOR_ERROR;
    } else
      sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                           ftsr, felm);
#else
    sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                         ftsr, felm);
#endif
    /*TAU_FSTART(zero_sum_padding);
    stat = zero_out_padding(ntid_A);
    TAU_FSTOP(zero_sum_padding);
    TAU_FSTART(zero_sum_padding);
    stat = zero_out_padding(ntid_B);
    TAU_FSTOP(zero_sum_padding);*/
    DEBUG_PRINTF("[%d] performing tensor sum\n", get_global_comm()->rank);
#if DEBUG >=3
    if (get_global_comm()->rank == 0){
      for (int i=0; i<tensors[ntid_A]->ndim; i++){
        printf("padding[%d] = %d\n",i, tensors[ntid_A]->padding[i]);
      }
      for (int i=0; i<tensors[ntid_B]->ndim; i++){
        printf("padding[%d] = %d\n",i, tensors[ntid_B]->padding[i]);
      }
    }
#endif

    TAU_FSTART(sum_func);
    /* Invoke the contraction algorithm */
    sumf->run();
    TAU_FSTOP(sum_func);
#ifndef SEQ
    stat = zero_out_padding(ntid_B);
#endif

#if VERIFY
    stat = allread_tsr(ntid_A, &nA, &uA);
    assert(stat == DIST_TENSOR_SUCCESS);
    stat = get_tsr_info(ntid_A, &ndim_A, &edge_len_A, &sym_A);
    assert(stat == DIST_TENSOR_SUCCESS);

    stat = allread_tsr(ntid_B, &nB, &uB);
    assert(stat == DIST_TENSOR_SUCCESS);
    stat = get_tsr_info(ntid_B, &ndim_B, &edge_len_B, &sym_B);
    assert(stat == DIST_TENSOR_SUCCESS);

    if (nsA != nA) { printf("nsA = %lld, nA = %lld\n",nsA,nA); ABORT; }
    if (nsB != nB) { printf("nsB = %lld, nB = %lld\n",nsB,nB); ABORT; }
    for (i=0; (ulong_int)i<nA; i++){
      if (fabs(uA[i] - sA[i]) > 1.E-6){
        printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
      }
    }

    cpy_sym_sum(alpha, uA, ndim_A, edge_len_A, edge_len_A, sym_A, map_A,
                beta, sB, ndim_B, edge_len_B, edge_len_B, sym_B, map_B);
    assert(stat == DIST_TENSOR_SUCCESS);

    for (i=0; (ulong_int)i<nB; i++){
      if (fabs(uB[i] - sB[i]) > 1.E-6){
        printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
      }
      assert(fabs(sB[i] - uB[i]) < 1.E-6);
    }
    CTF_free(uA);
    CTF_free(uB);
    CTF_free(sA);
    CTF_free(sB);
#endif

    delete sumf;
    if (ntid_A != tid_A) del_tsr(ntid_A);
    for (int i=nst_B-1; i>=0; i--){
      int ret = extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
      LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
      del_tsr(ntid_B);
      ntid_B = dstack_tid_B[i];
    }
    LIBT_ASSERT(ntid_B == tid_B);
  }
  CTF_free(map_A);
  CTF_free(map_B);
  CTF_free(dstack_map_B);
  CTF_free(dstack_tid_B);

  TAU_FSTOP(sum_tensors);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] ftsr pointer to sequential block contract function
 * \param[in] felm pointer to sequential element-wise contract function
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::
     home_contract(CTF_ctr_type_t const *    stype,
                   fseq_tsr_ctr<dtype> const ftsr,
                   fseq_elm_ctr<dtype> const felm,
                   dtype const               alpha,
                   dtype const               beta,
                   int const                 map_inner){
#ifndef HOME_CONTRACT
  return sym_contract(stype, ftsr, felm, alpha, beta, map_inner);
#else
  int ret, new_tid;
  int was_home_A, was_home_B, was_home_C;
  int was_padded_C, was_cyclic_C;
  long_int old_size_C;
  int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
  int * old_padding_C, * old_edge_len_C;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tensor<dtype> * ntsr_A, * ntsr_B, * ntsr_C;
  tsr_A = tensors[stype->tid_A];
  tsr_B = tensors[stype->tid_B];
  tsr_C = tensors[stype->tid_C];
  unmap_inner(tsr_A);
  unmap_inner(tsr_B);
  unmap_inner(tsr_C);
  
  if (tsr_A->has_zero_edge_len || 
      tsr_B->has_zero_edge_len || 
      tsr_C->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }

  contract_mst();

  if (stype->tid_A == stype->tid_B || stype->tid_A == stype->tid_C){
    clone_tensor(stype->tid_A, 1, &new_tid);
    CTF_ctr_type_t new_type = *stype;
    new_type.tid_A = new_tid;
    ret = home_contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    return ret;
  } else if (stype->tid_B == stype->tid_C){
    clone_tensor(stype->tid_B, 1, &new_tid);
    CTF_ctr_type_t new_type = *stype;
    new_type.tid_B = new_tid;
    ret = home_contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    return ret;
  } 

  CTF_ctr_type_t ntype = *stype;

  was_home_A = tsr_A->is_home;
  was_home_B = tsr_B->is_home;
  was_home_C = tsr_C->is_home;

  if (was_home_A){
    clone_tensor(stype->tid_A, 0, &ntype.tid_A, 0);
    ntsr_A = tensors[ntype.tid_A];
    ntsr_A->data = tsr_A->data;
    ntsr_A->home_buffer = tsr_A->home_buffer;
    ntsr_A->is_home = 1;
    ntsr_A->is_mapped = 1;
    ntsr_A->itopo = tsr_A->itopo;
    copy_mapping(tsr_A->ndim, tsr_A->edge_map, ntsr_A->edge_map);
    set_padding(ntsr_A);
  }     
  if (was_home_B){
    clone_tensor(stype->tid_B, 0, &ntype.tid_B, 0);
    ntsr_B = tensors[ntype.tid_B];
    ntsr_B->data = tsr_B->data;
    ntsr_B->home_buffer = tsr_B->home_buffer;
    ntsr_B->is_home = 1;
    ntsr_B->is_mapped = 1;
    ntsr_B->itopo = tsr_B->itopo;
    copy_mapping(tsr_B->ndim, tsr_B->edge_map, ntsr_B->edge_map);
    set_padding(ntsr_B);
  }
  if (was_home_C){
    clone_tensor(stype->tid_C, 0, &ntype.tid_C, 0);
    ntsr_C = tensors[ntype.tid_C];
    ntsr_C->data = tsr_C->data;
    ntsr_C->home_buffer = tsr_C->home_buffer;
    ntsr_C->is_home = 1;
    ntsr_C->is_mapped = 1;
    ntsr_C->itopo = tsr_C->itopo;
    copy_mapping(tsr_C->ndim, tsr_C->edge_map, ntsr_C->edge_map);
    set_padding(ntsr_C);
  }

  ret = sym_contract(&ntype, ftsr, felm, alpha, beta, map_inner);
  if (ret!= DIST_TENSOR_SUCCESS) return ret;
  if (was_home_A) unmap_inner(ntsr_A);
  if (was_home_B) unmap_inner(ntsr_B);
  if (was_home_C) unmap_inner(ntsr_C);

  if (was_home_C && !ntsr_C->is_home){
    if (global_comm->rank == 0)
      DPRINTF(2,"Migrating tensor %d back to home\n", stype->tid_C);
    save_mapping(ntsr_C,
                 &old_phase_C, &old_rank_C, 
                 &old_virt_dim_C, &old_pe_lda_C, 
                 &old_size_C, &was_padded_C, 
                 &was_cyclic_C, &old_padding_C, 
                 &old_edge_len_C, &topovec[ntsr_C->itopo]);
    tsr_C->data = ntsr_C->data;
    tsr_C->is_home = 0;
    remap_tensor(stype->tid_C, tsr_C, &topovec[tsr_C->itopo], old_size_C, 
                 old_phase_C, old_rank_C, old_virt_dim_C, 
                 old_pe_lda_C, was_padded_C, was_cyclic_C, 
                 old_padding_C, old_edge_len_C, global_comm);
    memcpy(tsr_C->home_buffer, tsr_C->data, tsr_C->size*sizeof(dtype));
    CTF_free(tsr_C->data);
    tsr_C->data = tsr_C->home_buffer;
    tsr_C->is_home = 1;
    ntsr_C->is_data_aliased = 1;
    del_tsr(ntype.tid_C);
    CTF_free(old_phase_C);
    CTF_free(old_rank_C);
    CTF_free(old_virt_dim_C);
    CTF_free(old_pe_lda_C);
    CTF_free(old_padding_C);
    CTF_free(old_edge_len_C);
  } else if (was_home_C) {
/*    tsr_C->itopo = ntsr_C->itopo;
    copy_mapping(tsr_C->ndim, ntsr_C->edge_map, tsr_C->edge_map);
    set_padding(tsr_C);*/
    LIBT_ASSERT(ntsr_C->data == tsr_C->data);
    ntsr_C->is_data_aliased = 1;
    del_tsr(ntype.tid_C);
  }
  if (was_home_A && !ntsr_A->is_home){
    ntsr_A->has_home = 0;
    del_tsr(ntype.tid_A);
  } else if (was_home_A) {
    ntsr_A->is_data_aliased = 1;
    del_tsr(ntype.tid_A);
  }
  if (was_home_B && !ntsr_B->is_home){
    ntsr_B->has_home = 0;
    del_tsr(ntype.tid_B);
  } else if (was_home_B) {
    ntsr_B->is_data_aliased = 1;
    del_tsr(ntype.tid_B);
  }
  return DIST_TENSOR_SUCCESS;
#endif
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] ftsr pointer to sequential block contract function
 * \param[in] felm pointer to sequential element-wise contract function
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::
     sym_contract(CTF_ctr_type_t const *    stype,
                  fseq_tsr_ctr<dtype> const ftsr,
                  fseq_elm_ctr<dtype> const felm,
                  dtype const               alpha,
                  dtype const               beta,
                  int const                 map_inner){
  int i;
  //int ** scl_idx_maps_C;
  //dtype * scl_alpha_C;
  int stat, new_tid;
  int * new_idx_map;
  int * map_A, * map_B, * map_C, * dstack_tid_C;
  int ** dstack_map_C;
  int ntid_A, ntid_B, ntid_C, nst_C;
  CTF_ctr_type_t unfold_type, ntype = *stype;
  CTF_ctr_type_t * type = &ntype;
  std::vector<CTF_ctr_type_t> perm_types;
  std::vector<dtype> signs;
  dtype dbeta;
  ctr<dtype> * ctrf;
  check_contraction(stype);
  if (tensors[type->tid_A]->has_zero_edge_len || tensors[type->tid_B]->has_zero_edge_len
      || tensors[type->tid_C]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  ntid_A = type->tid_A;
  ntid_B = type->tid_B;
  ntid_C = type->tid_C;
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_A]->ndim,   (void**)&map_A);
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->ndim,   (void**)&map_B);
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_C]->ndim,   (void**)&map_C);
  CTF_alloc_ptr(sizeof(int*)*tensors[ntid_C]->ndim,   (void**)&dstack_map_C);
  CTF_alloc_ptr(sizeof(int)*tensors[ntid_C]->ndim,   (void**)&dstack_tid_C);
  memcpy(map_A, type->idx_map_A, tensors[ntid_A]->ndim*sizeof(int));
  memcpy(map_B, type->idx_map_B, tensors[ntid_B]->ndim*sizeof(int));
  memcpy(map_C, type->idx_map_C, tensors[ntid_C]->ndim*sizeof(int));
  while (extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    if (ntid_A != type->tid_A) del_tsr(ntid_A);
    CTF_free(map_A);
    ntid_A = new_tid;
    map_A = new_idx_map;
  }
  while (extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    if (ntid_B != type->tid_B) del_tsr(ntid_B);
    CTF_free(map_B);
    ntid_B = new_tid;
    map_B = new_idx_map;
  }
  nst_C = 0;
  while (extract_diag(ntid_C, map_C, 1, &new_tid, &new_idx_map) == DIST_TENSOR_SUCCESS){
    dstack_map_C[nst_C] = map_C;
    dstack_tid_C[nst_C] = ntid_C;
    nst_C++;
    ntid_C = new_tid;
    map_C = new_idx_map;
  }
  type->tid_A = ntid_A;
  type->tid_B = ntid_B;
  type->tid_C = ntid_C;
  type->idx_map_A = map_A;
  type->idx_map_B = map_B;
  type->idx_map_C = map_C;

  unmap_inner(tensors[ntid_A]);
  unmap_inner(tensors[ntid_B]);
  unmap_inner(tensors[ntid_C]);
  if (ntid_A == ntid_B || ntid_A == ntid_C){
    clone_tensor(ntid_A, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_A = new_tid;
    stat = sym_contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    LIBT_ASSERT(stat == DIST_TENSOR_SUCCESS);
  } else if (ntid_B == ntid_C){
    clone_tensor(ntid_B, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_B = new_tid;
    stat = sym_contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    LIBT_ASSERT(stat == DIST_TENSOR_SUCCESS);
  } else {

    double alignfact = align_symmetric_indices(tensors[ntid_A]->ndim,
                                              map_A,
                                              tensors[ntid_A]->sym,
                                              tensors[ntid_B]->ndim,
                                              map_B,
                                              tensors[ntid_B]->sym,
                                              tensors[ntid_C]->ndim,
                                              map_C,
                                              tensors[ntid_C]->sym);

    /*
     * Apply a factor of n! for each set of n symmetric indices which are contracted over
     */
    double ocfact = overcounting_factor(tensors[ntid_A]->ndim,
                                       map_A,
                                       tensors[ntid_A]->sym,
                                       tensors[ntid_B]->ndim,
                                       map_B,
                                       tensors[ntid_B]->sym,
                                       tensors[ntid_C]->ndim,
                                       map_C,
                                       tensors[ntid_C]->sym);

    //std::cout << alpha << ' ' << alignfact << ' ' << ocfact << std::endl;

    if (unfold_broken_sym(type, NULL) != -1){
      if (global_comm->rank == 0)
        DPRINTF(1,"Contraction index is broken\n");

      unfold_broken_sym(type, &unfold_type);
#if PERFORM_DESYM
      if (map_tensors(&unfold_type, 
                      ftsr, felm, alpha, beta, &ctrf, 0) == DIST_TENSOR_SUCCESS){
#else
      int * sym, dim, sy;
      sy = 0;
      sym = get_sym(ntid_A);
      dim = get_dim(ntid_A);
      for (i=0; i<dim; i++){
        if (sym[i] == SY) sy = 1;
      }
      sym = get_sym(ntid_B);
      dim = get_dim(ntid_B);
      for (i=0; i<dim; i++){
        if (sym[i] == SY) sy = 1;
      }
      sym = get_sym(ntid_C);
      dim = get_dim(ntid_C);
      for (i=0; i<dim; i++){
        if (sym[i] == SY) sy = 1;
      }
      if (sy && map_tensors(&unfold_type,
                            ftsr, felm, alpha, beta, &ctrf, 0) == DIST_TENSOR_SUCCESS){
#endif
        desymmetrize(ntid_A, unfold_type.tid_A, 0);
        desymmetrize(ntid_B, unfold_type.tid_B, 0);
        desymmetrize(ntid_C, unfold_type.tid_C, 1);
        if (global_comm->rank == 0)
          DPRINTF(1,"Performing index desymmetrization\n");
        sym_contract(&unfold_type, ftsr, felm,
                     alpha*alignfact, beta, map_inner);
        symmetrize(ntid_C, unfold_type.tid_C);
        unmap_inner(tensors[unfold_type.tid_A]);
        unmap_inner(tensors[unfold_type.tid_B]);
        unmap_inner(tensors[unfold_type.tid_C]);
        dealias(ntid_A, unfold_type.tid_A);
        dealias(ntid_B, unfold_type.tid_B);
        dealias(ntid_C, unfold_type.tid_C);
        del_tsr(unfold_type.tid_A);
        del_tsr(unfold_type.tid_B);
        del_tsr(unfold_type.tid_C);
        CTF_free(unfold_type.idx_map_A);
        CTF_free(unfold_type.idx_map_B);
        CTF_free(unfold_type.idx_map_C);
      } else {
        get_sym_perms(type, alpha*alignfact*ocfact, 
                      perm_types, signs);
                      //&nscl_C, &scl_maps_C, &scl_alpha_C);
        dbeta = beta;
        for (i=0; i<(int)perm_types.size(); i++){
          contract(&perm_types[i], ftsr, felm,
                    signs[i], dbeta, map_inner);
          free_type(&perm_types[i]);
          dbeta = 1.0;
       }
      perm_types.clear();
      signs.clear();
      }
    } else {
      contract(type, ftsr, felm, alpha*alignfact*ocfact, beta, map_inner);
    }
    if (ntid_A != type->tid_A) del_tsr(ntid_A);
    if (ntid_B != type->tid_B) del_tsr(ntid_B);
    for (i=nst_C-1; i>=0; i--){
      extract_diag(dstack_tid_C[i], dstack_map_C[i], 0, &ntid_C, &new_idx_map);
      del_tsr(ntid_C);
      ntid_C = dstack_tid_C[i];
    }
    LIBT_ASSERT(ntid_C == type->tid_C);
  }

  CTF_free(map_A);
  CTF_free(map_B);
  CTF_free(map_C);
  CTF_free(dstack_map_C);
  CTF_free(dstack_tid_C);

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] ftsr pointer to sequential block contract function
 * \param[in] felm pointer to sequential element-wise contract function
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::
     contract(CTF_ctr_type_t const *      type,
              fseq_tsr_ctr<dtype> const   ftsr,
              fseq_elm_ctr<dtype> const   felm,
              dtype const                 alpha,
              dtype const                 beta,
              int const                   map_inner){
  int stat, new_tid;
  long_int membytes;
  ctr<dtype> * ctrf;

  if (tensors[type->tid_A]->has_zero_edge_len || tensors[type->tid_B]->has_zero_edge_len
      || tensors[type->tid_C]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  if (type->tid_A == type->tid_B || type->tid_A == type->tid_C){
    clone_tensor(type->tid_A, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_A = new_tid;
    stat = contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }
  if (type->tid_B == type->tid_C){
    clone_tensor(type->tid_B, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_B = new_tid;
    stat = contract(&new_type, ftsr, felm, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }
#if DEBUG >= 1
  if (get_global_comm()->rank == 0)
    printf("Contraction permutation:\n");
  print_ctr(type, alpha, beta);
#endif

  TAU_FSTART(contract);
#if VERIFY
  long_int nsA, nsB;
  long_int nA, nB, nC, up_nC;
  dtype * sA, * sB, * ans_C;
  dtype * uA, * uB, * uC;
  dtype * up_C, * up_ans_C, * pup_C;
  int ndim_A, ndim_B, ndim_C, i, pass;
  int * edge_len_A, * edge_len_B, * edge_len_C;
  int * sym_A, * sym_B, * sym_C;
  int * sym_tmp;
  stat = allread_tsr(type->tid_A, &nsA, &sA);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = allread_tsr(type->tid_B, &nsB, &sB);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = allread_tsr(type->tid_C, &nC, &ans_C);
  assert(stat == DIST_TENSOR_SUCCESS);
#endif
  /* Check if the current tensor mappings can be contracted on */
#if REDIST
  stat = map_tensors(type, ftsr, felm, alpha, beta, &ctrf);
  if (stat == DIST_TENSOR_ERROR) {
    printf("Failed to map tensors to physical grid\n");
    return DIST_TENSOR_ERROR;
  }
#else
  if (check_contraction_mapping(type) == 0) {
    /* remap if necessary */
    stat = map_tensors(type, ftsr, felm, alpha, beta, &ctrf);
    if (stat == DIST_TENSOR_ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return DIST_TENSOR_ERROR;
    }
  } else {
    /* Construct the tensor algorithm we would like to use */
#if DEBUG >= 2
    if (get_global_comm()->rank == 0)
      printf("Keeping mappings:\n");
    print_map(stdout, type->tid_A);
    print_map(stdout, type->tid_B);
    print_map(stdout, type->tid_C);
#endif
    ctrf = construct_contraction(type, ftsr, felm, alpha, beta);
  }
#endif
  LIBT_ASSERT(check_contraction_mapping(type));
#if FOLD_TSR
  if (felm.func_ptr == NULL && map_inner && can_fold(type)){
    iparam prm;
    TAU_FSTART(map_fold);
    stat = map_fold(type, &prm);
    TAU_FSTOP(map_fold);
    if (stat == DIST_TENSOR_ERROR){
      return DIST_TENSOR_ERROR;
    }
    if (stat == DIST_TENSOR_SUCCESS){
      delete ctrf;
      ctrf = construct_contraction(type, ftsr, felm, alpha, beta, 2, &prm);
    }
  }
#endif
#if INNER_MAP
  if (map_inner){
    iparam prm;
    TAU_FSTART(map_inner);
    stat = map_inner(type, &prm);
    TAU_FSTOP(map_inner);
    if (stat == DIST_TENSOR_ERROR){
      return DIST_TENSOR_ERROR;
    }
    if (stat == DIST_TENSOR_SUCCESS){
      delete ctrf;
      ctrf = construct_contraction(type, ftsr, felm, alpha, beta, 1, &prm);
    }
  }
#endif
#if DEBUG >=2
  if (get_global_comm()->rank == 0)
    ctrf->print();
#endif
  membytes = ctrf->mem_rec();

  if (get_global_comm()->rank == 0){
    DPRINTF(1,"[%d] performing contraction\n",
        get_global_comm()->rank);
    DPRINTF(1,"%E bytes of buffer space will be needed for this contraction\n",
      (double)membytes);
    DPRINTF(1,"System memory = %E bytes total, %E bytes used, %E bytes available.\n",
      (double)proc_bytes_total(),
      (double)proc_bytes_used(),
      (double)proc_bytes_available());
  }
/*  print_map(stdout, type->tid_A);
  print_map(stdout, type->tid_B);
  print_map(stdout, type->tid_C);*/
//  stat = zero_out_padding(type->tid_A);
//  stat = zero_out_padding(type->tid_B);
  TAU_FSTART(ctr_func);
  /* Invoke the contraction algorithm */
  ctrf->run();
  TAU_FSTOP(ctr_func);
#ifndef SEQ
  if (tensors[type->tid_C]->is_cyclic)
    stat = zero_out_padding(type->tid_C);
#endif
  if (get_global_comm()->rank == 0){
    DPRINTF(1, "Contraction completed.\n");
  }


#if VERIFY
  stat = allread_tsr(type->tid_A, &nA, &uA);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = get_tsr_info(type->tid_A, &ndim_A, &edge_len_A, &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = allread_tsr(type->tid_B, &nB, &uB);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = get_tsr_info(type->tid_B, &ndim_B, &edge_len_B, &sym_B);
  assert(stat == DIST_TENSOR_SUCCESS);

  if (nsA != nA) { printf("nsA = %lld, nA = %lld\n",nsA,nA); ABORT; }
  if (nsB != nB) { printf("nsB = %lld, nB = %lld\n",nsB,nB); ABORT; }
  for (i=0; (ulong_int)i<nA; i++){
    if (fabs(uA[i] - sA[i]) > 1.E-6){
      printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
    }
  }
  for (i=0; (ulong_int)i<nB; i++){
    if (fabs(uB[i] - sB[i]) > 1.E-6){
      printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
    }
  }

  stat = allread_tsr(type->tid_C, &nC, &uC);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = get_tsr_info(type->tid_C, &ndim_C, &edge_len_C, &sym_C);
  assert(stat == DIST_TENSOR_SUCCESS);
  DEBUG_PRINTF("packed size of C is %lld (should be %lld)\n", nC,
    sy_packed_size(ndim_C, edge_len_C, sym_C));

  pup_C = (dtype*)CTF_alloc(nC*sizeof(dtype));

  cpy_sym_ctr(alpha,
        uA, ndim_A, edge_len_A, edge_len_A, sym_A, type->idx_map_A,
        uB, ndim_B, edge_len_B, edge_len_B, sym_B, type->idx_map_B,
        beta,
    ans_C, ndim_C, edge_len_C, edge_len_C, sym_C, type->idx_map_C);
  assert(stat == DIST_TENSOR_SUCCESS);

#if ( DEBUG>=5)
  for (i=0; i<nC; i++){
//    if (fabs(C[i]-ans_C[i]) > 1.E-6){
      printf("PACKED: C[%d] = %lf, ans_C[%d] = %lf\n",
       i, C[i], i, ans_C[i]);
//     }
  }
#endif

  punpack_tsr(uC, ndim_C, edge_len_C,
        sym_C, 1, &sym_tmp, &up_C);
  punpack_tsr(ans_C, ndim_C, edge_len_C,
        sym_C, 1, &sym_tmp, &up_ans_C);
  punpack_tsr(up_ans_C, ndim_C, edge_len_C,
        sym_C, 0, &sym_tmp, &pup_C);
  for (i=0; (ulong_int)i<nC; i++){
    assert(fabs(pup_C[i] - ans_C[i]) < 1.E-6);
  }
  pass = 1;
  up_nC = 1;
  for (i=0; i<ndim_C; i++){ up_nC *= edge_len_C[i]; };

  for (i=0; i<(int)up_nC; i++){
    if (fabs((up_C[i]-up_ans_C[i])/up_ans_C[i]) > 1.E-6 &&
  fabs((up_C[i]-up_ans_C[i])) > 1.E-6){
      printf("C[%d] = %lf, ans_C[%d] = %lf\n",
       i, up_C[i], i, up_ans_C[i]);
      pass = 0;
    }
  }
  if (!pass) ABORT;

#endif

  delete ctrf;

  TAU_FSTOP(contract);
  return DIST_TENSOR_SUCCESS;


}

