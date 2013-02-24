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
  long_int i, j;
  double acc;
  double * aux_arr;
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
      get_buffer_space(tsr->size*sizeof(double), (void**)&aux_arr);
      if (tsr->is_mapped){
        if (idx_lyr == 0){
          for (i=0; i<tsr->size; i++){
            aux_arr[i] = tsr->data[i]*tsr->data[i];
          }
        } else
          std::fill(aux_arr,aux_arr+tsr->size,0.0);
            } else {
        for (i=0; i<tsr->size; i++){
          aux_arr[i] = tsr->pairs[i].d*tsr->pairs[i].d;
        }
            }
            for (i=1; i<tsr->size; i*=2){
        for (j=0; j<tsr->size; j+= i*2){
          aux_arr[j] = aux_arr[j] + aux_arr[j+i];
        }
      }
      ALLREDUCE(aux_arr, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);
      free_buffer_space(aux_arr);
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

  get_buffer_space(tsr->ndim*sizeof(int), (void**)&idx);

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
    free_buffer_space(prs);
    if (stat != DIST_TENSOR_SUCCESS) return stat;
  }
  free_buffer_space(idx);
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
 * \param[in] func_ptr pointer to sequential scale function
 */
template<typename dtype>
int dist_tensor<dtype>::
     scale_tsr(dtype const                alpha,
               int const                  tid,
               int const *                idx_map,
               fseq_tsr_scl<dtype> const  func_ptr){
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

  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&blk_len);
  get_buffer_space(sizeof(int)*tsr->ndim, (void**)&virt_blk_len);
  get_buffer_space(sizeof(int)*ndim_tot, (void**)&virt_dim);

  if (!check_self_mapping(tid, idx_map)){
    save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda,
                 &old_size, &was_padded, &was_cyclic, &old_padding, &old_edge_len, &topovec[tsr->itopo]);
    tsr->need_remap = 0;
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
    free(old_phase);
    free(old_rank);
    free(old_virt_dim);
    free(old_pe_lda);
    if (was_padded)
      free(old_padding);
    free(old_edge_len);
#if DEBUG >=2
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
  sclseq->alpha   = alpha;
  sclseq->ndim    = tsr->ndim;
  sclseq->idx_map = idx_map;
  sclseq->edge_len  = virt_blk_len;
  sclseq->sym   = tsr->sym;
  sclseq->func_ptr  = func_ptr;

  hscl->A   = tsr->data;
  hscl->alpha   = alpha;

  free_buffer_space(idx_arr);
  free_buffer_space(blk_len);

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
 * \param[in] func_ptr sequential funciton pointer
 * \return tsum summation class to run
*/
template<typename dtype>
tsum<dtype> * dist_tensor<dtype>::
    construct_sum(dtype const     alpha,
      dtype const   beta,
      int const     tid_A,
      int const *   idx_A,
      int const     tid_B,
      int const *   idx_B,
      fseq_tsr_sum<dtype> const func_ptr,
      int const   inner_stride){
  int nvirt, i, iA, iB, ndim_tot, is_top, sA, sB, need_rep, i_A, i_B, j;
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

  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&blk_len_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&blk_len_B);
  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&virt_blk_len_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&virt_blk_len_B);
  get_buffer_space(sizeof(int)*ndim_tot, (void**)&virt_dim);
  get_buffer_space(sizeof(int)*nphys_dim*2, (void**)&phys_mapped);
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
      get_buffer_space(sizeof(CommData_t*)*rtsum->ncdt_A, (void**)&rtsum->cdt_A);
    if (rtsum->ncdt_B > 0)
      get_buffer_space(sizeof(CommData_t*)*rtsum->ncdt_B, (void**)&rtsum->cdt_B);
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
  } else free(virt_dim);

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
              virt_blk_len_A[j] = 1;
              j--;
            } while (j>=0 && tsr_A->sym[j] != NS);
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
              virt_blk_len_B[j] = 1;
              j--;
            } while (j>=0 && tsr_B->sym[j] != NS);
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
  tsumseq->ndim_A   = tsr_A->ndim;
  tsumseq->idx_map_A  = idx_A;
  tsumseq->edge_len_A = virt_blk_len_A;
  tsumseq->sym_A  = tsr_A->sym;
  tsumseq->ndim_B = tsr_B->ndim;
  tsumseq->idx_map_B  = idx_B;
  tsumseq->edge_len_B = virt_blk_len_B;
  tsumseq->sym_B  = tsr_B->sym;
  tsumseq->func_ptr = func_ptr;

  htsum->A  = tsr_A->data;
  htsum->B  = tsr_B->data;
  htsum->alpha  = alpha;
  htsum->beta   = beta;

  free_buffer_space(idx_arr);
  free_buffer_space(blk_len_A);
  free_buffer_space(blk_len_B);
  free_buffer_space(phys_mapped);

  return htsum;
}


/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
 *  seq_func needed to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer
 * \param[in] func_ptr sequential ctr func pointer
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
    construct_contraction(CTF_ctr_type_t const * type,
        dtype *     buffer,
        int const   buffer_len,
        fseq_tsr_ctr<dtype> func_ptr,
        dtype const   alpha,
        dtype const   beta,
        int const   is_inner,
        iparam const *    inner_params,
        int *     nvirt_all){
  int num_tot, i, i_A, i_B, i_C, is_top, j, nphys_dim, nstep;
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

  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&virt_blk_len_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&virt_blk_len_B);
  get_buffer_space(sizeof(int)*tsr_C->ndim, (void**)&virt_blk_len_C);

  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&blk_len_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&blk_len_B);
  get_buffer_space(sizeof(int)*tsr_C->ndim, (void**)&blk_len_C);
  get_buffer_space(sizeof(int)*num_tot, (void**)&virt_dim);
  get_buffer_space(sizeof(int)*nphys_dim*3, (void**)&phys_mapped);
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
        printf("ERROR: ALL-TENSOR REPLICATION NO LONGER DONE\n");
        ABORT;
        LIBT_ASSERT(rctr->num_lyr == 1);
        hctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i]->rank;
        hctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;
        rctr->idx_lyr = topovec[tsr_A->itopo].dim_comm[i]->rank;
        rctr->num_lyr = topovec[tsr_A->itopo].dim_comm[i]->np;
      }
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
    if (rctr->ncdt_A > 0)
      get_buffer_space(sizeof(CommData_t*)*rctr->ncdt_A, (void**)&rctr->cdt_A);
    if (rctr->ncdt_B > 0)
      get_buffer_space(sizeof(CommData_t*)*rctr->ncdt_B, (void**)&rctr->cdt_B);
    if (rctr->ncdt_C > 0)
      get_buffer_space(sizeof(CommData_t*)*rctr->ncdt_C, (void**)&rctr->cdt_C);
    rctr->ncdt_A = 0;
    rctr->ncdt_B = 0;
    rctr->ncdt_C = 0;
    for (i=0; i<nphys_dim; i++){
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
    free(virt_dim);

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
    ctrseq->func_ptr  = func_ptr;
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
    free(iphase);
    itsr = tensors[tsr_B->rec_tid];
    iphase = calc_phase<dtype>(itsr);
    for (i=0; i<tsr_B->ndim; i++){
      if (virt_blk_len_B[i]%iphase[i] > 0)
        virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i]+1;
      else
        virt_blk_len_B[i] = virt_blk_len_B[i]/iphase[i];
    }
    free(iphase);
    itsr = tensors[tsr_C->rec_tid];
    iphase = calc_phase<dtype>(itsr);
    for (i=0; i<tsr_C->ndim; i++){
      if (virt_blk_len_C[i]%iphase[i] > 0)
        virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i]+1;
      else
        virt_blk_len_C[i] = virt_blk_len_C[i]/iphase[i];
    }
    free(iphase);
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
    i_A = 0;
    for (i=0; i<tsr_A->ndim; i++){
      if (tsr_A->sym[i] == NS){
        for (j=0; j<itsr->ndim; j++){
          if (tsr_A->inner_ordering[j] == i_A){
            j=i;
            do {
              virt_blk_len_A[j] = 1;
              j--;
            } while (j>=0 && tsr_A->sym[j] != NS);
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
              virt_blk_len_B[j] = 1;
              j--;
            } while (j>=0 && tsr_B->sym[j] != NS);
            break;
          }
        }
        i_B++;
      }
    }
    itsr = tensors[tsr_C->rec_tid];
    i_C = 0;
    for (i=0; i<tsr_C->ndim; i++){
      if (tsr_C->sym[i] == NS){
        for (j=0; j<itsr->ndim; j++){
          if (tsr_C->inner_ordering[j] == i_C){
            j=i;
            do {
              virt_blk_len_C[j] = 1;
              j--;
            } while (j>=0 && tsr_C->sym[j] != NS);
            break;
          }
        }
        i_C++;
      }
    }
  }
  ctrseq->alpha   = alpha;
  ctrseq->ndim_A  = tsr_A->ndim;
  ctrseq->idx_map_A = type->idx_map_A;
  ctrseq->edge_len_A  = virt_blk_len_A;
  ctrseq->sym_A   = tsr_A->sym;
  ctrseq->ndim_B  = tsr_B->ndim;
  ctrseq->idx_map_B = type->idx_map_B;
  ctrseq->edge_len_B  = virt_blk_len_B;
  ctrseq->sym_B   = tsr_B->sym;
  ctrseq->ndim_C  = tsr_C->ndim;
  ctrseq->idx_map_C = type->idx_map_C;
  ctrseq->edge_len_C  = virt_blk_len_C;
  ctrseq->sym_C   = tsr_C->sym;

  hctr->A   = tsr_A->data;
  hctr->B   = tsr_B->data;
  hctr->C   = tsr_C->data;
  hctr->beta  = beta;
/*  if (global_comm->rank == 0){
    int64_t n,m,k;
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

  free_buffer_space(idx_arr);
  free_buffer_space(blk_len_A);
  free_buffer_space(blk_len_B);
  free_buffer_space(blk_len_C);
  free_buffer_space(phys_mapped);
  TAU_FSTOP(construct_contraction);
  return hctr;
}

/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] func_ptr sequential ctr func pointer
 */
template<typename dtype>
int dist_tensor<dtype>::sum_tensors( dtype const    alpha_,
                                     dtype const    beta,
                                     int const      tid_A,
                                     int const      tid_B,
                                     int const *    idx_map_A,
                                     int const *    idx_map_B,
                                     fseq_tsr_sum<dtype> const  func_ptr){
  int stat, new_tid;
  tsum<dtype> * sumf;
  if (tensors[tid_A]->has_zero_edge_len || tensors[tid_B]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  if (tid_A == tid_B){
    clone_tensor(tid_A, 1, &new_tid);
    stat = sum_tensors(alpha_, beta, new_tid, tid_B, idx_map_A, idx_map_B, func_ptr);
    del_tsr(new_tid);
    return stat;
  }

  dtype alpha = alpha_*align_symmetric_indices(tensors[tid_A]->ndim,
                                               (int*)idx_map_A,
                                               tensors[tid_A]->sym,
                                               tensors[tid_B]->ndim,
                                               (int*)idx_map_B,
                                               tensors[tid_B]->sym);

  CTF_sum_type_t type = {(int)tid_A, (int)tid_B,
                         (int*)idx_map_A, (int*)idx_map_B};
#if DEBUG >= 1
  print_sum(&type);
#endif

#if VERIFY
  long_int nsA, nsB;
  long_int nA, nB;
  dtype * sA, * sB;
  dtype * uA, * uB;
  int ndim_A, ndim_B,  i;
  int * edge_len_A, * edge_len_B;
  int * sym_A, * sym_B;
  stat = allread_tsr(tid_A, &nsA, &sA);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = allread_tsr(tid_B, &nsB, &sB);
  assert(stat == DIST_TENSOR_SUCCESS);
#endif

  TAU_FSTART(sum_tensors);

  /* Check if the current tensor mappings can be summed on */
#if REDIST
  if (1) {
#else
  if (check_sum_mapping(tid_A, idx_map_A, tid_B, idx_map_B) == 0) {
#endif
    /* remap if necessary */
    stat = map_tensor_pair(tid_A, idx_map_A, tid_B, idx_map_B);
    if (stat == DIST_TENSOR_ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return DIST_TENSOR_ERROR;
    }
  } else {
/*#if DEBUG >= 2
    if (get_global_comm()->rank == 0){
      printf("Keeping mappings:\n");
    }
    print_map(stdout, tid_A);
    print_map(stdout, tid_B);
#endif*/
  }
  /* Construct the tensor algorithm we would like to use */
  LIBT_ASSERT(check_sum_mapping(tid_A, idx_map_A, tid_B, idx_map_B));
#if FOLD_TSR
  if (can_fold(&type)){
    int inner_stride;
    TAU_FSTART(map_fold);
    stat = map_fold(&type, &inner_stride);
    TAU_FSTOP(map_fold);
    if (stat == DIST_TENSOR_ERROR){
      return DIST_TENSOR_ERROR;
    }
    if (stat == DIST_TENSOR_SUCCESS){
      sumf = construct_sum(alpha, beta, tid_A, idx_map_A, tid_B, idx_map_B,
                            func_ptr, inner_stride);
    }
  } else
    sumf = construct_sum(alpha, beta, tid_A, idx_map_A, tid_B, idx_map_B,
                          func_ptr);
#else
  sumf = construct_sum(alpha, beta, tid_A, idx_map_A, tid_B, idx_map_B,
                        func_ptr);
#endif
  /*TAU_FSTART(zero_sum_padding);
  stat = zero_out_padding(tid_A);
  TAU_FSTOP(zero_sum_padding);
  TAU_FSTART(zero_sum_padding);
  stat = zero_out_padding(tid_B);
  TAU_FSTOP(zero_sum_padding);*/
  DEBUG_PRINTF("[%d] performing tensor sum\n", get_global_comm()->rank);

  TAU_FSTART(sum_func);
  /* Invoke the contraction algorithm */
  sumf->run();
  TAU_FSTOP(sum_func);
#ifndef SEQ
  tensors[tid_B]->need_remap = 1;
#endif

#if VERIFY
  TAU_FSTART(zero_sum_padding);
  stat = zero_out_padding(tid_B);
  TAU_FSTOP(zero_sum_padding);
  stat = allread_tsr(tid_A, &nA, &uA);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = get_tsr_info(tid_A, &ndim_A, &edge_len_A, &sym_A);
  assert(stat == DIST_TENSOR_SUCCESS);

  stat = allread_tsr(tid_B, &nB, &uB);
  assert(stat == DIST_TENSOR_SUCCESS);
  stat = get_tsr_info(tid_B, &ndim_B, &edge_len_B, &sym_B);
  assert(stat == DIST_TENSOR_SUCCESS);

  if (nsA != nA) { printf("nsA = %lld, nA = %lld\n",nsA,nA); ABORT; }
  if (nsB != nB) { printf("nsB = %lld, nB = %lld\n",nsB,nB); ABORT; }
  for (i=0; (uint64_t)i<nA; i++){
    if (fabs(uA[i] - sA[i]) > 1.E-6){
      printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
    }
  }

  cpy_sym_sum(alpha, uA, ndim_A, edge_len_A, edge_len_A, sym_A, idx_map_A,
        beta, sB, ndim_B, edge_len_B, edge_len_B, sym_B, idx_map_B);
  assert(stat == DIST_TENSOR_SUCCESS);

  for (i=0; (uint64_t)i<nB; i++){
    if (fabs(uB[i] - sB[i]) > 1.E-6){
      printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
    }
#ifdef LIBT_ASSERT
    LIBT_ASSERT(fabs(sB[i] - uB[i]) < 1.E-6);
#else
    assert(fabs(sB[i] - uB[i]) < 1.E-6);
#endif
  }
  free(uA);
  free(uB);
  free(sA);
  free(sB);
#endif

  delete sumf;

  TAU_FSTOP(sum_tensors);
  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer
 * \param[in] func_ptr sequential ctr func pointer
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::
     sym_contract(CTF_ctr_type_t const *    type,
                  dtype *                   buffer,
                  int const                 buffer_len,
                  fseq_tsr_ctr<dtype> const func_ptr,
                  dtype const               alpha,
                  dtype const               beta,
                  int const                 map_inner){
  int i;
  //int ** scl_idx_maps_C;
  //dtype * scl_alpha_C;
  int stat, new_tid;
  CTF_ctr_type_t unfold_type;
  std::vector<CTF_ctr_type_t> perm_types;
  std::vector<dtype> signs;
  dtype dbeta;
  ctr<dtype> * ctrf;
  if (tensors[type->tid_A]->has_zero_edge_len || tensors[type->tid_B]->has_zero_edge_len
      || tensors[type->tid_C]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }

  unmap_inner(tensors[type->tid_A]);
  unmap_inner(tensors[type->tid_B]);
  unmap_inner(tensors[type->tid_C]);
  if (type->tid_A == type->tid_B || type->tid_A == type->tid_C){
    clone_tensor(type->tid_A, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_A = new_tid;
    stat = sym_contract(&new_type, buffer, buffer_len, func_ptr, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }
  if (type->tid_B == type->tid_C){
    clone_tensor(type->tid_B, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_B = new_tid;
    stat = sym_contract(&new_type, buffer, buffer_len, func_ptr, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }

  double alignfact = align_symmetric_indices(tensors[type->tid_A]->ndim,
                                            type->idx_map_A,
                                            tensors[type->tid_A]->sym,
                                            tensors[type->tid_B]->ndim,
                                            type->idx_map_B,
                                            tensors[type->tid_B]->sym,
                                            tensors[type->tid_C]->ndim,
                                            type->idx_map_C,
                                            tensors[type->tid_C]->sym);

  /*
   * Apply a factor of n! for each set of n symmetric indices which are contracted over
   */
  double ocfact = overcounting_factor(tensors[type->tid_A]->ndim,
                                     type->idx_map_A,
                                     tensors[type->tid_A]->sym,
                                     tensors[type->tid_B]->ndim,
                                     type->idx_map_B,
                                     tensors[type->tid_B]->sym,
                                     tensors[type->tid_C]->ndim,
                                     type->idx_map_C,
                                     tensors[type->tid_C]->sym);

  //std::cout << alpha << ' ' << alignfact << ' ' << ocfact << std::endl;

  if (unfold_broken_sym(type, NULL) != -1){
    if (global_comm->rank == 0)
      DPRINTF(1,"Contraction index is broken\n");

    unfold_broken_sym(type, &unfold_type);
#if PERFORM_DESYM
    if (map_tensors(&unfold_type, buffer, buffer_len,
        func_ptr, alpha, beta, &ctrf, 0) == DIST_TENSOR_SUCCESS){
#else
    int * sym, dim, sy;
    sy = 0;
    sym = get_sym(type->tid_A);
    dim = get_dim(type->tid_A);
    for (i=0; i<dim; i++){
      if (sym[i] == SY) sy = 1;
    }
    sym = get_sym(type->tid_B);
    dim = get_dim(type->tid_B);
    for (i=0; i<dim; i++){
      if (sym[i] == SY) sy = 1;
    }
    sym = get_sym(type->tid_C);
    dim = get_dim(type->tid_C);
    for (i=0; i<dim; i++){
      if (sym[i] == SY) sy = 1;
    }
    if (sy && map_tensors(&unfold_type, buffer, buffer_len,
        func_ptr, alpha, beta, &ctrf, 0) == DIST_TENSOR_SUCCESS){
#endif
      desymmetrize(type->tid_A, unfold_type.tid_A, 0);
      desymmetrize(type->tid_B, unfold_type.tid_B, 0);
      desymmetrize(type->tid_C, unfold_type.tid_C, 1);
      if (global_comm->rank == 0)
        DPRINTF(1,"Performing index desymmetrization\n");
      sym_contract(&unfold_type, buffer, buffer_len, func_ptr, 
                   alpha*alignfact, beta, map_inner);
      symmetrize(type->tid_C, unfold_type.tid_C);
      unmap_inner(tensors[unfold_type.tid_A]);
      unmap_inner(tensors[unfold_type.tid_B]);
      unmap_inner(tensors[unfold_type.tid_C]);
      dealias(type->tid_A, unfold_type.tid_A);
      dealias(type->tid_B, unfold_type.tid_B);
      dealias(type->tid_C, unfold_type.tid_C);
      del_tsr(unfold_type.tid_A);
      del_tsr(unfold_type.tid_B);
      del_tsr(unfold_type.tid_C);
      free(unfold_type.idx_map_A);
      free(unfold_type.idx_map_B);
      free(unfold_type.idx_map_C);
    } else {
      get_sym_perms(type, alpha*alignfact*ocfact, 
                    perm_types, signs);
                    //&nscl_C, &scl_idx_maps_C, &scl_alpha_C);
      dbeta = beta;
      for (i=0; i<(int)perm_types.size(); i++){
        contract(&perm_types[i], buffer, buffer_len, func_ptr,
                  signs[i], dbeta, map_inner);
        free_type(&perm_types[i]);
        dbeta = 1.0;
      }
      perm_types.clear();
      signs.clear();
    }
  } else {
    contract(type, buffer, buffer_len, func_ptr, alpha*alignfact*ocfact, beta, map_inner);
  }

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C.
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer
 * \param[in] func_ptr sequential ctr func pointer
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int dist_tensor<dtype>::
     contract(CTF_ctr_type_t const *  type,
              dtype *       buffer,
              int const     buffer_len,
              fseq_tsr_ctr<dtype> const func_ptr,
              dtype const     alpha,
              dtype const     beta,
              int const     map_inner){
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
    stat = contract(&new_type, buffer, buffer_len, func_ptr, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }
  if (type->tid_B == type->tid_C){
    clone_tensor(type->tid_B, 1, &new_tid);
    CTF_ctr_type_t new_type = *type;
    new_type.tid_B = new_tid;
    stat = contract(&new_type, buffer, buffer_len, func_ptr, alpha, beta, map_inner);
    del_tsr(new_tid);
    return stat;
  }
#if DEBUG >= 1
  if (get_global_comm()->rank == 0)
    printf("Contraction permutation:\n");
  print_ctr(type);
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
  stat = map_tensors(type, buffer, buffer_len, func_ptr, alpha, beta, &ctrf);
  if (stat == DIST_TENSOR_ERROR) {
    printf("Failed to map tensors to physical grid\n");
    return DIST_TENSOR_ERROR;
  }
#else
  if (check_contraction_mapping(type) == 0) {
    /* remap if necessary */
    stat = map_tensors(type, buffer, buffer_len, func_ptr, alpha, beta, &ctrf);
    if (stat == DIST_TENSOR_ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return DIST_TENSOR_ERROR;
    }
  } else {
    /* Construct the tensor algorithm we would like to use */
#if DEBUG >= 2
/*    if (get_global_comm()->rank == 0)
      printf("Keeping mappings:\n");
    print_map(stdout, type->tid_A);
    print_map(stdout, type->tid_B);
    print_map(stdout, type->tid_C);*/
#endif
    ctrf = construct_contraction(type, buffer, buffer_len,
                                 func_ptr, alpha, beta);
  }
#endif
  LIBT_ASSERT(check_contraction_mapping(type));
#if FOLD_TSR
  if (map_inner && can_fold(type)){
    iparam prm;
    TAU_FSTART(map_fold);
    stat = map_fold(type, &prm);
    TAU_FSTOP(map_fold);
    if (stat == DIST_TENSOR_ERROR){
      return DIST_TENSOR_ERROR;
    }
    if (stat == DIST_TENSOR_SUCCESS){
      delete ctrf;
      ctrf = construct_contraction(type, buffer, buffer_len,
               func_ptr, alpha, beta, 2, &prm);
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
      ctrf = construct_contraction(type, buffer, buffer_len,
               func_ptr, alpha, beta, 1, &prm);
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
  TAU_FSTART(ctr_func);
  /* Invoke the contraction algorithm */
  ctrf->run();
  TAU_FSTOP(ctr_func);
#ifndef SEQ
  tensors[type->tid_C]->need_remap = 1;
#endif
  if (get_global_comm()->rank == 0){
    DPRINTF(1, "Contraction completed.\n");
  }


#if VERIFY
  TAU_FSTART(zero_ctr_padding);
  stat = zero_out_padding(type->tid_C);
  TAU_FSTOP(zero_ctr_padding);
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
  for (i=0; (uint64_t)i<nA; i++){
    if (fabs(uA[i] - sA[i]) > 1.E-6){
      printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
    }
  }
  for (i=0; (uint64_t)i<nB; i++){
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

  pup_C = (dtype*)malloc(nC*sizeof(dtype));

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
  for (i=0; (uint64_t)i<nC; i++){
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

