/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "sym_indices.hxx"

/**
 * \brief Scale each tensor element by alpha
 * \param[in] alpha scaling factor
 * \param[in] tid handle to tensor
 */
template<typename dtype>
int dist_tensor<dtype>::scale_tsr(dtype const alpha, int const tid){
  if (global_comm.rank == 0)
    printf("FAILURE: scale_tsr currently only supported for tensors of type double\n");
  return CTF_ERROR;
}

template<> 
int dist_tensor<double>::scale_tsr(double const alpha, int const tid){
  int i;
  tensor<double> * tsr;

  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return CTF_SUCCESS;
  }

  if (tsr->is_mapped){
    cdscal(tsr->size, alpha, tsr->data, 1);
  } else {
    for (i=0; i<tsr->size; i++){
      tsr->pairs[i].d = tsr->pairs[i].d*alpha;
    }
  }

  return CTF_SUCCESS;
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
  if (global_comm.rank == 0)
    printf("FAILURE: dot_loc_tsr currently only supported for tensors of type double\n");
  return CTF_ERROR;
}

template<> 
int dist_tensor<double>::dot_loc_tsr(int const tid_A, int const tid_B, double *product){
  double dprod;
  tensor<double> * tsr_A, * tsr_B;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  if (tsr_A->has_zero_edge_len || tsr_B->has_zero_edge_len){
    *product = 0.0;
    return CTF_SUCCESS;
  }

  ASSERT(tsr_A->is_mapped && tsr_B->is_mapped);
  ASSERT(tsr_A->size == tsr_B->size);

  dprod = cddot(tsr_A->size, tsr_A->data, 1, tsr_B->data, 1);

  /* FIXME: Wont work for single precision */
  ALLREDUCE(&dprod, product, 1, COMM_DOUBLE_T, COMM_OP_SUM, global_comm);

  return CTF_SUCCESS;
}

/* Perform an elementwise reduction on a tensor. All processors
   end up with the final answer. */
template<typename dtype>
int dist_tensor<dtype>::red_tsr(int const tid, CTF_OP op, dtype * result){
  if (global_comm.rank == 0)
    printf("FAILURE: reductions currently only supported for tensors of type double\n");
  return CTF_ERROR;
}

void sum_abs(double const alpha, double const a, double & b){
  b += alpha*fabs(a);
}

/* Perform an elementwise reduction on a tensor. All processors
   end up with the final answer. */
template<> 
int dist_tensor<double>::red_tsr(int const tid, CTF_OP op, double * result){
  int64_t i;
  double acc;
  tensor<double> * tsr;
  mapping * map;
  int is_AS;
  int idx_lyr = 0;
  int tid_scal, is_asym;
  int * idx_map;
  fseq_tsr_sum<double> fs;
  fseq_elm_sum<double> felm;
  fseq_tsr_ctr<double> fcs;
  fseq_elm_ctr<double> fcelm;


  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    *result = 0.0;
    return CTF_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);

  if (tsr->is_mapped){
    idx_lyr = global_comm.rank;
    for (i=0; i<tsr->order; i++){
      map = &tsr->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        idx_lyr -= topovec[tsr->itopo].dim_comm[map->cdt].rank
        *topovec[tsr->itopo].lda[map->cdt];
      }
      while (map->has_child){
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          idx_lyr -= topovec[tsr->itopo].dim_comm[map->cdt].rank
               *topovec[tsr->itopo].lda[map->cdt];
        }
      }
    }
  }
  is_asym = 0;
  for (i=0; i<tsr->order; i++){
    if (tsr->sym[i] == AS)
      is_asym = 1;
  }

  switch (op){
    case CTF_OP_SUM:
      if (is_asym) {
        *result = 0.0;
      } else {
        CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&idx_map);
        for (i=0; i<tsr->order; i++){
          idx_map[i] = i;
        }
        define_tensor(0, NULL, NULL, &tid_scal, 1);
        fs.func_ptr=sym_seq_sum_ref<double>;
        felm.func_ptr = NULL;
        home_sum_tsr(1.0, 0.0, tid, tid_scal, idx_map, NULL, fs, felm);
        if (global_comm.rank == 0)
          *result = tensors[tid_scal]->data[0];
        else
          *result = 0.0;
        POST_BCAST(result, sizeof(double), COMM_CHAR_T, 0, global_comm, 0);
        CTF_free(idx_map);
      }

/*      acc = 0.0;
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
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);*/
      break;

    case CTF_OP_NORM1:
    case CTF_OP_SUMABS:
      CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&idx_map);
      is_AS = 0;
      for (i=0; i<tsr->order; i++){
        idx_map[i] = i;
        if (tsr->sym[i] == AS) is_AS = 1;
      }
      define_tensor(0, NULL, NULL, &tid_scal, 1);
      fs.func_ptr=NULL;
      felm.func_ptr = sum_abs;
      if (is_AS){
        int * sh_sym, * save_sym;
        CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&sh_sym);
        for (i=0; i<tsr->order; i++){
          if (tsr->sym[i] == AS) sh_sym[i] = SH;
          else sh_sym[i] = tsr->sym[i];
        }
        /** FIXME: This ruins tensor meta data immutability */
        save_sym = tsr->sym;
        tsr->sym = sh_sym;
        home_sum_tsr(1.0, 0.0, tid, tid_scal, idx_map, NULL, fs, felm);
        tsr->sym = save_sym;
        CTF_free(sh_sym);
      } else {
        home_sum_tsr(1.0, 0.0, tid, tid_scal, idx_map, NULL, fs, felm);
      }
      if (global_comm.rank == 0)
        *result = tensors[tid_scal]->data[0];
      else
        *result = 0.0;

      POST_BCAST(result, sizeof(double), COMM_CHAR_T, 0, global_comm, 0);
      CTF_free(idx_map);

/*      acc = 0.0;
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
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);*/
      break;

    case CTF_OP_NORM2:
      CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&idx_map);
      for (i=0; i<tsr->order; i++){
        idx_map[i] = i;
      }
      define_tensor(0, NULL, NULL, &tid_scal, 1);
      CTF_ctr_type_t ctype;
      ctype.tid_A = tid; 
      ctype.tid_B = tid; 
      ctype.tid_C = tid_scal; 
      ctype.idx_map_A = idx_map; 
      ctype.idx_map_B = idx_map; 
      ctype.idx_map_C = NULL; 
      fcs.func_ptr=sym_seq_ctr_ref<double>;
#ifdef OFFLOAD
      fcs.is_offloadable = 0;
#endif
      fcelm.func_ptr = NULL;
      home_contract(&ctype, fcs, fcelm, 1.0, 0.0);
      if (global_comm.rank == 0)
        *result = sqrt(tensors[tid_scal]->data[0]);
      else
        *result = 0.0;

      POST_BCAST(result, sizeof(double), COMM_CHAR_T, 0, global_comm, 0);
      CTF_free(idx_map);

      /*if (tsr->is_mapped){
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
      ALLREDUCE(&acc, result, 1, MPI_DOUBLE, MPI_SUM, global_comm);*/
      break;

    case CTF_OP_MAX:
      if (is_asym) {
        red_tsr(tid, CTF_OP_MAXABS, result);
      } else {
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
      }
      break;

    /* FIXME: incorrect when there is padding and the actual MIN is > 0 */
    case CTF_OP_MIN:
      if (is_asym) {
        red_tsr(tid, CTF_OP_MAXABS, result);
        *result = -1.0 * (*result);
      } else {
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
      }
      break;

    case CTF_OP_MAXABS:
    case CTF_OP_NORM_INFTY:
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
      return CTF_ERROR;
      break;
  }
  return CTF_SUCCESS;
}

/**
 * \brief apply a function to each element to transform tensor
 * \param[in] tid handle to tensor
 * \param[in] map_func map function to apply to each element
 */
template<typename dtype>
int dist_tensor<dtype>::map_tsr(int const tid,
                                dtype (*map_func)(int const order,
                                                  int const * indices,
                                                  dtype const elem)){
  int64_t i, j, np, stat;
  int * idx;
  tensor<dtype> * tsr;
  key k;
  tkv_pair<dtype> * prs;

  tsr = tensors[tid];
  if (tsr->has_zero_edge_len){
    return CTF_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);

  CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&idx);

  /* Extract key-value pair representation */
  if (tsr->is_mapped){
    stat = read_local_pairs(tid, &np, &prs);
    if (stat != CTF_SUCCESS) return stat;
  } else {
    np = tsr->size;
    prs = tsr->pairs;
  }
  /* Extract location from key and map */
  for (i=0; i<np; i++){
    k = prs[i].k;
    for (j=0; j<tsr->order; j++){
      idx[j] = k%tsr->edge_len[j];
      k = k/tsr->edge_len[j];
    }
    prs[i].d = map_func(tsr->order, idx, prs[i].d);
  }
  /* Rewrite pairs to packed layout */
  if (tsr->is_mapped){
    stat = write_pairs(tid, np, 1.0, 0.0, prs,'w');
    CTF_free(prs);
    if (stat != CTF_SUCCESS) return stat;
  }
  CTF_free(idx);
  return CTF_SUCCESS;
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
  if (global_comm.rank == 0)
    printf("FAILURE: daxpy currently only supported for tensors of type double\n");
  return CTF_ERROR;
}

template<> 
int dist_tensor<double>::
    daxpy_local_tensor_pair(double alpha, const int tid_A, const int tid_B){
  tensor<double> * tsr_A, * tsr_B;
  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  if (tsr_A->has_zero_edge_len || tsr_B->has_zero_edge_len){
    return CTF_SUCCESS;
  }
  ASSERT(tsr_A->size == tsr_B->size);
  cdaxpy(tsr_A->size, alpha, tsr_A->data, 1, tsr_B->data, 1);
  return CTF_SUCCESS;
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
 * \param[in] is_used whether this ctr pointer will actually be run
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
                          int *                       nvirt_all,
                          int                         is_used){
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
                   dtype const               beta){

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
                  dtype const               beta){

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
              dtype const                 beta){

