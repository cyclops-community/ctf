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

/**
 * \brief convert index maps from arbitrary indices to the smallest possible
 *
 * \param[in] ndim dimension of tensor 
 * \param[in] cidx old index map of tensor 
 * \param[out] iidx new index map of tensor 
 */
inline 
int conv_idx(int const    ndim,
             int const *  cidx,
             int **       iidx){
  int i, j, n;
  int c;

  *iidx = (int*)malloc(sizeof(int)*ndim);

  n = 0;
  for (i=0; i<ndim; i++){
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
 * \param[in] ndim_A dimension of tensor A
 * \param[in] cidx_A old index map of tensor A
 * \param[out] iidx_A new index map of tensor A
 * \param[in] ndim_B dimension of tensor B
 * \param[in] cidx_B old index map of tensor B
 * \param[out] iidx_B new index map of tensor B
 */
inline 
int  conv_idx(int const   ndim_A,
              int const * cidx_A,
              int **      iidx_A,
              int const   ndim_B,
              int const * cidx_B,
              int **      iidx_B){
  int i, j, n;
  int c;

  *iidx_B = (int*)malloc(sizeof(int)*ndim_B);

  n = conv_idx(ndim_A, cidx_A, iidx_A);
  for (i=0; i<ndim_B; i++){
    c = cidx_B[i];
    for (j=0; j<ndim_A; j++){
      if (c == cidx_A[j]){
        (*iidx_B)[i] = (*iidx_A)[j];
        break;
      }
    }
    if (j==ndim_A){
      for (j=0; j<i; j++){
        if (c == cidx_B[j]){
          (*iidx_B)[i] = (*iidx_B)[j];
          break;
        }
      }
      if (j==i){
        (*iidx_B)[i] = n;
        n++;
      }
    }
  }
  return n;
}

/**
 * \brief convert index maps from arbitrary indices to the smallest possible
 *
 * \param[in] ndim_A dimension of tensor A
 * \param[in] cidx_A old index map of tensor A
 * \param[out] iidx_A new index map of tensor A
 * \param[in] ndim_B dimension of tensor B
 * \param[in] cidx_B old index map of tensor B
 * \param[out] iidx_B new index map of tensor B
 * \param[in] ndim_C dimension of tensor C
 * \param[in] cidx_C old index map of tensor C
 * \param[out] iidx_C new index map of tensor C
 */
inline 
int  conv_idx(int const   ndim_A,
              int const * cidx_A,
              int **      iidx_A,
              int const   ndim_B,
              int const * cidx_B,
              int **      iidx_B,
              int const   ndim_C,
              int const * cidx_C,
              int **      iidx_C){
  int i, j, n;
  int c;

  *iidx_C = (int*)malloc(sizeof(int)*ndim_C);

  n = conv_idx(ndim_A, cidx_A, iidx_A,
               ndim_B, cidx_B, iidx_B);

  for (i=0; i<ndim_C; i++){
    c = cidx_C[i];
    for (j=0; j<ndim_B; j++){
      if (c == cidx_B[j]){
        (*iidx_C)[i] = (*iidx_B)[j];
        break;
      }
    }
    if (j==ndim_B){
      for (j=0; j<ndim_A; j++){
        if (c == cidx_A[j]){
          (*iidx_C)[i] = (*iidx_A)[j];
          break;
        }
      }
      if (j==ndim_A){
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
 * \brief permute an array
 *
 * \param ndim number of elements
 * \param perm permutation array
 * \param arr array to permute
 */
inline 
void permute(int const    ndim,
             int const *  perm,
             int *        arr){
  int i;
  int * swap;
  get_buffer_space(ndim*sizeof(int), (void**)&swap);

  for (i=0; i<ndim; i++){
    swap[i] = arr[perm[i]];
  }
  for (i=0; i<ndim; i++){
    arr[i] = swap[i];
  }

  free(swap);
}

/**
 * \brief permutes a permutation array 
 *
 * \param ndim number of elements in perm
 * \param ndim_perm number of elements in arr
 * \param perm permutation array
 * \param arr permutation array to permute
 */
inline 
void permute_target(int const   ndim,
                    int const   ndim_perm,
                    int const * perm,
                    int *       arr){
  int i, j;
  int * swap;
  get_buffer_space(ndim*sizeof(int), (void**)&swap);

  for (i=0; i<ndim; i++){
    for (j=0; j<ndim_perm; j++){
      if (arr[j] == i) break;
    }
    swap[perm[i]] = j;
  }
  for (i=0; i<ndim; i++){
    arr[swap[i]] = i;
  }

  free(swap);
}

/**
 * \brief calculate the dimensions of the matrix 
 *    the contraction gets reduced to
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
void calc_fold_nmk( CTF_ctr_type_t const *  type, 
                    int const *             ordering_A, 
                    int const *             ordering_B, 
                    tensor<dtype> const *   tsr_A, 
                    tensor<dtype> const *   tsr_B,
                    tensor<dtype> const *   tsr_C,
                    iparam *                inner_prm) {
  int i, num_ctr, num_tot;
  int * idx_arr;
  int * edge_len_A, * edge_len_B;
  iparam prm;

    
  edge_len_A = tsr_A->edge_len;
  edge_len_B = tsr_B->edge_len;

  inv_idx(tsr_A->ndim, type->idx_map_A, NULL,
          tsr_B->ndim, type->idx_map_B, NULL,
          tsr_C->ndim, type->idx_map_C, NULL,
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
      prm.m = prm.m * edge_len_A[ordering_A[i]];
    else 
      prm.k = prm.k * edge_len_A[ordering_A[i]];
  }
  for (i=0; i<tsr_B->ndim; i++){
    if (i >= num_ctr)
      prm.n = prm.n * edge_len_B[ordering_B[i]];
  }
  /* This gets set later */
  prm.sz_C = 0;
  free(idx_arr);
  *inner_prm = prm;  
}

/**
 * \brief transposes a non-symmetric (folded) tensor
 *
 * \param[in] ndim dimension of tensor
 * \param[in] new_order new ordering of dimensions
 * \param[in] edge_len original edge lengths
 * \param[in] data data tp transpose
 * \param[in] dir which way are we going?
 * \param[in] max_ntd how many threads to use
 * \param[out] tswap_data tranposed data
 * \param[out] chunk_size chunk sizes of tranposed data
 */
template<typename dtype>
void nosym_transpose(int const          ndim,
                     int const *        new_order,
                     int const *        edge_len,
                     dtype const *      data,
                     int const          dir,
                     int const          max_ntd,
                     dtype **           tswap_data,
                     int *              chunk_size){
  int64_t local_size;
  int64_t j, last_dim;
  int64_t * lda, * new_lda;

  TAU_FSTART(nosym_transpose_thr);
  get_buffer_space(ndim*sizeof(int64_t), (void**)&lda);
  get_buffer_space(ndim*sizeof(int64_t), (void**)&new_lda);
  
  if (dir){
    last_dim = new_order[ndim-1];
  } else {
    last_dim = ndim - 1;
  }
//  last_dim = ndim-1;

  lda[0] = 1;
  for (j=1; j<ndim; j++){
    lda[j] = lda[j-1]*edge_len[j-1];
  }
  local_size = lda[ndim-1]*edge_len[ndim-1];
  new_lda[new_order[0]] = 1;
  for (j=1; j<ndim; j++){
    new_lda[new_order[j]] = new_lda[new_order[j-1]]*edge_len[new_order[j-1]];
  }
  LIBT_ASSERT(local_size == new_lda[new_order[ndim-1]]*edge_len[new_order[ndim-1]]);
#ifdef USE_OMP
  #pragma omp parallel num_threads(max_ntd)
#endif
  {
    int64_t i, off_old, off_new, tid, ntd, last_max, toff_new, toff_old;
    int64_t tidx_off;
    int64_t thread_chunk_size;
    int64_t * idx;
    dtype * swap_data;
    get_buffer_space(ndim*sizeof(int64_t), (void**)&idx);
    memset(idx, 0, ndim*sizeof(int64_t));

#ifdef USE_OMP
    tid = omp_get_thread_num();
    ntd = omp_get_num_threads();
#else
    tid = 0;
    ntd = 1;
    thread_chunk_size = local_size;
#endif
    last_max = 1;
    tidx_off = 0;
    off_old = 0;
    off_new = 0;
    toff_old = 0;
    toff_new = 0;
    if (ndim != 1){
      tidx_off = (edge_len[last_dim]/ntd)*tid;
      idx[last_dim] = tidx_off;
      last_max = (edge_len[last_dim]/ntd)*(tid+1);
      if (tid == ntd-1) last_max = edge_len[last_dim];
      off_old = idx[last_dim]*lda[last_dim];
      off_new = idx[last_dim]*new_lda[last_dim];
      toff_old = off_old;
      toff_new = off_new;
    //  print64_tf("%d %d %d %d %d\n", tid, ntd, idx[last_dim], last_max, edge_len[last_dim]);
      thread_chunk_size = (local_size*(last_max-tidx_off))/edge_len[last_dim];
    } else {
      thread_chunk_size = local_size;
      last_dim = 2;
    } 
    chunk_size[tid] = 0;
    if (last_max != 0 && tidx_off != last_max && (ndim != 1 || tid == 0)){
      chunk_size[tid] = thread_chunk_size;
      if (thread_chunk_size <= 0) 
        printf("ERRORR thread_chunk_size = %lld, tid = %lld, local_size = %lld\n", thread_chunk_size, tid, local_size);
      get_buffer_space(thread_chunk_size*sizeof(dtype), (void**)&tswap_data[tid]);
      swap_data = tswap_data[tid];
      for (;;){
        if (last_dim != 0){
          for (idx[0] = 0; idx[0] < edge_len[0]; idx[0]++){
            if (dir)
              swap_data[off_new-toff_new] = data[off_old];
            else
              swap_data[off_old-toff_old] = data[off_new];
        
            off_old += lda[0];
            off_new += new_lda[0];
          }
          off_old -= edge_len[0]*lda[0];
          off_new -= edge_len[0]*new_lda[0];

          idx[0] = 0;
        } else {
          for (idx[0] = tidx_off; idx[0] < last_max; idx[0]++){
            if (dir)
              swap_data[off_new-toff_new] = data[off_old];
            else
              swap_data[off_old-toff_old] = data[off_new];
        
            off_old += lda[0];
            off_new += new_lda[0];
          }
          off_old -= last_max*lda[0];
          off_new -= last_max*new_lda[0];

          idx[0] = tidx_off;
          off_old += idx[0]*lda[0];
          off_new += idx[0]*new_lda[0];
        } 

        for (i=1; i<ndim; i++){
          off_old -= idx[i]*lda[i];
          off_new -= idx[i]*new_lda[i];
          if (i == last_dim){
            idx[i] = (idx[i]+1)%last_max;
            if (idx[i] == 0) idx[i] = tidx_off;
            off_old += idx[i]*lda[i];
            off_new += idx[i]*new_lda[i];
            if (idx[i] != tidx_off) break;
          } else {
            idx[i] = (idx[i]+1)%edge_len[i];
            off_old += idx[i]*lda[i];
            off_new += idx[i]*new_lda[i];
            if (idx[i] != 0) break;
          }
        }
        if (i==ndim) break;
      }
    }
    free(idx);
  }
  free(lda);
  free(new_lda);
  TAU_FSTOP(nosym_transpose_thr);
}

/**
 * \brief transposes a non-symmetric (folded) tensor
 *
 * \param[in] ndim dimension of tensor
 * \param[in] new_order new ordering of dimensions
 * \param[in] edge_len original edge lengths
 * \param[in,out] data data tp transpose
 * \param[in] dir which way are we going?
 */
template<typename dtype>
void nosym_transpose(int const          ndim,
                     int const *        new_order,
                     int const *        edge_len,
                     dtype *            data,
                     int const          dir){
  int * chunk_size;
  dtype ** tswap_data;

  if (ndim == 0) return;
  TAU_FSTART(nosym_transpose);
#ifdef USE_OMP
  int max_ntd = MIN(16,omp_get_max_threads());
  get_buffer_space(max_ntd*sizeof(dtype*), (void**)&tswap_data);
  get_buffer_space(max_ntd*sizeof(int), (void**)&chunk_size);
#else
  int max_ntd=1;
  get_buffer_space(sizeof(dtype*), (void**)&tswap_data);
  get_buffer_space(sizeof(int), (void**)&chunk_size);
#endif
  nosym_transpose(ndim, new_order, edge_len, data, dir, max_ntd, tswap_data, chunk_size);
#ifdef USE_OMP
  #pragma omp parallel num_threads(max_ntd)
#endif
  {
    int tid;
#ifdef USE_OMP
    tid = omp_get_thread_num();
#else
    tid = 0;
#endif
    int thread_chunk_size = chunk_size[tid];
    int i;
    dtype * swap_data = tswap_data[tid];
    int toff = 0;
    for (i=0; i<tid; i++) toff += chunk_size[i];
    if (thread_chunk_size > 0){
      memcpy(data+toff,swap_data,sizeof(dtype)*thread_chunk_size);
      free(swap_data);
    }
  }

  free(tswap_data);
  free(chunk_size);
  TAU_FSTOP(nosym_transpose);
}


/**
 * \brief finds and return all summation indices which can be folded into
 *    dgemm, for which they must (1) not break symmetry (2) belong to 
 *    exactly two of (A,B).
 * \param[in] type contraction specification
 * \param[out] num_fold number of indices that can be folded
 * \param[out] fold_idx indices that can be folded
 */
template<typename dtype>
void dist_tensor<dtype>::get_fold_indices(CTF_sum_type_t const *  type,
                                          int *                   num_fold,
                                          int **                  fold_idx){
  int i, in, num_tot, nfold, broken;
  int iA, iB, inA, inB, iiA, iiB;
  int * idx_arr, * idx;
  tensor<dtype> * tsr_A, * tsr_B;
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          &num_tot, &idx_arr);
  get_buffer_space(num_tot*sizeof(int), (void**)&idx);

  for (i=0; i<num_tot; i++){
    idx[i] = 1;
  }
  
  for (iA=0; iA<tsr_A->ndim; iA++){
    i = type->idx_map_A[iA];
    iB = idx_arr[2*i+1];
    broken = 0;
    inA = iA;
    do {
      in = type->idx_map_A[inA];
      inB = idx_arr[2*in+1];
      if (((inA>=0) + (inB>=0) != 2) ||
          (iB != -1 && inB - iB != in-i) ||
          (iB != -1 && tsr_A->sym[inA] != tsr_B->sym[inB])){
        broken = 1;
      }
      inA++;
    } while (tsr_A->sym[inA-1] != NS);
    if (broken){
      for (iiA=iA;iiA<inA;iiA++){
        idx[type->idx_map_A[iiA]] = 0;
      }
    }
  }

  for (iB=0; iB<tsr_B->ndim; iB++){
    i = type->idx_map_B[iB];
    iA = idx_arr[2*i+0];
    broken = 0;
    inB = iB;
    do {
      in = type->idx_map_B[inB];
      inA = idx_arr[2*in+0];
      if (((inB>=0) + (inA>=0) != 2) ||
          (iA != -1 && inA - iA != in-i) ||
          (iA != -1 && tsr_B->sym[inB] != tsr_A->sym[inA])){
        broken = 1;
      }
      inB++;
    } while (tsr_B->sym[inB-1] != NS);
    if (broken){
      for (iiB=iB;iiB<inB;iiB++){
        idx[type->idx_map_B[iiB]] = 0;
      }
    }
  }
  

  nfold = 0;
  for (i=0; i<num_tot; i++){
    if (idx[i] == 1){
      idx[nfold] = i;
      nfold++;
    }
  }
  *num_fold = nfold;
  *fold_idx = idx;
  free(idx_arr);
}


/**
 * \brief finds and return all contraction indices which can be folded into
 *    dgemm, for which they must (1) not break symmetry (2) belong to 
 *    exactly two of (A,B,C).
 * \param[in] type contraction specification
 * \param[out] num_fold number of indices that can be folded
 * \param[out] fold_idx indices that can be folded
 */
template<typename dtype>
void dist_tensor<dtype>::get_fold_indices(CTF_ctr_type_t const *  type,
                                          int *                   num_fold,
                                          int **                  fold_idx){
  int i, in, num_tot, nfold, broken;
  int iA, iB, iC, inA, inB, inC, iiA, iiB, iiC;
  int * idx_arr, * idx;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];
  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
    tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
    tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
    &num_tot, &idx_arr);
  get_buffer_space(num_tot*sizeof(int), (void**)&idx);

  for (i=0; i<num_tot; i++){
    idx[i] = 1;
  }

  for (iA=0; iA<tsr_A->ndim; iA++){
    i = type->idx_map_A[iA];
    iB = idx_arr[3*i+1];
    iC = idx_arr[3*i+2];
    broken = 0;
    inA = iA;
    do {
      in = type->idx_map_A[inA];
      inB = idx_arr[3*in+1];
      inC = idx_arr[3*in+2];
      if (((iA>=0) + (iB>=0) + (iC>=0) != 2) ||
          ((inB == -1) ^ (iB == -1)) ||
          ((inC == -1) ^ (iC == -1)) ||
          (iB != -1 && inB - iB != in-i) ||
          (iC != -1 && inC - iC != in-i) ||
          (iB != -1 && tsr_A->sym[inA] != tsr_B->sym[inB]) ||
          (iC != -1 && tsr_A->sym[inA] != tsr_C->sym[inC])){
        broken = 1;
      }
      inA++;
    } while (tsr_A->sym[inA-1] != NS);
    if (broken){
      for (iiA=iA;iiA<inA;iiA++){
        idx[type->idx_map_A[iiA]] = 0;
      }
    }
  }
  
  for (iC=0; iC<tsr_C->ndim; iC++){
    i = type->idx_map_C[iC];
    iA = idx_arr[3*i+0];
    iB = idx_arr[3*i+1];
    broken = 0;
    inC = iC;
    do {
      in = type->idx_map_C[inC];
      inA = idx_arr[3*in+0];
      inB = idx_arr[3*in+1];
      if (((iC>=0) + (iA>=0) + (iB>=0) != 2) ||
          ((inA == -1) ^ (iA == -1)) ||
          ((inB == -1) ^ (iB == -1)) ||
          (iA != -1 && inA - iA != in-i) ||
          (iB != -1 && inB - iB != in-i) ||
          (iA != -1 && tsr_C->sym[inC] != tsr_A->sym[inA]) ||
          (iB != -1 && tsr_C->sym[inC] != tsr_B->sym[inB])){
        broken = 1;
      }
      inC++;
    } while (tsr_C->sym[inC-1] != NS);
    if (broken){
      for (iiC=iC;iiC<inC;iiC++){
        idx[type->idx_map_C[iiC]] = 0;
      }
    }
  }
  
  for (iB=0; iB<tsr_B->ndim; iB++){
    i = type->idx_map_B[iB];
    iC = idx_arr[3*i+2];
    iA = idx_arr[3*i+0];
    broken = 0;
    inB = iB;
    do {
      in = type->idx_map_B[inB];
      inC = idx_arr[3*in+2];
      inA = idx_arr[3*in+0];
      if (((iB>=0) + (iC>=0) + (iA>=0) != 2) ||
          ((inC == -1) ^ (iC == -1)) ||
          ((inA == -1) ^ (iA == -1)) ||
          (iC != -1 && inC - iC != in-i) ||
          (iA != -1 && inA - iA != in-i) ||
          (iC != -1 && tsr_B->sym[inB] != tsr_C->sym[inC]) ||
          (iA != -1 && tsr_B->sym[inB] != tsr_A->sym[inA])){
        broken = 1;
      }
      inB++;
    } while (tsr_B->sym[inB-1] != NS);
    if (broken){
      for (iiB=iB;iiB<inB;iiB++){
        idx[type->idx_map_B[iiB]] = 0;
      }
    }
  }

  nfold = 0;
  for (i=0; i<num_tot; i++){
    if (idx[i] == 1){
      idx[nfold] = i;
      nfold++;
    }
  }
  *num_fold = nfold;
  *fold_idx = idx;
  free(idx_arr);

}


/**
 * \brief determines whether this contraction can be folded
 * \param[in] type contraction specification
 * \return whether we can fold this contraction
 */
template<typename dtype>
int dist_tensor<dtype>::can_fold(CTF_ctr_type_t const * type){
  int nfold, * fold_idx, i, j;
  tensor<dtype> * tsr;
  tsr = tensors[type->tid_A];
  for (i=0; i<tsr->ndim; i++){
    for (j=i+1; j<tsr->ndim; j++){
      if (type->idx_map_A[i] == type->idx_map_A[j]) return 0;
    }
  }
  tsr = tensors[type->tid_B];
  for (i=0; i<tsr->ndim; i++){
    for (j=i+1; j<tsr->ndim; j++){
      if (type->idx_map_B[i] == type->idx_map_B[j]) return 0;
    }
  }
  tsr = tensors[type->tid_C];
  for (i=0; i<tsr->ndim; i++){
    for (j=i+1; j<tsr->ndim; j++){
      if (type->idx_map_C[i] == type->idx_map_C[j]) return 0;
    }
  }
  get_fold_indices(type, &nfold, &fold_idx);
  free(fold_idx);
  /* FIXME: 1 folded index is good enough for now, in the future model */
  return nfold > 0;
}

template<typename dtype>
int dist_tensor<dtype>::can_fold(CTF_sum_type_t const * type){
  int i, j, nfold, * fold_idx;
  tensor<dtype> * tsr;
  tsr = tensors[type->tid_A];
  for (i=0; i<tsr->ndim; i++){
    for (j=i+1; j<tsr->ndim; j++){
      if (type->idx_map_A[i] == type->idx_map_A[j]) return 0;
    }
  }
  tsr = tensors[type->tid_B];
  for (i=0; i<tsr->ndim; i++){
    for (j=i+1; j<tsr->ndim; j++){
      if (type->idx_map_B[i] == type->idx_map_B[j]) return 0;
    }
  }
  get_fold_indices(type, &nfold, &fold_idx);
  free(fold_idx);
  /* FIXME: 1 folded index is good enough for now, in the future model */
  return nfold > 0;
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
  
  LIBT_ASSERT(tsr->is_inner_mapped == 0);
  if (tsr->is_folded != 0) unfold_tsr(tsr);
  
  get_buffer_space(tsr->ndim*sizeof(int), (void**)&sub_edge_len);

  allfold_dim = 0, fold_dim = 0;
  for (j=0; j<tsr->ndim; j++){
    if (tsr->sym[j] == NS){
      allfold_dim++;
      for (i=0; i<nfold; i++){
        if (fold_idx[i] == idx_map[j])
          fold_dim++;
      }
    }
  }
  get_buffer_space(allfold_dim*sizeof(int), (void**)&all_edge_len);
  get_buffer_space(allfold_dim*sizeof(int), (void**)&dim_order);
  get_buffer_space(fold_dim*sizeof(int), (void**)&fold_edge_len);
  get_buffer_space(fold_dim*sizeof(int), (void**)&fold_sym);

  calc_dim(tsr->ndim, tsr->size, tsr->edge_len, tsr->edge_map,
     NULL, sub_edge_len, NULL);

  allfold_dim = 0, fdim = 0;
  for (j=0; j<tsr->ndim; j++){
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

  tsr->is_folded    = 1;
  tsr->rec_tid      = fold_tid;
  tsr->inner_ordering     = dim_order;

  *all_fdim = allfold_dim;
  *all_flen = all_edge_len;

  free(fold_edge_len);
  free(fold_sym);
  
  free(sub_edge_len);
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
    get_buffer_space(tsr->ndim*sizeof(int), (void**)&all_edge_len);
    get_buffer_space(tsr->ndim*sizeof(int), (void**)&sub_edge_len);
    calc_dim(tsr->ndim, tsr->size, tsr->edge_len, tsr->edge_map,
             NULL, sub_edge_len, NULL);
    allfold_dim = 0;
    for (i=0; i<tsr->ndim; i++){
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
    free(tsr->inner_ordering);
    free(all_edge_len);
    free(sub_edge_len);

  }  
  tsr->is_folded = 0;
}

/**
 * \brief find ordering of indices of tensor to reduce to DAXPY
 *
 * \param[in] type summation specification
 * \param[out] new_ordering_A the new ordering for indices of A
 * \param[out] new_ordering_B the new ordering for indices of B
 */
template<typename dtype>
void dist_tensor<dtype>::get_len_ordering(
                                      CTF_sum_type_t const *  type,
                                      int **      new_ordering_A,
                                      int **      new_ordering_B){
  int i, num_tot;
  int * ordering_A, * ordering_B, * idx_arr;
  tensor<dtype> * tsr_A, * tsr_B;
  
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&ordering_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&ordering_B);

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          &num_tot, &idx_arr);
  for (i=0; i<num_tot; i++){
    ordering_A[i] = idx_arr[2*i];
    ordering_B[i] = idx_arr[2*i+1];
  }
  free_buffer_space(idx_arr);
  *new_ordering_A = ordering_A;
  *new_ordering_B = ordering_B;
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
void dist_tensor<dtype>::get_len_ordering(
          CTF_ctr_type_t const *  type,
          int **      new_ordering_A,
          int **      new_ordering_B,
          int **      new_ordering_C){
  int i, num_tot, num_ctr, idx_ctr, num_no_ctr_A;
  int idx_no_ctr_A, idx_no_ctr_B;
  int * ordering_A, * ordering_B, * ordering_C, * idx_arr;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];
  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&ordering_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&ordering_B);
  get_buffer_space(sizeof(int)*tsr_C->ndim, (void**)&ordering_C);

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
  free_buffer_space(idx_arr);
  *new_ordering_A = ordering_A;
  *new_ordering_B = ordering_B;
  *new_ordering_C = ordering_C;
}

/**
 * \brief folds tensors for summation
 * \param[in] type contraction specification
 * \param[out] inner_stride inner stride (daxpy size)
 */
template<typename dtype>
int dist_tensor<dtype>::map_fold(CTF_sum_type_t const * type,
                                 int *                  inner_stride) {

  int i, j, nfold, nf, all_fdim_A, all_fdim_B;
  int nvirt_A, nvirt_B;
  int * fold_idx, * fidx_map_A, * fidx_map_B;
  int * fnew_ord_A, * fnew_ord_B;
  int * all_flen_A, * all_flen_B;
  tensor<dtype> * tsr_A, * tsr_B;
  tensor<dtype> * ftsr_A, * ftsr_B;
  CTF_sum_type_t fold_type;
  int inr_stride;

  get_fold_indices(type, &nfold, &fold_idx);
  if (nfold == 0){
    free(fold_idx);
    return DIST_TENSOR_ERROR;
  }

  /* overestimate this space to not bother with it later */
  get_buffer_space(nfold*sizeof(int), (void**)&fidx_map_A);
  get_buffer_space(nfold*sizeof(int), (void**)&fidx_map_B);

  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];

  fold_tsr(tsr_A, nfold, fold_idx, type->idx_map_A, 
           &all_fdim_A, &all_flen_A);
  fold_tsr(tsr_B, nfold, fold_idx, type->idx_map_B, 
           &all_fdim_B, &all_flen_B);

  nf = 0;
  for (i=0; i<tsr_A->ndim; i++){
    for (j=0; j<nfold; j++){
      if (tsr_A->sym[i] == NS && type->idx_map_A[i] == fold_idx[j]){
        fidx_map_A[nf] = j;
        nf++;
      }
    }
  }
  nf = 0;
  for (i=0; i<tsr_B->ndim; i++){
    for (j=0; j<nfold; j++){
      if (tsr_B->sym[i] == NS && type->idx_map_B[i] == fold_idx[j]){
        fidx_map_B[nf] = j;
        nf++;
      }
    }
  }

  ftsr_A = tensors[tsr_A->rec_tid];
  ftsr_B = tensors[tsr_B->rec_tid];

  fold_type.tid_A = tsr_A->rec_tid;
  fold_type.tid_B = tsr_B->rec_tid;

  conv_idx(ftsr_A->ndim, fidx_map_A, &fold_type.idx_map_A,
           ftsr_B->ndim, fidx_map_B, &fold_type.idx_map_B);

#if DEBUG>=2
  if (global_comm->rank == 0){
    printf("Folded summation type:\n");
  }
  print_sum(&fold_type);
#endif
  
  get_len_ordering(&fold_type, &fnew_ord_A, &fnew_ord_B); 

  permute_target(ftsr_A->ndim, all_fdim_A, fnew_ord_A, tsr_A->inner_ordering);
  permute_target(ftsr_B->ndim, all_fdim_B, fnew_ord_B, tsr_B->inner_ordering);
  

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

  inr_stride = 1;
  for (i=0; i<ftsr_A->ndim; i++){
    inr_stride *= ftsr_A->edge_len[i];
  }

  *inner_stride = inr_stride; 

  free(fidx_map_A);
  free(fidx_map_B);
  free(fold_type.idx_map_A);
  free(fold_type.idx_map_B);
  free(fnew_ord_A);
  free(fnew_ord_B);
  free(all_flen_A);
  free(all_flen_B);
  free(fold_idx);

  return DIST_TENSOR_SUCCESS;
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
    free(fold_idx);
    return DIST_TENSOR_ERROR;
  }

  /* overestimate this space to not bother with it later */
  get_buffer_space(nfold*sizeof(int), (void**)&fidx_map_A);
  get_buffer_space(nfold*sizeof(int), (void**)&fidx_map_B);
  get_buffer_space(nfold*sizeof(int), (void**)&fidx_map_C);

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
  for (i=0; i<tsr_A->ndim; i++){
    for (j=0; j<nfold; j++){
      if (tsr_A->sym[i] == NS && type->idx_map_A[i] == fold_idx[j]){
        fidx_map_A[nf] = j;
        nf++;
      }
    }
  }
  nf = 0;
  for (i=0; i<tsr_B->ndim; i++){
    for (j=0; j<nfold; j++){
      if (tsr_B->sym[i] == NS && type->idx_map_B[i] == fold_idx[j]){
        fidx_map_B[nf] = j;
        nf++;
      }
    }
  }
  nf = 0;
  for (i=0; i<tsr_C->ndim; i++){
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

  conv_idx(ftsr_A->ndim, fidx_map_A, &fold_type.idx_map_A,
           ftsr_B->ndim, fidx_map_B, &fold_type.idx_map_B,
           ftsr_C->ndim, fidx_map_C, &fold_type.idx_map_C);

#if DEBUG>=2
  if (global_comm->rank == 0){
    printf("Folded contraction type:\n");
  }
  print_ctr(&fold_type);
#endif
  
  get_len_ordering(&fold_type, &fnew_ord_A, &fnew_ord_B, &fnew_ord_C); 

  permute_target(ftsr_A->ndim, all_fdim_A, fnew_ord_A, tsr_A->inner_ordering);
  permute_target(ftsr_B->ndim, all_fdim_B, fnew_ord_B, tsr_B->inner_ordering);
  permute_target(ftsr_C->ndim, all_fdim_C, fnew_ord_C, tsr_C->inner_ordering);
  

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
  free(fidx_map_A);
  free(fidx_map_B);
  free(fidx_map_C);
  free(fold_type.idx_map_A);
  free(fold_type.idx_map_B);
  free(fold_type.idx_map_C);
  free(fnew_ord_A);
  free(fnew_ord_B);
  free(fnew_ord_C);
  free(all_flen_A);
  free(all_flen_B);
  free(all_flen_C);
  free(fold_idx);

  *inner_prm = iprm;
  return DIST_TENSOR_SUCCESS;
}


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
  int i, num_tot, iA, iB, iC;
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
    
    get_buffer_space(tsr_A->ndim*sizeof(int), (void**)&new_type->idx_map_A);
    get_buffer_space(tsr_B->ndim*sizeof(int), (void**)&new_type->idx_map_B);
    get_buffer_space(tsr_C->ndim*sizeof(int), (void**)&new_type->idx_map_C);

    memcpy(new_type->idx_map_A, type->idx_map_A, tsr_A->ndim*sizeof(int));
    memcpy(new_type->idx_map_B, type->idx_map_B, tsr_B->ndim*sizeof(int));
    memcpy(new_type->idx_map_C, type->idx_map_C, tsr_C->ndim*sizeof(int));
  }

  inv_idx(tsr_A->ndim, type->idx_map_A, tsr_A->edge_map,
          tsr_B->ndim, type->idx_map_B, tsr_B->edge_map,
          tsr_C->ndim, type->idx_map_C, tsr_C->edge_map,
          &num_tot, &idx_arr);

  for (i=0; i<tsr_A->ndim; i++){
    if (tsr_A->sym[i] != NS){
      iA = type->idx_map_A[i];
      if (idx_arr[3*iA+1] != -1){
        if (tsr_B->sym[idx_arr[3*iA+1]] == NS ||
            type->idx_map_A[i+1] != type->idx_map_B[idx_arr[3*iA+1]+1]){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          free(idx_arr); 
          return 3*i;
        }
      }
      if (idx_arr[3*iA+2] != -1){
        if (tsr_C->sym[idx_arr[3*iA+2]] == NS ||
            type->idx_map_A[i+1] != type->idx_map_C[idx_arr[3*iA+2]+1]){
          if (new_type != NULL)
            tsr_A->sym[i] = NS;
          free(idx_arr); 
          return 3*i;
        }
      }
    }
  } 
  for (i=0; i<tsr_B->ndim; i++){
    if (tsr_B->sym[i] != NS){
      iB = type->idx_map_B[i];
      if (idx_arr[3*iB+0] != -1){
        if (tsr_A->sym[idx_arr[3*iB+0]] == NS ||
            type->idx_map_B[i+1] != type->idx_map_A[idx_arr[3*iB+0]+1]){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          free(idx_arr); 
          return 3*i+1;
        }
      }
      if (idx_arr[3*iB+2] != -1){
        if (tsr_C->sym[idx_arr[3*iB+2]] == NS || 
            type->idx_map_B[i+1] != type->idx_map_C[idx_arr[3*iB+2]+1]){
          if (new_type != NULL)
            tsr_B->sym[i] = NS;
          free(idx_arr); 
          return 3*i+1;
        }
      }
    }
  } 
  for (i=0; i<tsr_C->ndim; i++){
    if (tsr_C->sym[i] != NS){
      iC = type->idx_map_C[i];
      if (idx_arr[3*iC+1] != -1){
        if (tsr_B->sym[idx_arr[3*iC+1]] == NS ||
            type->idx_map_C[i+1] != type->idx_map_B[idx_arr[3*iC+1]+1]){
          if (new_type != NULL)
            tsr_C->sym[i] = NS;
          free(idx_arr); 
          return 3*i+2;
        }
      }
      if (idx_arr[3*iC+0] != -1){
        if (tsr_A->sym[idx_arr[3*iC+0]] == NS ||
            type->idx_map_C[i+1] != type->idx_map_A[idx_arr[3*iC+0]+1]){
          if (new_type != NULL)
            tsr_C->sym[i] = NS;
          free(idx_arr); 
          return 3*i+2;
        }
      }
    }
  }
  free(idx_arr);
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
    copy_mapping(tsr_nonsym->ndim, tsr_nonsym->edge_map, 
                 tsr_sym->edge_map);
    tsr_sym->need_remap = tsr_nonsym->need_remap;
    tsr_sym->data = tsr_nonsym->data;
    set_padding(tsr_sym);
  }
}



/**
 * \brief unfolds the data of a tensor
 *
 * \param[in] sym_tid id of starting symmetric tensor (where data starts)
 * \param[in] nonsym_tid id of new tensor with a potentially unfolded symmetry
 */
template<typename dtype>
void dist_tensor<dtype>::desymmetrize(int const sym_tid, 
                                      int const nonsym_tid, 
                                      int const is_C){
  int i, j, sym_dim, scal_diag, num_sy;
  int * idx_map_A, * idx_map_B;
  tensor<dtype> * tsr_sym, * tsr_nonsym;
  double rev_sign;

  TAU_FSTART(desymmetrize);
  
  tsr_sym = tensors[sym_tid];
  tsr_nonsym = tensors[nonsym_tid];

  sym_dim = -1;
  rev_sign = 1.0;
  scal_diag = 0;
  num_sy=0;
  for (i=0; i<tsr_sym->ndim; i++){
    if (tsr_sym->sym[i] != tsr_nonsym->sym[i]){
      sym_dim = i;
      if (tsr_sym->sym[i] == AS) rev_sign = -1.0;
      if (tsr_sym->sym[i] == SY){
        scal_diag = 1;
        num_sy = 1;
        i++;
        while (i<tsr_sym->ndim && tsr_sym->sym[i] == SY){
          num_sy++;
          i++;
        }
      }
      break;
    }
  }
  clear_mapping(tsr_nonsym);
  set_padding(tsr_nonsym);
  copy_mapping(tsr_sym->ndim, tsr_sym->edge_map, tsr_nonsym->edge_map);
  tsr_nonsym->is_mapped = 1;
  tsr_nonsym->itopo   = tsr_sym->itopo;
  set_padding(tsr_nonsym);

  if (sym_dim == -1) {
    tsr_nonsym->size    = tsr_sym->size;
    tsr_nonsym->data    = tsr_sym->data;
    tsr_nonsym->need_remap = tsr_sym->need_remap;
    tsr_nonsym->is_data_aliased = 1;
    TAU_FSTOP(desymmetrize);
    return;
  }

  get_buffer_space(tsr_nonsym->size*sizeof(dtype), (void**)&tsr_nonsym->data);
  std::fill(tsr_nonsym->data, tsr_nonsym->data+tsr_nonsym->size, get_zero<dtype>());

  get_buffer_space(tsr_sym->ndim*sizeof(int), (void**)&idx_map_A);
  get_buffer_space(tsr_sym->ndim*sizeof(int), (void**)&idx_map_B);

  for (i=0; i<tsr_sym->ndim; i++){
    idx_map_A[i] = i;
    idx_map_B[i] = i;
  }
  fseq_tsr_sum<dtype> fs;
  fs.func_ptr=sym_seq_sum_ref<dtype>;

  sum_tensors(1.0, 1.0, sym_tid, nonsym_tid, idx_map_A, idx_map_B, fs);

  if (!is_C){
    idx_map_A[sym_dim] = sym_dim+1;
    idx_map_A[sym_dim+1] = sym_dim;
    
    sum_tensors(rev_sign, 1.0, sym_tid, nonsym_tid, idx_map_A, idx_map_B, fs);
    
//  idx_map_A[sym_dim] = sym_dim;
//    idx_map_A[sym_dim+1] = sym_dim+1;
  }

  /* Do not diagonal rescaling since sum has beta=0 and overwrites diagonal */
//  if (false){
//    print_tsr(stdout, nonsym_tid);
    if (!is_C && scal_diag){
      for (i=0; i<num_sy; i++){
        idx_map_A[sym_dim] = sym_dim;
        idx_map_A[sym_dim+i+1] = sym_dim;
        for (j=sym_dim+i+2; j<tsr_sym->ndim; j++){
          idx_map_A[j] = j-i-1;
        }
        fseq_tsr_scl<dtype> fss;
        fss.func_ptr=sym_seq_scl_ref<dtype>;
        int ret = scale_tsr(((double)(num_sy-i))/(num_sy-i+1.), nonsym_tid, idx_map_A, fss);
        if (ret != DIST_TENSOR_SUCCESS) ABORT;
      }
    }  
//    print_tsr(stdout, nonsym_tid);
//  }
  free(idx_map_A);
  free(idx_map_B);  

/*  switch (tsr_sym->edge_map[sym_dim].type){
    case NOT_MAPPED:
      LIBT_ASSERT(tsr_sym->edge_map[sym_dim+1].type == NOT_MAPPED);
      rw_smtr<dtype>(tsr_sym->ndim, tsr_sym->edge_len, 1.0, 0, 
         tsr_sym->sym, tsr_nonsym->sym,
         tsr_sym->data, tsr_nonsym->data);
      rw_smtr<dtype>(tsr_sym->ndim, tsr_sym->edge_len, rev_sign, 1, 
         tsr_sym->sym, tsr_nonsym->sym,
         tsr_sym->data, tsr_nonsym->data);
      break;

    case VIRTUAL_MAP:
      if (tsr_sym->edge_map[sym_dim+1].type == VIRTUAL_MAP){
  nvirt = 

      } else {
  LIBT_ASSERT(tsr_sym->edge_map[sym_dim+1].type == PHYSICAL_MAP);

      }
      break;
    
    case PHYSICAL_MAP:
      if (tsr_sym->edge_map[sym_dim+1].type == VIRTUAL_MAP){

      } else {
  LIBT_ASSERT(tsr_sym->edge_map[sym_dim+1].type == PHYSICAL_MAP);

      }
      break;
  }*/
  TAU_FSTOP(desymmetrize);

}


/**
 * \brief folds the data of a tensor
 *
 * \param[in] sym_tid id of ending symmetric tensor (where data ends)
 * \param[in] nonsym_tid id of new tensor with a potentially unfolded symmetry
 */
template<typename dtype>
void dist_tensor<dtype>::symmetrize(int const sym_tid, int const nonsym_tid){
  int i, j, sym_dim, scal_diag, num_sy;
  int * idx_map_A, * idx_map_B;
  tensor<dtype> * tsr_sym, * tsr_nonsym;
  double rev_sign;

  TAU_FSTART(symmetrize);
  
  tsr_sym = tensors[sym_tid];
  tsr_nonsym = tensors[nonsym_tid];

  sym_dim = -1;
  rev_sign = 1.0;
  scal_diag = 0;
  num_sy=0;
  for (i=0; i<tsr_sym->ndim; i++){
    if (tsr_sym->sym[i] != tsr_nonsym->sym[i]){
      sym_dim = i;
      if (tsr_sym->sym[i] == AS) rev_sign = -1.0;
      if (tsr_sym->sym[i] == SY){
        scal_diag = 1;
        num_sy = 1;
        i++;
        while (i<tsr_sym->ndim && tsr_sym->sym[i] == SY){
          num_sy++;
          i++;
        }
      }
      break;
    }
  }
  if (sym_dim == -1) {
    tsr_sym->itopo    = tsr_nonsym->itopo;
    tsr_sym->is_mapped    = 1;
    copy_mapping(tsr_nonsym->ndim, tsr_nonsym->edge_map, tsr_sym->edge_map);
    set_padding(tsr_sym);
    tsr_sym->size     = tsr_nonsym->size;
    tsr_sym->data     = tsr_nonsym->data;
    tsr_sym->need_remap = tsr_nonsym->need_remap;
    TAU_FSTOP(symmetrize);
    return;
  }

  std::fill(tsr_sym->data, tsr_sym->data+tsr_sym->size, get_zero<dtype>());
  get_buffer_space(tsr_sym->ndim*sizeof(int), (void**)&idx_map_A);
  get_buffer_space(tsr_sym->ndim*sizeof(int), (void**)&idx_map_B);

  for (i=0; i<tsr_sym->ndim; i++){
    idx_map_A[i] = i;
    idx_map_B[i] = i;
  }
  fseq_tsr_sum<dtype> fs;
  fs.func_ptr=sym_seq_sum_ref<dtype>;
 
  
  idx_map_A[sym_dim] = sym_dim+1;
  idx_map_A[sym_dim+1] = sym_dim;

  sum_tensors(rev_sign, 0.0, nonsym_tid, sym_tid, idx_map_A, idx_map_B, fs);

  idx_map_A[sym_dim] = sym_dim;
  idx_map_A[sym_dim+1] = sym_dim+1;
    
  sum_tensors(1.0, 1.0, nonsym_tid, sym_tid, idx_map_A, idx_map_B, fs);

  //  if (false){ 
    if (scal_diag){
    //  printf("symmetrizing diagonal=%d\n",num_sy);
      for (i=0; i<num_sy; i++){
        idx_map_B[sym_dim] = sym_dim;
        idx_map_B[sym_dim+i+1] = sym_dim;
        for (j=sym_dim+i+2; j<tsr_sym->ndim; j++){
          idx_map_B[j] = j-i-1;
        }
        fseq_tsr_scl<dtype> fss;
        fss.func_ptr=sym_seq_scl_ref<dtype>;
        int ret = scale_tsr(((double)(num_sy-i))/(num_sy-i+1.), sym_tid, idx_map_B, fss);
        if (ret != DIST_TENSOR_SUCCESS) ABORT;
      }
    }  
//  }

  free(idx_map_A);
  free(idx_map_B);


  TAU_FSTOP(symmetrize);
}


/**
 * \brief finds all permutations of a tensor according to a symmetry
 *
 * \param[in] ndim dimension of tensor
 * \param[in] sym symmetry specification of tensor
 * \param[out] nperm number of symmeitrc permutations to do
 * \param[out] perm the permutation
 * \param[out] sign sign of each permutation
 */
inline
void cmp_sym_perms(int const    ndim,
                   int const *  sym,
                   int *        nperm,
                   int **       perm,
                   double *     sign){
  int i, np;
  int * pm;
  double sgn;

  LIBT_ASSERT(sym[0] != NS);
  get_buffer_space(sizeof(int)*ndim, (void**)&pm);

  np=0;
  sgn=1.0;
  for (i=0; i<ndim; i++){
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
  for (i=np; i<ndim; i++){
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

  get_buffer_space(sizeof(int)*tsr_A->ndim, (void**)&new_type->idx_map_A);
  get_buffer_space(sizeof(int)*tsr_B->ndim, (void**)&new_type->idx_map_B);
  get_buffer_space(sizeof(int)*tsr_C->ndim, (void**)&new_type->idx_map_C);

  memcpy(new_type->idx_map_A, old_type->idx_map_A, sizeof(int)*tsr_A->ndim);
  memcpy(new_type->idx_map_B, old_type->idx_map_B, sizeof(int)*tsr_B->ndim);
  memcpy(new_type->idx_map_C, old_type->idx_map_C, sizeof(int)*tsr_C->ndim);

}

template<typename dtype>
void dist_tensor<dtype>::free_type(CTF_ctr_type_t * type){
  free(type->idx_map_A);
  free(type->idx_map_B);
  free(type->idx_map_C);
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

  for (i=0; i<tsr_A->ndim; i++){
    if (type_A->idx_map_A[i] != type_B->idx_map_A[i]) return 0;
  }
  for (i=0; i<tsr_B->ndim; i++){
    if (type_A->idx_map_B[i] != type_B->idx_map_B[i]) return 0;
  }
  for (i=0; i<tsr_C->ndim; i++){
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
                                    dtype &                add_sign,
                                    int &                 mod){
  int  iA, jA, iB, iC, jB, jC, iiB, iiC, broken, tmp;

  //find all symmetries in A
  for (iA=0; iA<tsr_A->ndim; iA++){
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
            if (tsr_B->sym[iiB] != tsr_A->sym[iA]) broken = 1;
          }
        } 
        if ((iB == -1) ^ (jB == -1)) broken = 1;
        jC = idx_arr[3*idx_map_A[jA]+off_C];
        //if (iC == jC) broken = 1;
        if (iC != -1 && jC != -1){
          for (iiC=MIN(iC,jC); iiC<MAX(iC,jC); iiC++){
            if (tsr_C->sym[iiC] != tsr_A->sym[iA]) broken = 1;
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
  
  inv_idx(tsr_A->ndim, norm_ord_perm.idx_map_A, NULL,
          tsr_B->ndim, norm_ord_perm.idx_map_B, NULL,
          tsr_C->ndim, norm_ord_perm.idx_map_C, NULL,
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
/*    for (iB=0; iB<tsr_B->ndim; iB++){
      if (tsr_B->sym[iB] != NS){
        jB=iB;
        iC = idx_arr[3*norm_ord_perm.idx_map_B[iB]+2];
        iA = idx_arr[3*norm_ord_perm.idx_map_B[iB]];
        while (tsr_B->sym[jB] != NS){
          broken = 0;
          jB++;
          jC = idx_arr[3*norm_ord_perm.idx_map_B[jB]+2];
          jA = idx_arr[3*norm_ord_perm.idx_map_B[jB]];
          if (jA != -1) broken = 1;
          //if (iC == jC) broken = 1;
          if (iC != -1 && jC != -1){
            for (iiC=MIN(iC,jC); iiC<MAX(iC,jC); iiC++){
              if (tsr_C->sym[iiC] != tsr_B->sym[iB]) broken = 1;
            }
          } 
          if ((iC == -1) ^ (jC == -1)) broken = 1;
          //if the symmetry is preserved, make sure index map is ordered
          if (!broken){
            if (norm_ord_perm.idx_map_B[iB] > norm_ord_perm.idx_map_B[jB]){
              idx_arr[3*norm_ord_perm.idx_map_B[iB]+1] = jB;
              idx_arr[3*norm_ord_perm.idx_map_B[jB]+1] = iB;
              tmp                          = norm_ord_perm.idx_map_B[iB];
              norm_ord_perm.idx_map_B[iB] = norm_ord_perm.idx_map_B[jB];
              norm_ord_perm.idx_map_B[jB] = tmp;
              if (tsr_B->sym[iB] == AS) add_sign *= -1.0;
              mod = 1;
            } 
            if (iC != -1 && jC != -1 &&
                norm_ord_perm.idx_map_C[iC] > norm_ord_perm.idx_map_C[jC]){
              idx_arr[3*norm_ord_perm.idx_map_C[iC]+2] = jC;
              idx_arr[3*norm_ord_perm.idx_map_C[jC]+2] = iC;
              tmp                          = norm_ord_perm.idx_map_C[iC];
              norm_ord_perm.idx_map_C[iC] = norm_ord_perm.idx_map_C[jC];
              norm_ord_perm.idx_map_C[jC] = tmp;
              if (tsr_C->sym[iC] == AS) add_sign *= -1.0;
              mod = 1;
            } 
          }
        }
      }
    }
    for (iC=0; iC<tsr_C->ndim; iC++){
      if (tsr_C->sym[iC] != NS){
        jC=iC;
        iA = idx_arr[3*norm_ord_perm.idx_map_C[iC]];
        iB = idx_arr[3*norm_ord_perm.idx_map_C[iC]+1];
        if (iA != -1 || iB != -1) continue;
        while (tsr_C->sym[jC] != NS){
          broken = 0;
          jC++;
          jA = idx_arr[3*norm_ord_perm.idx_map_C[jC]];
          jB = idx_arr[3*norm_ord_perm.idx_map_C[jC]+1];
          if (jA != -1 || jB != -1) broken = 1;
          //if the symmetry is preserved, make sure index map is ordered
          if (!broken){
            if (norm_ord_perm.idx_map_C[iC] > norm_ord_perm.idx_map_C[jC]){
              idx_arr[3*norm_ord_perm.idx_map_C[iC]+2] = jC;
              idx_arr[3*norm_ord_perm.idx_map_C[jC]+2] = iC;
              tmp                          = norm_ord_perm.idx_map_C[iC];
              norm_ord_perm.idx_map_C[iC] = norm_ord_perm.idx_map_C[jC];
              norm_ord_perm.idx_map_C[jC] = tmp;
              if (tsr_C->sym[iC] == AS) add_sign *= -1.0;
              mod = 1;
            } 
          }
        }
      }
    }*/
  } while (mod);

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
//  get_buffer_space(sizeof(dtype)*ndim_C, (void**)&scl_alpha_C);
//  get_buffer_space(sizeof(int*)*ndim_C, (void**)&scl_idx_maps_C);

  int i, j, k, tmp;
  CTF_ctr_type_t new_type;
  dtype sign;
  tensor<dtype> * tsr_A, * tsr_B, * tsr_C;
  tsr_A = tensors[type->tid_A];
  tsr_B = tensors[type->tid_B];
  tsr_C = tensors[type->tid_C];

  copy_type(type, &new_type);
  add_sym_perm(perms, signs, &new_type, alpha);

  for (i=0; i<tsr_A->ndim; i++){
    j=i;
    while (tsr_A->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
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
  for (i=0; i<tsr_B->ndim; i++){
    j=i;
    while (tsr_B->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
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
  
  for (i=0; i<tsr_C->ndim; i++){
    j=i;
    while (tsr_C->sym[j] != NS){
      j++;
      for (k=0; k<(int)perms.size(); k++){
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



