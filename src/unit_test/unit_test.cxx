/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "unit_test.h"
#include "../shared/util.h"

/**
 * \brief zeros out tensor symmetric padding
 * \param[in,out] A tensor to fix-up
 * \param[in] ndim number of dimensions of tensor A
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetries of tensor
 */
void zeroout_tsr(double *       A,
                 int const      ndim,
                 int const *    edge_len,
                 int const *    sym){
  int * idx, * edge_lda;
  int act_lda, i, j, idx_offset, zero_out;

 
  idx = (int*)malloc(ndim*sizeof(int));
  memset(idx, 0, ndim*sizeof(int));
  edge_lda = (int*)malloc(sizeof(int)*ndim);
  edge_lda[0] = 1;
  for (i=1; i<ndim; i++){
    edge_lda[i] = edge_lda[i-1]*edge_len[i-1];
  }

  idx_offset = 0;
  for (;;){
    for (i=0; i<edge_len[0]; i++){
      zero_out = 0;
      for (j=0; j<ndim; j++){
        if ((sym[j] == 2 || sym[j] == 3) && idx[j] >= idx[j+1])
          zero_out = 1;
        if (sym[j] == 1 && idx[j] > idx[j+1])
          zero_out = 1;
      }
      if (zero_out)
        A[idx_offset+i] = 0.0;
    }
    for (act_lda=1; act_lda < ndim; act_lda++){
      idx_offset -= idx[act_lda]*edge_lda[act_lda];
      idx[act_lda]++;
      if (idx[act_lda] >= edge_len[act_lda])
        idx[act_lda] = 0;
      idx_offset += idx[act_lda]*edge_lda[act_lda];
    }
    if (act_lda >= ndim) break;
  }
}
    
/**
 * \brief packs or unpacks a tensor from nonsymmetric to 
 *      symmetric packed layout
 * \param[in] A tensor to read from
 * \param[in] ndim number of dimensions of tensor A
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] pup 1 if going to unpacked, 0 in going to packed
 * \param[in,out] ptr_new_sym a sym array with all -1s allocated if pup=1
 *                              used if pup=0
 * \param[out] ptr_up this pointer is set to point to newly set buffer
 */
void punpack_tsr(double const * A,
                 int const      ndim,
                 int const *    edge_len,
                 int const *    sym,
                 int const      pup,
                 int **         ptr_new_sym,
                 double **      ptr_up){
  int * idx, * edge_lda, * new_sym, * min_idx;
  int act_lda, act_max, i, imax, size, buf_offset, idx_offset;
  double * up;

 
  if (pup){ 
    new_sym = (int*)malloc(ndim*sizeof(int));
    std::fill(new_sym, new_sym+ndim, 0);
    size = 1;
    for (i=0; i<ndim; i++){
      size *= edge_len[i];
    }
    up = (double*)malloc(sizeof(double)*size);
    std::fill(up, up+size, 0.0);
  } else {
    new_sym = *ptr_new_sym;
    up = *ptr_up;
  }
  if (ndim == 0){
    *ptr_new_sym = new_sym;
    up[0] = A[0];
    *ptr_up = up;
    return;
  }

  idx = (int*)malloc(ndim*sizeof(int));
  min_idx = (int*)malloc(ndim*sizeof(int));
  memset(idx, 0, ndim*sizeof(int));
  edge_lda = (int*)malloc(sizeof(int)*ndim);
  edge_lda[0] = 1;
  for (i=1; i<ndim; i++){
    edge_lda[i] = edge_lda[i-1]*edge_len[i-1];
  }
  idx_offset = 0;
  for (i=0; i<ndim; i++){
    if(sym[i] >= 2)
      idx[i+1] = idx[i]+1;
    idx_offset += idx[i]*edge_lda[i];
    if (idx[i] >= edge_len[i]) {
      if (pup){ 
        *ptr_new_sym = new_sym;
        *ptr_up = up;
      }
      return;
    }
  }
  memcpy(min_idx, idx, ndim*sizeof(int));

  buf_offset = 0;
  imax = edge_len[0];
  for (;;){
    if(sym[0] == 1)
      imax = idx[1]+1;
    if(sym[0] >= 2)
      imax = idx[1];
    for (i=0; i<imax; i++){
      if (pup)
        up[idx_offset+i] = A[buf_offset+i];
      else
        up[buf_offset+i] = A[idx_offset+i];
    }
//    printf("buf_offset = %d\n",buf_offset);
    buf_offset +=imax;
    for (act_lda=1; act_lda < ndim; act_lda++){
      idx_offset -= idx[act_lda]*edge_lda[act_lda];
      idx[act_lda]++;
      act_max = edge_len[act_lda];
      if (sym[act_lda] == 1) act_max = idx[act_lda+1]+1;
      if (sym[act_lda] >= 2) act_max = idx[act_lda+1];
      if (idx[act_lda] >= act_max)
        idx[act_lda] = min_idx[act_lda];
      idx_offset += idx[act_lda]*edge_lda[act_lda];
      if (idx[act_lda] > min_idx[act_lda])
        break;
    }
    if (act_lda >= ndim) break;
  }
  
  if (pup){ 
    *ptr_new_sym = new_sym;
    *ptr_up = up;
  }
  
}

/**
 * \brief naive sequential non-symmetric contraction function
 */
int sim_seq_ctr(double const    alpha,
               double const *   A,
               int const        ndim_A,
               int const *      edge_len_A,
               int const *      _lda_A,
               int const *      sym_A,
               int const *      idx_map_A,
               double const *   B,
               int const        ndim_B,
               int const *      edge_len_B,
               int const *      _lda_B,
               int const *      sym_B,
               int const *      idx_map_B,
               double const     beta,
               double *         C,
               int const        ndim_C,
               int const *      edge_len_C,
               int const *      _lda_C,
               int const *      sym_C,
               int const *      idx_map_C){
  int * idx_arr, * lda_A, * lda_B, * lda_C, * beta_arr, * edge_len_arr;
  int i, ndim_tot, off_A, off_B, off_C, nb_A, nb_B, nb_C; 
  double * dC;
  double dbeta, dA, dB;

  ndim_tot=-1;
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] > ndim_tot) ndim_tot = idx_map_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_map_B[i] > ndim_tot) ndim_tot = idx_map_B[i];
  }
  for (i=0; i<ndim_C; i++){
    if (idx_map_C[i] > ndim_tot) ndim_tot = idx_map_C[i];
  }
  ndim_tot++;

  idx_arr = (int*)malloc(sizeof(int)*ndim_tot);
  edge_len_arr = (int*)malloc(sizeof(int)*ndim_tot);
  lda_A = (int*)malloc(sizeof(int)*ndim_tot);
  lda_B = (int*)malloc(sizeof(int)*ndim_tot);
  lda_C = (int*)malloc(sizeof(int)*ndim_tot);

  for (i=0; i<ndim_A; i++){
    edge_len_arr[idx_map_A[i]] = edge_len_A[i];
  }
  for (i=0; i<ndim_B; i++){
    edge_len_arr[idx_map_B[i]] = edge_len_B[i];
  }
  /*printf("[%d] [%d] [%d] [%d] == [%d] [%d] [%d] [%d]\n",
          edge_len_arr[idx_map_C[0]],
          edge_len_arr[idx_map_C[1]],
          edge_len_arr[idx_map_C[2]],
          edge_len_arr[idx_map_C[3]],
          edge_len_C[0],
          edge_len_C[1],
          edge_len_C[2],
          edge_len_C[3]);*/
  for (i=0; i<ndim_C; i++){
    LIBT_ASSERT(edge_len_arr[idx_map_C[i]] == edge_len_C[i]);
  }

#define SET_LDA_X(__X)                                          \
  do {                                                          \
    memset(lda_##__X, 0, sizeof(int)*ndim_tot);                 \
    nb_##__X = 1;                                               \
    for (i=0; i<ndim_##__X; i++){                               \
      lda_##__X[idx_map_##__X[i]] += nb_##__X;                  \
      nb_##__X = nb_##__X*edge_len_##__X[i];                    \
    }                                                           \
  } while (0)
  SET_LDA_X(A);
  SET_LDA_X(B);
  SET_LDA_X(C);
#undef SET_LDA_X
   
  /* dynammically determined size */ 
  beta_arr = (int*)malloc(sizeof(int)*nb_C);
  memset(beta_arr,0,sizeof(int)*nb_C);
  memset(idx_arr,0,sizeof(int)*ndim_tot);
  off_A = 0; 
  off_B = 0; 
  off_C = 0;

  for (;;){
    dA  = A[off_A];
    dB  = B[off_B];
    dC  = C + off_C;

    assert(nb_C>off_C);
    dbeta               = beta_arr[off_C]>0 ? 1.0 : beta;
    beta_arr[off_C]     = 1;

  //  printf("off_C = %d\n",off_C);
    (*dC) = dbeta*(*dC);
    (*dC) += alpha*dA*dB;

    for (i=0; i<ndim_tot; i++){
      off_A -= lda_A[i]*idx_arr[i];
      off_B -= lda_B[i]*idx_arr[i];
      off_C -= lda_C[i]*idx_arr[i];
      idx_arr[i]++;
      if (idx_arr[i] >= edge_len_arr[i])
        idx_arr[i] = 0;
      off_A += lda_A[i]*idx_arr[i];
      off_B += lda_B[i]*idx_arr[i];
      off_C += lda_C[i]*idx_arr[i];
      if (idx_arr[i] != 0) break;
    }
    if (i==ndim_tot) break;
  }
  free(idx_arr);
  free(edge_len_arr);
  free(lda_A);
  free(lda_B);
  free(lda_C);
  free(beta_arr);
  return 0;
}



/**
 * \brief performs symmetric contraction by unpacking, 
 *      useful for correctness, but very slow
 */
int cpy_sym_ctr(double const    alpha,
               double const *   A,
               int const        ndim_A,
               int const *      edge_len_A,
               int const *      lda_A,
               int const *      sym_A,
               int const *      idx_map_A,
               double const *   B,
               int const        ndim_B,
               int const *      edge_len_B,
               int const *      lda_B,
               int const *      sym_B,
               int const *      idx_map_B,
               double const     beta,
               double *         C,
               int const        ndim_C,
               int const *      edge_len_C,
               int const *      lda_C,
               int const *      sym_C,
               int const *      idx_map_C){
  double * up_A, * up_B, * up_C;
  int * new_sym_A, * new_sym_B, * new_sym_C;

  punpack_tsr(A, ndim_A, edge_len_A, sym_A, 1,
              &new_sym_A, &up_A);
  punpack_tsr(B, ndim_B, edge_len_B, sym_B, 1,
              &new_sym_B, &up_B);
  punpack_tsr(C, ndim_C, edge_len_C, sym_C, 1,
              &new_sym_C, &up_C);

  sim_seq_ctr(alpha, up_A, ndim_A, edge_len_A, edge_len_A,
              new_sym_A, idx_map_A,
              up_B, ndim_B, edge_len_B, edge_len_B, 
              new_sym_B, idx_map_B,
              beta, up_C, ndim_C, edge_len_C, edge_len_C,
              new_sym_C, idx_map_C);

  //zeroout_tsr(up_C, ndim_C, edge_len_C, sym_C);

  punpack_tsr(up_C, ndim_C, edge_len_C, sym_C, 0,
              &new_sym_C, &C);
  free(up_A);
  free(up_B);
  free(up_C);
  free(new_sym_A);
  free(new_sym_B);
  free(new_sym_C);
  return 0;
}

/**
 * \brief performs symmetric summation by unpacking, 
 *      useful for correctness, but very slow
 */
int  cpy_sym_sum(double const   alpha,
                 double const * A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    lda_A,
                 int const *    sym_A,
                 int const *    idx_map_A,
                 double const   beta,
                 double *       B,
                 int const      ndim_B,
                 int const *    edge_len_B,
                 int const *    lda_B,
                 int const *    sym_B,
                 int const *    idx_map_B){

  double * up_A, * up_B;
  int * new_sym_A, * new_sym_B;

  punpack_tsr(A, ndim_A, edge_len_A, sym_A, 1,
              &new_sym_A, &up_A);
  punpack_tsr(B, ndim_B, edge_len_B, sym_B, 1,
              &new_sym_B, &up_B);

  sim_seq_sum(alpha, up_A, ndim_A, edge_len_A, 
              new_sym_A, idx_map_A,
              beta, up_B, ndim_B, edge_len_B, 
              new_sym_B, idx_map_B);

  punpack_tsr(up_B, ndim_B, edge_len_B, sym_B, 0,
              &new_sym_B, &B);
  free(up_A);
  free(up_B);
  free(new_sym_A);
  free(new_sym_B);
  return 0;
}
/**
 * \brief naive sequential non-symmetric summation function
 */
int  sim_seq_sum(double const   alpha,
                 double const * A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    sym_A,
                 int const *    idx_map_A,
                 double const   beta,
                 double *       B,
                 int const      ndim_B,
                 int const *    edge_len_B,
                 int const *    sym_B,
                 int const *    idx_map_B){
  int * idx_arr, * lda_A, * lda_B, * beta_arr, * edge_len_arr;
  int i, ndim_tot, off_A, off_B, nb_A, nb_B; 
  double * dB;
  double dbeta, dA;

  ndim_tot=-1;
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] > ndim_tot) ndim_tot = idx_map_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_map_B[i] > ndim_tot) ndim_tot = idx_map_B[i];
  }
  ndim_tot++;

  idx_arr = (int*)malloc(sizeof(int)*ndim_tot);
  edge_len_arr = (int*)malloc(sizeof(int)*ndim_tot);
  lda_A = (int*)malloc(sizeof(int)*ndim_tot);
  lda_B = (int*)malloc(sizeof(int)*ndim_tot);

  for (i=0; i<ndim_A; i++){
    edge_len_arr[idx_map_A[i]] = edge_len_A[i];
  }
  for (i=0; i<ndim_B; i++){
    edge_len_arr[idx_map_B[i]] = edge_len_B[i];
  }

#define SET_LDA_X(__X)                                          \
  do {                                                          \
    memset(lda_##__X, 0, sizeof(int)*ndim_tot);                 \
    nb_##__X = 1;                                               \
    for (i=0; i<ndim_##__X; i++){                               \
      lda_##__X[idx_map_##__X[i]] += nb_##__X;                  \
      nb_##__X = nb_##__X*edge_len_##__X[i];                    \
    }                                                           \
  } while (0)
  SET_LDA_X(A);
  SET_LDA_X(B);
#undef SET_LDA_X
   
  /* dynammically determined size */ 
  beta_arr = (int*)malloc(sizeof(int)*nb_B);
  memset(beta_arr,0,sizeof(int)*nb_B);
  memset(idx_arr,0,sizeof(int)*ndim_tot);
  off_A = 0; 
  off_B = 0; 


  for (;;){
    dA  = A[off_A];
    dB  = B + off_B;

    assert(nb_B>off_B);
    dbeta               = beta_arr[off_B]>0 ? 1.0 : beta;
    beta_arr[off_B]     = 1;

    (*dB) = dbeta*(*dB) + alpha*dA;

    for (i=0; i<ndim_tot; i++){
      off_A -= lda_A[i]*idx_arr[i];
      off_B -= lda_B[i]*idx_arr[i];
      idx_arr[i]++;
      if (idx_arr[i] >= edge_len_arr[i])
        idx_arr[i] = 0;
      off_A += lda_A[i]*idx_arr[i];
      off_B += lda_B[i]*idx_arr[i];
      if (idx_arr[i] != 0) break;
    }
    if (i==ndim_tot) break;
  }
  free(idx_arr);
  free(edge_len_arr);
  free(lda_A);
  free(lda_B);
  free(beta_arr);
  return 0;
}






