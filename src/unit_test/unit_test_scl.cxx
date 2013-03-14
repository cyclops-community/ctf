/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "unit_test.h"
#include "unit_test_scl.h"
#include "../shared/util.h"

/**
 * \brief main function for unit tests
 */
int main(int argc, char **argv){
  int myRank, numPes;
  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);
  test_scale(argc, argv, numPes, myRank, cdt_glb);
  COMM_EXIT;
}

/**
 * \brief naive sequential non-symmetric scale function
 */
int  sim_seq_scl(double const   alpha,
                 double *       A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    sym_A,
                 int const *    idx_map_A){
  int * idx_arr, * lda_A, * edge_len_arr;
  int i, ndim_tot, off_A, nb_A; 

  ndim_tot=-1;
  for (i=0; i<ndim_A; i++){
    if (idx_map_A[i] > ndim_tot) ndim_tot = idx_map_A[i];
  }
  ndim_tot++;

  idx_arr = (int*)malloc(sizeof(int)*ndim_tot);
  edge_len_arr = (int*)malloc(sizeof(int)*ndim_tot);
  lda_A = (int*)malloc(sizeof(int)*ndim_tot);

  for (i=0; i<ndim_A; i++){
    edge_len_arr[idx_map_A[i]] = edge_len_A[i];
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
#undef SET_LDA_X
   
  /* dynammically determined size */ 
  memset(idx_arr,0,sizeof(int)*ndim_tot);
  off_A = 0; 



  for (;;){
    A[off_A] *= alpha;

    for (i=0; i<ndim_tot; i++){
      off_A -= lda_A[i]*idx_arr[i];
      idx_arr[i]++;
      if (idx_arr[i] >= edge_len_arr[i])
        idx_arr[i] = 0;
      off_A += lda_A[i]*idx_arr[i];
      if (idx_arr[i] != 0) break;
    }
    if (i==ndim_tot) break;
  }
  free(idx_arr);
  free(edge_len_arr);
  free(lda_A);
  return 0;
}






/**
 * \brief performs symmetric scale by unpacking, 
 *      useful for correctness, but very slow
 */
int  cpy_sym_scl(double const   alpha,
                 double *       A,
                 int const      ndim_A,
                 int const *    edge_len_A,
                 int const *    lda_A,
                 int const *    sym_A,
                 int const *    idx_map_A){

  double * up_A;
  int * new_sym_A;

  punpack_tsr(A, ndim_A, edge_len_A, sym_A, 1,
              &new_sym_A, &up_A);

  sim_seq_scl(alpha, up_A, ndim_A, edge_len_A, 
              new_sym_A, idx_map_A);

  punpack_tsr(up_A, ndim_A, edge_len_A, sym_A, 0,
              &new_sym_A, &A);
  free(up_A);
  free(new_sym_A);
  return 0;
}
