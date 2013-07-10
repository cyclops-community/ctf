/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __SYM_SEQ_CTR_SHARED_HXX__
#define __SYM_SEQ_CTR_SHARED_HXX__
#include "../shared/util.h"
#include <limits.h>


/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[in] ndim_C number of dimensions of C
 * \param[in] idx_C index map of C
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 3*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             int const          ndim_B,
             int const *        idx_B,
             int const          ndim_C,
             int const *        idx_C,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_B[i] > dim_max) dim_max = idx_B[i];
  }
  for (i=0; i<ndim_C; i++){
    if (idx_C[i] > dim_max) dim_max = idx_C[i];
  }
  dim_max++;
  *ndim_tot = dim_max;
  *idx_arr = (int*)CTF_alloc(sizeof(int)*3*dim_max);
  std::fill((*idx_arr), (*idx_arr)+3*dim_max, -1);  

  for (i=0; i<ndim_A; i++){
    (*idx_arr)[3*idx_A[i]] = i;
  }
  for (i=0; i<ndim_B; i++){
    (*idx_arr)[3*idx_B[i]+1] = i;
  }
  for (i=0; i<ndim_C; i++){
    (*idx_arr)[3*idx_C[i]+2] = i;
  }
}

/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 2*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             int const          ndim_B,
             int const *        idx_B,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_B[i] > dim_max) dim_max = idx_B[i];
  }
  dim_max++;
  *ndim_tot = dim_max;
  *idx_arr = (int*)CTF_alloc(sizeof(int)*2*dim_max);
  std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);  

  for (i=0; i<ndim_A; i++){
    (*idx_arr)[2*idx_A[i]] = i;
  }
  for (i=0; i<ndim_B; i++){
    (*idx_arr)[2*idx_B[i]+1] = i;
  }
}


/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 2*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  dim_max++;
  *ndim_tot = dim_max;
  *idx_arr = (int*)CTF_alloc(sizeof(int)*dim_max);
  std::fill((*idx_arr), (*idx_arr)+dim_max, -1);  

  for (i=0; i<ndim_A; i++){
    (*idx_arr)[idx_A[i]] = i;
  }
}


#define GET_MIN_MAX(__X,nr,wd)                                                  \
do{                                                                             \
      i##__X = rev_idx_map[wd*idx+nr];                                          \
      if (i##__X != -1){                                                        \
        imax = MIN(imax, edge_len_##__X[i##__X]);                               \
        /*if (sym_##__X[i##__X] > -1){                                          \
          imax = MIN(imax, idx_glb[idx_map_##__X[sym_##__X[i##__X]]]+1);        \
        }                                                                       \
        if (i##__X > 0 && sym_##__X[i##__X-1] > -1){                            \
          imin = MAX(imin, idx_glb[idx_map_##__X[i##__X-1]]);                   \
        }*/                                                                     \
      }                                                                         \
} while (0);

#ifdef SEQ
#define CHECK_SYM(__X)                              \
do {                                                \
        sym_pass = 1;                               \
        for (i=0; i<ndim_##__X; i++){               \
          if (sym_##__X[i] == AS || sym_##__X[i] == SH){         \
            if (idx_glb[idx_map_##__X[i+1]] <=      \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
          if (sym_##__X[i] == SY){                  \
            if (idx_glb[idx_map_##__X[i+1]] <       \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
        }                                           \
} while(0)
#else
#define CHECK_SYM(__X)                              \
do {                                                \
        sym_pass = 1;                               \
        for (i=0; i<ndim_##__X; i++){               \
          if (sym_##__X[i] != NS){                  \
            if (idx_glb[idx_map_##__X[i+1]] <       \
                      idx_glb[idx_map_##__X[i]]) {  \
              sym_pass = 0;                         \
              break;                                \
            }                                       \
          }                                         \
        }                                           \
} while(0)
#endif

#define RESET_IDX(__X)                                                  \
do {                                                                    \
        idx_##__X = idx_glb[idx_map_##__X[0]];                          \
        off_idx = 0, off_lda = 1;                                       \
        for (i=1; i<ndim_##__X; i++){                                   \
          if (sym_##__X[i-1] == NS){                                    \
            off_idx = i;                                                \
            off_lda = sy_packed_size(i, dlen_##__X, sym_##__X); \
            idx_##__X += off_lda*idx_glb[idx_map_##__X[i]];             \
          } else if (idx_glb[idx_map_##__X[i]]!=0) {                    \
            k = 1;                                                      \
            dlen_##__X[i] = idx_glb[idx_map_##__X[i]];                  \
            do {                                                        \
              dlen_##__X[i-k] = idx_glb[idx_map_##__X[i]];              \
              k++;                                                      \
            } while (i>=k && sym_##__X[i-k] != NS);                     \
            idx_##__X += off_lda*sy_packed_size(i+1-off_idx,            \
                          dlen_##__X+off_idx,sym_##__X+off_idx);        \
            for (j=0; j<k; j++){                                        \
              dlen_##__X[i-j] = edge_len_##__X[i-j];                    \
            }                                                           \
          }                                                             \
        }                                                               \
} while (0)

#endif
