#ifndef __ITER_TSR_H__
#define __ITER_TSR_H__

#include "util.h"
namespace CTF_int{
  class algstrct;
  //lives in contraction/sym_seq_ctr
  void compute_syoff(int              r,
                     int64_t          len,
                     algstrct const * sr,
                     int64_t const *  edge_len,
                     int const *      sym,
                     uint64_t *       offsets);
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
        for (i=0; i<order_##__X; i++){               \
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
        for (i=0; i<order_##__X; i++){               \
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
        for (i=1; i<order_##__X; i++){                                   \
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
