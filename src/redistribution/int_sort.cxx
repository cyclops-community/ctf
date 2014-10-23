/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "int_sort.h"
#include "../shared/util.h"

namespace CTF_int {
  void permute_keys(int                         order,
                    int                         num_pair,
                    int const *                 edge_len,
                    int const *                 new_edge_len,
                    int * const *               permutation,
                    pair                        pairs,
                    int64_t *                   new_num_pair,
                    semiring                    sr){
    TAU_FSTART(permute_keys);
  #ifdef USE_OMP
    int mntd = omp_get_max_threads();
  #else
    int mntd = 1;
  #endif
    int64_t counts[mntd];
    std::fill(counts,counts+mntd,0);
  #ifdef USE_OMP
    #pragma omp parallel
  #endif
    { 
      int i, j, tid, ntd, outside;
      int64_t lda, wkey, knew, kdim, tstart, tnum_pair, cnum_pair;
  #ifdef USE_OMP
      tid = omp_get_thread_num();
      ntd = omp_get_num_threads();
  #else
      tid = 0;
      ntd = 1;
  #endif
      tnum_pair = num_pair/ntd;
      tstart = tnum_pair * tid + MIN(tid, num_pair % ntd);
      if (tid < num_pair % ntd) tnum_pair++;

      std::vector< tkv_pair<dtype> > my_pairs;
      cnum_pair = 0;

      for (i=tstart; i<tstart+tnum_pair; i++){
        wkey = pairs[i].k;
        lda = 1;
        knew = 0;
        outside = 0;
        for (j=0; j<order; j++){
          kdim = wkey%edge_len[j];
          if (permutation[j] != NULL){
            if (permutation[j][kdim] == -1){
              outside = 1;
            } else{
              knew += lda*permutation[j][kdim];
            }
          } else {
            knew += lda*kdim;
          }
          lda *= new_edge_len[j];
          wkey = wkey/edge_len[j];
        }
        if (!outside){
          tkv_pair<dtype> tkp;
          tkp.k = knew;
          tkp.d = pairs[i].d;
          cnum_pair++;
          my_pairs.push_back(tkp);
        }
      }
      counts[tid] = cnum_pair;
      {
  #ifdef USE_OMP
        #pragma omp barrier
  #endif
        int64_t pfx = 0;
        for (i=0; i<tid; i++){
          pfx += counts[i];
        }
        std::copy(my_pairs.begin(),my_pairs.begin()+cnum_pair,pairs+pfx);
        my_pairs.clear();
      }
    } 
    *new_num_pair = 0;
    for (int i=0; i<mntd; i++){
      *new_num_pair += counts[i];
    }
    TAU_FSTOP(permute_keys);
  }

  void depermute_keys(int                         order,
                      int                         num_pair,
                      int const *                 edge_len,
                      int const *                 new_edge_len,
                      int * const *               permutation,
                      pair *                      pairs,
                      semiring                    sr){
    TAU_FSTART(depermute_keys);
  #ifdef USE_OMP
    int mntd = omp_get_max_threads();
  #else
    int mntd = 1;
  #endif
    int64_t counts[mntd];
    std::fill(counts,counts+mntd,0);
    int ** depermutation = (int**)CTF_alloc(order*sizeof(int*));
    TAU_FSTART(form_depermutation);
    for (int d=0; d<order; d++){
      if (permutation[d] == NULL){
        depermutation[d] = NULL;
      } else {
        depermutation[d] = (int*)CTF_alloc(new_edge_len[d]*sizeof(int));
        std::fill(depermutation[d],depermutation[d]+new_edge_len[d], -1);
        for (int i=0; i<edge_len[d]; i++){
          depermutation[d][permutation[d][i]] = i;
        }
      }
    }
    TAU_FSTOP(form_depermutation);
  #ifdef USE_OMP
    #pragma omp parallel
  #endif
    { 
      int i, j, tid, ntd;
      int64_t lda, wkey, knew, kdim, tstart, tnum_pair;
  #ifdef USE_OMP
      tid = omp_get_thread_num();
      ntd = omp_get_num_threads();
  #else
      tid = 0;
      ntd = 1;
  #endif
      tnum_pair = num_pair/ntd;
      tstart = tnum_pair * tid + MIN(tid, num_pair % ntd);
      if (tid < num_pair % ntd) tnum_pair++;

      std::vector< tkv_pair<dtype> > my_pairs;

      for (i=tstart; i<tstart+tnum_pair; i++){
        wkey = pairs[i].k;
        lda = 1;
        knew = 0;
        for (j=0; j<order; j++){
          kdim = wkey%new_edge_len[j];
          if (depermutation[j] != NULL){
            ASSERT(depermutation[j][kdim] != -1);
            knew += lda*depermutation[j][kdim];
          } else {
            knew += lda*kdim;
          }
          lda *= edge_len[j];
          wkey = wkey/new_edge_len[j];
        }
        pairs[i].k = knew;
      }
    }
    for (int d=0; d<order; d++){
      if (permutation[d] != NULL)
        CTF_free(depermutation[d]);
    }
    CTF_free(depermutation);

    TAU_FSTOP(depermute_keys);
  }
}
