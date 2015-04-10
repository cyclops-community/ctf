/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include "assert.h"
#include "util.h"

namespace CTF_int {
  int64_t sy_packed_size(const int order, const int* len, const int* sym){
    int i, k, mp;
    int64_t size, tmp;

    if (order == 0) return 1;

    k = 1;
    tmp = 1;
    size = 1;
    mp = len[0];
    for (i = 0;i < order;i++){
      tmp = (tmp * mp) / k;
      k++;
      mp ++;
      
      if (sym[i] == 0){
        size *= tmp;
        k = 1;
        tmp = 1;
        if (i < order - 1) mp = len[i + 1];
      }
    }
    size *= tmp;

    return size;
  }


  int64_t packed_size(const int order, const int* len, const int* sym){

    int i, k, mp;
    int64_t size, tmp;

    if (order == 0) return 1;

    k = 1;
    tmp = 1;
    size = 1;
    if (order > 0)
      mp = len[0];
    else
      mp = 1;
    for (i = 0;i < order;i++){
      tmp = (tmp * mp) / k;
      k++;
      if (sym[i] != 1)
        mp--;
      else
        mp ++;
      
      if (sym[i] == 0){
        size *= tmp;
        k = 1;
        tmp = 1;
        if (i < order - 1) mp = len[i + 1];
      }
    }
    size *= tmp;

    return size;
  }

  void calc_idx_arr(int         order,
                    int const * lens,
                    int const * sym,
                    int64_t     idx,
                    int *       idx_arr){
    int64_t idx_rem = idx;
    memset(idx_arr, 0, order*sizeof(int));
    for (int dim=order-1; dim>=0; dim--){
      if (idx_rem == 0) break;
      if (dim == 0 || sym[dim-1] == NS){
        int64_t lda = packed_size(dim, lens, sym);
        idx_arr[dim] = idx_rem/lda;
        idx_rem -= idx_arr[dim]*lda;
      } else {
        int plen[dim+1];
        memcpy(plen, lens, (dim+1)*sizeof(int));
        int sg = 2;
        int fsg = 2;
        while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
        int64_t lda = packed_size(dim-sg+1, lens, sym);
        double fsg_idx = (((double)idx_rem)*fsg)/lda;
        int kidx = (int)pow(fsg_idx,1./sg);
        //if (sym[dim-1] != SY) 
        kidx += sg+1;
        int mkidx = kidx;
  #if DEBUG >= 1
        for (int idim=dim-sg+1; idim<=dim; idim++){
          plen[idim] = mkidx+1;
        }
        int64_t smidx = packed_size(dim+1, plen, sym);
        ASSERT(smidx > idx_rem);
  #endif
        int64_t midx = 0;
        for (; mkidx >= 0; mkidx--){
          for (int idim=dim-sg+1; idim<=dim; idim++){
            plen[idim] = mkidx;
          }
          midx = packed_size(dim+1, plen, sym);
          if (midx <= idx_rem) break;
        }
        if (midx == 0) mkidx = 0;
        idx_arr[dim] = mkidx;
        idx_rem -= midx;
      }
    }
    ASSERT(idx_rem == 0);
  }


  void sy_calc_idx_arr(int         order,
                       int const * lens,
                       int const * sym,
                       int64_t     idx,
                       int *       idx_arr){
    int64_t idx_rem = idx;
    memset(idx_arr, 0, order*sizeof(int));
    for (int dim=order-1; dim>=0; dim--){
      if (idx_rem == 0) break;
      if (dim == 0 || sym[dim-1] == NS){
        int64_t lda = sy_packed_size(dim, lens, sym);
        idx_arr[dim] = idx_rem/lda;
        idx_rem -= idx_arr[dim]*lda;
      } else {
        int plen[dim+1];
        memcpy(plen, lens, (dim+1)*sizeof(int));
        int sg = 2;
        int fsg = 2;
        while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
        int64_t lda = sy_packed_size(dim-sg+1, lens, sym);
        double fsg_idx = (((double)idx_rem)*fsg)/lda;
        int kidx = (int)pow(fsg_idx,1./sg);
        //if (sym[dim-1] != SY) 
        kidx += sg+1;
        int mkidx = kidx;
  #if DEBUG >= 1
        for (int idim=dim-sg+1; idim<=dim; idim++){
          plen[idim] = mkidx+1;
        }
        int64_t smidx = sy_packed_size(dim+1, plen, sym);
        ASSERT(smidx > idx_rem);
  #endif
        int64_t midx = 0;
        for (; mkidx >= 0; mkidx--){
          for (int idim=dim-sg+1; idim<=dim; idim++){
            plen[idim] = mkidx;
          }
          midx = sy_packed_size(dim+1, plen, sym);
          if (midx <= idx_rem) break;
        }
        if (midx == 0) mkidx = 0;
        idx_arr[dim] = mkidx;
        idx_rem -= midx;
      }
    }
    ASSERT(idx_rem == 0);
  }


  void factorize(int n, int *nfactor, int **factor){
    int tmp, nf, i;
    int * ff;
    tmp = n;
    nf = 0;
    while (tmp > 1){
      for (i=2; i<=n; i++){
        if (tmp % i == 0){
          nf++;
          tmp = tmp/i;
          break;
        }
      }
    }
    if (nf == 0){
      *nfactor = nf;
    } else {
      ff  = (int*)CTF_int::alloc(sizeof(int)*nf);
      tmp = n;
      nf = 0;
      while (tmp > 1){
        for (i=2; i<=n; i++){
          if (tmp % i == 0){
            ff[nf] = i;
            nf++;
            tmp = tmp/i;
            break;
          }
        }
      }
      *factor = ff;
      *nfactor = nf;
    }
  }

  void permute(int          order,
               int const *  perm,
               int *        arr){
    int i;
    int * swap;
    CTF_int::alloc_ptr(order*sizeof(int), (void**)&swap);

    for (i=0; i<order; i++){
      swap[i] = arr[perm[i]];
    }
    for (i=0; i<order; i++){
      arr[i] = swap[i];
    }

    CTF_int::cfree(swap);
  }

  void permute_target(int         order,
                      int const * perm,
                      int *       arr){
    int i;
    int * swap;
    CTF_int::alloc_ptr(order*sizeof(int), (void**)&swap);

    for (i=0; i<order; i++){
      swap[i] = arr[perm[i]];
    }
    for (i=0; i<order; i++){
      arr[i] = swap[i];
    }

    CTF_int::cfree(swap);
  }

}
