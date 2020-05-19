/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include <assert.h>
#include "util.h"

namespace CTF_int {
  int64_t sy_packed_size(int order, const int64_t * len, const int* sym){
    int i, k;
    int64_t size, mp, tmp;

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


  int64_t packed_size(int order, const int64_t * len, const int* sym){

    int i, k;
    int64_t size, mp, tmp;

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

  void calc_idx_arr(int             order,
                    int64_t const * lens,
                    int const *     sym,
                    int64_t         idx,
                    int64_t *       idx_arr){
    int64_t idx_rem = idx;
    memset(idx_arr, 0, order*sizeof(int64_t));
    for (int dim=order-1; dim>=0; dim--){
      if (idx_rem == 0) break;
      if (dim == 0 || sym[dim-1] == NS){
        int64_t lda = packed_size(dim, lens, sym);
        idx_arr[dim] = idx_rem/lda;
        idx_rem -= idx_arr[dim]*lda;
      } else {
        int64_t plen[dim+1];
        memcpy(plen, lens, (dim+1)*sizeof(int64_t));
        int sg = 2;
        int64_t fsg = 2;
        while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
        int64_t lda = packed_size(dim-sg+1, lens, sym);
        double fsg_idx = (((double)idx_rem)*fsg)/lda;
        int64_t kidx = (int64_t)pow(fsg_idx,1./sg);
        //if (sym[dim-1] != SY) 
        kidx += sg+1;
        int64_t mkidx = kidx;
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


  void sy_calc_idx_arr(int             order,
                       int64_t const * lens,
                       int const *     sym,
                       int64_t         idx,
                       int64_t *       idx_arr){
    int64_t idx_rem = idx;
    memset(idx_arr, 0, order*sizeof(int64_t));
    for (int dim=order-1; dim>=0; dim--){
      if (idx_rem == 0) break;
      if (dim == 0 || sym[dim-1] == NS){
        int64_t lda = sy_packed_size(dim, lens, sym);
        idx_arr[dim] = idx_rem/lda;
        idx_rem -= idx_arr[dim]*lda;
      } else {
        int64_t plen[dim+1];
        memcpy(plen, lens, (dim+1)*sizeof(int));
        int sg = 2;
        int64_t fsg = 2;
        while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
        int64_t lda = sy_packed_size(dim-sg+1, lens, sym);
        double fsg_idx = (((double)idx_rem)*fsg)/lda;
        int64_t kidx = (int64_t)pow(fsg_idx,1./sg);
        //if (sym[dim-1] != SY) 
        kidx += sg+1;
        int64_t mkidx = kidx;
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

    CTF_int::cdealloc(swap);
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

    CTF_int::cdealloc(swap);
  }

  void permute(int          order,
               int const *  perm,
               int64_t *    arr){
    int i;
    int64_t * swap;
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&swap);

    for (i=0; i<order; i++){
      swap[i] = arr[perm[i]];
    }
    for (i=0; i<order; i++){
      arr[i] = swap[i];
    }

    CTF_int::cdealloc(swap);
  }

  void permute_target(int         order,
                      int const * perm,
                      int64_t *   arr){
    int i;
    int64_t * swap;
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&swap);

    for (i=0; i<order; i++){
      swap[i] = arr[perm[i]];
    }
    for (i=0; i<order; i++){
      arr[i] = swap[i];
    }

    CTF_int::cdealloc(swap);
  }


  void socopy(int64_t         m,
              int64_t         n,
              int64_t         lda_a,
              int64_t         lda_b,
              int64_t const * sizes_a,
              int64_t *&      sizes_b,
              int64_t *&      offsets_b){
    sizes_b = (int64_t*)alloc(sizeof(int64_t)*m*n);
    offsets_b = (int64_t*)alloc(sizeof(int64_t)*m*n);

    int64_t last_offset = 0;
    for (int i=0; i<n; i++){
      for (int j=0; j<m; j++){
        sizes_b[lda_b*i+j]    = sizes_a[lda_a*i+j];
        offsets_b[lda_b*i+j]  = last_offset;
        last_offset           = last_offset+sizes_a[lda_a*i+j];
      }
    }
  }

  void spcopy(int64_t         m,
              int64_t         n,
              int64_t         lda_a,
              int64_t         lda_b,
              int64_t const * sizes_a,
              int64_t const * offsets_a,
              char const *    a,
              int64_t const * sizes_b,
              int64_t const * offsets_b,
              char *          b){
    for (int i=0; i<n; i++){
      for (int j=0; j<m; j++){
        memcpy(b+offsets_b[lda_b*i+j],a+offsets_a[lda_a*i+j],sizes_a[lda_a*i+j]);
      }
    }
  }
     
  int64_t fact(int64_t n){
    int64_t f = 1;
    for (int64_t i=1; i<=n; i++){
      f*=i;
    }
    return f;
  }
  
  int64_t choose(int64_t n, int64_t k){
    return fact(n)/(fact(k)*fact(n-k));
  }

  void get_choice(int n, int k, int ch, int64_t * chs){
    if (k==0) return;
    if (k==1){
      chs[0] = ch;
      return;
    }
    int64_t lens[k];
    std::fill(lens, lens+k, n);
    int sym[k];
    std::fill(sym, sym+k-1, SH);
    sym[k-1] = NS;

    calc_idx_arr(k,lens,sym,ch,chs);
    //FIXME add 1?
  }
  
  int64_t chchoose(int64_t n, int64_t k){
    return fact(n+k-1)/(fact(k)*fact(n-1));
  }


}
