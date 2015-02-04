#include "folding.h"
#include "../shared/util.h"

namespace CTF_int {
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

  void nosym_transpose(int              order,
                       int const *      new_order,
                       int const *      edge_len,
                       char *           data,
                       int              dir,
                       algstrct const & sr){
    int * chunk_size;
    char ** tswap_data;

    TAU_FSTART(nosym_transpose);
    if (order == 0){
      TAU_FSTOP(nosym_transpose);
      return;
    }
  #ifdef USE_OMP
    int max_ntd = MIN(16,omp_get_max_threads());
    CTF_int::alloc_ptr(max_ntd*sizeof(char*), (void**)&tswap_data);
    CTF_int::alloc_ptr(max_ntd*sizeof(int),   (void**)&chunk_size);
  #else
    int max_ntd=1;
    CTF_int::alloc_ptr(sizeof(char*), (void**)&tswap_data);
    CTF_int::alloc_ptr(sizeof(int),   (void**)&chunk_size);
  #endif
    nosym_transpose(order, new_order, edge_len, data, dir, max_ntd, tswap_data, chunk_size, sr);
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
      char * swap_data = tswap_data[tid];
      int toff = 0;
      for (i=0; i<tid; i++) toff += chunk_size[i];
      if (thread_chunk_size > 0){
        memcpy(data+sr.el_size*(toff),swap_data,sr.el_size*thread_chunk_size);
      }
    }
    for (int i=0; i<max_ntd; i++) {
      int thread_chunk_size = chunk_size[i];
      if (thread_chunk_size > 0)
        CTF_int::cfree(tswap_data[i],i);
    }

    CTF_int::cfree(tswap_data);
    CTF_int::cfree(chunk_size);
    TAU_FSTOP(nosym_transpose);
  }


  void nosym_transpose(int              order,
                       int const *      new_order,
                       int const *      edge_len,
                       char const *     data,
                       int              dir,
                       int              max_ntd,
                       char **          tswap_data,
                       int *            chunk_size,
                       algstrct const & sr){
    int64_t local_size;
    int64_t j, last_dim;
    int64_t * lda, * new_lda;

    TAU_FSTART(nosym_transpose_thr);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&lda);
    CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&new_lda);
    
    if (dir){
      last_dim = new_order[order-1];
    } else {
      last_dim = order - 1;
    }
  //  last_dim = order-1;

    lda[0] = 1;
    for (j=1; j<order; j++){
      lda[j] = lda[j-1]*edge_len[j-1];
    }
    local_size = lda[order-1]*edge_len[order-1];
    new_lda[new_order[0]] = 1;
    for (j=1; j<order; j++){
      new_lda[new_order[j]] = new_lda[new_order[j-1]]*edge_len[new_order[j-1]];
    }
    ASSERT(local_size == new_lda[new_order[order-1]]*edge_len[new_order[order-1]]);
  #ifdef USE_OMP
    #pragma omp parallel num_threads(max_ntd)
  #endif
    {
      int64_t i, off_old, off_new, tid, ntd, last_max, toff_new, toff_old;
      int64_t tidx_off;
      int64_t thread_chunk_size;
      int64_t * idx;
      char * swap_data;
      CTF_int::alloc_ptr(order*sizeof(int64_t), (void**)&idx);
      memset(idx, 0, order*sizeof(int64_t));

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
      if (order != 1){
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
        last_dim = 1;
      } 
      chunk_size[tid] = 0;
      if (last_max != 0 && tidx_off != last_max && (order != 1 || tid == 0)){
        chunk_size[tid] = thread_chunk_size;
        if (thread_chunk_size <= 0) 
          printf("ERRORR thread_chunk_size = " PRId64 ", tid = " PRId64 ", local_size = " PRId64 "\n", thread_chunk_size, tid, local_size);
        CTF_int::alloc_ptr(thread_chunk_size*sr.el_size, (void**)&tswap_data[tid]);
        swap_data = tswap_data[tid];
        for (;;){
          if (last_dim != 0){
            if (dir)
              sr.copy(edge_len[0], data+sr.el_size*(off_old), lda[0], swap_data+sr.el_size*(off_new-toff_new), new_lda[0]);
            else
              sr.copy(edge_len[0], data+sr.el_size*(off_new), new_lda[0], swap_data+sr.el_size*(off_old-toff_old), lda[0]);

            idx[0] = 0;
          } else {
            if (dir)
              sr.copy(last_max-tidx_off, data+sr.el_size*(off_old), lda[0], swap_data+sr.el_size*(off_new-toff_new), new_lda[0]);
            else
              sr.copy(last_max-tidx_off, data+sr.el_size*(off_new), new_lda[0], swap_data+sr.el_size*(off_old-toff_old), lda[0]);

            idx[0] = tidx_off;
          } 

          for (i=1; i<order; i++){
            off_old -= idx[i]*lda[i];
            off_new -= idx[i]*new_lda[i];
            if (i == last_dim){
              idx[i] = (idx[i] == last_max-1 ? tidx_off : idx[i]+1);
              off_old += idx[i]*lda[i];
              off_new += idx[i]*new_lda[i];
              if (idx[i] != tidx_off) break;
            } else {
              idx[i] = (idx[i] == edge_len[i]-1 ? 0 : idx[i]+1);
              off_old += idx[i]*lda[i];
              off_new += idx[i]*new_lda[i];
              if (idx[i] != 0) break;
            }
          }
          if (i==order) break;
        }
      }
      CTF_int::cfree(idx);
    }
    CTF_int::cfree(lda);
    CTF_int::cfree(new_lda);
    TAU_FSTOP(nosym_transpose_thr);
  }
}

