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
                       algstrct const * sr){
    int64_t * chunk_size;
    char ** tswap_data;

    TAU_FSTART(nosym_transpose);
    if (order == 0){
      TAU_FSTOP(nosym_transpose);
      return;
    }
  #ifdef USE_OMP
    int max_ntd = MIN(16,omp_get_max_threads());
    CTF_int::alloc_ptr(max_ntd*sizeof(char*),   (void**)&tswap_data);
    CTF_int::alloc_ptr(max_ntd*sizeof(int64_t), (void**)&chunk_size);
    std::fill(chunk_size, chunk_size+max_ntd, 0);
  #else
    int max_ntd=1;
    CTF_int::alloc_ptr(sizeof(char*),   (void**)&tswap_data);
    CTF_int::alloc_ptr(sizeof(int64_t), (void**)&chunk_size);
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
      int64_t thread_chunk_size = chunk_size[tid];
      int i;
      char * swap_data = tswap_data[tid];
      int64_t toff = 0;
      for (i=0; i<tid; i++) toff += chunk_size[i];
      if (thread_chunk_size > 0){
        memcpy(data+sr->el_size*(toff),swap_data,sr->el_size*thread_chunk_size);
      }
    }
    for (int i=0; i<max_ntd; i++) {
      int64_t thread_chunk_size = chunk_size[i];
      if (thread_chunk_size > 0)
        CTF_int::cfree(tswap_data[i],i);
    }

    CTF_int::cfree(tswap_data);
    CTF_int::cfree(chunk_size);
    TAU_FSTOP(nosym_transpose);
  }

#define CACHELINE 16

  template <int idim>
  void nosym_transpose_opt(int const *      new_order,
                           int const *      edge_len,
                           char const *     data,
                           char *           swap_data,
                           int              dir,
                           int              idx_new_lda1,
                           int64_t *        chunk_size,
                           int64_t const *  lda,
                           int64_t const *  new_lda,
                           int64_t          off_old,
                           int64_t          off_new,
                           int              i_new_lda1,
                           algstrct const * sr){
    if (idim == idx_new_lda1){
      for (int i=0; i<edge_len[idim]; i+=CACHELINE){
        nosym_transpose_opt<idim-1>(new_order, edge_len, data, swap_data, dir, idx_new_lda1, chunk_size, lda, new_lda, off_old+i*lda[idim], off_new+i, i, sr);
      }
    } else {
      for (int i=0; i<edge_len[idim]; i++){
        nosym_transpose_opt<idim-1>(new_order, edge_len, data, swap_data, dir, idx_new_lda1, chunk_size, lda, new_lda, off_old+i*lda[idim], off_new+i*new_lda[idim], i_new_lda1, sr);
      }
    }
  }

  template <> 
  inline void nosym_transpose_opt<0>(int const *      new_order,
                                     int const *      edge_len,
                                     char const *     data,
                                     char *           swap_data,
                                     int              dir,
                                     int              idx_new_lda1,
                                     int64_t *        chunk_size,
                                     int64_t const *  lda,
                                     int64_t const *  new_lda,
                                     int64_t          off_old,
                                     int64_t          off_new,
                                     int              i_new_lda1,
                                     algstrct const * sr){
    //FIXME: prealloc?
    char buf[sr->el_size*CACHELINE*CACHELINE];

    if (dir) {
      int new_lda1_n = std::min(edge_len[idx_new_lda1]-i_new_lda1*CACHELINE,CACHELINE);
      for (int i=0; i<edge_len[0]-CACHELINE+1; i+=CACHELINE){
        for (int j=0; j<new_lda1_n; j++){
          sr->copy(CACHELINE, data+sr->el_size*(off_old+j*lda[idx_new_lda1]+i), 1, buf, 1);
        }
        for (int j=0; j<CACHELINE; j++){
          sr->copy(new_lda1_n, buf, CACHELINE, swap_data+sr->el_size*(off_new+(i+j)*new_lda[0]), 1);
        }
      }
      int lda1_n = edge_len[0]%CACHELINE;
      for (int j=0; j<new_lda1_n; j++){
        sr->copy(lda1_n, data+sr->el_size*(off_old+j)*lda[idx_new_lda1]+edge_len[0]-lda1_n, 1, buf, 1);
      }
      for (int j=0; j<lda1_n; j++){
        sr->copy(new_lda1_n, buf, lda1_n, swap_data+sr->el_size*(off_new+(edge_len[0]-lda1_n+j)*new_lda[0]), 1);
      }
    } else {
      int new_lda1_n = std::max(edge_len[idx_new_lda1]-i_new_lda1*CACHELINE,CACHELINE);
      for (int i=0; i<edge_len[0]; i+=CACHELINE){
        int lda1_n = std::max(edge_len[0]-i*CACHELINE,CACHELINE);
        for (int j=0; j<lda1_n; j++){
          sr->copy(new_lda1_n, data+sr->el_size*(off_new+j*new_lda[0]+i), 1, buf, 1);
        }
        for (int j=0; j<new_lda1_n; j++){
          sr->copy(lda1_n, buf, new_lda1_n, swap_data+sr->el_size*(off_old+(i+j)*lda[idx_new_lda1]), 1);
        }
      }
    }
  }

  template
  void nosym_transpose_opt<8>(int const *      new_order,
                              int const *      edge_len,
                              char const *     data,
                              char *           swap_data,
                              int              dir,
                              int              idx_new_lda1,
                              int64_t *        chunk_size,
                              int64_t const *  lda,
                              int64_t const *  new_lda,
                              int64_t          off_old,
                              int64_t          off_new,
                              int              i_new_lda1,
                              algstrct const * sr);

  void nosym_transpose(int              order,
                       int const *      new_order,
                       int const *      edge_len,
                       char const *     data,
                       int              dir,
                       int              max_ntd,
                       char **          tswap_data,
                       int64_t *        chunk_size,
                       algstrct const * sr){
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
/*
    if (order <= 8){
      int idx_new_lda1 = -1;
      for (int i=0; i<order; i++){
        if(new_order[i] == 0){
          idx_new_lda1 = i;
          break;
        }
      }
      switch (order){
        case 1:
        nosym_transpose_opt<0>(new_order,edge_len,data,tswap_data[0],dir,idx_new_lda1,&local_size,lda,new_lda,0,0,0,sr);
        break;
      }
    }
*/


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
          printf("ERRORR thread_chunk_size = %ld, tid = %ld, local_size = %ld\n", thread_chunk_size, tid, local_size);
        CTF_int::alloc_ptr(thread_chunk_size*sr->el_size, (void**)&tswap_data[tid]);
        swap_data = tswap_data[tid];
        for (;;){
          if (last_dim != 0){
            if (dir) {
              sr->copy(edge_len[0], data+sr->el_size*(off_old), lda[0], swap_data+sr->el_size*(off_new-toff_new), new_lda[0]);
            } else
              sr->copy(edge_len[0], data+sr->el_size*(off_new), new_lda[0], swap_data+sr->el_size*(off_old-toff_old), lda[0]);

            idx[0] = 0;
          } else {
            if (dir)
              sr->copy(last_max-tidx_off, data+sr->el_size*(off_old), lda[0], swap_data+sr->el_size*(off_new-toff_new), new_lda[0]);
            else
              sr->copy(last_max-tidx_off, data+sr->el_size*(off_new), new_lda[0], swap_data+sr->el_size*(off_old-toff_old), lda[0]);
/*            printf("Wrote following values from");
            for (int asi=0; asi<lda[0]; asi++){
              printf("\n %ld to %ld\n",(off_new)+asi,(off_old-toff_old)+asi*lda[0]);
              sr->print(data+sr->el_size*(off_new+asi*lda[0]));
            }
            printf("\n");*/
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
