/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __DIST_TENSOR_SORT_HXX__
#define __DIST_TENSOR_SORT_HXX__
#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include "../ctr_comm/strp_tsr.h"
#ifdef USE_OMP
#include "omp.h"
#endif


/**
 * \brief retrieves the unpadded pairs
 * \param[in] ndim tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetry types of tensor
 * \param[in] padding padding of tensor (included in edge_len)
 * \param[in] prepadding padding at start of tensor (included in edge_len)
 * \param[in] pairs padded array of pairs
 * \param[out] new_pairs unpadded pairs
 * \param[out] new_num_pair number of unpadded pairs
 */
#ifdef USE_OMP
template<typename dtype>
void depad_tsr(int const                ndim,
               long_int const           num_pair,
               int const *              edge_len,
               int const *              sym,
               int const *              padding,
               int const *              prepadding,
               tkv_pair<dtype> const *  pairs,
               tkv_pair<dtype> *        new_pairs,
               long_int *               new_num_pair){
  TAU_FSTART(depad_tsr);

  long_int num_ins;
  int ntd = omp_get_max_threads();
  long_int * num_ins_t = (long_int*)CTF_alloc(sizeof(long_int)*ntd);
  long_int * pre_ins_t = (long_int*)CTF_alloc(sizeof(long_int)*ntd);

  TAU_FSTART(depad_tsr_cnt);
  #pragma omp parallel
  {
    long_int i, j, st, end, tid;
    key * kparts;
    key k;
    CTF_alloc_ptr(sizeof(key)*ndim, (void**)&kparts);
    tid = omp_get_thread_num();

    st = (num_pair/ntd)*tid;
    if (tid == ntd-1)
      end = num_pair;
    else
      end = (num_pair/ntd)*(tid+1);

    num_ins_t[tid] = 0;
    for (i=st; i<end; i++){
      k = pairs[i].k;
      for (j=0; j<ndim; j++){
        kparts[j] = k%(edge_len[j]+padding[j]);
        if (kparts[j] >= (key)edge_len[j] ||
            kparts[j] < prepadding[j]) break;
        k = k/(edge_len[j]+padding[j]);
      } 
      if (j==ndim){
        for (j=0; j<ndim; j++){
          if (sym[j] == SY){
            if (kparts[j+1] < kparts[j])
              break;
          }
          if (sym[j] == AS || sym[j] == SH){
            if (kparts[j+1] <= kparts[j])
              break;
          }
        }
        if (j==ndim){
          num_ins_t[tid]++;
        }
      }
    }
    CTF_free(kparts);
  }
  TAU_FSTOP(depad_tsr_cnt);

  pre_ins_t[0] = 0;
  for (int j=1; j<ntd; j++){
    pre_ins_t[j] = num_ins_t[j-1] + pre_ins_t[j-1];
  }

  TAU_FSTART(depad_tsr_move);
  #pragma omp parallel
  {
    long_int i, j, st, end, tid;
    key * kparts;
    key k;
    CTF_alloc_ptr(sizeof(key)*ndim, (void**)&kparts);
    tid = omp_get_thread_num();

    st = (num_pair/ntd)*tid;
    if (tid == ntd-1)
      end = num_pair;
    else
      end = (num_pair/ntd)*(tid+1);

    for (i=st; i<end; i++){
      k = pairs[i].k;
      for (j=0; j<ndim; j++){
        kparts[j] = k%(edge_len[j]+padding[j]);
        if (kparts[j] >= (key)edge_len[j] ||
            kparts[j] < prepadding[j]) break;
        k = k/(edge_len[j]+padding[j]);
      } 
      if (j==ndim){
        for (j=0; j<ndim; j++){
          if (sym[j] == SY){
            if (kparts[j+1] < kparts[j])
              break;
          }
          if (sym[j] == AS || sym[j] == SH){
            if (kparts[j+1] <= kparts[j])
              break;
          }
        }
        if (j==ndim){
          new_pairs[pre_ins_t[tid]] = pairs[i];
          pre_ins_t[tid]++;
        }
      }
    }
    CTF_free(kparts);
  }
  TAU_FSTOP(depad_tsr_move);
  num_ins = pre_ins_t[ntd-1];

  *new_num_pair = num_ins;
  CTF_free(pre_ins_t);
  CTF_free(num_ins_t);

  TAU_FSTOP(depad_tsr);
}
#else
template<typename dtype>
void depad_tsr(int const                ndim,
               long_int const           num_pair,
               int const *              edge_len,
               int const *              sym,
               int const *              padding,
               tkv_pair<dtype> const *  pairs,
               tkv_pair<dtype> *        new_pairs,
               long_int *               new_num_pair){
  
  TAU_FSTART(depad_tsr);
  long_int i, j, num_ins;
  key * kparts;
  key k;


  CTF_alloc_ptr(sizeof(key)*ndim, (void**)&kparts);

  num_ins = 0;
  for (i=0; i<num_pair; i++){
    k = pairs[i].k;
    for (j=0; j<ndim; j++){
      kparts[j] = k%(edge_len[j]+padding[j]);
      if (kparts[j] >= (key)edge_len[j]) break;
      k = k/(edge_len[j]+padding[j]);
    } 
    if (j==ndim){
      for (j=0; j<ndim; j++){
        if (sym[j] == SY){
          if (kparts[j+1] < kparts[j])
            break;
        }
        if (sym[j] == AS || sym[j] == SH){
          if (kparts[j+1] <= kparts[j])
            break;
        }
      }
      if (j==ndim){
        new_pairs[num_ins] = pairs[i];
        num_ins++;
      }
    }
  }
  *new_num_pair = num_ins;
  CTF_free(kparts);

  TAU_FSTOP(depad_tsr);
}
#endif

/**
 * \brief applies padding to keys
 * \param[in] ndim tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len tensor edge lengths
 * \param[in] padding padding of tensor (included in edge_len)
 */
template<typename dtype>
void pad_key(int const        ndim,
             long_int const   num_pair,
             int const *      edge_len,
             int const *      padding,
             tkv_pair<dtype> *  pairs){
  long_int i, j, lda;
  key knew, k;
  TAU_FSTART(pad_key);
#ifdef USE_OMP
  #pragma omp parallel for private(knew, k, lda, i, j)
#endif
  for (i=0; i<num_pair; i++){
    k = pairs[i].k;
    lda = 1;
    knew = 0;
    for (j=0; j<ndim; j++){
      knew += lda*(k%edge_len[j]);
      lda *= (edge_len[j]+padding[j]);
      k = k/edge_len[j];
    }
    pairs[i].k = knew;
  }
  TAU_FSTOP(pad_key);
}

/**
 * \brief pads a tensor
 * \param[in] ndim tensor dimension
 * \param[in] num_pair number of pairs
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] padding padding of tensor (included in edge_len)
 * \param[in] phys_phase phase of the tensor on virtualized processor grid
 * \param[in] virt_phase_rank physical phase rank multiplied by virtual phase
 * \param[in] virt_phase virtual phase in each dimension
 * \param[in] old_data array of input pairs
 * \param[out] new_pairs padded pairs
 * \param[out] new_size number of new padded pairs
 */
template<typename dtype>
void pad_tsr(int const                ndim,
             long_int const           size,
             int const *              edge_len,
             int const *              sym,
             int const *              padding,
             int const *              phys_phase,
             int *                    virt_phys_rank,
             int const *              virt_phase,
             tkv_pair<dtype> const *  old_data,
             tkv_pair<dtype> **       new_pairs,
             long_int *               new_size){
  int i, imax, act_lda;
  long_int new_el, pad_el;
  int pad_max, virt_lda, outside, offset, edge_lda;
  int * idx;  
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&idx);
  tkv_pair<dtype> * padded_pairs;
  
  pad_el = 0;
 
  for (;;){ 
    memset(idx, 0, ndim*sizeof(int));
    for (;;){
      if (sym[0] != NS)
        pad_max = idx[1]+1;
      else
        pad_max = (edge_len[0]+padding[0])/phys_phase[0];
      pad_el+=pad_max;
      for (act_lda=1; act_lda<ndim; act_lda++){
        idx[act_lda]++;
        imax = (edge_len[act_lda]+padding[act_lda])/phys_phase[act_lda];
        if (sym[act_lda] != NS)
          imax = idx[act_lda+1]+1;
        if (idx[act_lda] >= imax) 
          idx[act_lda] = 0;
        if (idx[act_lda] != 0) break;      
      }
      if (act_lda == ndim) break;

    }
    for (act_lda=0; act_lda<ndim; act_lda++){
      virt_phys_rank[act_lda]++;
      if (virt_phys_rank[act_lda]%virt_phase[act_lda]==0)
        virt_phys_rank[act_lda] -= virt_phase[act_lda];
      if (virt_phys_rank[act_lda]%virt_phase[act_lda]!=0) break;      
    }
    if (act_lda == ndim) break;
  }
  CTF_alloc_ptr(pad_el*sizeof(tkv_pair<dtype>), (void**)&padded_pairs);
  new_el = 0;
  offset = 0;
  outside = -1;
  virt_lda=1;
  for (i=0; i<ndim; i++){
    offset += virt_phys_rank[i]*virt_lda;
    virt_lda*=(edge_len[i]+padding[i]);
  }

  for (;;){
    memset(idx, 0, ndim*sizeof(int));
    for (;;){
      if (sym[0] != NS){
        if (idx[1] < edge_len[0]/phys_phase[0]) {
          imax = idx[1];
          if (sym[0] != SY && virt_phys_rank[0] < virt_phys_rank[1])
            imax++;
          if (sym[0] == SY && virt_phys_rank[0] <= virt_phys_rank[1])
            imax++;
        } else {
          imax = edge_len[0]/phys_phase[0];
          if (virt_phys_rank[0] < edge_len[0]%phys_phase[0])
            imax++;
        }
        pad_max = idx[1]+1;
      } else {
        imax = edge_len[0]/phys_phase[0];
        if (virt_phys_rank[0] < edge_len[0]%phys_phase[0])
          imax++;
        pad_max = (edge_len[0]+padding[0])/phys_phase[0];
      }
      if (outside == -1){
        for (i=0; i<pad_max-imax; i++){
          padded_pairs[new_el+i].k = offset + (imax+i)*phys_phase[0];
          padded_pairs[new_el+i].d = get_zero<dtype>();
        }
        new_el+=pad_max-imax;
      }  else {
        for (i=0; i<pad_max; i++){
          padded_pairs[new_el+i].k = offset + i*phys_phase[0];
          padded_pairs[new_el+i].d = get_zero<dtype>();
        }
        new_el += pad_max;
      }

      edge_lda = edge_len[0]+padding[0];
      for (act_lda=1; act_lda<ndim; act_lda++){
        offset -= idx[act_lda]*edge_lda*phys_phase[act_lda];
        idx[act_lda]++;
        imax = (edge_len[act_lda]+padding[act_lda])/phys_phase[act_lda];
        if (sym[act_lda] != NS && idx[act_lda+1]+1 <= imax){
          imax = idx[act_lda+1]+1;
      //    if (virt_phys_rank[act_lda] < virt_phys_rank[sym[act_lda]])
      //      imax++;
        } 
        if (idx[act_lda] >= imax)
          idx[act_lda] = 0;
        offset += idx[act_lda]*edge_lda*phys_phase[act_lda];
        if (idx[act_lda] > edge_len[act_lda]/phys_phase[act_lda] ||
            (idx[act_lda] == edge_len[act_lda]/phys_phase[act_lda] &&
            (edge_len[act_lda]%phys_phase[act_lda] <= virt_phys_rank[act_lda]))){
          if (outside < act_lda)
            outside = act_lda;
        } else {
          if (outside == act_lda)
            outside = -1;
        }
        if (sym[act_lda] != NS && idx[act_lda] == idx[act_lda+1]){
          if (sym[act_lda] != SY && 
              virt_phys_rank[act_lda] >= virt_phys_rank[act_lda+1]){
            if (outside < act_lda)
              outside = act_lda;
          } 
          if (sym[act_lda] == SY && 
              virt_phys_rank[act_lda] > virt_phys_rank[act_lda+1]){
            if (outside < act_lda)
              outside = act_lda;
          } 
        }
        if (idx[act_lda] != 0) break;      
        edge_lda*=(edge_len[act_lda]+padding[act_lda]);
      }
      if (act_lda == ndim) break;

    }
    virt_lda = 1;
    for (act_lda=0; act_lda<ndim; act_lda++){
      offset -= virt_phys_rank[act_lda]*virt_lda;
      virt_phys_rank[act_lda]++;
      if (virt_phys_rank[act_lda]%virt_phase[act_lda]==0)
        virt_phys_rank[act_lda] -= virt_phase[act_lda];
      offset += virt_phys_rank[act_lda]*virt_lda;
      if (virt_phys_rank[act_lda]%virt_phase[act_lda]!=0) break;      
      virt_lda*=(edge_len[act_lda]+padding[act_lda]);
    }
    if (act_lda == ndim) break;
    
  }
  DEBUG_PRINTF("ndim = %d new_el=%lld, size = %lld, pad_el = %lld\n", ndim, new_el, size, pad_el);
  LIBT_ASSERT(new_el + size == pad_el);
  memcpy(padded_pairs+new_el, old_data,  size*sizeof(tkv_pair<dtype>));
  *new_pairs = padded_pairs;
  *new_size = pad_el;
      

}


/**
 * \brief assigns keys to an array of values
 * \param[in] ndim tensor dimension
 * \param[in] size number of values
 * \param[in] nvirt total virtualization factor
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] phase phase of the tensor on virtualized processor grid
 * \param[in] virt_dim virtual phase in each dimension
 * \param[in] phase_rank physical phase rank multiplied by virtual phase
 * \param[in] vdata array of input values
 * \param[out] vpairs pairs of keys and inputted values
 */
template<typename dtype>
void assign_keys(int const          ndim,
                 long_int const     size,
                 int const          nvirt,
                 int const *        edge_len,
                 int const *        sym,
                 int const *        phase,
                 int const *        virt_dim,
                 int *              phase_rank,
                 dtype const *      vdata,
                 tkv_pair<dtype> *  vpairs){
  int i, imax, act_lda, idx_offset, act_max, buf_offset;
  long_int p;
  int * idx, * virt_rank, * edge_lda;  
  dtype const * data;
  tkv_pair<dtype> * pairs;
  if (ndim == 0){
    LIBT_ASSERT(size <= 1);
    if (size == 1){
      vpairs[0].k = 0;
      vpairs[0].d = vdata[0];
    }
    return;
  }

  TAU_FSTART(assign_keys);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&idx);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&virt_rank);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&edge_lda);
  
  memset(virt_rank, 0, sizeof(int)*ndim);
  
  edge_lda[0] = 1;
  for (i=1; i<ndim; i++){
    edge_lda[i] = edge_lda[i-1]*edge_len[i-1];
  }
  for (p=0;;p++){
    data = vdata + p*(size/nvirt);
    pairs = vpairs + p*(size/nvirt);

    idx_offset = 0, buf_offset = 0;
    for (act_lda=1; act_lda<ndim; act_lda++){
      idx_offset += phase_rank[act_lda]*edge_lda[act_lda];
    } 
  
    //printf("size = %d\n", size); 
    memset(idx, 0, ndim*sizeof(int));
    imax = edge_len[0]/phase[0];
    for (;;){
      if (sym[0] != NS)
        imax = idx[1]+1;
            /* Increment virtual bucket */
            for (i=0; i<imax; i++){
        LIBT_ASSERT(buf_offset+i<size);
        if (p*(size/nvirt) + buf_offset + i >= size){ 
          printf("exceeded how much I was supposed to read read %lld/%lld\n", p*(size/nvirt)+buf_offset+i,size);
          ABORT;
        }
        pairs[buf_offset+i].k = idx_offset+i*phase[0]+phase_rank[0];
        pairs[buf_offset+i].d = data[buf_offset+i];
      }
      buf_offset += imax;
      /* Increment indices and set up offsets */
      for (act_lda=1; act_lda < ndim; act_lda++){
        idx_offset -= (idx[act_lda]*phase[act_lda]+phase_rank[act_lda])
                *edge_lda[act_lda];
        idx[act_lda]++;
        act_max = edge_len[act_lda]/phase[act_lda];
        if (sym[act_lda] != NS) act_max = idx[act_lda+1]+1;
        if (idx[act_lda] >= act_max)
          idx[act_lda] = 0;
        idx_offset += (idx[act_lda]*phase[act_lda]+phase_rank[act_lda])
                *edge_lda[act_lda];
        LIBT_ASSERT(edge_len[act_lda]%phase[act_lda] == 0);
        if (idx[act_lda] > 0)
          break;
      }
      if (act_lda >= ndim) break;
    }
    for (act_lda=0; act_lda < ndim; act_lda++){
      phase_rank[act_lda] -= virt_rank[act_lda];
      virt_rank[act_lda]++;
      if (virt_rank[act_lda] >= virt_dim[act_lda])
        virt_rank[act_lda] = 0;
      phase_rank[act_lda] += virt_rank[act_lda];
      if (virt_rank[act_lda] > 0)
        break;
    }
    if (act_lda >= ndim) break;
  }
  LIBT_ASSERT(buf_offset == size/nvirt);
  CTF_free(idx);
  CTF_free(virt_rank);
  CTF_free(edge_lda);
  TAU_FSTOP(assign_keys);
}
       

/**
 * \brief optimized version of bucket_by_pe
 */
template <typename dtype>
void bucket_by_pe( int const                ndim,
                   long_int const           num_pair,
                   int const                np,
                   int const *              phase,
                   int const *              virt_phase,
                   int const *              bucket_lda,
                   int const *              edge_len,
                   tkv_pair<dtype> const *  mapped_data,
                   int *                    bucket_counts,
                   int *                    bucket_off,
                   tkv_pair<dtype> *        bucket_data){
  long_int i, j, loc;
//  int * inv_edge_len, * inv_virt_phase;
  key k;

/*  CTF_alloc_ptr(ndim*sizeof(int), (void**)&inv_edge_len);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&inv_virt_phase);


  for (i=0; i<ndim; i++){
    inv_edge_len[i]
  }*/

  memset(bucket_counts, 0, sizeof(int)*np); 
#ifdef USE_OMP
  int * sub_counts, * sub_offs;
  CTF_alloc_ptr(np*sizeof(int)*omp_get_max_threads(), (void**)&sub_counts);
  CTF_alloc_ptr(np*sizeof(int)*omp_get_max_threads(), (void**)&sub_offs);
  memset(sub_counts, 0, np*sizeof(int)*omp_get_max_threads());
#endif


  TAU_FSTART(bucket_by_pe_count);
  /* Calculate counts */
#ifdef USE_OMP
  #pragma omp parallel for schedule(static,256) private(i, j, loc, k)
#endif
  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
//    int tmp_arr[ndim];
    for (j=0; j<ndim; j++){
/*      tmp_arr[j] = (k%edge_len[j])%phase[j];
      tmp_arr[j] = tmp_arr[j]/virt_phase[j];
      tmp_arr[j] = tmp_arr[j]*bucket_lda[j];*/
      loc += ((k%phase[j])/virt_phase[j])*bucket_lda[j];
      k = k/edge_len[j];
    }
/*    for (j=0; j<ndim; j++){
      loc += tmp_arr[j];
    }*/
    LIBT_ASSERT(loc<np);
#ifdef USE_OMP
    sub_counts[loc+omp_get_thread_num()*np]++;
#else
    bucket_counts[loc]++;
#endif
  }
  TAU_FSTOP(bucket_by_pe_count);

#ifdef USE_OMP
  for (j=0; j<omp_get_max_threads(); j++){
    for (i=0; i<np; i++){
      bucket_counts[i] = sub_counts[j*np+i] + bucket_counts[i];
    }
  }
#endif

  /* Prefix sum to get offsets */
  bucket_off[0] = 0;
  for (i=1; i<np; i++){
    bucket_off[i] = bucket_counts[i-1] + bucket_off[i-1];
  }
  
  /* reset counts */
#ifdef USE_OMP
  memset(sub_offs, 0, sizeof(int)*np);
  for (i=1; i<omp_get_max_threads(); i++){
    for (j=0; j<np; j++){
      sub_offs[j+i*np]=sub_counts[j+(i-1)*np]+sub_offs[j+(i-1)*np];
    }
  }
#else
  memset(bucket_counts, 0, sizeof(int)*np); 
#endif

  /* bucket data */
  TAU_FSTART(bucket_by_pe_move);
#ifdef USE_OMP
  #pragma omp parallel for schedule(static,256) private(i, j, loc, k)
#endif
  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
    for (j=0; j<ndim; j++){
      loc += ((k%phase[j])/virt_phase[j])*bucket_lda[j];
      k = k/edge_len[j];
    }
#ifdef USE_OMP
    bucket_data[bucket_off[loc] + sub_offs[loc+omp_get_thread_num()*np]] 
        = mapped_data[i];
    sub_offs[loc+omp_get_thread_num()*np]++;
#else
    bucket_data[bucket_off[loc] + bucket_counts[loc]] = mapped_data[i];
    bucket_counts[loc]++;
#endif
  }
#ifdef USE_OMP
  CTF_free(sub_counts);
  CTF_free(sub_offs);
#endif
  TAU_FSTOP(bucket_by_pe_move);
}

/**
 * \brief optimized version of bucket_by_virt
 */
template <typename dtype>
void bucket_by_virt(int const               ndim,
                    int const               num_virt,
                    long_int const          num_pair,
                    int const *             virt_phase,
                    int const *             edge_len,
                    tkv_pair<dtype> const * mapped_data,
                    tkv_pair<dtype> *       bucket_data){
  long_int i, j, loc;
  int * virt_counts, * virt_prefix, * virt_lda;
  key k;
  TAU_FSTART(bucket_by_virt);
  
  CTF_alloc_ptr(num_virt*sizeof(int), (void**)&virt_counts);
  CTF_alloc_ptr(num_virt*sizeof(int), (void**)&virt_prefix);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&virt_lda);
 
 
  if (ndim > 0){
    virt_lda[0] = 1;
    for (i=1; i<ndim; i++){
      LIBT_ASSERT(virt_phase[i] > 0);
      virt_lda[i] = virt_phase[i-1]*virt_lda[i-1];
    }
  }

  memset(virt_counts, 0, sizeof(int)*num_virt); 
#ifdef USE_OMP
  int * sub_counts, * sub_offs;
  CTF_alloc_ptr(num_virt*sizeof(int)*omp_get_max_threads(), (void**)&sub_counts);
  CTF_alloc_ptr(num_virt*sizeof(int)*omp_get_max_threads(), (void**)&sub_offs);
  memset(sub_counts, 0, num_virt*sizeof(int)*omp_get_max_threads());
#endif

  /* bucket data */
#ifdef USE_OMP
  TAU_FSTART(bucket_by_virt_omp_cnt);
  #pragma omp parallel for schedule(static) private(j, loc, k, i)
  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
    //#pragma unroll
    for (j=0; j<ndim; j++){
      loc += (k%virt_phase[j])*virt_lda[j];
      k = k/edge_len[j];
    }
    
    //bucket_data[loc*num_pair_virt + virt_counts[loc]] = mapped_data[i];
    sub_counts[loc+omp_get_thread_num()*num_virt]++;
  }
  TAU_FSTOP(bucket_by_virt_omp_cnt);
  TAU_FSTART(bucket_by_virt_assemble_offsets);
  for (j=0; j<omp_get_max_threads(); j++){
    for (i=0; i<num_virt; i++){
      virt_counts[i] = sub_counts[j*num_virt+i] + virt_counts[i];
    }
  }
  virt_prefix[0] = 0;
  for (i=1; i<num_virt; i++){
    virt_prefix[i] = virt_prefix[i-1] + virt_counts[i-1];
  }

  memset(sub_offs, 0, sizeof(int)*num_virt);
  for (i=1; i<omp_get_max_threads(); i++){
    for (j=0; j<num_virt; j++){
      sub_offs[j+i*num_virt]=sub_counts[j+(i-1)*num_virt]+sub_offs[j+(i-1)*num_virt];
    }
  }
  TAU_FSTOP(bucket_by_virt_assemble_offsets);
  TAU_FSTART(bucket_by_virt_move);
  #pragma omp parallel for schedule(static) private(j, loc, k, i)
  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
    //#pragma unroll
    for (j=0; j<ndim; j++){
      loc += (k%virt_phase[j])*virt_lda[j];
      k = k/edge_len[j];
    }
    bucket_data[virt_prefix[loc] + sub_offs[loc+omp_get_thread_num()*num_virt]] 
    = mapped_data[i];
    sub_offs[loc+omp_get_thread_num()*num_virt]++;
  }
  TAU_FSTOP(bucket_by_virt_move);
#else
  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
    for (j=0; j<ndim; j++){
      loc += (k%virt_phase[j])*virt_lda[j];
      k = k/edge_len[j];
    }
    virt_counts[loc]++;
  }

  virt_prefix[0] = 0;
  for (i=1; i<num_virt; i++){
    virt_prefix[i] = virt_prefix[i-1] + virt_counts[i-1];
  }
  memset(virt_counts, 0, sizeof(int)*num_virt); 

  for (i=0; i<num_pair; i++){
    k = mapped_data[i].k;
    loc = 0;
    for (j=0; j<ndim; j++){
      loc += (k%virt_phase[j])*virt_lda[j];
      k = k/edge_len[j];
    }
    bucket_data[virt_prefix[loc] + virt_counts[loc]] = mapped_data[i];
    virt_counts[loc]++;
  }
#endif

  TAU_FSTART(bucket_by_virt_sort);
#ifdef USE_OMP
  #pragma omp parallel for
#endif
  for (i=0; i<num_virt; i++){
    std::sort(bucket_data+virt_prefix[i],
        bucket_data+(virt_prefix[i]+virt_counts[i]));
  }
  TAU_FSTOP(bucket_by_virt_sort);
#if DEBUG >= 1
  for (i=1; i<num_pair; i++){
    LIBT_ASSERT(bucket_data[i].k != bucket_data[i-1].k);
  }
#endif
#ifdef USE_OMP
  CTF_free(sub_counts);
  CTF_free(sub_offs);
#endif
  CTF_free(virt_prefix);
  CTF_free(virt_counts);
  CTF_free(virt_lda);
  TAU_FSTOP(bucket_by_virt);
}

/**
 * \brief assigns keys to an array of values
 * \param[in] ndim tensor dimension
 * \param[in] size number of values
 * \param[in] nvirt total virtualization factor
 * \param[in] edge_len tensor edge lengths with padding
 * \param[in] sym symmetries of tensor
 * \param[in] padding how much of the edge lengths is padding
 * \param[in] phase phase of the tensor on virtualized processor grid
 * \param[in] virt_dim virtual phase in each dimension
 * \param[in] phase_rank physical phase rank multiplied by virtual phase
 * \param[in,out] vdata array of all local data
 */
template<typename dtype>
void zero_padding( int const          ndim,
                   long_int const     size,
                   int const          nvirt,
                   int const *        edge_len,
                   int const *        sym,
                   int const *        padding,
                   int const *        phase,
                   int const *        virt_dim,
                   int const *        cphase_rank,
                   dtype *            vdata){
  if (ndim == 0) return;
  TAU_FSTART(zero_padding);
#ifdef USE_OMP
#pragma omp parallel
#endif
{
  int tid, ntd, vst, vend;
#ifdef USE_OMP
  tid = omp_get_thread_num();
  ntd = omp_get_num_threads();
#else
  tid = 0;
  ntd = 1;
#endif
  vst = (nvirt/ntd)*tid;
  vst += MIN(nvirt%ntd,tid);
  vend = vst+(nvirt/ntd);
  if (tid < nvirt % ntd) vend++;
  LIBT_ASSERT(tid != ntd-1 || vend == nvirt);
  int i, act_lda, act_max, buf_offset, curr_idx, sym_idx;
  int is_outside;
  long_int p;
  int * idx, * virt_rank, * phase_rank;
  dtype* data;

  CTF_alloc_ptr(ndim*sizeof(int), (void**)&idx);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&virt_rank);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&phase_rank);

  memcpy(phase_rank, cphase_rank, ndim*sizeof(int));
  memset(virt_rank, 0, sizeof(int)*ndim);
  for (p=0; p<nvirt; p++){
    if (p>=vst && p<vend){
      buf_offset = 0;
      data = vdata + p*(size/nvirt);

      //printf("size = %d\n", size); 
      memset(idx, 0, ndim*sizeof(int));
      for (;;){
        is_outside = 0;
        for (i=0; i<ndim; i++){
          curr_idx = idx[i]*phase[i]+phase_rank[i];
          if (curr_idx >= edge_len[i] - padding[i]){
            is_outside = 1;
            break;
          } else if (i < ndim-1) {
            sym_idx   = idx[i+1]*phase[i+1]+phase_rank[i+1];
            if (((sym[i] == AS || sym[i] == SH) && curr_idx >= sym_idx) ||
                ( sym[i] == SY                  && curr_idx >  sym_idx) ) {
              is_outside = 1;
              break;
            }
          }
        }
/*        for (i=0; i<ndim; i++){
          printf("phase_rank[%d] = %d, idx[%d] = %d, ",i,phase_rank[i],i,idx[i]);
        }
        printf("\n");
        printf("data[%lld]=%lf is_outside = %d\n", buf_offset+p*(size/nvirt), data[buf_offset], is_outside);*/
        if (is_outside)
          data[buf_offset] = 0.0;
        buf_offset++;
        /* Increment indices and set up offsets */
        for (i=0; i < ndim; i++){
          idx[i]++;
          act_max = edge_len[i]/phase[i];
          if (sym[i] != NS){
//            sym_idx   = idx[i+1]*phase[i+1]+phase_rank[i+1];
//            act_max   = MIN(act_max,((sym_idx-phase_rank[i])/phase[i]+1));
            act_max = MIN(act_max,idx[i+1]+1);
          }
          if (idx[i] >= act_max)
            idx[i] = 0;
          LIBT_ASSERT(edge_len[i]%phase[i] == 0);
          if (idx[i] > 0)
            break;
        }
        if (i >= ndim) break;
      }
    }
    for (act_lda=0; act_lda < ndim; act_lda++){
      phase_rank[act_lda] -= virt_rank[act_lda];
      virt_rank[act_lda]++;
      if (virt_rank[act_lda] >= virt_dim[act_lda])
        virt_rank[act_lda] = 0;
      phase_rank[act_lda] += virt_rank[act_lda];
      if (virt_rank[act_lda] > 0)
        break;
    }
  }
  //LIBT_ASSERT(buf_offset == size/nvirt);
  CTF_free(idx);
  CTF_free(virt_rank);
  CTF_free(phase_rank);
}
  TAU_FSTOP(zero_padding);
}
     
#endif
