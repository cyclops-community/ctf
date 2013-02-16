/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#ifndef __DIST_TENSOR_RW_HXX__
#define __DIST_TENSOR_RW_HXX__

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#ifdef USE_OMP
#include "omp.h"
#endif

/**
 * \brief read or write pairs from / to tensor
 * \param[in] ndim tensor dimension
 * \param[in] size number of pairs
 * \param[in] alpha multiplier for new value
 * \param[in] beta multiplier for old value
 * \param[in] edge_len tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] phase total phase in each dimension
 * \param[in] virt_dim virtualization in each dimension
 * \param[in] phase_rank virtualized rank in total phase
 * \param[in,out] vdata data to read from or write to
 * \param[in,out] pairs pairs to read or write
 * \param[in] rw whether to read 'r' or write 'w'
 */
template<typename dtype>
void readwrite(int const        ndim,
               long_int const   size,
               double const     alpha,
               double const     beta,
               int const        nvirt,
               int const *      edge_len,
               int const *      sym,
               int const *      phase,
               int const *      virt_dim,
               int *            phase_rank,
               dtype *          vdata,
               tkv_pair<dtype> *pairs,
               char const       rw){
  int i, imax, act_lda;
  long_int idx_offset, act_max, buf_offset, pr_offset, p;
  int * idx, * virt_rank, * edge_lda;  
  dtype * data;
  
  if (ndim == 0){
    if (size > 0){
      LIBT_ASSERT(size == 1);
      if (rw == 'r'){
        pairs[0].d = vdata[0];
      } else {
        vdata[0] = pairs[0].d;
      }
    }
    return;
  }
  TAU_FSTART(readwrite);
  get_buffer_space(ndim*sizeof(int), (void**)&idx);
  get_buffer_space(ndim*sizeof(int), (void**)&virt_rank);
  get_buffer_space(ndim*sizeof(int), (void**)&edge_lda);
  
  memset(virt_rank, 0, sizeof(int)*ndim);
  edge_lda[0] = 1;
  for (i=1; i<ndim; i++){
    edge_lda[i] = edge_lda[i-1]*edge_len[i-1];
  }

  pr_offset = 0;
  buf_offset = 0;
  data = vdata;// + buf_offset;

  for (p=0;;p++){
    data = data + buf_offset;
    idx_offset = 0, buf_offset = 0;
    for (act_lda=1; act_lda<ndim; act_lda++){
      idx_offset += phase_rank[act_lda]*edge_lda[act_lda];
    } 
   
    memset(idx, 0, ndim*sizeof(int));
    imax = edge_len[0]/phase[0];
    for (;;){
      if (sym[0] != NS)
        imax = idx[1]+1;
      /* Increment virtual bucket */
      for (i=0; i<imax; i++){
        if (pr_offset >= size)
          break;
        else {
          if (pairs[pr_offset].k == (key)(idx_offset
                +i*phase[0]+phase_rank[0])){
            if (rw == 'r'){
              /* FIXME: should it be the opposite? */
              pairs[pr_offset].d = alpha*data[buf_offset+i]+beta*pairs[pr_offset].d;
            } else {
              LIBT_ASSERT(rw =='w');
              data[(int)buf_offset+i] = beta*data[(int)buf_offset+i]+alpha*pairs[pr_offset].d;
            }
            pr_offset++;
          } else {
/*          DEBUG_PRINTF("%d key[%d] %d not matched with %d\n", 
                          (int)pairs[pr_offset-1].k,
                          pr_offset, (int)pairs[pr_offset].k,
                          (idx_offset+i*phase[0]+phase_rank[0]));*/
          }
        }
      }
      buf_offset += imax;
      if (pr_offset >= size)
        break;
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
      if (act_lda == ndim) break;
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
    if (act_lda == ndim) break;
  }
  TAU_FSTOP(readwrite);
  //printf("pr_offset = %lld/%lld\n",pr_offset,size);
  LIBT_ASSERT(pr_offset == size);
  free_buffer_space(idx);
  free_buffer_space(virt_rank);
  free_buffer_space(edge_lda);
}

/**
 * \brief read or write pairs from / to tensor
 * \param[in] ndim tensor dimension
 * \param[in] np number of processors
 * \param[in] nwrite number of pairs
 * \param[in] alpha multiplier for new value
 * \param[in] beta multiplier for old value
 * \param[in] need_pad whether tensor is padded
 * \param[in] rw whether to read 'r' or write 'w'
 * \param[in] num_virt new total virtualization factor
 * \param[in] sym symmetries of tensor
 * \param[in] edge_len tensor edge lengths
 * \param[in] padding padding of tensor
 * \param[in] phys_phase total phase in each dimension
 * \param[in] virt_phase virtualization in each dimension
 * \param[in] virt_phase_rank virtualized rank in total phase
 * \param[in] bucket_lda prefix sum of the processor grid
 * \param[in,out] wr_pairs pairs to read or write
 * \param[in,out] rw_data data to read from or write to
 * \param[in] glb_comm the global communicator
 */
template<typename dtype>
void wr_pairs_layout(int const          ndim,
                     int const          np,
                     long_int const     nwrite,
                     double const       alpha,  
                     double const       beta,  
                     int const          need_pad,
                     char const         rw,
                     int const          num_virt,
                     int const *        sym,
                     int const *        edge_len,
                     int const *        padding,
                     int const *        phys_phase,
                     int const *        virt_phase,
                     int *              virt_phys_rank,
                     int const *        bucket_lda,
                     tkv_pair<dtype> *  wr_pairs,
                     dtype *            rw_data,
                     CommData_t *       glb_comm){

  long_int i, new_num_pair;
  int * bucket_counts, * recv_counts;
  int * recv_displs, * send_displs;
  int * depadding, * depad_edge_len;
  tkv_pair<dtype> * swap_data, * buf_data, * el_loc;

  get_buffer_space(nwrite*sizeof(tkv_pair<dtype>),      (void**)&buf_data);
  get_buffer_space(nwrite*sizeof(tkv_pair<dtype>),      (void**)&swap_data);
  get_buffer_space(np*sizeof(int),                      (void**)&bucket_counts);
  get_buffer_space(np*sizeof(int),                      (void**)&recv_counts);
  get_buffer_space(np*sizeof(int),                      (void**)&send_displs);
  get_buffer_space(np*sizeof(int),                      (void**)&recv_displs);

  TAU_FSTART(wr_pairs_layout);

  /* Copy out the input data, do not touch that array */
  memcpy(swap_data, wr_pairs, nwrite*sizeof(tkv_pair<dtype>));

  /* If the packed tensor is padded, pad keys */
  if (need_pad){
    get_buffer_space(ndim*sizeof(int), (void**)&depad_edge_len);
    for (i=0; i<ndim; i++){
      depad_edge_len[i] = edge_len[i] - padding[i];
    } 
    pad_key(ndim, nwrite, depad_edge_len, padding, swap_data);
    free_buffer_space(depad_edge_len);
  }

  /* Figure out which processor the value in a packed layout, lies for each key */
  bucket_by_pe(ndim, nwrite, np, 
               phys_phase, virt_phase, bucket_lda, 
               edge_len, swap_data, bucket_counts, 
               send_displs, buf_data);

  /* Exchange send counts */
  ALL_TO_ALL(bucket_counts, 1, COMM_INT_T, 
             recv_counts, 1, COMM_INT_T, glb_comm);

  /* calculate offsets */
  recv_displs[0] = 0;
  for (i=1; i<np; i++){
    recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
  }
  new_num_pair = recv_displs[np-1] + recv_counts[np-1];

  for (i=0; i<np; i++){
    bucket_counts[i] = bucket_counts[i]*sizeof(tkv_pair<dtype>);
    send_displs[i] = send_displs[i]*sizeof(tkv_pair<dtype>);
    recv_counts[i] = recv_counts[i]*sizeof(tkv_pair<dtype>);
    recv_displs[i] = recv_displs[i]*sizeof(tkv_pair<dtype>);
  }

  if (new_num_pair > nwrite){
    free_buffer_space(swap_data);
    get_buffer_space(sizeof(tkv_pair<dtype>)*new_num_pair, (void**)&swap_data);
  }

  /* Exchange data according to counts/offsets */
  ALL_TO_ALLV(buf_data, bucket_counts, send_displs, MPI_CHAR,
              swap_data, recv_counts, recv_displs, MPI_CHAR, glb_comm);
  

  /*printf("[%d] old_num_pair = %d new_num_pair = %d\n", 
                glb_comm->rank, nwrite, new_num_pair);*/

  if (new_num_pair > nwrite){
    free_buffer_space(buf_data);
    get_buffer_space(sizeof(tkv_pair<dtype>)*new_num_pair, (void**)&buf_data);
  }

  /* Figure out what virtual bucket each key belongs to. Bucket
     and sort them accordingly */
  bucket_by_virt(ndim, num_virt, new_num_pair, virt_phase, 
                     edge_len, swap_data, buf_data);

  /* Write or read the values corresponding to the keys */
  readwrite(ndim,       new_num_pair,   alpha,
            beta,       num_virt,       edge_len,   
            sym,        phys_phase,     virt_phase,
            virt_phys_rank,     
            rw_data,    buf_data,       rw);

  /* If we want to read the keys, we must return them to where they
     were requested */
  if (rw == 'r'){
    get_buffer_space(ndim*sizeof(int), (void**)&depadding);
    /* Sort the key-value pairs we determine*/
    std::sort(buf_data, buf_data+new_num_pair);
    /* Search for the keys in the order in which we received the keys */
    for (i=0; i<new_num_pair; i++){
      el_loc = std::lower_bound(buf_data, 
                                buf_data+new_num_pair, 
                                swap_data[i]);
#if (DEBUG>=5)
      if (el_loc < buf_data || el_loc >= buf_data+new_num_pair){
        DEBUG_PRINTF("swap_data[%d].k = %d, not found\n", i, (int)swap_data[i].k);
        LIBT_ASSERT(0);
      }
#endif
      swap_data[i].d = el_loc->d;
    }
  
    /* Inverse the transpose we did above to get the keys back to requestors */
    ALL_TO_ALLV(swap_data, recv_counts, recv_displs, MPI_CHAR,
                buf_data, bucket_counts, send_displs, MPI_CHAR, glb_comm);
    

    /* unpad the keys if necesary */
    if (need_pad){
      for (i=0; i<ndim; i++){
        depadding[i] = -padding[i];
      } 
      pad_key(ndim, nwrite, edge_len, depadding, buf_data);
    }

    /* Sort the pairs that were sent out, now with correct values */
    std::sort(buf_data, buf_data+nwrite);
    /* Search for the keys in the same order they were requested */
    for (i=0; i<nwrite; i++){
      el_loc = std::lower_bound(buf_data, buf_data+nwrite, wr_pairs[i]);
      wr_pairs[i].d = el_loc[0].d;
    }
    free_buffer_space(depadding);
  }
  TAU_FSTOP(wr_pairs_layout);

  free_buffer_space(swap_data);
  free_buffer_space(buf_data);
  free_buffer_space((void*)bucket_counts);
  free_buffer_space((void*)recv_counts);
  free_buffer_space((void*)send_displs);
  free_buffer_space((void*)recv_displs);

}

/**
 * \brief read tensor pairs local to processor
 * \param[in] ndim tensor dimension
 * \param[in] nval number of local values
 * \param[in] pad whether tensor is padded
 * \param[in] num_virt new total virtualization factor
 * \param[in] sym symmetries of tensor
 * \param[in] edge_len tensor edge lengths
 * \param[in] padding padding of tensor
 * \param[in] virt_dim virtualization in each dimension
 * \param[in] virt_phase total phase in each dimension
 * \param[in] virt_phase_rank virtualized rank in total phase
 * \param[in] bucket_lda prefix sum of the processor grid
 * \param[out] nread number of local pairs read
 * \param[in] tensor data data to read from
 * \param[out] pairs local pairs read
 */
template<typename dtype>
void read_loc_pairs(int const           ndim,
                    long_int const      nval,
                    int const           pad,
                    int const           num_virt,
                    int const *         sym,
                    int const *         edge_len,
                    int const *         padding,
                    int const *         virt_dim,
                    int const *         virt_phase,
                    int *               virt_phase_rank,
                    long_int *          nread,
                    dtype const *       data,
                    tkv_pair<dtype> **  pairs){

  long_int i;
  tkv_pair<dtype> * dpairs;
  get_buffer_space(sizeof(tkv_pair<dtype>)*nval, (void**)&dpairs);
  /* Iterate through packed layout and form key value pairs */
  assign_keys(ndim,             nval,           num_virt,
              edge_len,         sym,
              virt_phase,       virt_dim,       virt_phase_rank,
              data,             dpairs);

  /* If we need to unpad */
  if (pad){
    long_int new_num_pair;
    int * depadding, * pad_len;
    tkv_pair<dtype> * new_pairs;
    get_buffer_space(sizeof(tkv_pair<dtype>)*nval, (void**)&new_pairs);
    get_buffer_space(sizeof(int)*ndim, (void**)&depadding);
    get_buffer_space(sizeof(int)*ndim, (void**)&pad_len);

    for (i=0; i<ndim; i++){
      pad_len[i] = edge_len[i]-padding[i];
    }
    /* Get rid of any padded values */
    depad_tsr(ndim, nval, pad_len, sym, padding,
              dpairs, new_pairs, &new_num_pair);

    free_buffer_space(dpairs);
    *pairs = new_pairs;
    *nread = new_num_pair;

    for (i=0; i<ndim; i++){
      depadding[i] = -padding[i];
    }
    
    /* Adjust keys to remove padding */
    pad_key(ndim, new_num_pair, edge_len, depadding, new_pairs);
    free_buffer_space((void*)pad_len);
    free_buffer_space((void*)depadding);
  } else {
    *pairs = dpairs;
    *nread = nval;
  }
}

/**
 * \brief desymmetrizes one index of a tensor
 * \param[in] ndim dimension of tensor
 * \param[in] edge_len edge lengths of tensor
 * \param[in] sign -1 if subtraction +1 if adding
 * \param[in] permute whether to permute symmetic indices
 * \param[in] sym_read symmetry of the symmetrized tensor
 * \param[in] sym_write symmetry of the desymmetrized tensor
 * \param[in] data_read data of the symmetrized tensor
 * \param[in,out] data_write data of the desymmetrized tensor 
 */
template<typename dtype>
void rw_smtr(int const          ndim,
             int const *        edge_len,
             double const       sign,
             int const          permute,
             int const *        sym_read,
             int const *        sym_write,
             dtype const *      data_read,
             dtype *            data_write){

  /***** UNTESTED AND LIKELY UNNECESSARY *****/

  int sym_idx, j, k, is_symm;
  long_int i, l, bottom_size, top_size;
  long_int write_idx, read_idx;
  
  /* determine which symmetry is broken and return if none */
  sym_idx = -1;
  is_symm = 0;
  for (i=0; i<ndim; i++){
    if (sym_read[i] != sym_write[i]){
      sym_idx = i;
    }
  }
  if (sym_idx == -1) return;

  /* determine the size of the index spaces below and above the symmetric indices */
  bottom_size   = packed_size(sym_idx, edge_len, sym_read);
  top_size      = packed_size(ndim-sym_idx-2, edge_len+sym_idx+2, sym_read+sym_idx+2);

  read_idx = 0;
  write_idx = 0;
  /* iterate on top of the symmetric indices which are the same for both tensors */
  for (i=0; i<top_size; i++){
    for (;;){
      /* iterate among the two indices whose symmetry is broken */
      for (j=0; j<edge_len[sym_idx]; j++){
        for (k=0; k<edge_len[sym_idx+1]; k++){
          if (j<=k){
            for (l=0; l<bottom_size; l++){
              data_write[write_idx] += sign*data_read[read_idx];
              write_idx++;
              read_idx++;
            } 
          } else {
            read_idx+=bottom_size;
          }
        }
        /* increment symmetrization mirror index in a transposed way */
        if (permute)
          write_idx+=(edge_len[sym_idx]-1)*bottom_size;
      }
      if (permute)
        write_idx-=(edge_len[sym_idx]-1)*bottom_size;
    }
  }     
}



#endif
