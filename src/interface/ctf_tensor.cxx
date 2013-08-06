/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "../../include/ctf.hpp"
#include "../shared/util.h"

template<typename dtype>
tCTF_Tensor<dtype>::tCTF_Tensor(const tCTF_Tensor<dtype>& A,
                                const bool                copy){
  int ret;
  world = A.world;
  name = NULL;

  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  //printf("Defined tensor %d to be the same as %d, copy=%d\n", tid, A.tid, (int)copy);

  if (copy){
    ret = world->ctf->copy_tensor(A.tid, tid);
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  }
}

template<typename dtype>
tCTF_Tensor<dtype>::tCTF_Tensor(const int           ndim_,
                                const int *         len_,
                                const int *         sym_,
                                tCTF_World<dtype> & world_,
                                char const *        name_,
                                int const           profile_){
  int ret;
  world = &world_;
  name = name_;
  ret = world->ctf->define_tensor(ndim_, len_, sym_, &tid, name_, profile_);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  ret = world->ctf->info_tensor(tid, &ndim, &len, &sym);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
tCTF_Tensor<dtype>::~tCTF_Tensor(){
  free(sym);
  free(len);
  world->ctf->clean_tensor(tid);
}

template<typename dtype>
dtype * tCTF_Tensor<dtype>::get_raw_data(long_int * size) {
  int ret;
  dtype * data;
  ret = world->ctf->get_raw_data(tid, &data, size);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  
  return data;
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_local_data(long_int *   npair, 
                                        long_int **  global_idx, 
                                        dtype **   data) const {
  tkv_pair< dtype > * pairs;
  int ret, i;
  ret = world->ctf->read_local_tensor(tid, npair, &pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  /* FIXME: careful with CTF_alloc */
  *global_idx = (long_int*)malloc((*npair)*sizeof(long_int));
  *data = (dtype*)malloc((*npair)*sizeof(dtype));
  for (i=0; i<(*npair); i++){
    (*global_idx)[i] = pairs[i].k;
    (*data)[i] = pairs[i].d;
  }
  if (*npair > 0)
    free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_remote_data(long_int const    npair, 
                                         long_int const *  global_idx, 
                                         dtype *          data) const {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)CTF_alloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
  }
  ret = world->ctf->read_tensor(tid, npair, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  for (i=0; i<npair; i++){
    data[i] = pairs[i].d;
  }
  CTF_free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::write_remote_data(long_int const    npair, 
                                           long_int const *  global_idx, 
                                           dtype const *    data) {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)CTF_alloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  CTF_free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::add_remote_data(long_int const    npair, 
                                         dtype const      alpha, 
                                         dtype const      beta,
                                         long_int const *  global_idx, 
                                         dtype const *    data) {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)CTF_alloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  CTF_free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_all_data(long_int * npair, dtype ** vals) const {
  int ret;
  ret = world->ctf->allread_tensor(tid, npair, vals);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::contract(const dtype                   alpha,
                                  const tCTF_Tensor<dtype>&     A,
                                  const char *                  idx_A,
                                  const tCTF_Tensor<dtype>&     B,
                                  const char *                  idx_B,
                                  const dtype                   beta,
                                  const char *                  idx_C,
                                  tCTF_fctr<dtype>              fseq){
  int ret;
  CTF_ctr_type_t tp;
  tp.tid_A = A.tid;
  tp.tid_B = B.tid;
  tp.tid_C = tid;
  conv_idx(A.ndim, idx_A, &tp.idx_map_A,
           B.ndim, idx_B, &tp.idx_map_B,
           ndim, idx_C, &tp.idx_map_C);
  if (fseq.func_ptr == NULL)
    ret = world->ctf->contract(&tp, alpha, beta);
  else {
    fseq_elm_ctr<dtype> fs;
    fs.func_ptr = fseq.func_ptr;
    ret = world->ctf->contract(&tp, fs, alpha, beta);
  }
  CTF_free(tp.idx_map_A);
  CTF_free(tp.idx_map_B);
  CTF_free(tp.idx_map_C);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::set_name(char const * name_) {
  name = name_;
  world->ctf->set_name(tid, name_);
}

template<typename dtype>
void tCTF_Tensor<dtype>::profile_on() {
  world->ctf->profile_on(tid);
}

template<typename dtype>
void tCTF_Tensor<dtype>::profile_off() {
  world->ctf->profile_off(tid);
}

template<typename dtype>
void tCTF_Tensor<dtype>::print(FILE* fp) const{
  world->ctf->print_tensor(fp, tid);
}

template<typename dtype>
void tCTF_Tensor<dtype>::sum(const dtype                alpha,
                             const tCTF_Tensor<dtype>&  A,
                             const char *               idx_A,
                             const dtype                beta,
                             const char *               idx_B,
                             tCTF_fsum<dtype>           fseq){
  int ret;
  int * idx_map_A, * idx_map_B;
  CTF_sum_type_t st;
  conv_idx(A.ndim, idx_A, &idx_map_A,
           ndim, idx_B, &idx_map_B);
  st.idx_map_A = idx_map_A;
  st.idx_map_B = idx_map_B;
  st.tid_A = A.tid;
  st.tid_B = tid;
  if (fseq.func_ptr == NULL)
    ret = world->ctf->sum_tensors(&st, alpha, beta);
  else {
    fseq_elm_sum<dtype> fs;
    fs.func_ptr = fseq.func_ptr;
    ret = world->ctf->sum_tensors(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fs);
  }
  CTF_free(idx_map_A);
  CTF_free(idx_map_B);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::scale(const dtype            alpha, 
                               const char *           idx_A,
                               tCTF_fscl<dtype>       fseq){
  int ret;
  int * idx_map_A;
  conv_idx(ndim, idx_A, &idx_map_A);
  if (fseq.func_ptr == NULL)
    ret = world->ctf->scale_tensor(alpha, tid, idx_map_A);
  else{
    fseq_elm_scl<dtype> fs;
    fs.func_ptr = fseq.func_ptr;
    ret = world->ctf->scale_tensor(alpha, tid, idx_map_A, fs);
  }
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::sum_slice(int const *    offsets,
                                   int const *    ends,
                                   double         beta,
                                   tCTF_Tensor &  A,
                                   int const *    offsets_A,
                                   int const *    ends_A,
                                   double         alpha){
  int ret =  world->ctf->slice_tensor(A.tid, offsets_A, ends_A, alpha,
                                      tid, offsets, ends, beta);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}


template<typename dtype>
tCTF_Tensor<dtype> tCTF_Tensor<dtype>::slice(int const * offsets,
                                               int const * ends){
  int i;
  int * new_lens = (int*)CTF_alloc(sizeof(int)*ndim);
  int * new_sym = (int*)CTF_alloc(sizeof(int)*ndim);
  for (i=0; i<ndim; i++){
    LIBT_ASSERT(ends[i] - offsets[i] > 0 && 
                offsets[i] >= 0 && 
                ends[i] <= len[i]);
    if (sym[i] != NS){
      if (offsets[i] == offsets[i+1] && ends[i] == ends[i+1]){
        new_sym[i] = sym[i];
      } else {
        LIBT_ASSERT(ends[i+1] >= offsets[i]);
        new_sym[i] = NS;
      }
    } else new_sym[i] = NS;
    new_lens[i] = ends[i] - offsets[i];
  }
  tCTF_Tensor<dtype> new_tsr(ndim, new_lens, new_sym, *world);
  std::fill(new_sym, new_sym+ndim, 0);
  new_tsr.sum_slice(new_sym, new_lens, 0.0, *this, offsets, ends, 1.0);
  CTF_free(new_lens);
  CTF_free(new_sym);
  return new_tsr;
}

template<typename dtype>
void tCTF_Tensor<dtype>::align(const tCTF_Tensor& A){
  int ret = world->ctf->align(tid, A.tid);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
dtype tCTF_Tensor<dtype>::reduce(CTF_OP op){
  int ret;
  dtype ans;
  ans = 0.0;
  ret = world->ctf->reduce_tensor(tid, op, &ans);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  return ans;
}
template<typename dtype>
void tCTF_Tensor<dtype>::get_max_abs(int const  n,
                                     dtype *    data){
  int ret;
  ret = world->ctf->get_max_abs(tid, n, data);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
tCTF_Tensor<dtype>& tCTF_Tensor<dtype>::operator=(const dtype val){
  long_int size;
  dtype* raw_data = get_raw_data(&size);
  std::fill(raw_data, raw_data+size, val);
  return *this;
}

template<typename dtype>
void tCTF_Tensor<dtype>::operator=(tCTF_Tensor<dtype> A){
  int ret;  

  world = A.world;

  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

  ret = world->ctf->copy_tensor(A.tid, tid);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}
    

template<typename dtype>
tCTF_Idx_Tensor<dtype>& tCTF_Tensor<dtype>::operator[](const char * idx_map_){
  tCTF_Idx_Tensor<dtype> * itsr = new tCTF_Idx_Tensor<dtype>(this, idx_map_);
  return *itsr;
}


template class tCTF_Tensor<double>;
template class tCTF_Tensor< std::complex<double> >;
