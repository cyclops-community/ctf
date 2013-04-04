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

  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);

  if (copy){
    ret = world->ctf->copy_tensor(A.tid, tid);
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  }
}

template<typename dtype>
tCTF_Tensor<dtype>::tCTF_Tensor(const int           ndim_,
                                const int *         len_,
                                const int *         sym_,
                                tCTF_World<dtype> & world_){
  int ret;
  world = &world_;

  ret = world->ctf->define_tensor(ndim_, len_, sym_, &tid);
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
dtype * tCTF_Tensor<dtype>::get_raw_data(int64_t * size) {
  int ret;
  dtype * data;
  ret = world->ctf->get_raw_data(tid, &data, size);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  
  return data;
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_local_data(int64_t *   npair, 
                                        int64_t **  global_idx, 
                                        dtype **   data) const {
  tkv_pair< dtype > * pairs;
  int ret, i;
  ret = world->ctf->read_local_tensor(tid, npair, &pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  /* FIXME: careful with malloc */
  *global_idx = (int64_t*)malloc((*npair)*sizeof(int64_t));
  *data = (dtype*)malloc((*npair)*sizeof(dtype));
  for (i=0; i<(*npair); i++){
    (*global_idx)[i] = pairs[i].k;
    (*data)[i] = pairs[i].d;
  }
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_remote_data(int64_t const    npair, 
                                         int64_t const *  global_idx, 
                                         dtype *          data) const {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)malloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
  }
  ret = world->ctf->read_tensor(tid, npair, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  for (i=0; i<npair; i++){
    data[i] = pairs[i].d;
  }
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::write_remote_data(int64_t const    npair, 
                                           int64_t const *  global_idx, 
                                           dtype const *    data) {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)malloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::add_remote_data(int64_t const    npair, 
                                         dtype const      alpha, 
                                         dtype const      beta,
                                         int64_t const *  global_idx, 
                                         dtype const *    data) {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)malloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_all_data(int64_t * npair, dtype ** vals) const {
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
  free(tp.idx_map_A);
  free(tp.idx_map_B);
  free(tp.idx_map_C);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
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
  free(idx_map_A);
  free(idx_map_B);
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
tCTF_Tensor<dtype>& tCTF_Tensor<dtype>::operator=(const dtype val){
  int64_t size;
  dtype* raw_data = get_raw_data(&size);
  std::fill(raw_data, raw_data+size, val);
  return *this;
}


template<typename dtype>
tCTF_Idx_Tensor<dtype>& tCTF_Tensor<dtype>::operator[](const char * idx_map_){
  tCTF_Idx_Tensor<dtype> * itsr = new tCTF_Idx_Tensor<dtype>(this, idx_map_);
  return *itsr;
}


template class tCTF_Tensor<double>;
template class tCTF_Tensor< std::complex<double> >;
