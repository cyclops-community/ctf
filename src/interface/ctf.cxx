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

//#ifdef ENABLE_ASSERT
#define DTASSERT(...) assert(__VA_ARGS__)
//#else
//#define DTASSERT(...)
//#endif

template<typename T>
int conv_idx(int const  ndim,
             T const *  cidx,
             int **     iidx){
  int i, j, n;
  T c;

  *iidx = (int*)malloc(sizeof(int)*ndim);

  n = 0;
  for (i=0; i<ndim; i++){
    c = cidx[i];
    for (j=0; j<i; j++){
      if (c == cidx[j]){
        (*iidx)[i] = (*iidx)[j];
        break;
      }
    }
    if (j==i){
      (*iidx)[i] = n;
      n++;
    }
  }
  return n;
}

template<typename T>
int  conv_idx(int const         ndim_A,
              T const *         cidx_A,
              int **            iidx_A,
              int const         ndim_B,
              T const *         cidx_B,
              int **            iidx_B){
  int i, j, n;
  T c;

  *iidx_B = (int*)malloc(sizeof(int)*ndim_B);

  n = conv_idx(ndim_A, cidx_A, iidx_A);
  for (i=0; i<ndim_B; i++){
    c = cidx_B[i];
    for (j=0; j<ndim_A; j++){
      if (c == cidx_A[j]){
        (*iidx_B)[i] = (*iidx_A)[j];
        break;
      }
    }
    if (j==ndim_A){
      for (j=0; j<i; j++){
        if (c == cidx_B[j]){
          (*iidx_B)[i] = (*iidx_B)[j];
          break;
        }
      }
      if (j==i){
        (*iidx_B)[i] = n;
        n++;
      }
    }
  }
  return n;
}


template<typename T>
int  conv_idx(int const         ndim_A,
              T const *         cidx_A,
              int **            iidx_A,
              int const         ndim_B,
              T const *         cidx_B,
              int **            iidx_B,
              int const         ndim_C,
              T const *         cidx_C,
              int **            iidx_C){
  int i, j, n;
  T c;

  *iidx_C = (int*)malloc(sizeof(int)*ndim_C);

  n = conv_idx(ndim_A, cidx_A, iidx_A,
               ndim_B, cidx_B, iidx_B);

  for (i=0; i<ndim_C; i++){
    c = cidx_C[i];
    for (j=0; j<ndim_B; j++){
      if (c == cidx_B[j]){
        (*iidx_C)[i] = (*iidx_B)[j];
        break;
      }
    }
    if (j==ndim_B){
      for (j=0; j<ndim_A; j++){
        if (c == cidx_A[j]){
          (*iidx_C)[i] = (*iidx_A)[j];
          break;
        }
      }
      if (j==ndim_A){
        for (j=0; j<i; j++){
          if (c == cidx_C[j]){
            (*iidx_C)[i] = (*iidx_C)[j];
            break;
          }
        }
        if (j==i){
          (*iidx_C)[i] = n;
          n++;
        }
      }
    }
  }
  return n;
}

template<typename dtype>
tCTF_World<dtype>::tCTF_World(MPI_Comm comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, MACHINE_BGQ, rank, np);
#else
#ifdef BGP
  ctf->init(comm, MACHINE_BGP, rank, np);
#else
  ctf->init(comm, MACHINE_8D, rank, np);
#endif
#endif
}

template<typename dtype>
tCTF_World<dtype>::tCTF_World(int const   ndim, 
                              int const * lens, 
                              MPI_Comm    comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
  ctf->init(comm, rank, np, ndim, lens);
}

template<typename dtype>
tCTF_World<dtype>::~tCTF_World(){
  delete ctf;
}

template<typename dtype>
tCTF_Tensor<dtype>::tCTF_Tensor(const tCTF_Tensor<dtype>& A,
                                const bool                copy){
  int ret;
  world = A.world;

  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);

  if (copy){
    ret = world->ctf->copy_tensor(A.tid, tid);
    DTASSERT(ret == DIST_TENSOR_SUCCESS);
  }
}

template<typename dtype>
tCTF_Tensor<dtype>::tCTF_Tensor(const int           ndim_,
                                const int *         len_,
                                const int *         sym_,
                                tCTF_World<dtype> * world_){
  int ret;
  world = world_;

  ret = world->ctf->define_tensor(ndim_, len_, sym_, &tid);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  ret = world->ctf->info_tensor(tid, &ndim, &len, &sym);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
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
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  
  return data;
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_local_data(int64_t *   npair, 
                                        int64_t **  global_idx, 
                                        dtype **   data) const {
  tkv_pair< dtype > * pairs;
  int ret, i;
  ret = world->ctf->read_local_tensor(tid, npair, &pairs);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
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
  ret = world->ctf->write_tensor(tid, npair, pairs);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  for (i=0; i<npair; i++){
    data[i] = pairs[i].d;
  }
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::write_remote_data(int64_t const    npair, 
                                           int64_t const *  global_idx, 
                                           dtype const *    data) const {
  int ret, i;
  tkv_pair< dtype > * pairs;
  pairs = (tkv_pair< dtype >*)malloc(npair*sizeof(tkv_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, pairs);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::add_remote_data(int64_t const    npair, 
                                         double const     alpha, 
                                         double const     beta,
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
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  free(pairs);
}

template<typename dtype>
void tCTF_Tensor<dtype>::get_all_data(int64_t * npair, dtype ** vals) const {
  int ret;
  ret = world->ctf->allread_tensor(tid, npair, vals);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::contract(const dtype                   alpha,
                                  const tCTF_Tensor<dtype>&     A,
                                  const char *                  idx_A,
                                  const tCTF_Tensor<dtype>&     B,
                                  const char *                  idx_B,
                                  const dtype                   beta,
                                  const char *                  idx_C) {
  int ret;
  CTF_ctr_type_t tp;
  tp.tid_A = A.tid;
  tp.tid_B = B.tid;
  tp.tid_C = tid;
  conv_idx(A.ndim, idx_A, &tp.idx_map_A,
           B.ndim, idx_B, &tp.idx_map_B,
           ndim, idx_C, &tp.idx_map_C);
  ret = world->ctf->contract(&tp, alpha, beta);
  free(tp.idx_map_A);
  free(tp.idx_map_B);
  free(tp.idx_map_C);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
tCTF_Tensor<dtype>& tCTF_Tensor<dtype>::operator=(const dtype val){
  int64_t size;
  dtype* raw_data = get_raw_data(&size);
  std::fill(raw_data, raw_data+size, val);
  return *this;
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
                             const char *               idx_B){
  int ret;
  int * idx_map_A, * idx_map_B;
  CTF_sum_type_t st;
  conv_idx(A.ndim, idx_A, &idx_map_A,
           ndim, idx_B, &idx_map_B);
  st.idx_map_A = idx_map_A;
  st.idx_map_B = idx_map_B;
  st.tid_A = A.tid;
  st.tid_B = tid;
  ret = world->ctf->sum_tensors(&st, alpha, beta);
  free(idx_map_A);
  free(idx_map_B);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
void tCTF_Tensor<dtype>::scale(const dtype alpha, const char * idx_A){
  int ret;
  int * idx_map_A;
  conv_idx(ndim, idx_A, &idx_map_A);
  ret = world->ctf->scale_tensor(alpha, tid, idx_map_A);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
}

template<typename dtype>
dtype tCTF_Tensor<dtype>::reduce(CTF_OP op){
  int ret;
  dtype ans;
  ans = 0.0;
  ret = world->ctf->reduce_tensor(tid, op, &ans);
  DTASSERT(ret == DIST_TENSOR_SUCCESS);
  return ans;
}

template class tCTF_World<double>;
template class tCTF_World< std::complex<double> >;
template class tCTF_Tensor<double>;
template class tCTF_Tensor< std::complex<double> >;
