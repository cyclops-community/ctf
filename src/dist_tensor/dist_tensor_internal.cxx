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

#include "dist_tensor_internal.h"
#include "dt_aux_permute.hxx"
#include "dt_aux_sort.hxx"
#include "dt_aux_rw.hxx"
#include "dt_aux_map.hxx"
#include "dt_aux_topo.hxx"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "../ctr_comm/ctr_comm.h"
#include "../ctr_comm/ctr_tsr.h"
#include "../ctr_comm/sum_tsr.h"
#include "../ctr_comm/strp_tsr.h"
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <vector>
#include <algorithm>

#define MAX_NVIRT 256
#ifndef MIN_NVIRT
#define MIN_NVIRT 1
#endif
//#define USE_VIRT_25D


template <> inline
double dist_tensor<double>::GET_REAL(double const d) const{
  return d;
}
template <> inline
double dist_tensor< std::complex<double> >::GET_REAL(std::complex<double> const d) const{
  return d.real();
}
template <typename dtype>
double dist_tensor<dtype>::GET_REAL(dtype const d) const{
  ABORT;
  return 42.0;
}

/* accessors */
template<typename dtype>
CommData_t * dist_tensor<dtype>::get_global_comm(){ return global_comm; }
template<typename dtype>
void dist_tensor<dtype>::set_global_comm(CommData_t * cdt){ global_comm = cdt; }

/**
 * \brief deallocates all internal state
 */
template<typename dtype>
int dist_tensor<dtype>::dist_cleanup(){
  int j;
  std::vector<topology>::iterator iter;
  FREE_CDT(global_comm);
  free(global_comm);

  for (iter=topovec.begin(); iter<topovec.end(); iter++){
    for (j=0; j<iter->ndim; j++){
      FREE_CDT(iter->dim_comm[j]);
 //     free(iter->dim_comm[j]); //FIXME folded communicator pointers are replicated
    }
    free(iter->dim_comm);
    free(iter->lda);
  }
  topovec.clear();
#if INNER_MAP
  for (iter=inner_topovec.begin(); iter<inner_topovec.end(); iter++){
    free(iter->dim_comm);
  }
  inner_topovec.clear();
#endif
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief destructor
 */
template<typename dtype>
dist_tensor<dtype>::~dist_tensor(){
  int ret;
  ret = dist_cleanup();
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
}


/**
 * \brief  initializes library.
 * \param[in] ndim is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 * \param[in] inner_size is the total block size of dgemm calls
 */
template<typename dtype>
int dist_tensor<dtype>::initialize(CommData_t * cdt_global,
                                   int const    ndim,
                                   int const *  dim_len,
                                   int const    inner_sz){
  int i, rank, stride, cut;
  int * srt_dim_len;

  get_buffer_space(ndim*sizeof(int), (void**)&srt_dim_len);
  memcpy(srt_dim_len, dim_len, ndim*sizeof(int));

  rank = cdt_global->rank;

  /* setup global communicator */
  set_global_comm(cdt_global);

  /* setup dimensional communicators */
  CommData_t ** phys_comm = (CommData_t**)malloc(ndim*sizeof(CommData_t*));

/* FIXME: Sorting will fuck up dimensional ordering */
//  std::sort(srt_dim_len, srt_dim_len + ndim);

#if DEBUG >= 1
  if (cdt_global->rank == 0)
    printf("Setting up initial torus topology:\n");
#endif
  stride = 1, cut = 0;
  for (i=0; i<ndim; i++){
    LIBT_ASSERT(dim_len[i] != 1);
#if DEBUG >= 1
    if (cdt_global->rank == 0)
      printf("dim[%d] = %d:\n",i,srt_dim_len[i]);
#endif

    phys_comm[i] = (CommData_t*)malloc(sizeof(CommData_t));
    SETUP_SUB_COMM(cdt_global, phys_comm[i],
                   ((rank/stride)%srt_dim_len[ndim-i-1]),
                   (((rank/(stride*srt_dim_len[ndim-i-1]))*stride)+cut),
                   srt_dim_len[ndim-i-1], NREQ, NBCAST);
    stride*=srt_dim_len[ndim-i-1];
    cut = (rank - (rank/stride)*stride);
  }
  set_phys_comm(phys_comm, ndim);
  free(srt_dim_len);

#if INNER_MAP
  return init_inner_topology(inner_sz);
#else
  return DIST_TENSOR_SUCCESS;
#endif
}



/**
 * \brief sets the physical torus topology
 * \param[in] cdt grid communicator
 * \param[in] ndim number of dimensions
 */
template<typename dtype>
void dist_tensor<dtype>::set_phys_comm(CommData_t ** cdt, int const ndim){
  int i, lda;
  topology new_topo;


  new_topo.ndim = ndim;
  new_topo.dim_comm = cdt;

  /* do not duplicate topologies */
  if (find_topology(&new_topo, topovec) != -1){
    /*for (i=0; i<ndim; i++){
      FREE_CDT(cdt[i]);
    }*/
    free(cdt);
    return;
  }

  new_topo.lda = (int*)malloc(sizeof(int)*ndim);
  lda = 1;
  /* Figure out the lda of each dimension communicator */
  for (i=0; i<ndim; i++){
    LIBT_ASSERT(cdt[i]->np != 1);
    new_topo.lda[i] = lda;
    lda = lda*cdt[i]->np;
    LIBT_ASSERT(cdt[i]->np > 0);
  }
  topovec.push_back(new_topo);

  if (ndim > 1)
    fold_torus(&new_topo, global_comm, this);
}



/**
 * \brief  defines a tensor and retrieves handle
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] edge_len global edge lengths of tensor
 * \param[in] sym symmetry relations of tensor
 * \param[out] tensor_id the tensor index (handle)
 * \param[in] alloc_data whether this tensor's data should be alloced
 */
template<typename dtype>
int dist_tensor<dtype>::define_tensor( int const          ndim,
                                       int const *        edge_len, 
                                       int const *        sym,
                                       int *              tensor_id,
                                       int const          alloc_data){
  int i;

  tensor<dtype> * tsr = (tensor<dtype>*)malloc(sizeof(tensor<dtype>));
  get_buffer_space(ndim*sizeof(int), (void**)&tsr->padding);
  memset(tsr->padding, 0, ndim*sizeof(int));

  tsr->is_padded          = 1;
  tsr->is_mapped          = 0;
  tsr->itopo              = -1;
  tsr->is_alloced         = 1;
  tsr->is_cyclic          = 1;
  tsr->size               = 0;
  tsr->is_inner_mapped    = 0;
  tsr->is_folded          = 0;
  tsr->is_matrix          = 0;
  tsr->is_data_aliased    = 0;
  tsr->need_remap         = 0;
  tsr->has_zero_edge_len  = 0;

  tsr->pairs    = NULL;
  tsr->ndim     = ndim;
  tsr->edge_len = (int*)malloc(ndim*sizeof(int));
  memcpy(tsr->edge_len, edge_len, ndim*sizeof(int));
  tsr->sym      = (int*)malloc(ndim*sizeof(int));
  memcpy(tsr->sym, sym, ndim*sizeof(int));
//  memcpy(inner_sym, sym, ndim*sizeof(int));
/*  for (i=0; i<ndim; i++){
    if (tsr->sym[i] != NS)
      tsr->sym[i] = SY;
  }*/

  tsr->sym_table = (int*)calloc(ndim*ndim*sizeof(int),1);
  tsr->edge_map  = (mapping*)malloc(sizeof(mapping)*ndim);

  /* initialize map array and symmetry table */
  for (i=0; i<ndim; i++){
    if (tsr->edge_len[i] <= 0) tsr->has_zero_edge_len = 0;
    tsr->edge_map[i].type       = NOT_MAPPED;
    tsr->edge_map[i].has_child  = 0;
    tsr->edge_map[i].np         = 1;
    if (tsr->sym[i] != NS) {
      tsr->sym_table[(i+1)+i*ndim] = 1;
      tsr->sym_table[(i+1)*ndim+i] = 1;
    }
  }

  (*tensor_id) = tensors.size();
  tensors.push_back(tsr);

  /* Set tensor data to zero. */
  if (alloc_data)
    return set_zero_tsr(*tensor_id);
  else
    return DIST_TENSOR_SUCCESS;
}

template<typename dtype>
std::vector< tensor<dtype>* > * dist_tensor<dtype>::get_tensors(){ return &tensors; }

/**
 * \brief malloc abstraction
 * \param[in] len number of bytes
 * \param[in,out] ptr pointer to set to new allocation address
 */
inline
int get_buffer_space(int const len, void ** const ptr){
  int pm;
  //(*ptr) = malloc(len);
  pm = posix_memalign(ptr, ALIGN_BYTES, len);
  LIBT_ASSERT(pm == 0);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief free abstraction
 * \param[in,out] ptr pointer to set to address to free
 */
inline
int free_buffer_space(void * ptr){
  free(ptr);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief sets the data in the tensor
 * \param[in] tensor_id tensor handle,
 * \param[in] num_val new number of data values
 * \param[in] tsr_data new tensor data
 */
template<typename dtype>
int dist_tensor<dtype>::set_tsr_data( int const tensor_id,
                                      int const num_val,
                                      dtype * tsr_data){
  tensors[tensor_id]->data = tsr_data;
  tensors[tensor_id]->size = num_val;
  return DIST_TENSOR_SUCCESS;
}

template<typename dtype>
topology * dist_tensor<dtype>::get_topo(int const itopo) {
  return &topovec[itopo];
}

template<typename dtype>
int dist_tensor<dtype>::get_dim(int const tensor_id) const { return tensors[tensor_id]->ndim; }

template<typename dtype>
int * dist_tensor<dtype>::get_edge_len(int const tensor_id) const {
  int i;
  int * edge_len;
  get_buffer_space(tensors[tensor_id]->ndim*sizeof(int), (void**)&edge_len);

  if (tensors[tensor_id]->is_padded){
    for (i=0; i<tensors[tensor_id]->ndim; i++){
      edge_len[i] = tensors[tensor_id]->edge_len[i]
                   -tensors[tensor_id]->padding[i];
    }
  }
  else {
    memcpy(edge_len, tensors[tensor_id]->edge_len, 
           tensors[tensor_id]->ndim*sizeof(int));
  }

  return edge_len;
}


template<typename dtype>
int * dist_tensor<dtype>::get_sym(int const tensor_id) const {
  int * sym;
  get_buffer_space(tensors[tensor_id]->ndim*sizeof(int), (void**)&sym);
  memcpy(sym, tensors[tensor_id]->sym, tensors[tensor_id]->ndim*sizeof(int));

  return sym;
}

/* \brief get raw data pointer WARNING: includes padding
 * \param[in] tensor_id id of tensor
 * \return raw local data
 */
template<typename dtype>
dtype * dist_tensor<dtype>::get_raw_data(int const tensor_id, int64_t * size) {
  if (tensors[tensor_id]->has_zero_edge_len){
    *size = 0;
    return NULL;
  }
  *size = tensors[tensor_id]->size;
  return tensors[tensor_id]->data;
}

/**
 * \brief pulls out information about a tensor
 * \param[in] tensor_id tensor handle
 * \param[out] ndim the dimensionality of the tensor
 * \param[out] edge_len the number of processors along each dimension
 *             of the tensor
 * \param[out] sym the symmetries of the tensor
 */
template<typename dtype>
int dist_tensor<dtype>::get_tsr_info( int const tensor_id,
                                      int *             ndim,
                                      int **            edge_len,
                                      int **            sym) const{
  int i;
  int nd;
  int * el, * s;

  const tensor<dtype> * tsr = tensors[tensor_id];

  nd = tsr->ndim;
  get_buffer_space(nd*sizeof(int), (void**)&el);
  get_buffer_space(nd*sizeof(int), (void**)&s);
  if (tsr->is_padded){
    for (i=0; i<nd; i++){
      el[i] = tsr->edge_len[i] - tsr->padding[i];
    }
  } else
    memcpy(el, tsr->edge_len, nd*sizeof(int));
  memcpy(s, tsr->sym, nd*sizeof(int));

  *ndim = nd;
  *edge_len = el;
  *sym = s;

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
seq_tsr_scl<dtype>::seq_tsr_scl(scl<dtype> * other) : scl<dtype>(other) {
  seq_tsr_scl<dtype> * o = (seq_tsr_scl<dtype>*)other;
  
  ndim          = o->ndim;
  idx_map       = o->idx_map;
  sym           = o->sym;
  edge_len      = (int*)malloc(sizeof(int)*ndim);
  memcpy(edge_len, o->edge_len, sizeof(int)*ndim);

  func_ptr = o->func_ptr;
}

/**
 * \brief copies scl object
 */
template<typename dtype>
scl<dtype> * seq_tsr_scl<dtype>::clone() {
  return new seq_tsr_scl<dtype>(this);
}


template<typename dtype>
long_int seq_tsr_scl<dtype>::mem_fp(){ return 0; }

/**
 * \brief wraps user sequential function signature
 */
template<typename dtype>
void seq_tsr_scl<dtype>::run(){
  func_ptr.func_ptr(this->alpha,
                    this->A,
                    ndim,
                    edge_len,
                    edge_len,
                    sym,
                    idx_map);
}

/**
 * \brief copies sum object
 */
template<typename dtype>
seq_tsr_sum<dtype>::seq_tsr_sum(tsum<dtype> * other) : tsum<dtype>(other) {
  seq_tsr_sum<dtype> * o = (seq_tsr_sum<dtype>*)other;
  
  ndim_A        = o->ndim_A;
  idx_map_A     = o->idx_map_A;
  sym_A         = o->sym_A;
  edge_len_A    = (int*)malloc(sizeof(int)*ndim_A);
  memcpy(edge_len_A, o->edge_len_A, sizeof(int)*ndim_A);

  ndim_B        = o->ndim_B;
  idx_map_B     = o->idx_map_B;
  sym_B         = o->sym_B;
  edge_len_B    = (int*)malloc(sizeof(int)*ndim_B);
  memcpy(edge_len_B, o->edge_len_B, sizeof(int)*ndim_B);
  
  is_inner      = o->is_inner;
  inr_stride    = o->inr_stride;

  func_ptr = o->func_ptr;
}

/**
 * \brief copies sum object
 */
template<typename dtype>
tsum<dtype> * seq_tsr_sum<dtype>::clone() {
  return new seq_tsr_sum<dtype>(this);
}

template<typename dtype>
long_int seq_tsr_sum<dtype>::mem_fp(){ return 0; }

/**
 * \brief wraps user sequential function signature
 */
template<typename dtype>
void seq_tsr_sum<dtype>::run(){
  if (is_inner){
    sym_seq_sum_inr(this->alpha,
                    this->A,
                    ndim_A,
                    edge_len_A,
                    edge_len_A,
                    sym_A,
                    idx_map_A,
                    this->beta,
                    this->B,
                    ndim_B,
                    edge_len_B,
                    edge_len_B,
                    sym_B,
                    idx_map_B,
                    inr_stride);
  } else {
    func_ptr.func_ptr(this->alpha,
                      this->A,
                      ndim_A,
                      edge_len_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->beta,
                      this->B,
                      ndim_B,
                      edge_len_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B);
  }
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
seq_tsr_ctr<dtype>::seq_tsr_ctr(ctr<dtype> * other) : ctr<dtype>(other) {
  seq_tsr_ctr<dtype> * o = (seq_tsr_ctr<dtype>*)other;
  alpha = o->alpha;
  
  ndim_A        = o->ndim_A;
  idx_map_A     = o->idx_map_A;
  sym_A         = o->sym_A;
  edge_len_A    = (int*)malloc(sizeof(int)*ndim_A);
  memcpy(edge_len_A, o->edge_len_A, sizeof(int)*ndim_A);

  ndim_B        = o->ndim_B;
  idx_map_B     = o->idx_map_B;
  sym_B         = o->sym_B;
  edge_len_B    = (int*)malloc(sizeof(int)*ndim_B);
  memcpy(edge_len_B, o->edge_len_B, sizeof(int)*ndim_B);

  ndim_C        = o->ndim_C;
  idx_map_C     = o->idx_map_C;
  sym_C         = o->sym_C;
  edge_len_C    = (int*)malloc(sizeof(int)*ndim_C);
  memcpy(edge_len_C, o->edge_len_C, sizeof(int)*ndim_C);

  is_inner      = o->is_inner;
  inner_params  = o->inner_params;

  func_ptr = o->func_ptr;
}

/**
 * \brief copies ctr object
 */
template<typename dtype>
ctr<dtype> * seq_tsr_ctr<dtype>::clone() {
  return new seq_tsr_ctr<dtype>(this);
}


template<typename dtype>
long_int seq_tsr_ctr<dtype>::mem_fp(){ return 0; }

/**
 * \brief wraps user sequential function signature
 */
template<typename dtype>
void seq_tsr_ctr<dtype>::run(){
  if (is_inner){
    sym_seq_ctr_inr(this->alpha,
                    this->A,
                    ndim_A,
                    edge_len_A,
                    edge_len_A,
                    sym_A,
                    idx_map_A,
                    this->B,
                    ndim_B,
                    edge_len_B,
                    edge_len_B,
                    sym_B,
                    idx_map_B,
                    this->beta,
                    this->C,
                    ndim_C,
                    edge_len_C,
                    edge_len_C,
                    sym_C,
                    idx_map_C,
                    &inner_params);
  } else {
    func_ptr.func_ptr(this->alpha,
                      this->A,
                      ndim_A,
                      edge_len_A,
                      edge_len_A,
                      sym_A,
                      idx_map_A,
                      this->B,
                      ndim_B,
                      edge_len_B,
                      edge_len_B,
                      sym_B,
                      idx_map_B,
                      this->beta,
                      this->C,
                      ndim_C,
                      edge_len_C,
                      edge_len_C,
                      sym_C,
                      idx_map_C);
  }
}

/* \brief clone a tensor object
 * \param[in] tensor_id id of old tensor
 * \param[in] copy_data if 0 then leave tensor blank, if 1 copy data from old
 * \param[out] new_tensor_id id of new tensor
 */
template<typename dtype>
int dist_tensor<dtype>::clone_tensor( int const tensor_id,
                                      int const copy_data,
                                      int *       new_tensor_id,
                                      int const alloc_data){
  int ndim, * edge_len, * sym;
  get_tsr_info(tensor_id, &ndim, &edge_len, &sym);
  define_tensor(ndim, edge_len, sym, 
                new_tensor_id, alloc_data);
  free(edge_len), free(sym);
  if (copy_data){
    return cpy_tsr(tensor_id, *new_tensor_id);
  }
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief copies tensor A to tensor B
 * \param[in] tid_A handle to A
 * \param[in] tid_B handle to B
 */
template<typename dtype>
int dist_tensor<dtype>::cpy_tsr(int const tid_A, int const tid_B){
  int i;
  tensor<dtype> * tsr_A, * tsr_B;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  tsr_B->has_zero_edge_len = tsr_A->has_zero_edge_len;

  if (tsr_A->is_folded) unfold_tsr(tsr_A);

  if (tsr_A->is_mapped){
    if (tsr_B->is_mapped){
      if (tsr_B->size < tsr_A->size || tsr_B->size > 2*tsr_A->size){
        free_buffer_space(tsr_B->data);
        get_buffer_space(tsr_A->size*sizeof(dtype), (void**)&tsr_B->data);
      } 
    } else {
      if (tsr_B->pairs != NULL) 
        free_buffer_space(tsr_B->pairs);
      get_buffer_space(tsr_A->size*sizeof(dtype), (void**)&tsr_B->data);
    }
    memcpy(tsr_B->data, tsr_A->data, sizeof(dtype)*tsr_A->size);
  } else {
    if (tsr_B->is_mapped){
      free_buffer_space(tsr_B->data);
      get_buffer_space(tsr_A->size*sizeof(tkv_pair<dtype>), 
                       (void**)&tsr_B->pairs);
    } else {
      if (tsr_B->size < tsr_A->size || tsr_B->size > 2*tsr_A->size){
        free_buffer_space(tsr_B->pairs);
        get_buffer_space(tsr_A->size*sizeof(tkv_pair<dtype>), 
                         (void**)&tsr_B->pairs);
      }
    }
    memcpy(tsr_B->pairs, tsr_A->pairs, 
            sizeof(tkv_pair<dtype>)*tsr_A->size);
  } 
  if (tsr_B->is_inner_mapped || tsr_B->is_folded){
    del_tsr(tsr_B->rec_tid);
  }
  tsr_B->is_inner_mapped = tsr_A->is_inner_mapped;
  if (tsr_A->is_inner_mapped){
    int new_tensor_id;
    tensor<dtype> * itsr = tensors[tsr_A->rec_tid];
    define_tensor(tsr_A->ndim, itsr->edge_len, tsr_A->sym, 
                  &new_tensor_id, 0);
    cpy_tsr(tsr_A->rec_tid, new_tensor_id);
    tsr_B->rec_tid = new_tensor_id;
  }
  tsr_B->is_folded = tsr_A->is_folded;
  if (tsr_A->is_folded){
    int new_tensor_id;
    tensor<dtype> * itsr = tensors[tsr_A->rec_tid];
    define_tensor(tsr_A->ndim, itsr->edge_len, tsr_A->sym, 
                              &new_tensor_id, 0);
    get_buffer_space(sizeof(int)*tsr_A->ndim, 
                     (void**)&tsr_B->inner_ordering);
    for (i=0; i<tsr_A->ndim; i++){
      tsr_B->inner_ordering[i] = tsr_A->inner_ordering[i];
    }
    tsr_B->rec_tid = new_tensor_id;
  }

  if (tsr_A->ndim != tsr_B->ndim){
    free_buffer_space(tsr_B->edge_len);
    if (tsr_B->is_padded)
      free_buffer_space(tsr_B->padding);
    free_buffer_space(tsr_B->sym);
    free_buffer_space(tsr_B->sym_table);
    if (tsr_B->is_mapped)
      free_buffer_space(tsr_B->edge_map);

    get_buffer_space(tsr_A->ndim*sizeof(int), (void**)&tsr_B->edge_len);
    get_buffer_space(tsr_A->ndim*sizeof(int), (void**)tsr_B->padding);
    get_buffer_space(tsr_A->ndim*sizeof(int), (void**)tsr_B->sym);
    get_buffer_space(tsr_A->ndim*tsr_A->ndim*sizeof(int), (void**)tsr_B->sym_table);
    get_buffer_space(tsr_A->ndim*sizeof(mapping), (void**)tsr_B->edge_map);
  }

  tsr_B->ndim = tsr_A->ndim;
  memcpy(tsr_B->edge_len, tsr_A->edge_len, sizeof(int)*tsr_A->ndim);
  tsr_B->is_padded = tsr_A->is_padded;
  if (tsr_A->is_padded)
    memcpy(tsr_B->padding, tsr_A->padding, sizeof(int)*tsr_A->ndim);
  memcpy(tsr_B->sym, tsr_A->sym, sizeof(int)*tsr_A->ndim);
  memcpy(tsr_B->sym_table, tsr_A->sym_table, sizeof(int)*tsr_A->ndim*tsr_A->ndim);
  tsr_B->is_mapped      = tsr_A->is_mapped;
  tsr_B->is_cyclic      = tsr_A->is_cyclic;
  tsr_B->itopo          = tsr_A->itopo;
  tsr_B->need_remap     = tsr_A->need_remap;
  if (tsr_A->is_mapped)
    copy_mapping(tsr_A->ndim, tsr_A->edge_map, tsr_B->edge_map);
  tsr_B->size = tsr_A->size;

  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief  Read or write tensor data by <key, value> pairs where key is the
 *              global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to write
 * \param[in] alpha multiplier for new value
 * \param[in] beta multiplier for old value
 * \param[in,out] mapped_data pairs to read/write
 * \param[in] rw weather to read (r) or write (w)
 */
template<typename dtype>
int dist_tensor<dtype>::write_pairs(int const           tensor_id, 
                                    long_int const      num_pair,  
                                    double const        alpha,  
                                    double const        beta,  
                                    tkv_pair<dtype> *   mapped_data, 
                                    char const          rw){
  int i, num_virt, need_pad;
  int * phys_phase, * virt_phase, * bucket_lda;
  int * virt_phys_rank;
  mapping * map;
  tensor<dtype> * tsr;

  TAU_FSTART(write_pairs);

  tsr = tensors[tensor_id];
  
  if (tsr->has_zero_edge_len) return DIST_TENSOR_SUCCESS;
  unmap_inner(tsr);
  set_padding(tsr);

  if (tsr->is_mapped){
    get_buffer_space(tsr->ndim*sizeof(int),     (void**)&phys_phase);
    get_buffer_space(tsr->ndim*sizeof(int),     (void**)&virt_phys_rank);
    get_buffer_space(tsr->ndim*sizeof(int),     (void**)&bucket_lda);
    get_buffer_space(tsr->ndim*sizeof(int),     (void**)&virt_phase);
    num_virt = 1;
    need_pad = tsr->is_padded;
    /* Setup rank/phase arrays, given current mapping */
    for (i=0; i<tsr->ndim; i++){
      map               = tsr->edge_map + i;
      phys_phase[i]     = calc_phase(map);
      virt_phase[i]     = phys_phase[i]/calc_phys_phase(map);
      virt_phys_rank[i] = calc_phys_rank(map, &topovec[tsr->itopo])
                          *virt_phase[i];
      num_virt          = num_virt*virt_phase[i];
      if (map->type == PHYSICAL_MAP)
        bucket_lda[i] = topovec[tsr->itopo].lda[map->cdt];
      else
        bucket_lda[i] = 0;
    }

    wr_pairs_layout(tsr->ndim,
                    global_comm->np,
                    num_pair,
                    alpha,
                    beta,
                    need_pad,
                    rw,
                    num_virt,
                    tsr->sym,
                    tsr->edge_len,
                    tsr->padding,
                    phys_phase,
                    virt_phase,
                    virt_phys_rank,
                    bucket_lda,
                    mapped_data,
                    tsr->data,
                    global_comm);

    free_buffer_space(phys_phase);
    free_buffer_space(virt_phys_rank);
    free_buffer_space(bucket_lda);
    free_buffer_space(virt_phase);

  } else {
    DEBUG_PRINTF("SHOULD NOT BE HERE, ALWAYS MAP ME\n");
    TAU_FSTOP(write_pairs);
    return DIST_TENSOR_ERROR;
  }
  TAU_FSTOP(write_pairs);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief Retrieve local data in the form of key value pairs
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of pairs read
 * \param[out] mapped_data pairs read
 */
template<typename dtype>
int dist_tensor<dtype>::read_local_pairs(int                tensor_id, 
                                         long_int *         num_pair,  
                                         tkv_pair<dtype> ** mapped_data){
  int i, num_virt, idx_lyr;
  long_int np;
  int * virt_phase, * virt_phys_rank, * phys_phase;
  tensor<dtype> * tsr;
  tkv_pair<dtype> * pairs;
  mapping * map;

  TAU_FSTART(read_local_pairs);

  tsr = tensors[tensor_id];
  if (tsr->has_zero_edge_len){
    *num_pair = 0;
    return DIST_TENSOR_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);


  if (!tsr->is_mapped){
    *num_pair = tsr->size;
    *mapped_data = tsr->pairs;
    return DIST_TENSOR_SUCCESS;
  } else {
    np = tsr->size;

    get_buffer_space(sizeof(int)*tsr->ndim, (void**)&virt_phase);
    get_buffer_space(sizeof(int)*tsr->ndim, (void**)&phys_phase);
    get_buffer_space(sizeof(int)*tsr->ndim, (void**)&virt_phys_rank);


    num_virt = 1;
    idx_lyr = global_comm->rank;
    for (i=0; i<tsr->ndim; i++){
      /* Calcute rank and phase arrays */
      map               = tsr->edge_map + i;
      phys_phase[i]     = calc_phase(map);
      virt_phase[i]     = phys_phase[i]/calc_phys_phase(map);
      virt_phys_rank[i] = calc_phys_rank(map, &topovec[tsr->itopo])
                                              *virt_phase[i];
      num_virt          = num_virt*virt_phase[i];

      if (map->type == PHYSICAL_MAP)
        idx_lyr -= topovec[tsr->itopo].lda[map->cdt]
                                *virt_phys_rank[i]/virt_phase[i];
    }
    if (idx_lyr == 0){
      read_loc_pairs(tsr->ndim, np, tsr->is_padded, num_virt,
                     tsr->sym, tsr->edge_len, tsr->padding,
                     virt_phase, phys_phase, virt_phys_rank, num_pair,
                     tsr->data, &pairs); 
      *mapped_data = pairs;
    } else {
      *mapped_data = NULL;
      *num_pair = 0;
    }


    free_buffer_space((void*)virt_phase);
    free_buffer_space((void*)phys_phase);
    free_buffer_space((void*)virt_phys_rank);

    TAU_FSTOP(read_local_pairs);
    return DIST_TENSOR_SUCCESS;
  }
  TAU_FSTOP(read_local_pairs);
}


/**
 * \brief read entire tensor with each processor (in packed layout).
 *         WARNING: will use a lot of memory. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[in,out] mapped_data values read
 */
template<typename dtype>
int dist_tensor<dtype>::allread_tsr(int const     tid, 
                                    long_int *    num_val,  
                                    dtype **      all_data){
  int numPes;
  int * nXs;
  int nval, n, i;
  int * pXs;
  tkv_pair<dtype> * my_pairs, * all_pairs;
  dtype * whole_tsr;

  numPes = global_comm->np;
  if (tensors[tid]->has_zero_edge_len){
    *num_val = 0;
    return DIST_TENSOR_SUCCESS;
  }

  get_buffer_space(numPes*sizeof(int), (void**)&nXs);
  get_buffer_space(numPes*sizeof(int), (void**)&pXs);
  pXs[0] = 0;

  long_int ntt = 0;
  my_pairs = NULL;
  read_local_pairs(tid, &ntt, &my_pairs);
  n = (int)ntt;
  n*=sizeof(tkv_pair<dtype>);
  ALLGATHER(&n, 1, MPI_INT, nXs, 1, MPI_INT, global_comm);
  for (i=1; i<numPes; i++){
    pXs[i] = pXs[i-1]+nXs[i-1];
  }
  nval = pXs[numPes-1] + nXs[numPes-1];
  get_buffer_space(nval, (void**)&all_pairs);
  MPI_Allgatherv(my_pairs, n, MPI_CHAR,
                 all_pairs, nXs, pXs, MPI_CHAR, MPI_COMM_WORLD);
  nval = nval/sizeof(tkv_pair<dtype>);

  std::sort(all_pairs,all_pairs+nval);
  if (n>0)
    free_buffer_space(my_pairs);
  get_buffer_space(nval*sizeof(dtype), (void**)&whole_tsr);
  for (i=0; i<nval; i++){
    whole_tsr[i] = all_pairs[i].d;
  }
  *num_val = (long_int)nval;
  *all_data = whole_tsr;

  free_buffer_space(nXs);
  free_buffer_space(pXs);
  free_buffer_space(all_pairs);

  return DIST_TENSOR_SUCCESS;
}



/* \brief deletes a tensor and deallocs the data
 */
template<typename dtype>
int dist_tensor<dtype>::del_tsr(int const tid){
  tensor<dtype> * tsr;

  tsr = tensors[tid];
  if (tsr->is_alloced){
    unfold_tsr(tsr);
    free_buffer_space(tsr->edge_len);
    if (tsr->is_padded)
      free_buffer_space(tsr->padding);
    free_buffer_space(tsr->sym);
    free_buffer_space(tsr->sym_table);
    if (tsr->is_mapped){
      if (!tsr->is_data_aliased)
        free_buffer_space(tsr->data);
      clear_mapping(tsr);
    }
    free(tsr->edge_map);
    tsr->is_alloced = 0;
  }

  return DIST_TENSOR_SUCCESS;
}

/* WARNING: not a standard spec function, know what you are doing */
template<typename dtype>
int dist_tensor<dtype>::elementalize(int const      tid,
                                     int const      x_rank,
                                     int const      x_np,
                                     int const      y_rank,
                                     int const      y_np,
                                     long_int const blk_sz,
                                     dtype *          data){
  tensor<dtype> * tsr;
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda, * old_padding;
  int * new_phase, * new_rank, * new_virt_dim, * new_pe_lda, * new_edge_len;
  int * new_padding, * old_edge_len;
  dtype * shuffled_data;
  int repad, is_pad, i, j, pad, my_x_dim, my_y_dim, was_padded;
  long_int old_size;


  std::vector< tensor<dtype> > * tensors = get_tensors();

  tsr = &(*tensors)[tid];
  unmap_inner(tsr);
  set_padding(tsr);

  assert(tsr->is_mapped);
  assert(tsr->ndim == 2);
  assert(tsr->sym[0] == NS);
  assert(tsr->sym[1] == NS);

  save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda, 
                     &old_size, &was_padded, &old_padding, &old_edge_len, &topovec[tsr->itopo]);

  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_phase);
  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_rank);
  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_pe_lda);
  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_virt_dim);
  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_padding);
  get_buffer_space(sizeof(int)*tsr->ndim,       (void**)&new_edge_len);

  repad = 1;    
  is_pad = 1;   
  new_phase[0]          = x_np;
  new_rank[0]           = x_rank;
  new_virt_dim[0]       = 1;
  new_pe_lda[0]         = 1;

  new_phase[1]          = y_np;
  new_rank[1]           = y_rank;
  new_virt_dim[1]       = 1;
  new_pe_lda[1]         = x_np;

  for (j=0; j<tsr->ndim; j++){
    new_edge_len[j] = tsr->edge_len[j];
    if (tsr->is_padded){
      pad = (tsr->edge_len[j]-tsr->padding[j])%new_phase[j];
      if (pad != 0)
              pad = new_phase[j]-pad;
      if (pad != tsr->padding[j]){
              repad = 1;
      }
      if (pad != 0) is_pad = 1;
      new_padding[j] = pad;
    } else {
      pad = tsr->edge_len[j]%new_phase[j];
      if (pad != 0){
        pad = new_phase[j]-pad;
        is_pad = 1;
        repad = 1;
      }
      new_padding[j] = pad;
    }
  }

  if (false){
    padded_reshuffle(tid,
                     tsr->ndim,
                     old_size,
                     new_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     was_padded,
                     old_padding,
                     old_edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     is_pad,
                     new_padding,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     &shuffled_data,
                     get_global_comm());
  } else {
    ABORT;
/*    get_buffer_space(sizeof(dtype)*tsr->size, (void**)&shuffled_data);
    cyclic_reshuffle(tsr->ndim,
                     tsr->size,
                     new_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     shuffled_data,
                     get_global_comm());
    new_blk_sz = tsr->size;
    new_padding = tsr->padding;*/
  }

  if (!is_pad && !repad){
//    assert(new_blk_sz == blk_sz);
    memcpy(data, shuffled_data, blk_sz*sizeof(dtype));
  } else {
    my_x_dim = (new_edge_len[0]-new_padding[0])/x_np;
    if (x_rank < (new_edge_len[0]-new_padding[0])%x_np)
      my_x_dim++;
    my_y_dim = (new_edge_len[1]-new_padding[1])/y_np;
    if (y_rank < (new_edge_len[1]-new_padding[1])%y_np)
      my_y_dim++;

    assert(my_x_dim*my_y_dim == blk_sz);

    for (i=0; i<my_y_dim; i++){
      for (j=0; j<my_x_dim; j++){
              data[j+i*my_x_dim] = shuffled_data[j+i*new_edge_len[0]];
      }
    }
  }


  free_buffer_space((void*)new_phase);
  free_buffer_space((void*)new_rank);
  free_buffer_space((void*)new_virt_dim);
  free_buffer_space((void*)new_edge_len);
  free_buffer_space((void*)shuffled_data);

  return DIST_TENSOR_SUCCESS;
}

/* \brief set the tensor to zero, called internally for each defined tensor
 * \param tensor_id id of tensor to set to zero
 */
template<typename dtype>
int dist_tensor<dtype>::set_zero_tsr(int tensor_id){
  tensor<dtype> * tsr;
  int * restricted;
  int i, map_success, btopo;
  uint64_t nvirt, bnvirt;
  uint64_t memuse, bmemuse;
  tsr = tensors[tensor_id];

  if (tsr->is_mapped){
    std::fill(tsr->data, tsr->data + tsr->size, get_zero<dtype>());
  } else {
    if (tsr->pairs != NULL){
      for (i=0; i<tsr->size; i++) tsr->pairs[i].d = get_zero<dtype>();
    } else {
      get_buffer_space(tsr->ndim*sizeof(int), (void**)&restricted);
//      memset(restricted, 0, tsr->ndim*sizeof(int));

      /* Map the tensor if necessary */
      bnvirt = UINT64_MAX, btopo = -1;
      bmemuse = UINT64_MAX;
      for (i=global_comm->rank; i<(int)topovec.size(); i+=global_comm->np){
        clear_mapping(tsr);
        set_padding(tsr);
        memset(restricted, 0, tsr->ndim*sizeof(int));
        map_success = map_tensor(topovec[i].ndim, tsr->ndim, tsr->edge_len,
                                 tsr->sym_table, restricted,
                                 topovec[i].dim_comm, NULL, 0,
                                 tsr->edge_map);
        if (map_success == DIST_TENSOR_ERROR) {
          LIBT_ASSERT(0);
          return DIST_TENSOR_ERROR;
        } else if (map_success == DIST_TENSOR_SUCCESS){
          tsr->itopo = i;
          set_padding(tsr);
          memuse = (uint64_t)tsr->size;

          if ((uint64_t)memuse >= proc_bytes_available()){
            DPRINTF(1,"Not enough memory to map tensor on topo %d\n", i);
            continue;
          }

          nvirt = (uint64_t)calc_nvirt(tsr);
          LIBT_ASSERT(nvirt != 0);
          if (btopo == -1 || nvirt < bnvirt){
            bnvirt = nvirt;
            btopo = i;
            bmemuse = memuse;
          } else if (nvirt == bnvirt && memuse < bmemuse){
            btopo = i;
            bmemuse = memuse;
          }
        }
      }
      if (btopo == -1)
              bnvirt = UINT64_MAX;
      /* pick lower dimensional mappings, if equivalent */
      ///btopo = get_best_topo(bnvirt, btopo, global_comm, 0, bmemuse);
      btopo = get_best_topo(bmemuse, btopo, global_comm);

      if (btopo == -1 || btopo == INT_MAX) {
        if (global_comm->rank==0)
          printf("ERROR: FAILED TO MAP TENSOR\n");
        return DIST_TENSOR_ERROR;
      }

      memset(restricted, 0, tsr->ndim*sizeof(int));
      clear_mapping(tsr);
      set_padding(tsr);
      map_success = map_tensor(topovec[btopo].ndim, tsr->ndim,
                               tsr->edge_len, tsr->sym_table, restricted,
                               topovec[btopo].dim_comm, NULL, 0,
                               tsr->edge_map);
      LIBT_ASSERT(map_success == DIST_TENSOR_SUCCESS);

      tsr->itopo = btopo;

      free_buffer_space(restricted);

      tsr->is_mapped = 1;
      set_padding(tsr);

#if 0
      get_buffer_space(tsr->ndim*sizeof(int), (void**)&phys_phase);
      get_buffer_space(tsr->ndim*sizeof(int), (void**)&sub_edge_len);

      /* Pad the tensor */
      need_pad = 1;
      nvirt = 1;
      for (i=0; i<tsr->ndim; i++){
        if (tsr->edge_map[i].type == PHYSICAL_MAP){
          phys_phase[i] = tsr->edge_map[i].np;
          if (tsr->edge_map[i].has_child){
            phys_phase[i] = phys_phase[i]*tsr->edge_map[i].child->np;
            nvirt *= tsr->edge_map[i].child->np;
          } 
        } else {
          LIBT_ASSERT(tsr->edge_map[i].type == VIRTUAL_MAP);
          phys_phase[i] = tsr->edge_map[i].np;
          nvirt *= tsr->edge_map[i].np;
        }
        if (tsr->edge_len[i] % phys_phase[i] != 0){
          need_pad = 1;
        }
      }

      old_size = packed_size(tsr->ndim, tsr->edge_len, 
                             tsr->sym, tsr->sym_type);

      if (need_pad){    
        tsr->is_padded = 1;
        get_buffer_space(tsr->ndim*sizeof(int), (void**)&tsr->padding);
        for (i=0; i<tsr->ndim; i++){
          tsr->padding[i] = tsr->edge_len[i] % phys_phase[i];
          if (tsr->padding[i] != 0)
            tsr->padding[i] = phys_phase[i] - tsr->padding[i];
          tsr->edge_len[i] += tsr->padding[i];
        }
      }
      for (i=0; i<tsr->ndim; i++){
        sub_edge_len[i] = tsr->edge_len[i]/phys_phase[i];
      }
      /* Alloc and set the data to zero */
      DEBUG_PRINTF("tsr->size = nvirt = %llu * packed_size = %llu\n",
                    (unsigned long long int)nvirt, packed_size(tsr->ndim, sub_edge_len,
                                       tsr->sym, tsr->sym_type));
      if (global_comm->rank == 0){
        printf("Tensor %d initially mapped with virtualization factor of %llu\n",tensor_id,nvirt);
      }
      tsr->size =nvirt*packed_size(tsr->ndim, sub_edge_len, 
                                   tsr->sym, tsr->sym_type);
      if (global_comm->rank == 0){
        printf("Tensor %d is of size %lld, has factor of %lf growth due to padding\n", 
              tensor_id, tsr->size,
              global_comm->np*(tsr->size/(double)old_size));

      }
#endif
      DPRINTF(3,"size set to %lld\n",tsr->size);
      get_buffer_space(tsr->size*sizeof(dtype), (void**)&tsr->data);
      std::fill(tsr->data, tsr->data + tsr->size, get_zero<dtype>());
/*      free_buffer_space(phys_phase);
      free_buffer_space(sub_edge_len);*/
    }
  }
  return DIST_TENSOR_SUCCESS;
}
/*
 * \brief print tensor tid to stream
 * WARNING: serializes ALL data to ONE processor
 * \param stream output stream (stdout, stdin, FILE)
 * \param tid tensor handle
 */
template<typename dtype>
int dist_tensor<dtype>::print_tsr(FILE * stream, int const tid) {
  tensor<dtype> const * tsr;
  int i, j;
  long_int my_sz, tot_sz =0;
  int * recvcnts, * displs, * adj_edge_len, * idx_arr;
  tkv_pair<dtype> * my_data;
  tkv_pair<dtype> * all_data;
  key k;

  print_map(stdout, tid, 1, 0);

  tsr = tensors[tid];

  my_sz = 0;
  read_local_pairs(tid, &my_sz, &my_data);

  if (global_comm->rank == 0){
    get_buffer_space(global_comm->np*sizeof(int), (void**)&recvcnts);
    get_buffer_space(global_comm->np*sizeof(int), (void**)&displs);
    get_buffer_space(tsr->ndim*sizeof(int), (void**)&adj_edge_len);
    get_buffer_space(tsr->ndim*sizeof(int), (void**)&idx_arr);

    if (tsr->is_padded){
      for (i=0; i<tsr->ndim; i++){
              adj_edge_len[i] = tsr->edge_len[i] - tsr->padding[i];
      }
    } else {
      memcpy(adj_edge_len, tsr->edge_len, tsr->ndim*sizeof(int));
    }
  }

  GATHER(&my_sz, 1, COMM_INT_T, recvcnts, 1, COMM_INT_T, 0, global_comm);

  if (global_comm->rank == 0){
    for (i=0; i<global_comm->np; i++){
      recvcnts[i] *= sizeof(tkv_pair<dtype>);
    }
    displs[0] = 0;
    for (i=1; i<global_comm->np; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
    }
    tot_sz = (displs[global_comm->np-1] 
                    + recvcnts[global_comm->np-1])/sizeof(tkv_pair<dtype>);
    get_buffer_space(tot_sz*sizeof(tkv_pair<dtype>), (void**)&all_data);
  }

  if (my_sz == 0) my_data = NULL;
  GATHERV(my_data, my_sz*sizeof(tkv_pair<dtype>), COMM_CHAR_T, 
          all_data, recvcnts, displs, COMM_CHAR_T, 0, global_comm);

  if (global_comm->rank == 0){
    std::sort(all_data, all_data + tot_sz);
    for (i=0; i<tot_sz; i++){
      k = all_data[i].k;
      for (j=0; j<tsr->ndim; j++){
        idx_arr[tsr->ndim-j-1] = k%adj_edge_len[j];
        k = k/adj_edge_len[j];
      }
      for (j=0; j<tsr->ndim; j++){
              fprintf(stream,"[%d]",idx_arr[j]);
      }
      fprintf(stream," <%E>\n",GET_REAL(all_data[i].d));
    }
    free_buffer_space(recvcnts);
    free_buffer_space(displs);
    free_buffer_space(adj_edge_len);
    free_buffer_space(idx_arr);
    free_buffer_space(all_data);
  }
  //COMM_BARRIER(global_comm);
  return DIST_TENSOR_SUCCESS;
}

/*
 * \brief print mapping of tensor tid to stream
 * \param stream output stream (stdout, stdin, FILE)
 * \param tid tensor handle
 */
template<typename dtype>
int dist_tensor<dtype>::print_map(FILE *    stream,
                                  int const tid,
                                  int const all,
                                  int const is_inner) const{
  int i;
  tensor<dtype> const * tsr;
  mapping * map;
  tsr = tensors[tid];


  if (all)
    COMM_BARRIER(global_comm);
  if (tsr->is_mapped && (!all || global_comm->rank == 0)){
    printf("Tensor %d of dimension %d is mapped to a ", tid, tsr->ndim);
    if (is_inner){
      for (i=0; i<inner_topovec[tsr->itopo].ndim-1; i++){
              printf("%d-by-", inner_topovec[tsr->itopo].dim_comm[i]->np);
      }
      if (inner_topovec[tsr->itopo].ndim > 0)
              printf("%d inner topology.\n", inner_topovec[tsr->itopo].dim_comm[i]->np);
    } else {
      for (i=0; i<topovec[tsr->itopo].ndim-1; i++){
              printf("%d-by-", topovec[tsr->itopo].dim_comm[i]->np);
      }
      if (topovec[tsr->itopo].ndim > 0)
              printf("%d topology.\n", topovec[tsr->itopo].dim_comm[i]->np);
    }
    for (i=0; i<tsr->ndim; i++){
      switch (tsr->edge_map[i].type){
        case NOT_MAPPED:
          printf("Dimension %d of length %d and symmetry %d is not mapped\n",i,tsr->edge_len[i],tsr->sym[i]);
          break;

        case PHYSICAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped to physical dimension %d with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].cdt,tsr->edge_map[i].np);
          map = &tsr->edge_map[i];
          while (map->has_child){
            map = map->child;
            if (map->type == VIRTUAL_MAP)
              printf("\tDimension %d also has a virtualized child of phase %d\n", i, map->np);
            else
              printf("\tDimension %d also has a physical child mapped to physical dimension %d with phase %d\n",
                      i, map->cdt, map->np);
          }
          break;

        case VIRTUAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped virtually with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].np);
          break;
      }
    }
  }
  if (all)
    COMM_BARRIER(global_comm);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief sets the padded area of a tensor to zero, needed after contractions
 * \param[in] tensor_id identifies tensor
 */
template<typename dtype>
int dist_tensor<dtype>::zero_out_padding(int const tensor_id){
  long_int n;
  int stat;
  tkv_pair<dtype> * mapped_data;
  if (tensors[tensor_id]->has_zero_edge_len){
    return DIST_TENSOR_SUCCESS;
  }
  stat = read_local_pairs(tensor_id, &n, &mapped_data);
  std::fill(tensors[tensor_id]->data, 
            tensors[tensor_id]->data+tensors[tensor_id]->size, 
            get_zero<dtype>());
  if (stat != DIST_TENSOR_SUCCESS) return stat;
  return write_pairs(tensor_id, n, 1.0, 0.0, mapped_data, 'w');

}

template<typename dtype>
int dist_tensor<dtype>::print_ctr(CTF_ctr_type_t const * ctype) const {
  int dim_A, dim_B, dim_C;
  int * sym_A, * sym_B, * sym_C;
  int i,j,max,ex_A, ex_B,ex_C;
  COMM_BARRIER(global_comm);
  if (global_comm->rank == 0){
    printf("Contacting Tensor %d with %d into %d\n", ctype->tid_A, ctype->tid_B,
                                                                                         ctype->tid_C);
    dim_A = get_dim(ctype->tid_A);
    dim_B = get_dim(ctype->tid_B);
    dim_C = get_dim(ctype->tid_C);
    max = dim_A+dim_B+dim_C;
    sym_A = get_sym(ctype->tid_A);
    sym_B = get_sym(ctype->tid_B);
    sym_C = get_sym(ctype->tid_C);

    printf("Contraction index table:\n");
    printf("     A     B     C\n");
    for (i=0; i<max; i++){
      ex_A=0;
      ex_B=0;
      ex_C=0;
      printf("%d:   ",i);
      for (j=0; j<dim_A; j++){
        if (ctype->idx_map_A[j] == i){
          ex_A++;
          if (sym_A[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_A == 0)
        printf("      ");
      if (ex_A == 1)
        printf("   ");
      for (j=0; j<dim_B; j++){
        if (ctype->idx_map_B[j] == i){
          ex_B=1;
          if (sym_B[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_B == 0)
        printf("      ");
      if (ex_B == 1)
        printf("   ");
      for (j=0; j<dim_C; j++){
        if (ctype->idx_map_C[j] == i){
          ex_C=1;
          if (sym_C[j] != NS)
            printf("%d' ",j);
          else
            printf("%d ",j);
        }
      }
      printf("\n");
      if (ex_A + ex_B + ex_C == 0) break;
    }
  }
  return DIST_TENSOR_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::print_sum(CTF_sum_type_t const * stype) const {
  int dim_A, dim_B;
  int i,j,max,ex_A,ex_B;
  int * sym_A, * sym_B;
  COMM_BARRIER(global_comm);
  if (global_comm->rank == 0){
    printf("Summing Tensor %d with %d into %d\n",stype->tid_A, stype->tid_B,
                                                    stype->tid_B);
    dim_A = get_dim(stype->tid_A);
    dim_B = get_dim(stype->tid_B);
    max = dim_A+dim_B; //MAX(MAX((dim_A), (dim_B)), (dim_C));
    sym_A = get_sym(stype->tid_A);
    sym_B = get_sym(stype->tid_B);

    printf("Sum index table:\n");
    printf("     A     B \n");
    for (i=0; i<max; i++){
      ex_A=0;
      ex_B=0;
      printf("%d:   ",i);
      for (j=0; j<dim_A; j++){
        if (stype->idx_map_A[j] == i){
          ex_A++;
          if (sym_A[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_A == 0)
        printf("      ");
      if (ex_A == 1)
        printf("   ");
      for (j=0; j<dim_B; j++){
        if (stype->idx_map_B[j] == i){
          ex_B=1;
          if (sym_B[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      printf("\n");
      if (ex_A + ex_B == 0) break;
    }
  }
  COMM_BARRIER(global_comm);
  return DIST_TENSOR_SUCCESS;
}


#include "dist_tensor_map.cxx"
#include "dist_tensor_op.cxx"
#include "dist_tensor_inner.cxx"
#include "dist_tensor_fold.cxx"
