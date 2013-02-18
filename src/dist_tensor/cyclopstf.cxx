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
#include "cyclopstf.hpp"
#include "mach.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include <stdint.h>
#include <limits.h>
#if VERIFY
#include "../unit_test/unit_test.h"
#include "../unit_test/unit_test_ctr.h"
#endif


/** 
 * \brief destructor
 */
template<typename dtype>
tCTF<dtype>::~tCTF(){
  exit();
}

/** 
 * \brief constructor
 */
template<typename dtype>
tCTF<dtype>::tCTF(){
  initialized = 0;
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 */
template<typename dtype>
int tCTF<dtype>::init(MPI_Comm const  global_context,
                      int const       rank, 
                      int const       np){      
  int ret;
#ifdef BGQ
  ret = tCTF<dtype>::init(global_context, MACHINE_BGQ, rank, np);
#else
  #ifdef BGP
    ret = tCTF<dtype>::init(global_context, MACHINE_BGP, rank, np);
  #else
    ret = tCTF<dtype>::init(global_context, MACHINE_8D, rank, np);
  #endif
#endif
  return ret;
}

template<typename dtype>
MPI_Comm tCTF<dtype>::get_MPI_Comm(){
  return (dt->get_global_comm())->cm;
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] mach the type of machine we are running on
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] inner_size is the total block size of dgemm calls 
 */
template<typename dtype>
int tCTF<dtype>::init(MPI_Comm const  global_context,
                      CTF_MACHINE           mach,
                      int const       rank, 
                      int const       np,
                      int const       inner_size){      
  int ndim, ret;
  int * dim_len;
  get_topo(np, mach, &ndim, &dim_len);
  ret = tCTF<dtype>::init(global_context, rank, np, ndim, dim_len, inner_size);
  free(dim_len);
  return ret;
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] ndim is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 * \param[in] inner_size is the total block size of dgemm calls 
 */
template<typename dtype>
int tCTF<dtype>::init(MPI_Comm const  global_context,
                      int const       rank, 
                      int const       np, 
                      int const       ndim, 
                      int const *     dim_len,
                      int const       inner_size){
  initialized = 1;
  CommData_t * glb_comm = (CommData_t*)malloc(sizeof(CommData_t));
  SET_COMM(global_context, rank, np, glb_comm);
  dt = new dist_tensor<dtype>();
  return dt->initialize(glb_comm, ndim, dim_len, inner_size);
}


/**
 * \brief  defines a tensor and retrieves handle
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] edge_len global edge lengths of tensor
 * \param[in] sym symmetry relations of tensor
 * \param[out] tensor_id the tensor index (handle)
 */
template<typename dtype>
int tCTF<dtype>::define_tensor(int const          ndim,             
                               int const *      edge_len, 
                               int const *      sym,
                               int *        tensor_id){
  return dt->define_tensor(ndim, edge_len, sym, tensor_id);
}
    
/* \brief clone a tensor object
 * \param[in] tensor_id id of old tensor
 * \param[in] copy_data if 0 then leave tensor blank, if 1 copy data from old
 * \param[out] new_tensor_id id of new tensor
 */
template<typename dtype>
int tCTF<dtype>::clone_tensor(int const tensor_id,
                              int const copy_data,
                              int *     new_tensor_id){
  dt->clone_tensor(tensor_id, copy_data, new_tensor_id);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get dimension of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_dimension(int const tensor_id, int *ndim) const{
  *ndim = dt->get_dim(tensor_id);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get lengths of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] edge_len edge lengths of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_lengths(int const tensor_id, int **edge_len) const{
  *edge_len = dt->get_edge_len(tensor_id);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get symmetry of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] sym symmetries of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_symmetry(int const tensor_id, int **sym) const{
  *sym = dt->get_sym(tensor_id);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get raw data pointer WARNING: includes padding 
 * \param[in] tensor_id id of tensor
 * \param[out] data raw local data
 */
template<typename dtype>
int tCTF<dtype>::get_raw_data(int const tensor_id, dtype ** data, int64_t * size) {
  *data = dt->get_raw_data(tensor_id, size);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief get information about tensor
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 * \param[out] edge_len edge lengths of tensor
 * \param[out] sym symmetries of tensor
 */
template<typename dtype>
int tCTF<dtype>::info_tensor(int const  tensor_id,
                             int *      ndim,
                             int **     edge_len,
                             int **     sym) const{
  dt->get_tsr_info(tensor_id, ndim, edge_len, sym);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief  Input tensor data with <key, value> pairs where key is the
 *              global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to write
 * \param[in] mapped_data pairs to write
 */
template<typename dtype>
int tCTF<dtype>::write_tensor(int const               tensor_id, 
                              int64_t const           num_pair,  
                              tkv_pair<dtype> * const mapped_data){
  return dt->write_pairs(tensor_id, num_pair, 1.0, 0.0, mapped_data, 'w');
}

/** 
 * \brief  Add tensor data new=alpha*new+beta*old
 *         with <key, value> pairs where key is the 
 *         global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] 
 * \param[in] num_pair number of pairs to write
 * \param[in] mapped_data pairs to write
 */
template<typename dtype>
int tCTF<dtype>::write_tensor(int const               tensor_id, 
                              int64_t const           num_pair,  
                              double const            alpha,
                              double const            beta,
                              tkv_pair<dtype> * const mapped_data){
  return dt->write_pairs(tensor_id, num_pair, alpha, beta, mapped_data, 'w');
}

/**
 * \brief read tensor data with <key, value> pairs where key is the
 *              global index for the value, which gets filled in. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to read
 * \param[in,out] mapped_data pairs to read
 */
template<typename dtype>
int tCTF<dtype>::read_tensor(int const                tensor_id, 
                             int64_t const            num_pair, 
                             tkv_pair<dtype> * const  mapped_data){
  return dt->write_pairs(tensor_id, num_pair, 1.0, 0.0, mapped_data, 'r');
}

/**
 * \brief read entire tensor with each processor (in packed layout).
 *         WARNING: will use a lot of memory. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[in,out] mapped_data values read
 */
template<typename dtype>
int tCTF<dtype>::allread_tensor(int const   tensor_id, 
                                int64_t *   num_pair, 
                                dtype **    all_data){
  int ret;
  long_int np;
  ret = dt->allread_tsr(tensor_id, &np, all_data);
  *num_pair = np;
  return ret;
}

/* input tensor local data or set buffer for contract answer. */
/*int tCTF<dtype>::set_local_tensor(int const   tensor_id, 
                         int const      num_val, 
                         dtype *        tsr_data){
  return set_tsr_data(tensor_id, num_val, tsr_data);  
}*/

/**
 * \brief  map input tensor local data to zero
 * \param[in] tensor_id tensor handle
 */
template<typename dtype>
int tCTF<dtype>::set_zero_tensor(int const tensor_id){
  return dt->set_zero_tsr(tensor_id);
}

/**
 * \brief read tensor data pairs local to processor. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[out] mapped_data values read
 */
template<typename dtype>
int tCTF<dtype>::read_local_tensor(int const          tensor_id, 
                                   int64_t *          num_pair,  
                                   tkv_pair<dtype> ** mapped_data){
  int ret;
  long_int np;
  ret = dt->read_local_pairs(tensor_id, &np, mapped_data);
  *num_pair = np;
  return ret;
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *  type,
                          dtype const             alpha,
                          dtype const             beta){
  fseq_tsr_ctr<dtype> fs;
  fs.func_ptr=sym_seq_ctr_ref<dtype>;
  return contract(type, NULL, 0, fs, alpha, beta, 1);
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      accepts custom-sized buffer-space,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *  type,
                          dtype *                 buffer, 
                          int const               buffer_len, 
                          dtype const             alpha,
                          dtype const             beta){
  fseq_tsr_ctr<dtype> fs;
  fs.func_ptr=sym_seq_ctr_ref<dtype>;
  return contract(type, buffer, buffer_len, fs, alpha, beta, 1);
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C. 
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential op 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer 
 * \param[in] func_ptr sequential ctr func pointer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *    type,
                          dtype *                   buffer, 
                          int const                 buffer_len, 
                          fseq_tsr_ctr<dtype> const func_ptr, 
                          dtype const               alpha,
                          dtype const               beta,
                          int const                 map_inner){
#if DEBUG >= 1
  if (dt->get_global_comm()->rank == 0)
    printf("Head contraction :\n");
  dt->print_ctr(type);
#endif

  return dt->sym_contract(type, buffer, buffer_len, func_ptr, alpha,
                                      beta, map_inner);
}

/**
 * \brief copy tensor from one handle to another
 * \param[in] tid_A tensor handle to copy from
 * \param[in] tid_B tensor handle to copy to
 */
template<typename dtype>
int tCTF<dtype>::copy_tensor(int const tid_A, int const tid_B){
  return dt->cpy_tsr(tid_A, tid_B);
}

/**
 * \brief scales a tensor by alpha
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const alpha, int const tid){
  return dt->scale_tsr(alpha, tid);
}
/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const               alpha, 
                              int const                 tid, 
                              int const *               idx_map){
  fseq_tsr_scl<dtype> fs;
  fs.func_ptr=sym_seq_scl_ref<dtype>;
  return dt->scale_tsr(alpha, tid, idx_map, fs);
}


/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 * \param[in] func_ptr pointer to sequential scale function
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const               alpha, 
                              int const                 tid, 
                              int const *               idx_map,
                              fseq_tsr_scl<dtype> const func_ptr){
    return dt->scale_tsr(alpha, tid, idx_map, func_ptr);
  }

  /**
   * \brief computes a dot product of two tensors A dot B
   * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[out] product the result of the dot-product
 */
template<typename dtype>
int tCTF<dtype>::dot_tensor(int const tid_A, int const tid_B, dtype *product){
  int stat;
  /* check if the mappings of A and B are the same */
  stat = dt->check_pair_mapping(tid_A, tid_B);
  if (stat == 0){
    /* Align the mappings of A and B */
    stat = dt->map_tensor_pair(tid_A, tid_B);
    if (stat != DIST_TENSOR_SUCCESS)
      return stat;
  }
  /* Compute the dot product of A and B */
  return dt->dot_loc_tsr(tid_A, tid_B, product);
}

/**
 * \brief Performs an elementwise reduction on a tensor 
 * \param[in] tid tensor handle
 * \param[in] CTF::OP reduction operation to apply
 * \param[out] result result of reduction operation
 */
template<typename dtype>
int tCTF<dtype>::reduce_tensor(int const tid, CTF_OP op, dtype * result){
  return dt->red_tsr(tid, op, result);
}

/**
 * \brief Calls a mapping function on each element of the tensor 
 * \param[in] tid tensor handle
 * \param[in] map_func function pointer to apply to each element
 */
template<typename dtype>
int tCTF<dtype>::map_tensor(int const tid, 
                            dtype (*map_func)(int const   ndim, 
                                              int const * indices, 
                                              dtype const elem)){
  return dt->map_tsr(tid, map_func);
}
    
/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 *               uses standard summation pointer
 * \param[in] type idx_maps and tids of contraction
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(CTF_sum_type_t const * type,
                             dtype const            alpha,
                             dtype const            beta){
  
  fseq_tsr_sum<dtype> fs;
  fs.func_ptr=sym_seq_sum_ref<dtype>;
  return sum_tensors(alpha, beta, type->tid_A, type->tid_B, 
                     type->idx_map_A, type->idx_map_B, fs);

}
    
/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 * \param[in] type idx_maps and tids of contraction
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] func_ptr sequential ctr func pointer 
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(CTF_sum_type_t const *     type,
                             dtype const                alpha,
                             dtype const                beta,
                             fseq_tsr_sum<dtype> const  func_ptr){
  return sum_tensors(alpha, beta, type->tid_A, type->tid_B, 
                                 type->idx_map_A, type->idx_map_B, func_ptr);

}

/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] func_ptr sequential ctr func pointer 
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(dtype const                alpha,
                             dtype const                beta,
                             int const                  tid_A,
                             int const                  tid_B,
                             int const *                idx_map_A,
                             int const *                idx_map_B,
                             fseq_tsr_sum<dtype> const  func_ptr){
  return dt->sum_tensors(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, func_ptr);
}

/**
 * \brief daxpy tensors A and B, B = B+alpha*A
 * \param[in] alpha scaling factor
 * \param[in] tid_A tensor handle of A
 * \param[in] tid_B tensor handle of B
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(dtype const  alpha,
                             int const    tid_A,
                             int const    tid_B){
  int stat;
  
  /* check if the mappings of A and B are the same */
  stat = dt->check_pair_mapping(tid_A, tid_B);
  if (stat == 0){
    /* Align the mappings of A and B */
    stat = dt->map_tensor_pair(tid_A, tid_B);
    if (stat != DIST_TENSOR_SUCCESS)
      return stat;
  }
  /* Sum tensors */
  return dt->daxpy_local_tensor_pair(alpha, tid_A, tid_B);
}

template<typename dtype>
int tCTF<dtype>::print_tensor(FILE * stream, int const tid) {
  return dt->print_tsr(stream, tid);
}
/* Prints contraction type. */
template<typename dtype>
int tCTF<dtype>::print_ctr(CTF_ctr_type_t const * ctype) const {
  return dt->print_ctr(ctype);
}

/* Prints sum type. */
template<typename dtype>
int tCTF<dtype>::print_sum(CTF_sum_type_t const * stype) const{
  return dt->print_sum(stype);
}


/**
 * \brief removes all tensors, invalidates all handles
 */
template<typename dtype>
int tCTF<dtype>::clean_tensors(){
  unsigned int i;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  for (i=0; i<tensors->size(); i++){
    dt->del_tsr(i);
    free((*tensors)[i]);
  }
  tensors->clear();
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief removes a tensor, invalidates its handle
 * \param tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::clean_tensor(int const tid){
  return dt->del_tsr(tid);
}

/**
 * \brief removes all tensors, invalidates all handles, and exits library.
 *              Do not use library instance after executing this.
 */
template<typename dtype>
int tCTF<dtype>::exit(){
  int ret;
  if (initialized){
    ret = tCTF<dtype>::clean_tensors();
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
    delete dt;
    initialized = 0;
    return ret;
  } else
    return DIST_TENSOR_SUCCESS;
}

/* \brief ScaLAPACK back-end, see their DOC */
template<typename dtype>
int tCTF<dtype>::pgemm(char const   TRANSA, 
                       char const   TRANSB, 
                       int const    M, 
                       int const    N, 
                       int const    K, 
                       dtype const  ALPHA,
                       dtype *      A, 
                       int const    IA, 
                       int const    JA, 
                       int const *  DESCA, 
                       dtype *      B, 
                       int const    IB, 
                       int const    JB, 
                       int const *  DESCB, 
                       dtype const  BETA,
                       dtype *      C, 
                       int const    IC, 
                       int const    JC, 
                       int const *  DESCC){
  int ret, need_remap, i, j;
#if (!REDIST)
  int redist;
#endif
  int stid_A, stid_B, stid_C;
  int otid_A, otid_B, otid_C;
  long_int old_size_C;
  int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
  int * old_padding_C, * old_edge_len_C;
  int * need_free;
  int was_padded_C, was_cyclic_C;
  tensor<dtype> * tsr_nC, * tsr_oC;
  CTF_ctr_type ct;
  fseq_tsr_ctr<dtype> fs;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  get_buffer_space(3*sizeof(int), (void**)&need_free);
  ret = dt->pgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA,
                  B, IB, JB, DESCB,
                  BETA, C, IC, JC, DESCC, &ct, &fs, need_free);
  if (ret != DIST_TENSOR_SUCCESS)
    return ret;

  otid_A = ct.tid_A;
  otid_B = ct.tid_B;
  otid_C = ct.tid_C;
#if (!REDIST)
  ret = dt->try_topo_morph(otid_A, otid_B, otid_C);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
  redist = dt->check_contraction_mapping(&ct);
  if (redist == 0) {
    printf("REDISTRIBUTING\n");
#endif
    clone_tensor(ct.tid_A, 1, &stid_A);
    clone_tensor(ct.tid_B, 1, &stid_B);
    clone_tensor(ct.tid_C, 1, &stid_C);
    ct.tid_A = stid_A;
    ct.tid_B = stid_B;
    ct.tid_C = stid_C;
#if (!REDIST)
  }
#endif

  ret = this->contract(&ct, NULL, 0, fs, ALPHA, BETA);
  (*tensors)[ct.tid_C]->need_remap = 0;
  if (ret != DIST_TENSOR_SUCCESS)
    return ret;
#if (!REDIST)
  if (redist == 0){
#endif
    tsr_oC = (*tensors)[otid_C];
    tsr_nC = (*tensors)[stid_C];
    need_remap = 0;
    if (tsr_oC->itopo == tsr_nC->itopo){
      if (!comp_dim_map(&tsr_oC->edge_map[0],&tsr_nC->edge_map[0]))
        need_remap = 1;
      if (!comp_dim_map(&tsr_oC->edge_map[1],&tsr_nC->edge_map[1]))
        need_remap = 1;
    } else
      need_remap = 1;
    if (need_remap){
      save_mapping(tsr_nC, &old_phase_C, &old_rank_C, &old_virt_dim_C, 
                   &old_pe_lda_C, &old_size_C, &was_padded_C, &was_cyclic_C, 
                   &old_padding_C, &old_edge_len_C, 
                   dt->get_topo(tsr_nC->itopo));
      if (need_free[2])
        free(tsr_oC->data);
      tsr_oC->data = tsr_nC->data;
      remap_tensor(otid_C, tsr_oC, dt->get_topo(tsr_oC->itopo), old_size_C, 
                   old_phase_C, old_rank_C, old_virt_dim_C, 
                   old_pe_lda_C, was_padded_C, was_cyclic_C, 
                   old_padding_C, old_edge_len_C, dt->get_global_comm());
    } else{
      if (need_free[2])
              free(tsr_oC->data);
      tsr_oC->data = tsr_nC->data;
    }
    /* If this process owns any data */
    if (!need_free[2]){
      memcpy(C,tsr_oC->data,tsr_oC->size*sizeof(dtype));
    } else
      free(tsr_oC->data);
    if (need_free[0])
      dt->del_tsr(otid_A);
    if (need_free[1])
      dt->del_tsr(otid_B);
    dt->del_tsr(stid_A);
    dt->del_tsr(stid_B);
    (*tensors)[stid_A]->is_alloced = 0;
    (*tensors)[stid_B]->is_alloced = 0;
    (*tensors)[stid_C]->is_alloced = 0;
#if (!REDIST)
  }
#endif
  if ((*tensors)[otid_A]->padding[0] != 0 ||
      (*tensors)[otid_A]->padding[1] != 0){
    free((*tensors)[otid_A]->data);
  }
  if ((*tensors)[otid_B]->padding[0] != 0 ||
      (*tensors)[otid_B]->padding[1] != 0){
    free((*tensors)[otid_B]->data);
  }
  if ((*tensors)[otid_C]->padding[0] != 0 ||
      (*tensors)[otid_C]->padding[1] != 0){
    int brow, bcol;
    brow = DESCC[4];
    bcol = DESCC[5];
    for (i=0; i<bcol-(*tensors)[otid_C]->padding[1]; i++){
      for (j=0; j<brow-(*tensors)[otid_C]->padding[0]; j++){
        C[i*(brow-(*tensors)[otid_C]->padding[0])+j] 
          = (*tensors)[otid_C]->data[i*brow+j];
      }
    }
    free((*tensors)[otid_C]->data);
  }
  (*tensors)[otid_A]->is_alloced = 0;
  (*tensors)[otid_B]->is_alloced = 0;
  (*tensors)[otid_C]->is_alloced = 0;
  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief define matrix from ScaLAPACK descriptor
 *
 * \param[in] DESCA ScaLAPACK descriptor for a matrix
 * \param[in] data pointer to actual data
 * \param[out] tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::def_scala_mat(int const * DESCA,
                               dtype const * data,
                               int * tid){
  int ret, stid;
  ret = dt->load_matrix((dtype*)data, DESCA, &stid, NULL);
  if (ret != DIST_TENSOR_SUCCESS) return ret;
  clone_tensor(stid, 1, tid);
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  tensor<dtype> * stsr = (*tensors)[stid];
  tensor<dtype> * tsr = (*tensors)[*tid];
  free(stsr->data);
  stsr->is_alloced = 0;
  tsr->is_matrix = 1;
  tsr->slay = stid;
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief reads a ScaLAPACK matrix to the original data pointer
 *
 * \param[in] tid tensor handle
 * \param[in,out] data pointer to buffer data
 */
template<typename dtype>
int tCTF<dtype>::read_scala_mat(int const tid,
                                dtype * data){
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda;
  int * old_padding, * old_edge_len;
  int was_padded, was_cyclic;
  long_int old_size;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  tensor<dtype> * tsr = (*tensors)[tid];
  tensor<dtype> * stsr = (*tensors)[tsr->slay];
  dt->unmap_inner(tsr);
  save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, 
               &old_pe_lda, &old_size, &was_padded, &was_cyclic, 
               &old_padding, &old_edge_len, 
               dt->get_topo(tsr->itopo));
  LIBT_ASSERT(tsr->is_matrix);
  get_buffer_space(sizeof(dtype)*tsr->size, (void**)&stsr->data);
  memcpy(stsr->data, tsr->data, sizeof(dtype)*tsr->size);
  remap_tensor(tsr->slay, stsr, dt->get_topo(stsr->itopo), old_size, 
               old_phase, old_rank, old_virt_dim, 
               old_pe_lda, was_padded, was_cyclic, 
               old_padding, old_edge_len, dt->get_global_comm());
  if (data!=NULL)
    memcpy(data, stsr->data, stsr->size*sizeof(dtype));  
  free(stsr->data);
  return DIST_TENSOR_SUCCESS;
}
/**
 * \brief CTF interface for pgemm
 */
template<typename dtype>
int tCTF<dtype>::pgemm(char const   TRANSA, 
                       char const   TRANSB, 
                       int const    M, 
                       int const    N, 
                       int const    K, 
                       dtype const  ALPHA,
                       int const    tid_A,
                       int const    tid_B,
                       dtype const  BETA,
                       int const    tid_C){
  int herm_A, herm_B, ret;
  CTF_ctr_type ct;
  fseq_tsr_ctr<dtype> fs;
  ct.tid_A = tid_A;
  ct.tid_B = tid_B;
  ct.tid_C = tid_C;

  ct.idx_map_A = (int*)malloc(sizeof(int)*2);
  ct.idx_map_B = (int*)malloc(sizeof(int)*2);
  ct.idx_map_C = (int*)malloc(sizeof(int)*2);
  ct.idx_map_C[0] = 1;
  ct.idx_map_C[1] = 2;
  herm_A = 0;
  herm_B = 0;
  if (TRANSA == 'N' || TRANSA == 'n'){
    ct.idx_map_A[0] = 1;
    ct.idx_map_A[1] = 0;
  } else {
    LIBT_ASSERT(TRANSA == 'T' || TRANSA == 't' || TRANSA == 'c' || TRANSA == 'C');
    if (TRANSA == 'c' || TRANSA == 'C')
      herm_A = 1;
    ct.idx_map_A[0] = 0;
    ct.idx_map_A[1] = 1;
  }
  if (TRANSB == 'N' || TRANSB == 'n'){
    ct.idx_map_B[0] = 0;
    ct.idx_map_B[1] = 2;
  } else {
    LIBT_ASSERT(TRANSB == 'T' || TRANSB == 't' || TRANSB == 'c' || TRANSB == 'C');
    if (TRANSB == 'c' || TRANSB == 'C')
      herm_B = 1;
    ct.idx_map_B[0] = 2;
    ct.idx_map_B[1] = 0;
  }
  if (herm_A && herm_B)
    fs.func_ptr = &gemm_ctr<dtype,1,1>;
  else if (herm_A)
    fs.func_ptr = &gemm_ctr<dtype,1,0>;
  else if (herm_B)
    fs.func_ptr = &gemm_ctr<dtype,0,1>;
  else
    fs.func_ptr = &gemm_ctr<dtype,0,0>;
  ret = this->contract(&ct, NULL, 0, fs, ALPHA, BETA);
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  (*tensors)[ct.tid_C]->need_remap = 0;
  return ret;
};
  
/* Instantiate the ugly templates */
template class tCTF<double>;
#if (VERIFY==0)
template class tCTF< std::complex<double> >;
#endif


