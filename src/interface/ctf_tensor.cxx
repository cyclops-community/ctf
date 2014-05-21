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
#include "../shared/util_ext.h"
#include "../dist_tensor/cyclopstf.hpp"

template<typename dtype>
CTF_Tensor<dtype>::CTF_Tensor(){
  tid = -1;
  ndim = -1;
  sym = NULL;
  len = NULL;
  name = NULL;
  world = NULL;
}

template<typename dtype>
CTF_Tensor<dtype>::CTF_Tensor(const CTF_Tensor<dtype>& A,
                                 bool                copy){
  int ret;
  world = A.world;
  name = A.name;

  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  assert(ret == CTF_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid, name, name!=NULL);
  assert(ret == CTF_SUCCESS);

  //printf("Defined tensor %d to be the same as %d, copy=%d\n", tid, A.tid, (int)copy);

  if (copy){
    ret = world->ctf->copy_tensor(A.tid, tid);
    assert(ret == CTF_SUCCESS);
  }
}

template<typename dtype>
CTF_Tensor<dtype>::CTF_Tensor(const CTF_Tensor<dtype>& A,
                                CTF_World<dtype> & world_){
  int ret;
  world = &world_;
  name = A.name;

  ret = A.world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  assert(ret == CTF_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid, name, name!=NULL);
  assert(ret == CTF_SUCCESS);
}


template<typename dtype>
CTF_Tensor<dtype>::CTF_Tensor(int                 ndim_,
                                int const *         len_,
                                int const *         sym_,
                                CTF_World<dtype> & world_,
                                char const *        name_,
                                int const           profile_){
  int ret;
  world = &world_;
  name = name_;
  ret = world->ctf->define_tensor(ndim_, len_, sym_, &tid, name_, profile_);
  assert(ret == CTF_SUCCESS);
  ret = world->ctf->info_tensor(tid, &ndim, &len, &sym);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
CTF_Tensor<dtype>::~CTF_Tensor(){
/*  if (sym != NULL)
    CTF_free_cond(sym);
  if (len != NULL)
    CTF_free_cond(len);*/
  if (sym != NULL)
    free(sym);
  if (len != NULL)
    free(len);
  world->ctf->clean_tensor(tid);
}

template<typename dtype>
dtype * CTF_Tensor<dtype>::get_raw_data(int64_t * size) {
  int ret;
  dtype * data;
  ret = world->ctf->get_raw_data(tid, &data, size);
  assert(ret == CTF_SUCCESS);
  
  return data;
}

template<typename dtype>
void CTF_Tensor<dtype>::read_local(int64_t *   npair, 
                                    int64_t **  global_idx, 
                                    dtype **   data) const {
  CTF_pair< dtype > * pairs;
  int ret, i;
  ret = world->ctf->read_local_tensor(tid, npair, &pairs);
  assert(ret == CTF_SUCCESS);
  /* FIXME: careful with CTF_alloc */
  *global_idx = (int64_t*)malloc((*npair)*sizeof(int64_t));
  *data = (dtype*)malloc((*npair)*sizeof(dtype));
  for (i=0; i<(*npair); i++){
    (*global_idx)[i] = pairs[i].k;
    (*data)[i] = pairs[i].d;
  }
  free(pairs);
}

template<typename dtype>
void CTF_Tensor<dtype>::read_local(int64_t *   npair,
                                    CTF_pair<dtype> **   pairs) const {
  int ret = world->ctf->read_local_tensor(tid, npair, pairs);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::read(int64_t          npair, 
                              int64_t const *  global_idx, 
                              dtype *           data) const {
  int ret, i;
  CTF_pair< dtype > * pairs;
  pairs = (CTF_pair< dtype >*)CTF_alloc(npair*sizeof(CTF_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
  }
  ret = world->ctf->read_tensor(tid, npair, pairs);
  assert(ret == CTF_SUCCESS);
  for (i=0; i<npair; i++){
    data[i] = pairs[i].d;
  }
  CTF_free(pairs);
}

template<typename dtype>
void CTF_Tensor<dtype>::read(int64_t          npair,
                              CTF_pair<dtype> * pairs) const {
  int ret = world->ctf->read_tensor(tid, npair, pairs);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::write(int64_t          npair, 
                               int64_t const *  global_idx, 
                               dtype const *     data) {
  int ret, i;
  CTF_pair< dtype > * pairs;
  pairs = (CTF_pair< dtype >*)CTF_alloc(npair*sizeof(CTF_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, pairs);
  assert(ret == CTF_SUCCESS);
  CTF_free(pairs);
}

template<typename dtype>
void CTF_Tensor<dtype>::write(int64_t                npair,
                               CTF_pair<dtype> const * pairs) {
  int ret = world->ctf->write_tensor(tid, npair, pairs);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::write(int64_t         npair, 
                               dtype            alpha, 
                               dtype            beta,
                               int64_t const * global_idx, 
                               dtype const *    data) {
  int ret, i;
  CTF_pair< dtype > * pairs;
  pairs = (CTF_pair< dtype >*)CTF_alloc(npair*sizeof(CTF_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
  assert(ret == CTF_SUCCESS);
  CTF_free(pairs);
}

template<typename dtype>
void CTF_Tensor<dtype>::write(int64_t          npair,
                               dtype             alpha,
                               dtype             beta,
                               CTF_pair<dtype> const * pairs) {
  int ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::read(int64_t         npair, 
                              dtype            alpha, 
                              dtype            beta,
                              int64_t const * global_idx, 
                              dtype *          data) const{
  int ret, i;
  CTF_pair< dtype > * pairs;
  pairs = (CTF_pair< dtype >*)CTF_alloc(npair*sizeof(CTF_pair< dtype >));
  for (i=0; i<npair; i++){
    pairs[i].k = global_idx[i];
    pairs[i].d = data[i];
  }
  ret = world->ctf->read_tensor(tid, npair, alpha, beta, pairs);
  assert(ret == CTF_SUCCESS);
  for (i=0; i<npair; i++){
    data[i] = pairs[i].d;
  }
  CTF_free(pairs);
}

template<typename dtype>
void CTF_Tensor<dtype>::read(int64_t          npair,
                              dtype             alpha,
                              dtype             beta,
                              CTF_pair<dtype> * pairs) const{
  int ret = world->ctf->read_tensor(tid, npair, alpha, beta, pairs);
  assert(ret == CTF_SUCCESS);
}


template<typename dtype>
void CTF_Tensor<dtype>::read_all(int64_t * npair, dtype ** vals) const {
  int ret;
  ret = world->ctf->allread_tensor(tid, npair, vals);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
int64_t CTF_Tensor<dtype>::read_all(dtype * vals) const {
  int ret;
  int64_t npair;
  ret = world->ctf->allread_tensor(tid, &npair, vals);
  assert(ret == CTF_SUCCESS);
  return npair;
}

template<typename dtype>
int64_t CTF_Tensor<dtype>::estimate_cost(
                                  const CTF_Tensor<dtype>&     A,
                                  const char *                  idx_A,
                                  const CTF_Tensor<dtype>&     B,
                                  const char *                  idx_B,
                                  const char *                  idx_C){
  int * idx_map_A, * idx_map_B, * idx_map_C;
  conv_idx(A.ndim, idx_A, &idx_map_A,
           B.ndim, idx_B, &idx_map_B,
           ndim, idx_C, &idx_map_C);
  return world->ctf->estimate_cost(A.tid, idx_map_A, B.tid, idx_map_B, tid, idx_map_C);
}

template<typename dtype>
int64_t CTF_Tensor<dtype>::estimate_cost(
                                  const CTF_Tensor<dtype>&     A,
                                  const char *                  idx_A,
                                  const char *                  idx_B){
  int * idx_map_A, * idx_map_B;
  conv_idx(A.ndim, idx_A, &idx_map_A,
           ndim, idx_B, &idx_map_B);
  return world->ctf->estimate_cost(A.tid, idx_map_A, tid, idx_map_B);

  
}

template<typename dtype>
void CTF_Tensor<dtype>::contract(dtype                         alpha,
                                  const CTF_Tensor<dtype>&     A,
                                  const char *                  idx_A,
                                  const CTF_Tensor<dtype>&     B,
                                  const char *                  idx_B,
                                  dtype                         beta,
                                  const char *                  idx_C,
                                  CTF_fctr<dtype>              fseq){
  int ret;
  CTF_ctr_type_t tp;
  tp.tid_A = A.tid;
  tp.tid_B = B.tid;
  tp.tid_C = tid;
  conv_idx(A.ndim, idx_A, &tp.idx_map_A,
           B.ndim, idx_B, &tp.idx_map_B,
           ndim, idx_C, &tp.idx_map_C);
  assert(A.world->ctf == world->ctf);
  assert(B.world->ctf == world->ctf);
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
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::set_name(char const * name_) {
  name = name_;
  world->ctf->set_name(tid, name_);
}

template<typename dtype>
void CTF_Tensor<dtype>::profile_on() {
  world->ctf->profile_on(tid);
}

template<typename dtype>
void CTF_Tensor<dtype>::profile_off() {
  world->ctf->profile_off(tid);
}

template<typename dtype>
void CTF_Tensor<dtype>::print(FILE* fp, double cutoff) const{
  world->ctf->print_tensor(fp, tid, cutoff);
}

template<typename dtype>
void CTF_Tensor<dtype>::compare(const CTF_Tensor<dtype>& A, FILE* fp, double cutoff) const{
  world->ctf->compare_tensor(fp, tid, A.tid, cutoff);
}

template<typename dtype>
void CTF_Tensor<dtype>::sum(dtype                      alpha,
                             const CTF_Tensor<dtype>&  A,
                             const char *               idx_A,
                             dtype                      beta,
                             const char *               idx_B,
                             CTF_fsum<dtype>           fseq){
  int ret;
  int * idx_map_A, * idx_map_B;
  CTF_sum_type_t st;
  conv_idx(A.ndim, idx_A, &idx_map_A,
           ndim, idx_B, &idx_map_B);
  assert(A.world->ctf == world->ctf);
    
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
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::scale(dtype                  alpha, 
                               const char *           idx_A,
                               CTF_fscl<dtype>       fseq){
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
  CTF_free(idx_map_A);
  assert(ret == CTF_SUCCESS);
}
template<typename dtype>


void CTF_Tensor<dtype>::permute(dtype                  beta,
                                 CTF_Tensor &          A,
                                 int * const *           perms_A,
                                 dtype                  alpha){
  int ret = world->ctf->permute_tensor(A.tid, perms_A, alpha, A.world->ctf, 
                                       tid, NULL, beta, world->ctf);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::permute(int * const *           perms_B,
                                 dtype                  beta,
                                 CTF_Tensor &          A,
                                 dtype                  alpha){
  int ret = world->ctf->permute_tensor(A.tid, NULL, alpha, A.world->ctf, 
                                       tid, perms_B, beta, world->ctf);
  assert(ret == CTF_SUCCESS);
}
template<typename dtype>
void CTF_Tensor<dtype>::add_to_subworld(
                         CTF_Tensor<dtype> * tsr,
                         dtype alpha,
                         dtype beta) const {
  int ret;
  if (tsr == NULL)
    ret = world->ctf->add_to_subworld(tid, -1, NULL, alpha, beta);
  else
    ret = world->ctf->add_to_subworld(tid, tsr->tid, tsr->world->ctf, alpha, beta);
  assert(ret == CTF_SUCCESS);
}
template<typename dtype>
void CTF_Tensor<dtype>::add_to_subworld(
                         CTF_Tensor<dtype> * tsr) const {
  return add_to_subworld(tsr, get_one<dtype>(), get_one<dtype>());
}

template<typename dtype>
void CTF_Tensor<dtype>::add_from_subworld(
                         CTF_Tensor<dtype> * tsr,
                         dtype alpha,
                         dtype beta) const {
  int ret;
  if (tsr == NULL)
    ret = world->ctf->add_from_subworld(tid, -1, NULL, alpha, beta);
  else
    ret = world->ctf->add_from_subworld(tid, tsr->tid, tsr->world->ctf, alpha, beta);
  assert(ret == CTF_SUCCESS);
}
template<typename dtype>
void CTF_Tensor<dtype>::add_from_subworld(
                         CTF_Tensor<dtype> * tsr) const {
  return add_from_subworld(tsr, get_one<dtype>(), get_one<dtype>());
}



template<typename dtype>
void CTF_Tensor<dtype>::slice(int const *    offsets,
                               int const *    ends,
                               dtype          beta,
                               CTF_Tensor const &  A,
                               int const *    offsets_A,
                               int const *    ends_A,
                               dtype          alpha) const {
  int ret, np_A, np_B;
  if (A.world->comm != world->comm){
    MPI_Comm_size(A.world->comm, &np_A);
    MPI_Comm_size(world->comm,   &np_B);
    assert(np_A != np_B);
    ret = world->ctf->slice_tensor(
              A.tid, offsets_A, ends_A, alpha, A.world->ctf, 
              tid, offsets, ends, beta);
  } else {
    ret =  world->ctf->slice_tensor(A.tid, offsets_A, ends_A, alpha,
                                        tid, offsets, ends, beta);
  }
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
void CTF_Tensor<dtype>::slice(int64_t       corner_off,
                               int64_t       corner_end,
                               dtype          beta,
                               CTF_Tensor const &  A,
                               int64_t       corner_off_A,
                               int64_t       corner_end_A,
                               dtype          alpha) const {
  int * offsets, * ends, * offsets_A, * ends_A;
 
  conv_idx(this->ndim, this->len, corner_off, &offsets);
  conv_idx(this->ndim, this->len, corner_end, &ends);
  conv_idx(A.ndim, A.len, corner_off_A, &offsets_A);
  conv_idx(A.ndim, A.len, corner_end_A, &ends_A);
  
  slice(offsets, ends, beta, A, offsets_A, ends_A, alpha);

  CTF_free(offsets);
  CTF_free(ends);
  CTF_free(offsets_A);
  CTF_free(ends_A);
}

template<typename dtype>
CTF_Tensor<dtype> CTF_Tensor<dtype>::slice(int const *          offsets,
                                             int const *          ends) const {

  return slice(offsets, ends, world);
}

template<typename dtype>
CTF_Tensor<dtype> CTF_Tensor<dtype>::slice(int64_t corner_off,
                                             int64_t corner_end) const {

  return slice(corner_off, corner_end, world);
}




template<typename dtype>
CTF_Tensor<dtype> CTF_Tensor<dtype>::slice(int const *          offsets,
                                             int const *          ends,
                                             CTF_World<dtype> *  owrld) const {
  int i;
  int * new_lens = (int*)CTF_alloc(sizeof(int)*ndim);
  int * new_sym = (int*)CTF_alloc(sizeof(int)*ndim);
  for (i=0; i<ndim; i++){
    assert(ends[i] - offsets[i] > 0 && 
                offsets[i] >= 0 && 
                ends[i] <= len[i]);
    if (sym[i] != NS){
      if (offsets[i] == offsets[i+1] && ends[i] == ends[i+1]){
        new_sym[i] = sym[i];
      } else {
        assert(ends[i+1] >= offsets[i]);
        new_sym[i] = NS;
      }
    } else new_sym[i] = NS;
    new_lens[i] = ends[i] - offsets[i];
  }
  CTF_Tensor<dtype> new_tsr(ndim, new_lens, new_sym, *owrld);
  std::fill(new_sym, new_sym+ndim, 0);
  new_tsr.slice(new_sym, new_lens, 0.0, *this, offsets, ends, 1.0);
  CTF_free(new_lens);
  CTF_free(new_sym);
  return new_tsr;
}

template<typename dtype>
CTF_Tensor<dtype> CTF_Tensor<dtype>::slice(int64_t             corner_off,
                                             int64_t             corner_end,
                                             CTF_World<dtype> *  owrld) const {

  int * offsets, * ends;
 
  conv_idx(this->ndim, this->len, corner_off, &offsets);
  conv_idx(this->ndim, this->len, corner_end, &ends);
  
  CTF_Tensor tsr = slice(offsets, ends, owrld);

  CTF_free(offsets);
  CTF_free(ends);

  return tsr;
}

template<typename dtype>
void CTF_Tensor<dtype>::align(const CTF_Tensor& A){
  if (A.world->ctf != world->ctf) {
    printf("ERROR: cannot align tensors on different CTF instances\n");
    ABORT;
  }
  int ret = world->ctf->align(tid, A.tid);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
dtype CTF_Tensor<dtype>::reduce(CTF_OP op){
  int ret;
  dtype ans;
  ans = 0.0;
  ret = world->ctf->reduce_tensor(tid, op, &ans);
  assert(ret == CTF_SUCCESS);
  return ans;
}
template<typename dtype>
void CTF_Tensor<dtype>::get_max_abs(int        n,
                                     dtype *    data){
  int ret;
  ret = world->ctf->get_max_abs(tid, n, data);
  assert(ret == CTF_SUCCESS);
}

template<typename dtype>
CTF_Tensor<dtype>& CTF_Tensor<dtype>::operator=(dtype val){
  int64_t size;
  dtype* raw = get_raw_data(&size);
  std::fill(raw, raw+size, val);
  return *this;
}

template<typename dtype>
void CTF_Tensor<dtype>::operator=(CTF_Tensor<dtype> A){
  int ret;  

  world = A.world;
  name = A.name;

  if (sym != NULL)
    free(sym);
  if (len != NULL)
    free(len);
    //free(len);
  ret = world->ctf->info_tensor(A.tid, &ndim, &len, &sym);
  assert(ret == CTF_SUCCESS);

  ret = world->ctf->define_tensor(ndim, len, sym, &tid, name, name != NULL);
  assert(ret == CTF_SUCCESS);

  //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

  ret = world->ctf->copy_tensor(A.tid, tid);
  assert(ret == CTF_SUCCESS);
}
    

template<typename dtype>
CTF_Idx_Tensor<dtype> CTF_Tensor<dtype>::operator[](const char * idx_map_){
  CTF_Idx_Tensor<dtype> itsr(this, idx_map_);
  return itsr;
}


template<typename dtype>
CTF_Sparse_Tensor<dtype> CTF_Tensor<dtype>::operator[](std::vector<int64_t> indices){
  CTF_Sparse_Tensor<dtype> stsr(indices,this);
  return stsr;
}




/**
 * \brief a sparse subset of a tensor 
 */
template<typename dtype>
class CTF_Sparse_Tensor {
  public:
    CTF_Tensor<dtype> * parent;
    std::vector<int64_t> indices;
    dtype scale;

    /** 
      * \brief base constructor 
      */
    CTF_Sparse_Tensor();
    
    /**
     * \brief initialize a tensor which corresponds to a set of indices 
     * \param[in] indices a vector of global indices to tensor values
     * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
     */
    CTF_Sparse_Tensor(std::vector<int64_t> indices,
                       CTF_Tensor<dtype> * parent);

    /**
     * \brief initialize a tensor which corresponds to a set of indices 
     * \param[in] number of values this sparse tensor will have locally
     * \param[in] indices an array of global indices to tensor values
     * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
     */
    CTF_Sparse_Tensor(int64_t              n,
                       int64_t *            indices,
                       CTF_Tensor<dtype> * parent);

    /**
     * \brief set the sparse set of indices on the parent tensor to values
     *        forall(j) i = indices[j]; parent[i] = beta*parent[i] + alpha*values[j];
     * \param[in] alpha scaling factor on values array 
     * \param[in] values data, should be of same size as the number of indices (n)
     * \param[in] beta scaling factor to apply to previously existing data
     */
    void write(dtype              alpha, 
               dtype *            values,
               dtype              beta); 

    // C++ overload special-cases of above method
    void operator=(std::vector<dtype> values); 
    void operator+=(std::vector<dtype> values); 
    void operator-=(std::vector<dtype> values); 
    void operator=(dtype * values); 
    void operator+=(dtype * values); 
    void operator-=(dtype * values); 

    /**
     * \brief read the sparse set of indices on the parent tensor to values
     *        forall(j) i = indices[j]; values[j] = alpha*parent[i] + beta*values[j];
     * \param[in] alpha scaling factor on parent array 
     * \param[in] values data, should be preallocated to the same size as the number of indices (n)
     * \param[in] beta scaling factor to apply to previously existing data in values
     */
    void read(dtype              alpha, 
              dtype *            values,
              dtype              beta); 

    // C++ overload special-cases of above method
    operator std::vector<dtype>();
    operator dtype*();
};



