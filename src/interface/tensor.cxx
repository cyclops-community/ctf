/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../interface/common.h"
#include "world.h"

namespace CTF {

  template<typename dtype>
  Tensor<dtype>::Tensor() : CTF_int::tensor() { }

  template<typename dtype>
  Tensor<dtype>::Tensor(const Tensor<dtype>& A,
                        bool                 copy) 
    : CTF_int::tensor(&A, copy) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(const Tensor<dtype> & A,
                        World *               world_) 
    : CTF_int::tensor(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                 order,
                        int const *         lens,
                        int const *         sym,
                        World *             wrld,
                        char const *        name,
                        int const           profile)
    : CTF_int::tensor(Semiring<dtype>(), order, lens, sym, wrld, 1, name, profile) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                 order,
                        int const *         len,
                        int const *         sym,
                        World *             world,
                        Semiring<dtype>     sr,
                        char const *        name,
                        int const           profile)
    : CTF_int::tensor(sr, order, lens, sym, wrld, 1, name, profile) {}

  template<typename dtype>
  Tensor<dtype>::~Tensor(){ }

  template<typename dtype>
  dtype * Tensor<dtype>::get_raw_data(int64_t * size) {
    dtype * data;
    this->get_raw_data(&data, size);
    return data;
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *   npair, 
                                 int64_t **  global_idx, 
                                 dtype **   data) const {
    Pair< dtype > * pairs;
    int ret, i;
    ret = CTF_int::tensor::read_local(npair, &pairs);
    assert(ret == SUCCESS);
    /* FIXME: careful with alloc */
    *global_idx = (int64_t*)CTF_alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)CTF_alloc((*npair)*sizeof(dtype));
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k;
      (*data)[i] = pairs[i].d;
    }
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *      npair,
                                 Pair<dtype> ** pairs) const {
    int ret = CTF_int::tensor::read_local(npair, pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t          npair, 
                           int64_t const *  global_idx, 
                           dtype *          data) const {
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
    }
    ret = CTF_int::tensor::read(npair, pairs);
    assert(ret == SUCCESS);
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t          npair,
                           Pair<dtype> *    pairs) const {
    int ret = CTF_int::tensor::read(npair, pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t          npair, 
                            int64_t const *  global_idx, 
                            dtype const *    data) {
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::write(npair, pairs);
    assert(ret == SUCCESS);
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            Pair<dtype> const * pairs) {
    int ret = CTF_int::tensor::write(npair, pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t          npair, 
                            dtype            alpha, 
                            dtype            beta,
                            int64_t const *  global_idx, 
                            dtype const *    data) {
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::write(npair, &alpha, &beta, pairs);
    assert(ret == SUCCESS);
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            dtype               alpha,
                            dtype               beta,
                            Pair<dtype> const * pairs) {
    int ret = CTF_int::tensor::write(npair, &alpha, &beta, pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t         npair, 
                           dtype           alpha, 
                           dtype           beta,
                           int64_t const * global_idx, 
                           dtype *         data) const{
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::read(npair, &alpha, &beta, pairs);
    assert(ret == SUCCESS);
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t       npair,
                           dtype         alpha,
                           dtype         beta,
                           Pair<dtype> * pairs) const{
    int ret = CTF_int::tensor::read(npair, &alpha, &beta, pairs);
    assert(ret == SUCCESS);
  }


  template<typename dtype>
  void Tensor<dtype>::read_all(int64_t * npair, dtype ** vals) const {
    int ret;
    ret = CTF_int::tensor::allread(npair, vals);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::read_all(dtype * vals) const {
    int ret;
    int64_t npair;
    ret = CTF_int::tensor::allread(&npair, vals);
    assert(ret == SUCCESS);
    return npair;
  }

  template<typename dtype>
  void Tensor<dtype>::set_name(char const * name_) {
    name = name_;
    CTF_int::tensor::set_name(name_);
  }

  template<typename dtype>
  void Tensor<dtype>::profile_on() {
    CTF_int::tensor::profile_on();
  }

  template<typename dtype>
  void Tensor<dtype>::profile_off() {
    CTF_int::tensor::profile_off();
  }

  template<typename dtype>
  void Tensor<dtype>::print(FILE* fp, double cutoff) const{
    CTF_int::tensor::print(fp, cutoff);
  }

  template<typename dtype>
  void Tensor<dtype>::compare(const Tensor<dtype>& A, FILE* fp, double cutoff) const{
    CTF_int::tensor::compare(fp, &A, cutoff);
  }

  template<typename dtype>
  void Tensor<dtype>::permute(dtype             beta,
                              Tensor &          A,
                              int * const *     perms_A,
                              dtype             alpha){
    int ret = CTF_int::tensor::permute(&A, perms_A, &alpha, 
                                           NULL, &beta);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::permute(int * const * perms_B,
                              dtype         beta,
                              Tensor &      A,
                              dtype         alpha){
    int ret = CTF_int::tensor::permute(&A, NULL, &alpha,
                                           perms_B, &beta);
    assert(ret == SUCCESS);
  }
  template<typename dtype>
  void Tensor<dtype>::add_to_subworld(
                           Tensor<dtype> * tsr,
                           dtype alpha,
                           dtype beta) const {
    int ret;
    if (tsr == NULL)
      ret = CTF_int::tensor::add_to_subworld(NULL, alpha, beta);
    else
      ret = CTF_int::tensor::add_to_subworld(&tsr->tid, alpha, beta);
    assert(ret == SUCCESS);
  }
  template<typename dtype>
  void Tensor<dtype>::add_to_subworld(
                           Tensor<dtype> * tsr) const {
    return add_to_subworld(tsr, sr.mulid, sr.mulid);
  }

  template<typename dtype>
  void Tensor<dtype>::add_from_subworld(
                           Tensor<dtype> * tsr,
                           dtype alpha,
                           dtype beta) const {
    int ret;
    if (tsr == NULL)
      ret = CTF_int::tensor::add_from_subworld(-1, NULL, &alpha, &beta);
    else
      ret = CTF_int::tensor::add_from_subworld(tsr->tid, tsr->wrld, &alpha, &beta);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::add_from_subworld(
                           Tensor<dtype> * tsr) const {
    return add_from_subworld(tsr, sr.mulid, sr.mulid);
  }

  template<typename dtype>
  void Tensor<dtype>::slice(int const *    offsets,
                            int const *    ends,
                            dtype          beta,
                            Tensor const & A,
                            int const *    offsets_A,
                            int const *    ends_A,
                            dtype          alpha) const {
    int ret, np_A, np_B;
    if (A.wrld->comm != wrld->comm){
      MPI_Comm_size(A.wrld->comm, &np_A);
      MPI_Comm_size(wrld->comm,   &np_B);
      assert(np_A != np_B);
      //FIXME: was reversed?
      ret = CTF_int::tensor::slice(
                offsets, ends, beta, A,
                offsets_A, ends_A, alpha);
    } else {
      ret = CTF_int::tensor::slice(
                offsets, ends, beta, A,
                offsets_A, ends_A, alpha);
    }
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::slice(int64_t        corner_off,
                            int64_t        corner_end,
                            dtype          beta,
                            Tensor const & A,
                            int64_t        corner_off_A,
                            int64_t        corner_end_A,
                            dtype          alpha) const {
    int * offsets, * ends, * offsets_A, * ends_A;
   
    conv_idx(this->order, this->len, corner_off, &offsets);
    conv_idx(this->order, this->len, corner_end, &ends);
    conv_idx(A.order, A.len, corner_off_A, &offsets_A);
    conv_idx(A.order, A.len, corner_end_A, &ends_A);
    
    slice(offsets, ends, beta, A, offsets_A, ends_A, alpha);

    CTF_free(offsets);
    CTF_free(ends);
    CTF_free(offsets_A);
    CTF_free(ends_A);
  }

  template<typename dtype>
  Tensor<dtype> Tensor<dtype>::slice(int const * offsets,
                                     int const * ends) const {

    return slice(offsets, ends, wrld);
  }

  template<typename dtype>
  Tensor<dtype> Tensor<dtype>::slice(int64_t corner_off,
                                     int64_t corner_end) const {

    return slice(corner_off, corner_end, wrld);
  }
  
  template<typename dtype>
  Tensor<dtype> Tensor<dtype>::slice(int const *  offsets,
                                     int const *  ends,
                                     World *      owrld) const {
    int i;
    int * new_lens = (int*)CTF_alloc(sizeof(int)*order);
    int * new_sym = (int*)CTF_alloc(sizeof(int)*order);
    for (i=0; i<order; i++){
      assert(ends[i] - offsets[i] > 0 && 
                  offsets[i] >= 0 && 
                  ends[i] <= lens[i]);
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
    Tensor<dtype> new_tsr(order, new_lens, new_sym, *owrld);
    std::fill(new_sym, new_sym+order, 0);
    new_tsr.slice(new_sym, new_lens, 0.0, *this, offsets, ends, 1.0);
    CTF_free(new_lens);
    CTF_free(new_sym);
    return new_tsr;
  }

  template<typename dtype>
  Tensor<dtype> Tensor<dtype>::slice(int64_t  corner_off,
                                     int64_t  corner_end,
                                     World *  owrld) const {

    int * offsets, * ends;
   
    conv_idx(this->order, this->len, corner_off, &offsets);
    conv_idx(this->order, this->len, corner_end, &ends);
    
    Tensor tsr = slice(offsets, ends, owrld);

    CTF_free(offsets);
    CTF_free(ends);

    return tsr;
  }

  template<typename dtype>
  void Tensor<dtype>::align(const Tensor& A){
    if (A.wrld->global_comm.cm != wrld->global_comm.cm) {
      printf("ERROR: cannot align tensors on different CTF instances\n");
      assert(0);
    }
    int ret = CTF_int::tensor::align(A);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  dtype Tensor<dtype>::reduce(OP op){
    int ret;
    dtype ans;
    ans = 0.0;
    ret = CTF_int::tensor::reduce(op, &ans);
    assert(ret == SUCCESS);
    return ans;
  }

  template<typename dtype>
  void Tensor<dtype>::get_max_abs(int     n,
                                  dtype * data){
    int ret;
    ret = CTF_int::tensor::get_max_abs(n, data);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::estimate_cost(
                                    const Tensor<dtype>&     A,
                                    const char *             idx_A,
                                    const Tensor<dtype>&     B,
                                    const char *             idx_B,
                                    const char *             idx_C){
    int * idx_map_A, * idx_map_B, * idx_map_C;
    conv_idx(A.order, idx_A, &idx_map_A,
             B.order, idx_B, &idx_map_B,
             order, idx_C, &idx_map_C);
    return CTF_int::tensor::estimate_cost(A.tid, idx_map_A, B.tid, idx_map_B, tid, idx_map_C);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::estimate_cost(
                                    const Tensor<dtype>& A,
                                    const char *         idx_A,
                                    const char *         idx_B){
    int * idx_map_A, * idx_map_B;
    conv_idx(A.order, idx_A, &idx_map_A,
             order, idx_B, &idx_map_B);
    return CTF_int::tensor::estimate_cost(A.tid, idx_map_A, tid, idx_map_B);
    
  }

  template<typename dtype>
  void Tensor<dtype>::contract(dtype                 alpha,
                               const Tensor<dtype>&  A,
                               const char *          idx_A,
                               const Tensor<dtype>&  B,
                               const char *          idx_B,
                               dtype                 beta,
                               const char *          idx_C,
                               Bivar_Function<dtype> fseq){
    int ret;
    CTF_int::ctr_type_t tp;
    tp.tid_A = A.tid;
    tp.tid_B = B.tid;
    tp.tid_C = tid;
    conv_idx(A.order, idx_A, &tp.idx_map_A,
             B.order, idx_B, &tp.idx_map_B,
             order, idx_C, &tp.idx_map_C);
    assert(A.wrld->ctf == world->ctf);
    assert(B.wrld->ctf == world->ctf);
    ret = CTF_int::tensor::contract(&tp, fseq, alpha, beta);
  /*  else {
      fseq_elm_ctr<dtype> fs;
      fs.func_ptr = fseq.func_ptr;
      ret = CTF_int::tensor::contract(&tp, fs, alpha, beta);
    }*/
    CTF_free(tp.idx_map_A);
    CTF_free(tp.idx_map_B);
    CTF_free(tp.idx_map_C);
    assert(ret == SUCCESS);
  }


  template<typename dtype>
  void Tensor<dtype>::sum(dtype                  alpha,
                          const Tensor<dtype>&   A,
                          const char *           idx_A,
                          dtype                  beta,
                          const char *           idx_B,
                          Univar_Function<dtype> fseq){
    int ret;
    int * idx_map_A, * idx_map_B;
    CTF_int::sum_type_t st;
    conv_idx(A.order, idx_A, &idx_map_A,
             order, idx_B, &idx_map_B);
    assert(A.wrld->ctf == world->ctf);
      
    st.idx_map_A = idx_map_A;
    st.idx_map_B = idx_map_B;
    st.tid_A = A.tid;
    st.tid_B = tid;
    ret = CTF_int::tensor::sums(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fseq);
    CTF_free(idx_map_A);
    CTF_free(idx_map_B);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::scale(dtype                alpha, 
                            const char *         idx_A,
                            Endomorphism<dtype>  fseq){
    int ret;
    int * idx_map_A;
    conv_idx(order, idx_A, &idx_map_A);
    ret = CTF_int::tensor::scale(alpha, tid, idx_map_A, fseq);
    CTF_free(idx_map_A);
    assert(ret == SUCCESS);
  }
  template<typename dtype>
  template<typename dtype>
  Tensor<dtype>& Tensor<dtype>::operator=(dtype val){
    int64_t size;
    dtype* raw = get_raw_data(&size);
    std::fill(raw, raw+size, val);
    return *this;
  }

  template<typename dtype>
  void Tensor<dtype>::operator=(Tensor<dtype> A){
    int ret;  

//    FIXME: delete current data

    ret = CTF_int::tensor::init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile);
    assert(ret == SUCCESS);
/*
    sr = A.sr;
    world = A.wrld;
    name = A.name;

    if (sym != NULL)
      CTF_free(sym);
    if (len != NULL)
      CTF_free(len);
      //CTF_free(len);
    ret = CTF_int::tensor::info(&A, &order, &len, &sym);
    assert(ret == SUCCESS);

    ret = CTF_int::tensor::define(sr, order, len, sym, &tid, 1, name, name != NULL);
    assert(ret == SUCCESS);

    //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

    ret = CTF_int::tensor::copy(A.tid, tid);
    assert(ret == SUCCESS);*/
  }
      

  template<typename dtype>
  Idx_Tensor<dtype> Tensor<dtype>::operator[](const char * idx_map_){
    Idx_Tensor<dtype> idxtsr(this, idx_map_);
    return idxtsr;
  }


  template<typename dtype>
  Sparse_Tensor<dtype> Tensor<dtype>::operator[](std::vector<int64_t> indices){
    Sparse_Tensor<dtype> stsr(indices,this);
    return stsr;
  }

}

