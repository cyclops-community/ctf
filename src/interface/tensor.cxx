/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../interface/common.h"
#include "world.h"
#include "idx_tensor.h"
#include "../tensor/untyped_tensor.h"

namespace CTF {

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::Tensor() : CTF_int::tensor() { }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::Tensor(tensor const & A,
                                bool           copy)
    : CTF_int::tensor(&A, copy) { }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::Tensor(tensor const & A,
                                World &        world_)
    : CTF_int::tensor(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile) { }


  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::Tensor(int                       order,
                                int const *               len,
                                int const *               sym,
                                World &                   world,
                                char const *              name,
                                int const                 profile,
                                CTF_int::algstrct const & sr)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {}



  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::Tensor(int                       order,
                                int const *               len,
                                int const *               sym,
                                World &                   world,
                                CTF_int::algstrct const & sr,
                                char const *              name,
                                int const                 profile)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {}

  template<typename dtype, bool is_ord>
  Idx_Tensor Tensor<dtype, is_ord>::operator[](const char * idx_map_){
    ASSERT(strlen(idx_map_)==order);
    Idx_Tensor idxtsr(this, idx_map_);
    return idxtsr;
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>::~Tensor(){ }

  template<typename dtype, bool is_ord>
  dtype * Tensor<dtype, is_ord>::get_raw_data(int64_t * size) const {
    dtype * data;
    tensor::get_raw_data((char**)&data, size);
    return data;
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read_local(int64_t *  npair,
                                         int64_t ** global_idx,
                                         dtype **   data) const {
    Pair< dtype > * pairs;
    int ret, i;
    ret = CTF_int::tensor::read_local(npair,(char**)&pairs);
    assert(ret == SUCCESS);
    /* FIXME: careful with alloc */
    *global_idx = (int64_t*)CTF_int::alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)CTF_int::alloc((*npair)*sizeof(dtype));
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k;
      (*data)[i] = pairs[i].d;
    }
    if (pairs != NULL) CTF_int::cfree(pairs);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read_local(int64_t *      npair,
                                         Pair<dtype> ** pairs) const {
    int ret = CTF_int::tensor::read_local(npair, (char**)pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read(int64_t         npair,
                                   int64_t const * global_idx,
                                   dtype *         data){
    int ret;
    int64_t i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
    }
    ret = CTF_int::tensor::read(npair, (char*)pairs);
    assert(ret == SUCCESS);
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    CTF_int::cfree(pairs);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read(int64_t       npair,
                                   Pair<dtype> * pairs){
    int ret = CTF_int::tensor::read(npair, (char*)pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::write(int64_t         npair,
                                    int64_t const * global_idx,
                                    dtype const *   data) {
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::write(npair, sr->mulid(), sr->addid(), (char*)pairs);
    assert(ret == SUCCESS);
    CTF_int::cfree(pairs);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::write(int64_t             npair,
                                    Pair<dtype> const * pairs) {
    int ret = CTF_int::tensor::write(npair, 1.0, 0.0, (char const *)pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::write(int64_t         npair,
                                    dtype           alpha,
                                    dtype           beta,
                                    int64_t const * global_idx,
                                    dtype const *   data) {
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, (char*)pairs);
    assert(ret == SUCCESS);
    CTF_int::cfree(pairs);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::write(int64_t             npair,
                                    dtype               alpha,
                                    dtype               beta,
                                    Pair<dtype> const * pairs) {
    int ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, (char const *)pairs);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read(int64_t         npair,
                                   dtype           alpha,
                                   dtype           beta,
                                   int64_t const * global_idx,
                                   dtype *         data){
    int ret, i;
    Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, (char*)pairs);
    assert(ret == SUCCESS);
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    CTF_int::cfree(pairs);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read(int64_t       npair,
                                   dtype         alpha,
                                   dtype         beta,
                                   Pair<dtype> * pairs){
    int ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, (char*)pairs);
    assert(ret == SUCCESS);
  }


  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::read_all(int64_t * npair, dtype ** vals){
    int ret;
    ret = CTF_int::tensor::allread(npair, ((char**)vals));
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  int64_t Tensor<dtype, is_ord>::read_all(dtype * vals){
    int ret;
    int64_t npair;
    ret = CTF_int::tensor::allread(&npair, (char*)vals);
    assert(ret == SUCCESS);
    return npair;
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::set_name(char const * name_) {
    CTF_int::tensor::set_name(name_);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::profile_on() {
    CTF_int::tensor::profile_on();
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::profile_off() {
    CTF_int::tensor::profile_off();
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::print(FILE* fp, dtype cutoff) const{
    CTF_int::tensor::print(fp, (char *)&cutoff);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::print(FILE* fp) const{
    CTF_int::tensor::print(fp, NULL);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::compare(const Tensor<dtype, is_ord>& A, FILE* fp, double cutoff) const{
    CTF_int::tensor::compare(fp, &A, cutoff);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::permute(dtype         beta,
                                      Tensor &      A,
                                      int * const * perms_A,
                                      dtype         alpha){
    tensor t = tensor();
    t.sr = sr->clone();
    int ret = CTF_int::tensor::permute(&A, perms_A, (char*)&alpha,
                                           &t, (char*)&beta);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::permute(int * const * perms_B,
                                      dtype         beta,
                                      Tensor &      A,
                                      dtype         alpha){
    tensor t = tensor();
    t.sr = sr->clone();
    int ret = CTF_int::tensor::permute(&A, &t, (char*)&alpha,
                                           perms_B, (char*)&beta);
    assert(ret == SUCCESS);
  }
  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::add_to_subworld(
                                     Tensor<dtype, is_ord> * tsr,
                                     dtype                   alpha,
                                     dtype                   beta){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      CTF_int::tensor::add_to_subworld(&t, (char*)&alpha, (char*)&beta);
    } else
      CTF_int::tensor::add_to_subworld(tsr, (char*)&alpha, (char*)&beta);
  }
 
  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::add_to_subworld(
                           Tensor<dtype, is_ord> * tsr){
    return add_to_subworld(tsr, sr->mulid(), sr->mulid());
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::add_from_subworld(
                                 Tensor<dtype, is_ord> * tsr,
                                 dtype                   alpha,
                                 dtype                   beta){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      CTF_int::tensor::add_from_subworld(&t, (char*)&alpha, (char*)&beta);
    } else
      CTF_int::tensor::add_from_subworld(tsr, (char*)&alpha, (char*)&beta);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::add_from_subworld(
                           Tensor<dtype, is_ord> * tsr){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      return CTF_int::tensor::add_from_subworld(&t, sr->mulid(), sr->mulid());
    } else
      return CTF_int::tensor::add_from_subworld(tsr, sr->mulid(), sr->mulid());
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::slice(int const *    offsets,
                                    int const *    ends,
                                    dtype          beta,
                                    Tensor const & A,
                                    int const *    offsets_A,
                                    int const *    ends_A,
                                    dtype          alpha){
    int np_A, np_B;
    if (A.wrld->comm != wrld->comm){
      MPI_Comm_size(A.wrld->comm, &np_A);
      MPI_Comm_size(wrld->comm,   &np_B);
      assert(np_A != np_B);
      //FIXME: was reversed?
      CTF_int::tensor::slice(
          offsets, ends, (char*)&beta, (Tensor *)&A,
          offsets_A, ends_A, (char*)&alpha);
    } else {
      CTF_int::tensor::slice(
          offsets, ends, (char*)&beta, (Tensor *)&A,
          offsets_A, ends_A, (char*)&alpha);
    }
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::slice(int64_t        corner_off,
                                    int64_t        corner_end,
                                    dtype          beta,
                                    Tensor const & A,
                                    int64_t        corner_off_A,
                                    int64_t        corner_end_A,
                                    dtype          alpha){
    int * offsets, * ends, * offsets_A, * ends_A;
   
    conv_idx(this->order, this->len, corner_off, &offsets);
    conv_idx(this->order, this->len, corner_end, &ends);
    conv_idx(A.order, A.len, corner_off_A, &offsets_A);
    conv_idx(A.order, A.len, corner_end_A, &ends_A);
    
    slice(offsets, ends, beta, &A, offsets_A, ends_A, (char*)&alpha);

    CTF_int::cfree(offsets);
    CTF_int::cfree(ends);
    CTF_int::cfree(offsets_A);
    CTF_int::cfree(ends_A);
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord> Tensor<dtype, is_ord>::slice(int const * offsets,
                                                     int const * ends) const {

    return slice(offsets, ends, wrld);
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord> Tensor<dtype, is_ord>::slice(int64_t corner_off,
                                                     int64_t corner_end) const {

    return slice(corner_off, corner_end, wrld);
  }
  
  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord> Tensor<dtype, is_ord>::slice(int const *  offsets,
                                                     int const *  ends,
                                                     World *      owrld) const {
    int i;
    int * new_lens = (int*)CTF_int::alloc(sizeof(int)*order);
    int * new_sym = (int*)CTF_int::alloc(sizeof(int)*order);
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
    //FIXME: could discard sr qualifiers
    Tensor<dtype, is_ord> new_tsr(order, new_lens, new_sym, *owrld, *sr);
//   Tensor<dtype, is_ord> new_tsr = tensor(sr, order, new_lens, new_sym, owrld, 1);
    std::fill(new_sym, new_sym+order, 0);
    new_tsr.slice(new_sym, new_lens, *(dtype*)sr->addid(), *this, offsets, ends, *(dtype*)sr->mulid());
/*    new_tsr.slice(
        new_sym, new_lens, sr->addid(), this,
        offsets, ends, sr->mulid());*/
    CTF_int::cfree(new_lens);
    CTF_int::cfree(new_sym);
    return new_tsr;
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord> Tensor<dtype, is_ord>::slice(int64_t  corner_off,
                                                     int64_t  corner_end,
                                                     World *  owrld) const {

    int * offsets, * ends;
   
    conv_idx(this->order, this->len, corner_off, &offsets);
    conv_idx(this->order, this->len, corner_end, &ends);
    
    Tensor tsr = slice(offsets, ends, owrld);

    CTF_int::cfree(offsets);
    CTF_int::cfree(ends);

    return tsr;
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::align(const Tensor & A){
    if (A.wrld->cdt.cm != wrld->cdt.cm) {
      printf("ERROR: cannot align tensors on different CTF instances\n");
      assert(0);
    }
    int ret = CTF_int::tensor::align(&A);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  dtype Tensor<dtype, is_ord>::reduce(OP op){
    int ret;
    dtype ans;
    switch (op) {
      case OP_SUM:
        ret = reduce_sum((char*)&ans);
        break;
      case OP_SUMABS:
        ret = reduce_sumabs((char*)&ans);
        break;
      case OP_SUMSQ:
        ret = reduce_sumsq((char*)&ans);
        break;
      case OP_MAX:
        {
          dtype minval;
          sr->min((char*)&minval);
          Monoid<dtype, 1> mmax = Monoid<dtype, 1>(minval, default_max<dtype, 1>, MPI_MAX);
          ret = reduce_sum((char*)&ans, &mmax);
        }
        break;
      case OP_MIN:
        {
          dtype maxval;
          sr->max((char*)&maxval);
          Monoid<dtype, 1> mmin = Monoid<dtype, 1>(maxval, default_min<dtype, 1>, MPI_MIN);
          ret = reduce_sum((char*)&ans, &mmin);
        }
        break;
      case OP_MAXABS:
        {
          dtype minval;
          sr->min((char*)&minval);
          Monoid<dtype, 1> mmax = Monoid<dtype, 1>(minval, default_max<dtype, 1>, MPI_MAX);
          ret = reduce_sumabs((char*)&ans, &mmax);
        }
        break;
      case OP_MINABS:
        {
          dtype maxval;
          sr->max((char*)&maxval);
          Monoid<dtype, 1> mmin = Monoid<dtype, 1>(maxval, default_min<dtype, 1>, MPI_MIN);
          ret = reduce_sumabs((char*)&ans, &mmin);
        }
        break;
    }
    assert(ret == SUCCESS);
    return ans;
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::get_max_abs(int     n,
                                  dtype * data) const {
    int ret;
    ret = CTF_int::tensor::get_max_abs(n, data);
    assert(ret == SUCCESS);
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::contract(dtype                  alpha,
                                       Tensor<dtype, is_ord>& A,
                                       const char *           idx_A,
                                       Tensor<dtype, is_ord>& B,
                                       const char *           idx_B,
                                       dtype                  beta,
                                       const char *           idx_C){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    assert(B.wrld->cdt.cm == wrld->cdt.cm);
    CTF_int::contraction ctr 
      = CTF_int::contraction(&A, idx_A, &B, idx_B, (char*)&alpha, this, idx_C, (char*)&beta);
    ctr.execute();
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::contract(dtype                  alpha,
                                       Tensor<dtype, is_ord>& A,
                                       const char *           idx_A,
                                       Tensor<dtype, is_ord>& B,
                                       const char *           idx_B,
                                       dtype                  beta,
                                       const char *           idx_C,
                                       Bivar_Function<dtype>  fseq){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    assert(B.wrld->cdt.cm == wrld->cdt.cm);
    CTF_int::contraction ctr 
      = CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, this, idx_C, (char const *)&beta, &fseq);
    ctr.execute();
  }


  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::sum(dtype                  alpha,
                                  Tensor<dtype, is_ord>& A,
                                  const char *           idx_A,
                                  dtype                  beta,
                                  const char *           idx_B){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);

    CTF_int::summation sum 
      = CTF_int::summation(&A, idx_A, (char*)&alpha, this, idx_B, (char*)&beta);

    sum.execute();

  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::sum(dtype                  alpha,
                                  Tensor<dtype, is_ord>& A,
                                  const char *           idx_A,
                                  dtype                  beta,
                                  const char *           idx_B,
                                  Univar_Function<dtype> fseq){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    
    CTF_int::summation sum = CTF_int::summation(&A, idx_A, (char const *)&alpha, this, idx_B, (char const *)&beta &fseq);

    sum.execute();
  }

  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::scale(dtype        alpha,
                                    const char * idx_A){
    CTF_int::scaling scl = CTF_int::scaling(this, idx_A, (char*)&alpha);
    scl.execute();
  }


  template<typename dtype, bool is_ord>
  void Tensor<dtype, is_ord>::scale(dtype               alpha,
                                    const char *        idx_A,
                                    Endomorphism<dtype> fseq){
    CTF_int::scaling scl = CTF_int::scaling(this, idx_A, (char const *)&alpha, fseq);
    scl.execute();
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>& Tensor<dtype, is_ord>::operator=(dtype val){
    set(&val);
/*    int64_t size;
    dtype* raw = get_raw_data(&size);
    //FIXME: Uuuuh, padding?
    ASSERT(0);
    std::fill(raw, raw+size, val);*/
    return *this;
  }
 
  template<typename dtype, bool is_ord>
  double Tensor<dtype, is_ord>::estimate_time(
                                    Tensor<dtype, is_ord>& A,
                                    const char *   idx_A,
                                    Tensor<dtype, is_ord>& B,
                                    const char *   idx_B,
                                    const char *   idx_C){
    CTF_int::contraction ctr
      = CTF_int::contraction(&A, idx_A, &B, idx_B, sr->mulid(), this, idx_C, sr->addid());
    return ctr.estimate_time();
  }
    
  template<typename dtype, bool is_ord>
  double Tensor<dtype, is_ord>::estimate_time(
                                    Tensor<dtype, is_ord>& A,
                                    const char *   idx_A,
                                    const char *   idx_B){
    CTF_int::summation sum = CTF_int::summation(&A, idx_A, sr->mulid(), this, idx_B, sr->addid());

    return sum.estimate_time();
    
  }

  template<typename dtype, bool is_ord>
  Tensor<dtype, is_ord>& Tensor<dtype, is_ord>::operator=(Tensor<dtype, is_ord> A){

    free_self();
    init(A.sr, A.order, A.lens, A.sym, A.wrld, 1, A.name, A.profile);
    copy_tensor_data(&A);
    return *this;
/*
    sr = A.sr;
    world = A.wrld;
    name = A.name;

    if (sym != NULL)
      CTF_int::cfree(sym);
    if (len != NULL)
      CTF_int::cfree(len);
      //CTF_int::cfree(len);
    ret = CTF_int::tensor::info(&A, &order, &len, &sym);
    assert(ret == SUCCESS);

    ret = CTF_int::tensor::define(sr, order, len, sym, &tid, 1, name, name != NULL);
    assert(ret == SUCCESS);

    //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

    ret = CTF_int::tensor::copy(A.tid, tid);
    assert(ret == SUCCESS);*/
  }


  template<typename dtype, bool is_ord>
  Sparse_Tensor<dtype, is_ord> Tensor<dtype, is_ord>::operator[](std::vector<int64_t> indices){
    Sparse_Tensor<dtype, is_ord> stsr(indices,this);
    return stsr;
  }

}

