/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../interface/common.h"
#include "world.h"
#include "idx_tensor.h"
#include "../tensor/untyped_tensor.h"

namespace CTF {

  template<typename dtype>
  Tensor<dtype>::Tensor() : CTF_int::tensor() { }


  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        char const *              name,
                        bool                      profile,
                        CTF_int::algstrct const & sr)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {}

  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {}


  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        bool                      is_sparse,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile, is_sparse) {}



  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, NULL, &world, 1, name, profile) {}


  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        char const *              idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        char const *              name,
                        bool                      profile,
                        CTF_int::algstrct const & sr_)
    : CTF_int::tensor(&sr_, order, len, sym, &world, idx, prl, blk, name, profile) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(bool           copy,
                                tensor const & A)
    : CTF_int::tensor(&A, copy) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(Tensor<dtype> const & A)
    : CTF_int::tensor(&A, true) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(tensor const & A)
    : CTF_int::tensor(&A, true) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(tensor const & A,
                        World &        world_)
    : CTF_int::tensor(A.sr, A.order, A.lens, A.sym, &world_, 1, A.name, A.profile) { }

  template<typename dtype>
  Tensor<dtype>::Tensor(tensor &    A,
                        int const * new_sym)
    : CTF_int::tensor(&A, new_sym){ }

  template<typename dtype>
  Idx_Tensor Tensor<dtype>::operator[](const char * idx_map_){
    //ASSERT(strlen(idx_map_)==order);
    Idx_Tensor idxtsr(this, idx_map_);
    return idxtsr;
  }

  template<typename dtype>
  Tensor<dtype>::~Tensor(){ }

  template<typename dtype>
  dtype * Tensor<dtype>::get_raw_data(int64_t * size) const {
    dtype * data;
    tensor::get_raw_data((char**)&data, size);
    return data;
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *  npair,
                                 int64_t ** global_idx,
                                 dtype **   data) const {
    char * cpairs;
    int ret, i;
    ret = CTF_int::tensor::read_local(npair,&cpairs);
    assert(ret == CTF_int::SUCCESS);
    /* FIXME: careful with alloc */
    *global_idx = (int64_t*)CTF_int::alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)CTF_int::alloc((*npair)*sizeof(dtype));
    CTF_int::PairIterator pairs(sr, cpairs);
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k();
      pairs[i].read_val((char*)((*data)+i));
    }
    if (cpairs != NULL) CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *      npair,
                                         Pair<dtype> ** pairs) const {
    //FIXME raises mem consumption
    char * cpairs; 
    int ret = CTF_int::tensor::read_local(npair, &cpairs);
    *pairs = Pair<dtype>::cast_char_arr(cpairs, *npair);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::read_local_nnz(int64_t *  npair,
                                     int64_t ** global_idx,
                                     dtype **   data) const {
    char * cpairs;
    int ret, i;
    ret = CTF_int::tensor::read_local_nnz(npair,&cpairs);
    assert(ret == CTF_int::SUCCESS);
    /* FIXME: careful with alloc */
    *global_idx = (int64_t*)CTF_int::alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)CTF_int::alloc((*npair)*sizeof(dtype));
    CTF_int::PairIterator pairs(sr, cpairs);
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k();
      pairs[i].read_val((char*)((*data)+i));
    }
    if (cpairs != NULL) CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read_local_nnz(int64_t *      npair,
                                     Pair<dtype> ** pairs) const {
    //FIXME raises mem consumption
    char * cpairs; 
    int ret = CTF_int::tensor::read_local_nnz(npair, &cpairs);
    *pairs = Pair<dtype>::cast_char_arr(cpairs, *npair);
    assert(ret == CTF_int::SUCCESS);
  }


  template<typename dtype>
  void Tensor<dtype>::read(int64_t         npair,
                           int64_t const * global_idx,
                           dtype *         data){
    int ret;
    int64_t i;
    /*Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
    }*/
    char * cpairs = (char*)CTF_int::alloc(npair*sr->pair_size());
    CTF_int::PairIterator pairs = CTF_int::PairIterator(sr, cpairs);
    for (i=0; i<npair; i++){
      pairs[i].write_key(global_idx[i]);
    }
    ret = CTF_int::tensor::read(npair, cpairs);
    assert(ret == CTF_int::SUCCESS);
    for (i=0; i<npair; i++){
      pairs[i].read_val((char*)(data+i));
    }
    CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t       npair,
                           Pair<dtype> * pairs){
    //FIXME raises mem consumption
    char * cpairs = Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::read(npair, cpairs);
    if (cpairs != (char*)pairs){
      CTF_int::PairIterator ipairs = CTF_int::PairIterator(sr, cpairs);
      for (int64_t i=0; i<npair; i++){
        pairs[i].k = ipairs[i].k();
        ipairs[i].read_val((char*)&(pairs[i].d));
      }
      CTF_int::cdealloc(cpairs);
    }
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t         npair,
                            int64_t const * global_idx,
                            dtype const *   data) {
    int ret, i;
    /*Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }*/
    char * cpairs = (char*)CTF_int::alloc(npair*sr->pair_size());
    CTF_int::PairIterator pairs = CTF_int::PairIterator(sr, cpairs);
    for (i=0; i<npair; i++){
      pairs[i].write_key(global_idx[i]);
      pairs[i].write_val((char*)&(data[i]));
    }
    ret = CTF_int::tensor::write(npair, sr->mulid(), sr->addid(), cpairs);
    assert(ret == CTF_int::SUCCESS);
    CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            Pair<dtype> const * pairs) {

    //FIXME raises mem consumption
    char * cpairs = Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::write(npair, sr->mulid(), sr->addid(), cpairs);
    assert(ret == CTF_int::SUCCESS);
    if (cpairs != (char*)pairs)
      CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t         npair,
                            dtype           alpha,
                            dtype           beta,
                            int64_t const * global_idx,
                            dtype const *   data) {
    int ret, i;
    char * cpairs = (char*)CTF_int::alloc(npair*sr->pair_size());
    CTF_int::PairIterator pairs = CTF_int::PairIterator(sr, cpairs);
    for (i=0; i<npair; i++){
      pairs[i].write_key(global_idx[i]);
      pairs[i].write_val((char*)&(data[i]));
    }
    /*Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }*/
    ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, cpairs);
    assert(ret == CTF_int::SUCCESS);
    CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            dtype               alpha,
                            dtype               beta,
                            Pair<dtype> const * pairs) {
    char * cpairs = Pair<dtype>::scast_to_char_arr(pairs, npair);

    int ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, cpairs);
    if (cpairs != (char*)pairs) CTF_int::cdealloc(cpairs);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t         npair,
                                   dtype           alpha,
                                   dtype           beta,
                                   int64_t const * global_idx,
                                   dtype *         data){
    int ret, i;
    char * cpairs = (char*)CTF_int::alloc(npair*sr->pair_size());
    CTF_int::PairIterator pairs = CTF_int::PairIterator(sr, cpairs);
    for (i=0; i<npair; i++){
      pairs[i].write_key(global_idx[i]);
      pairs[i].write_val(data[i]);
    }
    ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, cpairs);
    assert(ret == CTF_int::SUCCESS);
    for (i=0; i<npair; i++){
      pairs[i].read_val((char*)((*data)+i));
    }
    CTF_int::cdealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t       npair,
                           dtype         alpha,
                           dtype         beta,
                           Pair<dtype> * pairs){
    char * cpairs = Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, cpairs);
    if (cpairs != (char*)pairs){
      CTF_int::PairIterator ipairs = CTF_int::PairIterator(sr, cpairs);
      for (int64_t i=0; i<npair; i++){
        pairs[i].k = ipairs[i].k();
        ipairs[i].read_val((char*)&(pairs[i].d));
      }
      CTF_int::cdealloc(cpairs);
    }
    assert(ret == CTF_int::SUCCESS);
  }


  template<typename dtype>
  void Tensor<dtype>::read_all(int64_t * npair, dtype ** vals, bool unpack){
    int ret;
    ret = CTF_int::tensor::allread(npair, ((char**)vals), unpack);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::read_all(dtype * vals, bool unpack){
    int ret;
    int64_t npair;
    ret = CTF_int::tensor::allread(&npair, (char*)vals, unpack);
    assert(ret == CTF_int::SUCCESS);
    return npair;
  }

  template<typename dtype>
  void Tensor<dtype>::set_name(char const * name_) {
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
  void Tensor<dtype>::print(FILE* fp, dtype cutoff) const{
    CTF_int::tensor::print(fp, (char *)&cutoff);
  }

  template<typename dtype>
  void Tensor<dtype>::print(FILE* fp) const{
    CTF_int::tensor::print(fp, NULL);
  }

  template<typename dtype>
  void Tensor<dtype>::compare(const Tensor<dtype>& A, FILE* fp, double cutoff){
    CTF_int::tensor::compare(&A, fp, (char const *)&cutoff);
  }

  template<typename dtype>
  void Tensor<dtype>::permute(dtype             beta,
                              CTF_int::tensor & A,
                              int * const *     perms_A,
                              dtype             alpha){
    int ret = CTF_int::tensor::permute(&A, perms_A, (char*)&alpha,
                                       NULL, (char*)&beta);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::permute(int * const * perms_B,
                              dtype         beta,
                              tensor &      A,
                              dtype         alpha){
    int ret = CTF_int::tensor::permute(&A, NULL, (char*)&alpha,
                                       perms_B, (char*)&beta);
    assert(ret == CTF_int::SUCCESS);
  }
  template<typename dtype>
  void Tensor<dtype>::add_to_subworld(
                                     Tensor<dtype> * tsr,
                                     dtype                   alpha,
                                     dtype                   beta){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      CTF_int::tensor::add_to_subworld(&t, (char*)&alpha, (char*)&beta);
      delete t.sr;
    } else
      CTF_int::tensor::add_to_subworld(tsr, (char*)&alpha, (char*)&beta);
  }
 
  template<typename dtype>
  void Tensor<dtype>::add_to_subworld(
                           Tensor<dtype> * tsr){
    return add_to_subworld(tsr, sr->mulid(), sr->mulid());
  }

  template<typename dtype>
  void Tensor<dtype>::add_from_subworld(
                                 Tensor<dtype> * tsr,
                                 dtype                   alpha,
                                 dtype                   beta){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      CTF_int::tensor::add_from_subworld(&t, (char*)&alpha, (char*)&beta);
      delete t.sr;
    } else
      CTF_int::tensor::add_from_subworld(tsr, (char*)&alpha, (char*)&beta);
  }

  template<typename dtype>
  void Tensor<dtype>::add_from_subworld(
                           Tensor<dtype> * tsr){
    if (tsr == NULL){
      tensor t = tensor();
      t.sr = sr->clone();
      CTF_int::tensor::add_from_subworld(&t, sr->mulid(), sr->mulid());
      delete t.sr;
    } else
      CTF_int::tensor::add_from_subworld(tsr, sr->mulid(), sr->mulid());
  }

  template<typename dtype>
  void Tensor<dtype>::slice(int const *    offsets,
                            int const *    ends,
                            dtype          beta,
                            tensor const & A,
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

  template<typename dtype>
  void Tensor<dtype>::slice(int64_t        corner_off,
                            int64_t        corner_end,
                            dtype          beta,
                            tensor const & A,
                            int64_t        corner_off_A,
                            int64_t        corner_end_A,
                            dtype          alpha){
    int * offsets, * ends, * offsets_A, * ends_A;
   
    CTF_int::cvrt_idx(this->order, this->len, corner_off, &offsets);
    CTF_int::cvrt_idx(this->order, this->len, corner_end, &ends);
    CTF_int::cvrt_idx(A.order, A.lens, corner_off_A, &offsets_A);
    CTF_int::cvrt_idx(A.order, A.lens, corner_end_A, &ends_A);
    
    slice(offsets, ends, beta, &A, offsets_A, ends_A, (char*)&alpha);

    CTF_int::cdealloc(offsets);
    CTF_int::cdealloc(ends);
    CTF_int::cdealloc(offsets_A);
    CTF_int::cdealloc(ends_A);
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
    Tensor<dtype> new_tsr(order, new_lens, new_sym, *owrld, *sr);
//   Tensor<dtype> new_tsr = tensor(sr, order, new_lens, new_sym, owrld, 1);
    std::fill(new_sym, new_sym+order, 0);
    new_tsr.slice(new_sym, new_lens, *(dtype*)sr->addid(), *this, offsets, ends, *(dtype*)sr->mulid());
/*    new_tsr.slice(
        new_sym, new_lens, sr->addid(), this,
        offsets, ends, sr->mulid());*/
    CTF_int::cdealloc(new_lens);
    CTF_int::cdealloc(new_sym);
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

    CTF_int::cdealloc(offsets);
    CTF_int::cdealloc(ends);

    return tsr;
  }

  template<typename dtype>
  void Tensor<dtype>::align(const CTF_int::tensor & A){
    if (A.wrld->cdt.cm != wrld->cdt.cm) {
      printf("ERROR: cannot align tensors on different CTF instances\n");
      assert(0);
    }
    int ret = CTF_int::tensor::align(&A);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  dtype Tensor<dtype>::reduce(OP op){
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
          Monoid<dtype, 1> mmax = Monoid<dtype, 1>(minval, CTF_int::default_max<dtype, 1>, MPI_MAX);
          ret = reduce_sum((char*)&ans, &mmax);
        }
        break;
      case OP_MIN:
        {
          dtype maxval;
          sr->max((char*)&maxval);
          Monoid<dtype, 1> mmin = Monoid<dtype, 1>(maxval, CTF_int::default_min<dtype, 1>, MPI_MIN);
          ret = reduce_sum((char*)&ans, &mmin);
        }
        break;
      case OP_MAXABS:
        {
          dtype minval;
          sr->min((char*)&minval);
          Monoid<dtype, 1> mmax = Monoid<dtype, 1>(minval, CTF_int::default_max<dtype, 1>, MPI_MAX);
          ret = reduce_sumabs((char*)&ans, &mmax);
        }
        break;
      case OP_MINABS:
        {
          dtype maxval;
          sr->max((char*)&maxval);
          Monoid<dtype, 1> mmin = Monoid<dtype, 1>(maxval, CTF_int::default_min<dtype, 1>, MPI_MIN);
          ret = reduce_sumabs((char*)&ans, &mmin);
        }
        break;
    }
    assert(ret == CTF_int::SUCCESS);
    return ans;
  }

  template<typename dtype>
  void Tensor<dtype>::get_max_abs(int     n,
                                  dtype * data) const {
    int ret;
    ret = CTF_int::tensor::get_max_abs(n, data);
    assert(ret == CTF_int::SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::fill_random(dtype rmin, dtype rmax){
    if (wrld->rank == 0) 
      printf("CTF ERROR: fill_random(rmin, rmax) not available for the type of tensor %s\n",name);
    assert(0);
  }

  template<>
  inline void Tensor<double>::fill_random(double rmin, double rmax){
    assert(!is_sparse);
    for (int64_t i=0; i<size; i++){
      ((double*)data)[i] = drand48()*(rmax-rmin)+rmin;
    }
    zero_out_padding();
  }


  template<typename dtype>
  void Tensor<dtype>::contract(dtype            alpha,
                               CTF_int::tensor& A,
                               const char *     idx_A,
                               CTF_int::tensor& B,
                               const char *     idx_B,
                               dtype            beta,
                               const char *     idx_C){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    assert(B.wrld->cdt.cm == wrld->cdt.cm);
    CTF_int::contraction ctr 
      = CTF_int::contraction(&A, idx_A, &B, idx_B, (char*)&alpha, this, idx_C, (char*)&beta);
    ctr.execute();
  }

  template<typename dtype>
  void Tensor<dtype>::contract(dtype                 alpha,
                               CTF_int::tensor&      A,
                               const char *          idx_A,
                               CTF_int::tensor&      B,
                               const char *          idx_B,
                               dtype                 beta,
                               const char *          idx_C,
                               Bivar_Function<dtype> fseq){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    assert(B.wrld->cdt.cm == wrld->cdt.cm);
    CTF_int::contraction ctr 
      = CTF_int::contraction(&A, idx_A, &B, idx_B, (char const *)&alpha, this, idx_C, (char const *)&beta, &fseq);
    ctr.execute();
  }


  template<typename dtype>
  void Tensor<dtype>::sum(dtype            alpha,
                          CTF_int::tensor& A,
                          const char *     idx_A,
                          dtype            beta,
                          const char *     idx_B){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);

    CTF_int::summation sum 
      = CTF_int::summation(&A, idx_A, (char*)&alpha, this, idx_B, (char*)&beta);

    sum.execute();

  }

  template<typename dtype>
  void Tensor<dtype>::sum(dtype                  alpha,
                          CTF_int::tensor&       A,
                          const char *           idx_A,
                          dtype                  beta,
                          const char *           idx_B,
                          Univar_Function<dtype> fseq){
    assert(A.wrld->cdt.cm == wrld->cdt.cm);
    
    CTF_int::summation sum = CTF_int::summation(&A, idx_A, (char const *)&alpha, this, idx_B, (char const *)&beta, fseq);

    sum.execute();
  }

  template<typename dtype>
  void Tensor<dtype>::scale(dtype        alpha,
                            const char * idx_A){
    CTF_int::scaling scl = CTF_int::scaling(this, idx_A, (char*)&alpha);
    scl.execute();
  }


  template<typename dtype>
  void Tensor<dtype>::scale(dtype               alpha,
                            const char *        idx_A,
                            Endomorphism<dtype> fseq){
    CTF_int::scaling scl = CTF_int::scaling(this, idx_A, &fseq, (char const *)&alpha);
    scl.execute();
  }

  template<typename dtype>
  dtype * Tensor<dtype>::read(char const *          idx,
                              Idx_Partition const & prl,
                              Idx_Partition const & blk,
                              bool                  unpack){
    return (dtype*)CTF_int::tensor::read(idx, prl, blk, unpack);
  }



  template<typename dtype>
  Tensor<dtype>& Tensor<dtype>::operator=(dtype val){
    set((char const*)&val);
/*    int64_t size;
    dtype* raw = get_raw_data(&size);
    //FIXME: Uuuuh, padding?
    ASSERT(0);
    std::fill(raw, raw+size, val);*/
    return *this;
  }
 
  template<typename dtype>
  double Tensor<dtype>::estimate_time(
                                    CTF_int::tensor& A,
                                    const char *     idx_A,
                                    CTF_int::tensor& B,
                                    const char *     idx_B,
                                    const char *     idx_C){
    CTF_int::contraction ctr
      = CTF_int::contraction(&A, idx_A, &B, idx_B, sr->mulid(), this, idx_C, sr->addid());
    return ctr.estimate_time();
  }
    
  template<typename dtype>
  double Tensor<dtype>::estimate_time(
                                    CTF_int::tensor& A,
                                    const char *     idx_A,
                                    const char *     idx_B){
    CTF_int::summation sum = CTF_int::summation(&A, idx_A, sr->mulid(), this, idx_B, sr->addid());

    return sum.estimate_time();
    
  }

  template<typename dtype>
  Tensor<dtype>& Tensor<dtype>::operator=(Tensor<dtype> A){

    free_self();
    init(A.sr, A.order, A.lens, A.sym, A.wrld, 0, A.name, A.profile, A.is_sparse);
    copy_tensor_data(&A);
    return *this;
/*
    sr = A.sr;
    world = A.wrld;
    name = A.name;

    if (sym != NULL)
      CTF_int::cdealloc(sym);
    if (len != NULL)
      CTF_int::cdealloc(len);
      //CTF_int::cdealloc(len);
    ret = CTF_int::tensor::info(&A, &order, &len, &sym);
    assert(ret == CTF_int::SUCCESS);

    ret = CTF_int::tensor::define(sr, order, len, sym, &tid, 1, name, name != NULL);
    assert(ret == CTF_int::SUCCESS);

    //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

    ret = CTF_int::tensor::copy(A.tid, tid);
    assert(ret == CTF_int::SUCCESS);*/
  }


  template<typename dtype>
  Sparse_Tensor<dtype> Tensor<dtype>::operator[](std::vector<int64_t> indices){
    Sparse_Tensor<dtype> stsr(indices,this);
    return stsr;
  }

}

