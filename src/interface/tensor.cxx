/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../interface/common.h"
#include "world.h"
#include "idx_tensor.h"
#include "../tensor/untyped_tensor.h"
#include "graph_io_aux.h"


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
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }


  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        bool                      is_sparse,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, sym, &world, 1, name, profile, is_sparse) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        bool                      is_sparse,
                        int const *               len,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, NULL, &world, 1, name, profile, is_sparse) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }


  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        int const *               len,
                        World &                   world,
                        CTF_int::algstrct const & sr,
                        char const *              name,
                        bool                      profile)
    : CTF_int::tensor(&sr, order, len, NULL, &world, 1, name, profile) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }


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
    : CTF_int::tensor(&sr_, order, 0, len, sym, &world, idx, prl, blk, name, profile) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                       order,
                        bool                      is_sparse_,
                        int const *               len,
                        int const *               sym,
                        World &                   world,
                        char const *              idx,
                        Idx_Partition const &     prl,
                        Idx_Partition const &     blk,
                        char const *              name,
                        bool                      profile,
                        CTF_int::algstrct const & sr_)
    : CTF_int::tensor(&sr_, order, is_sparse_, len, sym, &world, idx, prl, blk, name, profile) {
    IASSERT(sizeof(dtype)==this->sr->el_size);
  }


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
  Typ_Idx_Tensor<dtype> Tensor<dtype>::operator[](const char * idx_map_){
    //IASSERT(strlen(idx_map_)==order);
    Typ_Idx_Tensor<dtype> idxtsr(this, idx_map_);
    return idxtsr;
  }

  template<typename dtype>
  Typ_Idx_Tensor<dtype> Tensor<dtype>::i(const char * idx_map_){
    //IASSERT(strlen(idx_map_)==order);
    Typ_Idx_Tensor<dtype> idxtsr(this, idx_map_);
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
  void Tensor<dtype>::get_local_data(int64_t *  npair,
                                     int64_t ** global_idx,
                                     dtype **   data,
                                     bool       nonzeros_only,
                                     bool       unpack_sym) const {
    char * cpairs;
    int ret, i;
    if (nonzeros_only)
      ret = CTF_int::tensor::read_local_nnz(npair,&cpairs,unpack_sym);
    else
      ret = CTF_int::tensor::read_local(npair,&cpairs,unpack_sym);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_local\n"); IASSERT(0); return; }
    *global_idx = (int64_t*)CTF_int::alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)sr->alloc((*npair));
    CTF_int::PairIterator pairs(sr, cpairs);
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k();
      pairs[i].read_val((char*)((*data)+i));
    }
    if (cpairs != NULL) sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *  npair,
                                 int64_t ** global_idx,
                                 dtype **   data,
                                 bool       unpack_sym) const {
    char * cpairs;
    int ret, i;
    ret = CTF_int::tensor::read_local(npair,&cpairs,unpack_sym);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_local\n"); IASSERT(0); return; }
    *global_idx = (int64_t*)CTF_int::alloc((*npair)*sizeof(int64_t));
    *data = (dtype*)CTF_int::alloc((*npair)*sizeof(dtype));
    CTF_int::PairIterator pairs(sr, cpairs);
    for (i=0; i<(*npair); i++){
      (*global_idx)[i] = pairs[i].k();
      pairs[i].read_val((char*)((*data)+i));
    }
    if (cpairs != NULL) sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::get_local_pairs(int64_t *      npair,
                                      Pair<dtype> ** pairs,
                                      bool           nonzeros_only,
                                      bool           unpack_sym) const {
    char * cpairs;
    int ret;
    if (nonzeros_only)
      ret = CTF_int::tensor::read_local_nnz(npair,&cpairs,unpack_sym);
    else
      ret = CTF_int::tensor::read_local(npair,&cpairs,unpack_sym);
    *pairs = (Pair<dtype>*)cpairs; //Pair<dtype>::cast_char_arr(cpairs, *npair, sr);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_local\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *      npair,
                                 Pair<dtype> ** pairs,
                                 bool           unpack_sym) const {
    char * cpairs;
    int ret = CTF_int::tensor::read_local(npair, &cpairs, unpack_sym);
    *pairs = (Pair<dtype>*)cpairs; //Pair<dtype>::cast_char_arr(cpairs, *npair, sr);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_local\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t         npair,
                           int64_t const * global_idx,
                           dtype *         data){
    int ret;
    int64_t i;
    char * cpairs = sr->pair_alloc(npair);
    Pair< dtype > * pairs =(Pair< dtype >*)cpairs;
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::read(npair, cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read\n"); IASSERT(0); return; }
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t       npair,
                           Pair<dtype> * pairs){
    //FIXME raises mem consumption
    //char * cpairs = Pair<dtype>::scast_to_char_arr(pairs, npair);
    char * cpairs = (char*)pairs; //Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::read(npair, cpairs);
    IASSERT(cpairs == (char*)pairs);
    /*if (cpairs != (char*)pairs){
      for (int64_t i=0; i<npair; i++){
        pairs[i].k = ipairs[i].k();
        ipairs[i].read_val((char*)&(pairs[i].d));
      }
      sr->pair_dealloc(cpairs);
    }*/
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t         npair,
                            int64_t const * global_idx,
                            dtype const *   data) {
    int ret, i;
    char * cpairs = sr->pair_alloc(npair);
    Pair< dtype > * pairs =(Pair< dtype >*)cpairs;
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    /*char * cpairs = sr->pair_alloc(npair);
    CTF_int::PairIterator pairs = CTF_int::PairIterator(sr, cpairs);
    for (i=0; i<npair; i++){
      pairs[i].write_key(global_idx[i]);
      pairs[i].write_val((char*)&(data[i]));
    }*/
    ret = CTF_int::tensor::write(npair, sr->mulid(), sr->addid(), cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function write\n"); IASSERT(0); return; }
    sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            Pair<dtype> const * pairs) {

    //FIXME raises mem consumption
    char const * cpairs = (char const*)pairs; //Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::write(npair, sr->mulid(), sr->addid(), (char*)cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function write\n"); IASSERT(0); return; }
    /*if (cpairs != (char*)pairs)
      sr->pair_dealloc(cpairs);*/
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t         npair,
                            dtype           alpha,
                            dtype           beta,
                            int64_t const * global_idx,
                            dtype const *   data) {
    int ret, i;
    char * cpairs = sr->pair_alloc(npair);
    Pair< dtype > * pairs =(Pair< dtype >*)cpairs;

    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }

    /*Pair< dtype > * pairs;
    pairs = (Pair< dtype >*)CTF_int::alloc(npair*sizeof(Pair< dtype >));
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }*/
    ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function write\n"); IASSERT(0); return; }
    sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            dtype               alpha,
                            dtype               beta,
                            Pair<dtype> const * pairs) {
    char const * cpairs = (char const*)pairs; //Pair<dtype>::scast_to_char_arr(pairs, npair);

    int ret = CTF_int::tensor::write(npair, (char*)&alpha, (char*)&beta, (char*)cpairs);
    //if (cpairs != (char*)pairs) sr->pair_dealloc(cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function write\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t         npair,
                           dtype           alpha,
                           dtype           beta,
                           int64_t const * global_idx,
                           dtype *         data){
    int ret, i;
    char * cpairs = sr->pair_alloc(npair);
    Pair< dtype > * pairs =(Pair< dtype >*)cpairs;
    for (i=0; i<npair; i++){
      pairs[i].k = global_idx[i];
      pairs[i].d = data[i];
    }
    ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, cpairs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read\n"); IASSERT(0); return; }
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    sr->pair_dealloc(cpairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t       npair,
                           dtype         alpha,
                           dtype         beta,
                           Pair<dtype> * pairs){
    char * cpairs = (char*)pairs; //Pair<dtype>::scast_to_char_arr(pairs, npair);
    int ret = CTF_int::tensor::read(npair, (char*)&alpha, (char*)&beta, cpairs);
    IASSERT(cpairs == (char*)pairs);/*
    {
      CTF_int::PairIterator ipairs = CTF_int::PairIterator(sr, cpairs);
      for (int64_t i=0; i<npair; i++){
        pairs[i].k = ipairs[i].k();
        ipairs[i].read_val((char*)&(pairs[i].d()));
      }
      sr->pair_dealloc(cpairs);
    }*/
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read\n"); IASSERT(0); return; }
  }


  template<typename dtype>
  void Tensor<dtype>::read_all(int64_t * npair, dtype ** vals, bool unpack){
    int ret;
    ret = CTF_int::tensor::allread(npair, ((char**)vals), unpack);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_all\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  int64_t Tensor<dtype>::read_all(dtype * vals, bool unpack){
    int ret;
    int64_t npair;
    ret = CTF_int::tensor::allread(&npair, (char*)vals, unpack);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function read_all\n"); IASSERT(0); }
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
  void Tensor<dtype>::prnt() const{
    CTF_int::tensor::print(stdout, NULL);
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
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::permute(int * const *     perms_B,
                              dtype             beta,
                              CTF_int::tensor & A,
                              dtype             alpha){
    int ret = CTF_int::tensor::permute(&A, NULL, (char*)&alpha,
                                       perms_B, (char*)&beta);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function permute\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::sparsify(){
    int ret = CTF_int::tensor::sparsify();
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function sparsify\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::sparsify(dtype threshold, bool take_abs){
    int ret = CTF_int::tensor::sparsify((char*)&threshold, take_abs);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function sparsify\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::sparsify(std::function<bool(dtype)> filter){
    int ret = CTF_int::tensor::sparsify([&](char const * c){ return filter(((dtype*)c)[0]); });
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function sparisfy\n"); IASSERT(0); return; }
  }

  template <typename dtype>
  void read_sparse_from_file_base(const char * fpath, Tensor<dtype> * T){
    int64_t *inds = NULL;
    dtype *vals = NULL;
    uint64_t my_nvals = 0;

    char **leno;
    my_nvals = CTF_int::read_data_mpiio<dtype>(T->wrld->rank, T->wrld->np, fpath, &leno);
    CTF_int::process_tensor<dtype>(leno, T->order, T->lens, my_nvals, &inds, &vals);
    free(leno[0]);
    free(leno);

    T->write(my_nvals,inds,vals);
    free(inds);
    free(vals);
  }

  template<>
  inline void Tensor<int>::read_sparse_from_file(const char * fpath){
    read_sparse_from_file_base<int>(fpath, this);
  }

  template<>
  inline void Tensor<double>::read_sparse_from_file(const char * fpath){
    read_sparse_from_file_base<double>(fpath, this);
  }

  template<>
  inline void Tensor<float>::read_sparse_from_file(const char * fpath){
    read_sparse_from_file_base<float>(fpath, this);
  }

  template<>
  inline void Tensor<int64_t>::read_sparse_from_file(const char * fpath){
    read_sparse_from_file_base<int64_t>(fpath, this);
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
  void Tensor<dtype>::slice(int const *             offsets,
                            int const *             ends,
                            dtype                   beta,
                            CTF_int::tensor const & A,
                            int const *             offsets_A,
                            int const *             ends_A,
                            dtype                   alpha){
    int np_A, np_B;
    if (A.wrld->comm != wrld->comm){
      MPI_Comm_size(A.wrld->comm, &np_A);
      MPI_Comm_size(wrld->comm,   &np_B);
      if (np_A == np_B){
        printf("CTF ERROR: number of processors should not match in slice if worlds are different\n");
        IASSERT(0);
        return;
      }
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
  void Tensor<dtype>::slice(int64_t                 corner_off,
                            int64_t                 corner_end,
                            dtype                   beta,
                            CTF_int::tensor const & A,
                            int64_t                 corner_off_A,
                            int64_t                 corner_end_A,
                            dtype                   alpha){
    int * offsets, * ends, * offsets_A, * ends_A;

    CTF_int::cvrt_idx(this->order, this->lens, corner_off, &offsets);
    CTF_int::cvrt_idx(this->order, this->lens, corner_end, &ends);
    for (int i=0; i<order; i++){
      ends[i]++;
    }
    CTF_int::cvrt_idx(A.order, A.lens, corner_off_A, &offsets_A);
    CTF_int::cvrt_idx(A.order, A.lens, corner_end_A, &ends_A);
    for (int i=0; i<A.order; i++){
      ends_A[i]++;
    }

    CTF_int::tensor::slice(offsets, ends, (char*)&beta, (Tensor *)&A, offsets_A, ends_A, (char*)&alpha);

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
      if (!(ends[i] - offsets[i] > 0 &&
                  offsets[i] >= 0 &&
                  ends[i] <= lens[i])){
        printf("CTF ERROR: invalid slice dimensions\n");
        IASSERT(0);
        return Tensor<dtype>();
      }
      if (sym[i] != NS){
        if (offsets[i] == offsets[i+1] && ends[i] == ends[i+1]){
          new_sym[i] = sym[i];
        } else {
          if (!(ends[i+1] >= offsets[i])){
            printf("CTF ERROR: slice dimensions don't respect tensor symmetry\n");
            IASSERT(0);
            return Tensor<dtype>();
          }
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

    CTF_int::cvrt_idx(this->order, this->lens, corner_off, &offsets);
    CTF_int::cvrt_idx(this->order, this->lens, corner_end, &ends);
    for (int i=0; i<order; i++){
      ends[i]++;
    }

    Tensor<dtype> tsr = slice(offsets, ends, owrld);

    CTF_int::cdealloc(offsets);
    CTF_int::cdealloc(ends);

    return tsr;
  }

  template<typename dtype>
  void Tensor<dtype>::align(const CTF_int::tensor & A){
    if (A.wrld->cdt.cm != wrld->cdt.cm) {
      printf("CTF ERROR: cannot align tensors on different CTF instances\n");
      IASSERT(0);
      return;
    }
    int ret = CTF_int::tensor::align(&A);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function align\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  dtype Tensor<dtype>::reduce(OP op){
    int ret;
    dtype ans;
    switch (op) {
      case OP_SUM:
        if (sr->is_ordered()){
          Semiring<dtype,1> r = Semiring<dtype,1>();
          ret = reduce_sum((char*)&ans, &r);
        } else {
          Semiring<dtype,0> r = Semiring<dtype,0>();
          ret = reduce_sum((char*)&ans, &r);
        }
//        ret = reduce_sum((char*)&ans);
        break;
      case OP_SUMABS:
        if (sr->is_ordered()){
          Ring<dtype,1> r = Ring<dtype,1>();
          ret = reduce_sumabs((char*)&ans, &r);
        } else {
          Ring<dtype,0> r = Ring<dtype,0>();
          ret = reduce_sumabs((char*)&ans, &r);
        }
        break;
      case OP_SUMSQ:
/*        if (sr->is_ordered()){
          Ring<dtype,1> r = Ring<dtype,1>();
          ret = reduce_sumsq((char*)&ans, &r);
        } else {
          Ring<dtype,0> r = Ring<dtype,0>();
          ret = reduce_sumsq((char*)&ans, &r);
        }*/
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
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function reduce\n"); IASSERT(0); }
    return ans;
  }


  template<typename dtype>
  void real_norm1(Tensor<dtype> & A, double & nrm){
    char inds[A.order];
    for (int i=0; i<A.order; i++){
      inds[i] = 'a'+i;
    }
    nrm = Function<dtype,double>([](dtype a){ return (double)std::abs(a); })(A[inds]);
  }

  template<>
  inline void real_norm1<bool>(Tensor<bool> & A, double & nrm){
    char inds[A.order];
    for (int i=0; i<A.order; i++){
      inds[i] = 'a'+i;
    }
    nrm = A[inds];
  }



  template<typename dtype>
  void Tensor<dtype>::norm1(double & nrm){
    if (wrld->rank == 0)
      printf("CTF ERROR: norm not available for the type of tensor %s\n",name);
    IASSERT(0);
  }

#define NORM1_INST(dtype) \
  template<> \
  inline void Tensor<dtype>::norm1(double & nrm){ \
    real_norm1<dtype>(*this, nrm); \
  }

NORM1_INST(bool)
NORM1_INST(int8_t)
NORM1_INST(int16_t)
NORM1_INST(int)
NORM1_INST(int64_t)
NORM1_INST(float)
NORM1_INST(double)

  template<typename dtype>
  static void real_norm2(Tensor<dtype> & A, double & nrm){
    char inds[A.order];
    for (int i=0; i<A.order; i++){
      inds[i] = 'a'+i;
    }
    //CTF::Scalar<double> dnrm(A.dw);
    nrm = std::sqrt((double)Function<dtype,double>([](dtype a){ return (double)(a*a); })(A[inds]));
  }

  template<typename dtype>
  static void complex_norm2(Tensor<dtype> & A, double & nrm){
    char inds[A.order];
    for (int i=0; i<A.order; i++){
      inds[i] = 'a'+i;
    }
    nrm = std::sqrt((double)Function<dtype,double>([](dtype a){ return (double)std::norm(a); })(A[inds]));
  }


  template<typename dtype>
  void Tensor<dtype>::norm2(double & nrm){
    if (wrld->rank == 0)
      printf("CTF ERROR: norm not available for the type of tensor %s\n",name);
    IASSERT(0);
  }

#define NORM2_REAL_INST(dtype) \
  template<> \
  inline void Tensor<dtype>::norm2(double & nrm){ \
    real_norm2<dtype>(*this, nrm); \
  }

#define NORM2_COMPLEX_INST(dtype) \
  template<> \
  inline void Tensor< std::complex<dtype> >::norm2(double & nrm){ \
    complex_norm2< std::complex<dtype> >(*this, nrm); \
  }


NORM2_REAL_INST(bool)
NORM2_REAL_INST(int8_t)
NORM2_REAL_INST(int16_t)
NORM2_REAL_INST(int)
NORM2_REAL_INST(int64_t)
NORM2_REAL_INST(float)
NORM2_REAL_INST(double)
NORM2_COMPLEX_INST(float)
NORM2_COMPLEX_INST(double)

  template<typename dtype>
  void Tensor<dtype>::norm_infty(double & nrm){
    if (wrld->rank == 0)
      printf("CTF ERROR: norm not available for the type of tensor %s\n",name);
    IASSERT(0);
  }

#define NORM_INFTY_INST(dtype) \
  template<> \
  inline void Tensor<dtype>::norm_infty(double & nrm){ \
    nrm = this->norm_infty(); \
  }

NORM_INFTY_INST(bool)
NORM_INFTY_INST(int8_t)
NORM_INFTY_INST(int16_t)
NORM_INFTY_INST(int)
NORM_INFTY_INST(int64_t)
NORM_INFTY_INST(float)
NORM_INFTY_INST(double)

#undef NORM1_INST
#undef NORM2_REAL_INST
#undef NORM2_COMPLEX_INST
#undef NORM_INFTY_INST

  template<typename dtype>
  void Tensor<dtype>::get_max_abs(int     n,
                                  dtype * data) const {
    int ret;
    ret = CTF_int::tensor::get_max_abs(n, data);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function get_max_abs\n"); IASSERT(0); return; }
  }

  template<typename dtype>
  void Tensor<dtype>::fill_random(dtype rmin, dtype rmax){
    if (wrld->rank == 0)
      printf("CTF ERROR: fill_random(rmin, rmax) not available for the type of tensor %s\n",name);
    IASSERT(0);
  }

  template <typename dtype>
  void fill_random_base(dtype rmin, dtype rmax, Tensor<dtype> & T){
    if (T.is_sparse){
      printf("CTF ERROR: fill_random should not be called on a sparse tensor, use fill_random_sp instead\n");
      IASSERT(0);
      return;
    }
    for (int64_t i=0; i<T.size; i++){
      ((dtype*)T.data)[i] = CTF_int::get_rand48()*(rmax-rmin)+rmin;
    }
    T.zero_out_padding();
  }

  template<>
  inline void Tensor<double>::fill_random(double rmin, double rmax){
    fill_random_base<double>(rmin, rmax, *this);
  }

  template<>
  inline void Tensor<float>::fill_random(float rmin, float rmax){
    fill_random_base<float>(rmin, rmax, *this);
  }

  template<>
  inline void Tensor<int64_t>::fill_random(int64_t rmin, int64_t rmax){
    fill_random_base<int64_t>(rmin, rmax, *this);
  }

  template<>
  inline void Tensor<int>::fill_random(int rmin, int rmax){
    fill_random_base<int>(rmin, rmax, *this);
  }


  template<typename dtype>
  void Tensor<dtype>::fill_sp_random(dtype rmin, dtype rmax, double frac_sp){
    if (wrld->rank == 0)
      printf("CTF ERROR: fill_sp_random(rmin, rmax, frac_sp) not available for the type of tensor %s\n",name);
    IASSERT(0);
  }

  template <typename dtype>
  void fill_sp_random_base(dtype rmin, dtype rmax, double frac_sp, Tensor<dtype> * T){
    int64_t tot_size = 1; //CTF_int::packed_size(T.order, T.lens, T.sym);
    for (int i=0; i<T->order; i++) tot_size *= T->lens[i];
    double sf = tot_size*frac_sp;
    double dg = 0.0;
    //generate approximately tot_size*e^frac_sp rather than tot_size*frac_sp elements, to account for conflicts in writing them
    for (int i=2; i<20; i++){
      dg += sf;
      sf *= frac_sp/i;
    }
    int64_t gen_size = (int64_t)(dg+.5);
    int64_t my_gen_size = gen_size/T->wrld->np;
    if (gen_size % T->wrld->np > T->wrld->rank){
      my_gen_size++;
    }
    Pair<dtype> * pairs = (Pair<dtype>*)T->sr->pair_alloc(my_gen_size);
    for (int64_t i=0; i<my_gen_size; i++){
      pairs[i] = Pair<dtype>((int64_t)(CTF_int::get_rand48()*tot_size), 1.0);
    }
    T->write(my_gen_size,pairs);
    T->sr->pair_dealloc((char*)pairs);
    char str[T->order];
    for (int i=0; i<T->order; i++){
      str[i] = 'a'+i;
    }

    Transform<dtype>([=](dtype & d){ d=CTF_int::get_rand48()*(rmax-rmin)+rmin; })(T->operator[](str));

    /*std::vector<Pair<dtype>> pairs;


    pairs.reserve(size*frac_sp);
    int64_t npairs=0;
    for (int64_t i=wrld->rank; i<tot_sz; i+=wrld->np){
      if (CTF_int::get_rand48() < frac_sp){
        pairs.push_back(Pair<dtype>(i,CTF_int::get_rand48()*(rmax-rmin)+rmin));
        npairs++;
      }
    }
    this->write(npairs, pairs.data());*/

  }

  template<>
  inline void Tensor<double>::fill_sp_random(double rmin, double rmax, double frac_sp){
    fill_sp_random_base<double>(rmin, rmax, frac_sp, this);
  }

  template<>
  inline void Tensor<float>::fill_sp_random(float rmin, float rmax, double frac_sp){
    fill_sp_random_base<float>(rmin, rmax, frac_sp, this);
  }

  template<>
  inline void Tensor<int>::fill_sp_random(int rmin, int rmax, double frac_sp){
    fill_sp_random_base<int>(rmin, rmax, frac_sp, this);
  }

  template<>
  inline void Tensor<int64_t>::fill_sp_random(int64_t rmin, int64_t rmax, double frac_sp){
    fill_sp_random_base<int64_t>(rmin, rmax, frac_sp, this);
  }

  template<typename dtype>
  void Tensor<dtype>::contract(dtype            alpha,
                               CTF_int::tensor& A,
                               const char *     idx_A,
                               CTF_int::tensor& B,
                               const char *     idx_B,
                               dtype            beta,
                               const char *     idx_C){
    if (A.wrld->cdt.cm != wrld->cdt.cm || B.wrld->cdt.cm != wrld->cdt.cm){
      printf("CTF ERROR: worlds of contracted tensors must match\n");
      IASSERT(0);
      return;
    }
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
    if (A.wrld->cdt.cm != wrld->cdt.cm || B.wrld->cdt.cm != wrld->cdt.cm){
      printf("CTF ERROR: worlds of contracted tensors must match\n");
      IASSERT(0);
      return;
    }
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
    if (A.wrld->cdt.cm != wrld->cdt.cm){
      printf("CTF ERROR: worlds of summed tensors must match\n");
      IASSERT(0);
      return;
    }

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
    if (A.wrld->cdt.cm != wrld->cdt.cm){
      printf("CTF ERROR: worlds of summed tensors must match\n");
      IASSERT(0);
      return;
    }

    CTF_int::summation sum = CTF_int::summation(&A, idx_A, (char const *)&alpha, this, idx_B, (char const *)&beta, &fseq);

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
  dtype * Tensor<dtype>::get_mapped_data(char const *          idx,
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
    IASSERT(0);
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
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function\n"); IASSERT(0); return; }

    ret = CTF_int::tensor::define(sr, order, len, sym, &tid, 1, name, name != NULL);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function\n"); IASSERT(0); return; }

    //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

    ret = CTF_int::tensor::copy(A.tid, tid);
    if (ret != CTF_int::SUCCESS){ printf("CTF ERROR: failed to execute function\n"); IASSERT(0); return; }*/
  }


  template<typename dtype>
  Sparse_Tensor<dtype> Tensor<dtype>::operator[](std::vector<int64_t> indices){
    Sparse_Tensor<dtype> stsr(indices,this);
    return stsr;
  }

}


