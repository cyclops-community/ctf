/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../interface/common.h"
#include "../world/int_world.h"

namespace CTF {

  template<typename dtype>
  Tensor<dtype>::Tensor(){
    tid = -1;
    order = -1;
    sym = NULL;
    len = NULL;
    name = NULL;
    world = NULL;
    sr = Semiring<dtype>();
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(const Tensor<dtype>& A,
                        bool                 copy){
    int ret;
    world = A.world;
    name = A.name;
    sr = A.sr;

    ret = world->ctf->info_tensor(A.tid, &order, &len, &sym);
    assert(ret == SUCCESS);

    ret = world->ctf->define_tensor(sr, order, len, sym, &tid, 1, name, name!=NULL);
    assert(ret == SUCCESS);

    //printf("Defined tensor %d to be the same as %d, copy=%d\n", tid, A.tid, (int)copy);

    if (copy){
      ret = world->ctf->copy_tensor(A.tid, tid);
      assert(ret == SUCCESS);
    }
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(const Tensor<dtype> & A,
                        World &               world_){
    int ret;
    world = &world_;
    name = A.name;
    sr = A.sr;

    ret = A.world->ctf->info_tensor(A.tid, &order, &len, &sym);
    assert(ret == SUCCESS);

    ret = world->ctf->define_tensor(sr, order, len, sym, &tid, 1, name, name!=NULL);
    assert(ret == SUCCESS);
  }


  template<typename dtype>
  Tensor<dtype>::Tensor(int                 order_,
                        int const *         len_,
                        int const *         sym_,
                        World &             world_,
                        char const *        name_,
                        int const           profile_){
    int ret;
    world = &world_;
    name = name_;
    sr = Semiring<dtype>();
    ret = world->ctf->define_tensor(sr, order_, len_, sym_, &tid, 1, name_, profile_);
    assert(ret == SUCCESS);
    ret = world->ctf->info_tensor(tid, &order, &len, &sym);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  Tensor<dtype>::Tensor(int                 order_,
                        int const *         len_,
                        int const *         sym_,
                        World &             world_,
                        Semiring<dtype>     sr_,
                        char const *        name_,
                        int const           profile_){
    int ret;
    world = &world_;
    name = name_;
    sr = sr_;
    ret = world->ctf->define_tensor(sr, order_, len_, sym_, &tid, 1, name_, profile_);
    assert(ret == SUCCESS);
    ret = world->ctf->info_tensor(tid, &order, &len, &sym);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  Tensor<dtype>::~Tensor(){
  /*  if (sym != NULL)
      CTF_free_cond(sym);
    if (len != NULL)
      CTF_free_cond(len);*/
    if (sym != NULL)
      CTF_free(sym);
    if (len != NULL)
      CTF_free(len);
    world->ctf->clean_tensor(tid);
  }

  template<typename dtype>
  dtype * Tensor<dtype>::get_raw_data(int64_t * size) {
    int ret;
    dtype * data;
    ret = world->ctf->get_raw_data(tid, &data, size);
    assert(ret == SUCCESS);
    
    return data;
  }

  template<typename dtype>
  void Tensor<dtype>::read_local(int64_t *   npair, 
                                 int64_t **  global_idx, 
                                 dtype **   data) const {
    Pair< dtype > * pairs;
    int ret, i;
    ret = world->ctf->read_local_tensor(tid, npair, &pairs);
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
    int ret = world->ctf->read_local_tensor(tid, npair, pairs);
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
    ret = world->ctf->read_tensor(tid, npair, pairs);
    assert(ret == SUCCESS);
    for (i=0; i<npair; i++){
      data[i] = pairs[i].d;
    }
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::read(int64_t          npair,
                           Pair<dtype> *    pairs) const {
    int ret = world->ctf->read_tensor(tid, npair, pairs);
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
    ret = world->ctf->write_tensor(tid, npair, pairs);
    assert(ret == SUCCESS);
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            Pair<dtype> const * pairs) {
    int ret = world->ctf->write_tensor(tid, npair, pairs);
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
    ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
    assert(ret == SUCCESS);
    CTF_free(pairs);
  }

  template<typename dtype>
  void Tensor<dtype>::write(int64_t             npair,
                            dtype               alpha,
                            dtype               beta,
                            Pair<dtype> const * pairs) {
    int ret = world->ctf->write_tensor(tid, npair, alpha, beta, pairs);
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
    ret = world->ctf->read_tensor(tid, npair, alpha, beta, pairs);
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
    int ret = world->ctf->read_tensor(tid, npair, alpha, beta, pairs);
    assert(ret == SUCCESS);
  }


  template<typename dtype>
  void Tensor<dtype>::read_all(int64_t * npair, dtype ** vals) const {
    int ret;
    ret = world->ctf->allread_tensor(tid, npair, vals);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::read_all(dtype * vals) const {
    int ret;
    int64_t npair;
    ret = world->ctf->allread_tensor(tid, &npair, vals);
    assert(ret == SUCCESS);
    return npair;
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
    return world->ctf->estimate_cost(A.tid, idx_map_A, B.tid, idx_map_B, tid, idx_map_C);
  }

  template<typename dtype>
  int64_t Tensor<dtype>::estimate_cost(
                                    const Tensor<dtype>& A,
                                    const char *         idx_A,
                                    const char *         idx_B){
    int * idx_map_A, * idx_map_B;
    conv_idx(A.order, idx_A, &idx_map_A,
             order, idx_B, &idx_map_B);
    return world->ctf->estimate_cost(A.tid, idx_map_A, tid, idx_map_B);

    
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
    assert(A.world->ctf == world->ctf);
    assert(B.world->ctf == world->ctf);
    ret = world->ctf->contract(&tp, fseq, alpha, beta);
  /*  else {
      fseq_elm_ctr<dtype> fs;
      fs.func_ptr = fseq.func_ptr;
      ret = world->ctf->contract(&tp, fs, alpha, beta);
    }*/
    CTF_free(tp.idx_map_A);
    CTF_free(tp.idx_map_B);
    CTF_free(tp.idx_map_C);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::set_name(char const * name_) {
    name = name_;
    world->ctf->set_name(tid, name_);
  }

  template<typename dtype>
  void Tensor<dtype>::profile_on() {
    world->ctf->profile_on(tid);
  }

  template<typename dtype>
  void Tensor<dtype>::profile_off() {
    world->ctf->profile_off(tid);
  }

  template<typename dtype>
  void Tensor<dtype>::print(FILE* fp, double cutoff) const{
    world->ctf->print_tensor(fp, tid, cutoff);
  }

  template<typename dtype>
  void Tensor<dtype>::compare(const Tensor<dtype>& A, FILE* fp, double cutoff) const{
    world->ctf->compare_tensor(fp, tid, A.tid, cutoff);
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
    assert(A.world->ctf == world->ctf);
      
    st.idx_map_A = idx_map_A;
    st.idx_map_B = idx_map_B;
    st.tid_A = A.tid;
    st.tid_B = tid;
    ret = world->ctf->sum_tensors(alpha, beta, A.tid, tid, idx_map_A, idx_map_B, fseq);
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
    ret = world->ctf->scale_tensor(alpha, tid, idx_map_A, fseq);
    CTF_free(idx_map_A);
    assert(ret == SUCCESS);
  }
  template<typename dtype>


  void Tensor<dtype>::permute(dtype             beta,
                              Tensor &          A,
                              int * const *     perms_A,
                              dtype             alpha){
    int ret = world->ctf->permute_tensor(A.tid, perms_A, alpha, A.world->ctf, 
                                         tid, NULL, beta, world->ctf);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  void Tensor<dtype>::permute(int * const * perms_B,
                              dtype         beta,
                              Tensor &      A,
                              dtype         alpha){
    int ret = world->ctf->permute_tensor(A.tid, NULL, alpha, A.world->ctf, 
                                         tid, perms_B, beta, world->ctf);
    assert(ret == SUCCESS);
  }
  template<typename dtype>
  void Tensor<dtype>::add_to_subworld(
                           Tensor<dtype> * tsr,
                           dtype alpha,
                           dtype beta) const {
    int ret;
    if (tsr == NULL)
      ret = world->ctf->add_to_subworld(tid, -1, NULL, alpha, beta);
    else
      ret = world->ctf->add_to_subworld(tid, tsr->tid, tsr->world->ctf, alpha, beta);
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
      ret = world->ctf->add_from_subworld(tid, -1, NULL, alpha, beta);
    else
      ret = world->ctf->add_from_subworld(tid, tsr->tid, tsr->world->ctf, alpha, beta);
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

    return slice(offsets, ends, world);
  }

  template<typename dtype>
  Tensor<dtype> Tensor<dtype>::slice(int64_t corner_off,
                                     int64_t corner_end) const {

    return slice(corner_off, corner_end, world);
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
    if (A.world->ctf != world->ctf) {
      printf("ERROR: cannot align tensors on different CTF instances\n");
      assert(0);
    }
    int ret = world->ctf->align(tid, A.tid);
    assert(ret == SUCCESS);
  }

  template<typename dtype>
  dtype Tensor<dtype>::reduce(OP op){
    int ret;
    dtype ans;
    ans = 0.0;
    ret = world->ctf->reduce_tensor(tid, op, &ans);
    assert(ret == SUCCESS);
    return ans;
  }
  template<typename dtype>
  void Tensor<dtype>::get_max_abs(int     n,
                                  dtype * data){
    int ret;
    ret = world->ctf->get_max_abs(tid, n, data);
    assert(ret == SUCCESS);
  }

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

    sr = A.sr;
    world = A.world;
    name = A.name;

    if (sym != NULL)
      CTF_free(sym);
    if (len != NULL)
      CTF_free(len);
      //CTF_free(len);
    ret = world->ctf->info_tensor(A.tid, &order, &len, &sym);
    assert(ret == SUCCESS);

    ret = world->ctf->define_tensor(sr, order, len, sym, &tid, 1, name, name != NULL);
    assert(ret == SUCCESS);

    //printf("Set tensor %d to be the same as %d\n", tid, A.tid);

    ret = world->ctf->copy_tensor(A.tid, tid);
    assert(ret == SUCCESS);
  }
      

  template<typename dtype>
  Idx_Tensor<dtype> Tensor<dtype>::operator[](const char * idx_map_){
    Idx_Tensor<dtype> itsr(this, idx_map_);
    return itsr;
  }


  template<typename dtype>
  Sparse_Tensor<dtype> Tensor<dtype>::operator[](std::vector<int64_t> indices){
    Sparse_Tensor<dtype> stsr(indices,this);
    return stsr;
  }

}

