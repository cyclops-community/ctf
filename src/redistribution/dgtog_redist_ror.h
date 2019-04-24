
//to be included with proper ifdefs and inside a namespace

using namespace CTF_int;

#ifdef PUT_NOTIFY
typedef foMPI_Request CTF_Request;
#define MPI_Waitany(...) foMPI_Waitany(__VA_ARGS__)
#else
typedef MPI_Request CTF_Request;
#endif



template <int idim>
void isendrecv(int * const *    pe_offset,
               int * const *    bucket_offset,
               int const *      rep_phase,
               int64_t const *  counts,
               int64_t const *  displs,
               CTF_Request *    reqs,
#ifdef PUT_NOTIFY
               foMPI_Win &      cm,
#else
               MPI_Comm         cm,
#endif
               char *           buffer,
               algstrct const * sr,
               int              bucket_off,
               int              pe_off,
               int              dir){
  for (int r=0; r<rep_phase[idim]; r++){
    int rec_bucket_off = bucket_off+bucket_offset[idim][r];
    int rec_pe_off = pe_off+pe_offset[idim][r];
    isendrecv<idim-1>(pe_offset, bucket_offset, rep_phase, counts, displs, reqs, cm, buffer, sr, rec_bucket_off, rec_pe_off, dir);
  }
}

template <>
void isendrecv<0>
              (int * const *    pe_offset,
               int * const *    bucket_offset,
               int const *      rep_phase,
               int64_t const *  counts,
               int64_t const *  displs,
               CTF_Request *    reqs,
#ifdef PUT_NOTIFY
               foMPI_Win &      cm,
#else
               MPI_Comm         cm,
#endif
               char *           buffer,
               algstrct const * sr,
               int              bucket_off,
               int              pe_off,
               int              dir){
  for (int r=0; r<rep_phase[0]; r++){
    int bucket = bucket_off+r;
    int pe = pe_off+pe_offset[0][r];
#ifdef PUT_NOTIFY
    ASSERT(dir);
    foMPI_Notify_init(cm, pe, MTAG, 1, reqs+bucket);
    foMPI_Start(reqs+bucket);
#else
    if (dir)
      MPI_Irecv(buffer+displs[bucket]*sr->el_size, counts[bucket], sr->mdtype(), pe, MTAG, cm, reqs+bucket);
    else
      MPI_Isend(buffer+displs[bucket]*sr->el_size, counts[bucket], sr->mdtype(), pe, MTAG, cm, reqs+bucket);
#endif
  }
}

#ifdef ROR
template <int idim>
void redist_bucket_ror(int * const *        bucket_offset,
                       int64_t * const *    data_offset,
                       int * const *        ivmax_pre,
                       int const *          rep_phase,
                       int const *          rep_idx,
                       int                  virt_dim0,
                       bool                 data_to_buckets,
                       char * __restrict__  data,
                       char ** __restrict__ buckets,
                       int64_t *            counts,
                       algstrct const *     sr,
                       int64_t              data_off=0,
                       int                  bucket_off=0,
                       int                  prev_idx=0){
  int ivmax = ivmax_pre[idim][prev_idx];
  for (int iv=rep_idx[idim]; iv<=ivmax; iv+=rep_phase[idim]){
    int64_t rec_data_off = data_off + data_offset[idim][iv];
    redist_bucket_ror<idim-1>(bucket_offset, data_offset, ivmax_pre, rep_phase, rep_idx, virt_dim0, data_to_buckets, data, buckets, counts, sr, rec_data_off, bucket_off, iv);
  }
}

template <>
void redist_bucket_ror<0>
                      (int * const *        bucket_offset,
                       int64_t * const *    data_offset,
                       int * const *        ivmax_pre,
                       int const *          rep_phase,
                       int const *          rep_idx,
                       int                  virt_dim0,
                       bool                 data_to_buckets,
                       char * __restrict__  data,
                       char ** __restrict__ buckets,
                       int64_t *            counts,
                       algstrct const *     sr,
                       int64_t              data_off,
                       int                  bucket_off,
                       int                  prev_idx){
  if (rep_idx[0] == -1)
    redist_bucket<0>(bucket_offset, data_offset, ivmax_pre, rep_phase[0], virt_dim0, data_to_buckets, data, buckets, counts, sr, data_off, bucket_off, prev_idx);
  else
    CTF_int::redist_bucket_r0(bucket_offset, data_offset, ivmax_pre, rep_phase[0], rep_idx[0], virt_dim0, data_to_buckets, data, buckets, counts, sr, data_off, bucket_off, prev_idx);
}

#ifdef PUTREDIST
template <int idim>
void put_buckets(int const *                 rep_phase,
                 int * const *               pe_offset,
                 int * const *               bucket_offset,
                 char * const * __restrict__ buckets,
                 int64_t const *             counts,
                 algstrct const *            sr,
                 int64_t const *             put_displs,
                 CTF_Win &                   win,
                 int                         bucket_off,
                 int                         pe_off){
  for (int r=0; r<rep_phase[idim]; r++){
    int rec_bucket_off = bucket_off+bucket_offset[idim][r];
    int rec_pe_off = pe_off+pe_offset[idim][r];
    put_buckets<idim-1>(rep_phase, pe_offset, bucket_offset, buckets, counts, sr, put_displs, win, rec_bucket_off, rec_pe_off);
  }
}

template <>
void put_buckets<0>(
                 int const *                 rep_phase,
                 int * const *               pe_offset,
                 int * const *               bucket_offset,
                 char * const * __restrict__ buckets,
                 int64_t const *             counts,
                 algstrct const *            sr,
                 int64_t const *             put_displs,
                 CTF_Win &                   win,
                 int                         bucket_off,
                 int                         pe_off){
  for (int r=0; r<rep_phase[0]; r++){
    int rec_pe_off = pe_off + pe_offset[0][r];
    int rec_bucket_off = bucket_off + bucket_offset[0][r];
#ifdef PUT_NOTIFY
    foMPI_Put_notify(buckets[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), rec_pe_off, put_displs[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), win, MTAG);
#else
    MPI_Put(buckets[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), rec_pe_off, put_displs[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), win);
#endif
  }
}
#endif

template <int idim>
void redist_bucket_isr(int                  order,
                       int * const *        pe_offset,
                       int * const *        bucket_offset,
                       int64_t * const *    data_offset,
                       int * const *        ivmax_pre,
                       int const *          rep_phase,
                       int *                rep_idx,
                       int                  virt_dim0,
#ifdef IREDIST
                       CTF_Request *        rep_reqs,
                       MPI_Comm             cm,
#endif
#ifdef  PUTREDIST
                       int64_t const *      put_displs,
                       CTF_Win &            win,
#endif
                       bool                 data_to_buckets,
                       char * __restrict__  data,
                       char ** __restrict__ buckets,
                       int64_t *            counts,
                       algstrct const *     sr,
                       int                  bucket_off=0,
                       int                  pe_off=0){
#ifdef USE_OMP
  int tothi_rep_phase = 1;
  for (int id=1; id<=idim; id++){
    tothi_rep_phase *= rep_phase[id];
  }
  #pragma omp parallel for
  for (int t=0; t<tothi_rep_phase; t++){
    int rep_idx2[order];
    memcpy(rep_idx2, rep_idx, sizeof(int)*order);
    rep_idx[0] = -1;
    int rec_bucket_off = bucket_off;
    int rec_pe_off = pe_off;
    int tleft = t;
    for (int id=1; id<=idim; id++){
      int r = tleft%rep_phase[id];
      tleft = tleft / rep_phase[id];
      rep_idx2[id] = r;
      rec_bucket_off += bucket_offset[id][r];
      rec_pe_off += pe_offset[id][r];
    }
    redist_bucket_isr<0>(order, pe_offset, bucket_offset, data_offset, ivmax_pre, rep_phase, rep_idx2, virt_dim0,
#ifdef IREDIST
                         rep_reqs, cm,
#endif
#ifdef  PUTREDIST
                         put_displs, win,
#endif
                         data_to_buckets, data, buckets, counts, sr, rec_bucket_off, rec_pe_off);

  }
#else
  if (rep_phase[idim] == 1){
    int rec_bucket_off = bucket_off + bucket_offset[idim][0];
    int rec_pe_off = pe_off + pe_offset[idim][0];
    redist_bucket_isr<idim-1>(order, pe_offset, bucket_offset, data_offset, ivmax_pre, rep_phase, rep_idx, virt_dim0,
#ifdef IREDIST
                              rep_reqs, cm,
#endif
#ifdef  PUTREDIST
                              put_displs, win,
#endif
                              data_to_buckets, data, buckets, counts, sr, rec_bucket_off, rec_pe_off);
  } else {
    for (int r=0; r<rep_phase[idim]; r++){
      int rep_idx2[order];
      memcpy(rep_idx2, rep_idx, sizeof(int)*order);
      rep_idx[0] = -1;
      rep_idx2[idim] = r;
      int rec_bucket_off = bucket_off + bucket_offset[idim][r];
      int rec_pe_off = pe_off + pe_offset[idim][r];
      redist_bucket_isr<idim-1>(order, pe_offset, bucket_offset, data_offset, ivmax_pre, rep_phase, rep_idx2, virt_dim0,
#ifdef IREDIST
                                rep_reqs, cm,
#endif
#ifdef  PUTREDIST
                                put_displs, win,
#endif
                                data_to_buckets, data, buckets, counts, sr, rec_bucket_off, rec_pe_off);
    }
  }
#endif
}


template <>
void redist_bucket_isr<0>
                      (int                  order,
                       int * const *        pe_offset,
                       int * const *        bucket_offset,
                       int64_t * const *    data_offset,
                       int * const *        ivmax_pre,
                       int const *          rep_phase,
                       int *                rep_idx,
                       int                  virt_dim0,
#ifdef IREDIST
                       CTF_Request *        rep_reqs,
                       MPI_Comm             cm,
#endif
#ifdef  PUTREDIST
                       int64_t const *      put_displs,
                       CTF_Win &            win,
#endif
                       bool                 data_to_buckets,
                       char * __restrict__  data,
                       char ** __restrict__ buckets,
                       int64_t *            counts,
                       algstrct const *     sr,
                       int                  bucket_off,
                       int                  pe_off){
#ifndef WAITANY
#ifdef IREDIST
  if (!data_to_buckets){
    MPI_Waitall(rep_phase[0], rep_reqs+bucket_off, MPI_STATUSES_IGNORE);
  }
#endif
#endif
  SWITCH_ORD_CALL(redist_bucket_ror, order-1, bucket_offset, data_offset, ivmax_pre, rep_phase, rep_idx, virt_dim0, data_to_buckets, data, buckets, counts, sr, 0, bucket_off, 0)
  if (data_to_buckets){
#ifdef IREDIST
#ifndef PUT_NOTIFY
    for (int r=0; r<rep_phase[0]; r++){
      int bucket = bucket_off + bucket_offset[0][r];
      int pe = pe_off + pe_offset[0][r];
      MPI_Isend(buckets[bucket], counts[bucket], sr->mdtype(), pe, MTAG, cm, rep_reqs+bucket);
    }
    //progressss please
    if (bucket_off > 0){
      int flag;
      MPI_Testall(bucket_off, rep_reqs, &flag, MPI_STATUSES_IGNORE);
    }
#endif
#endif
#ifdef  PUTREDIST
    put_buckets<0>(rep_phase, pe_offset, bucket_offset, buckets, counts, sr, put_displs, win, bucket_off, pe_off);
#endif
  }
}
#endif

void dgtog_reshuffle(int const *          sym,
                     int64_t const *      edge_len,
                     distribution const & old_dist,
                     distribution const & new_dist,
                     char **              ptr_tsr_data,
                     char **              ptr_tsr_new_data,
                     algstrct const *     sr,
                     CommData             ord_glb_comm){
  int order = old_dist.order;

  char * tsr_data = *ptr_tsr_data;
  char * tsr_new_data = *ptr_tsr_new_data;

  if (order == 0){
    tsr_new_data = sr->alloc(1);
    if (ord_glb_comm.rank == 0){
      sr->copy(tsr_new_data, tsr_data);
    } else {
      sr->copy(tsr_new_data, sr->addid());
    }
    *ptr_tsr_new_data = tsr_new_data;
    sr->dealloc(tsr_data);
    return;
  }
#ifdef TUNE
  MPI_Barrier(ord_glb_comm.cm);
#endif
  TAU_FSTART(dgtog_reshuffle);
  double st_time = MPI_Wtime();

  int * old_virt_lda, * new_virt_lda;
  alloc_ptr(order*sizeof(int),     (void**)&old_virt_lda);
  alloc_ptr(order*sizeof(int),     (void**)&new_virt_lda);

  new_virt_lda[0] = 1;
  old_virt_lda[0] = 1;

  int old_idx_lyr = ord_glb_comm.rank - old_dist.perank[0]*old_dist.pe_lda[0];
  int new_idx_lyr = ord_glb_comm.rank - new_dist.perank[0]*new_dist.pe_lda[0];
  int new_nvirt=new_dist.virt_phase[0], old_nvirt=old_dist.virt_phase[0];
  for (int i=1; i<order; i++) {
    new_virt_lda[i] = new_nvirt;
    old_virt_lda[i] = old_nvirt;
    old_nvirt = old_nvirt*old_dist.virt_phase[i];
    new_nvirt = new_nvirt*new_dist.virt_phase[i];
    old_idx_lyr -= old_dist.perank[i]*old_dist.pe_lda[i];
    new_idx_lyr -= new_dist.perank[i]*new_dist.pe_lda[i];
  }
  int64_t old_virt_nelem = old_dist.size/old_nvirt;
  int64_t new_virt_nelem = new_dist.size/new_nvirt;

  int64_t *old_phys_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&old_phys_edge_len);
  for (int dim = 0;dim < order;dim++)
    old_phys_edge_len[dim] = old_dist.pad_edge_len[dim]/old_dist.phys_phase[dim];

  int64_t *new_phys_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&new_phys_edge_len);
  for (int dim = 0;dim < order;dim++)
    new_phys_edge_len[dim] = new_dist.pad_edge_len[dim]/new_dist.phys_phase[dim];

  int64_t *old_virt_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&old_virt_edge_len);
  for (int dim = 0;dim < order;dim++)
    old_virt_edge_len[dim] = old_phys_edge_len[dim]/old_dist.virt_phase[dim];

  int64_t *new_virt_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&new_virt_edge_len);
  for (int dim = 0;dim < order;dim++)
    new_virt_edge_len[dim] = new_phys_edge_len[dim]/new_dist.virt_phase[dim];

  int nold_rep = 1;
  int * old_rep_phase; alloc_ptr(sizeof(int)*order, (void**)&old_rep_phase);
  for (int i=0; i<order; i++){
    old_rep_phase[i] = lcm(old_dist.phys_phase[i], new_dist.phys_phase[i])/old_dist.phys_phase[i];
    nold_rep *= old_rep_phase[i];
  }

  int nnew_rep = 1;
  int * new_rep_phase; alloc_ptr(sizeof(int)*order, (void**)&new_rep_phase);
  for (int i=0; i<order; i++){
    new_rep_phase[i] = lcm(new_dist.phys_phase[i], old_dist.phys_phase[i])/new_dist.phys_phase[i];
    nnew_rep *= new_rep_phase[i];
  }

  int64_t * send_counts = (int64_t*)alloc(sizeof(int64_t)*nold_rep);
  std::fill(send_counts, send_counts+nold_rep, 0);
  calc_drv_displs(sym, edge_len, old_dist, new_dist, send_counts, old_idx_lyr);

  int64_t * recv_counts = (int64_t*)alloc(sizeof(int64_t)*nnew_rep);
  std::fill(recv_counts, recv_counts+nnew_rep, 0);
  calc_drv_displs(sym, edge_len, new_dist, old_dist, recv_counts, new_idx_lyr);
  int64_t * recv_displs = (int64_t*)alloc(sizeof(int64_t)*nnew_rep);

#ifdef IREDIST
  CTF_Request * recv_reqs = (CTF_Request*)alloc(sizeof(CTF_Request)*nnew_rep);
  CTF_Request * send_reqs = (CTF_Request*)alloc(sizeof(CTF_Request)*nold_rep);
#endif

  for (int i=0; i<nnew_rep; i++){
    if (i==0)
      recv_displs[0] = 0;
    else
      recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
  }

  int ** recv_bucket_offset; alloc_ptr(sizeof(int*)*order, (void**)&recv_bucket_offset);
  int ** recv_pe_offset; alloc_ptr(sizeof(int*)*order, (void**)&recv_pe_offset);
  int ** recv_ivmax_pre; alloc_ptr(sizeof(int*)*order, (void**)&recv_ivmax_pre);
  int64_t ** recv_data_offset; alloc_ptr(sizeof(int64_t*)*order, (void**)&recv_data_offset);
  precompute_offsets(new_dist, old_dist, sym, edge_len, new_rep_phase, new_phys_edge_len, new_virt_edge_len, new_dist.virt_phase, new_virt_lda, new_virt_nelem, recv_pe_offset, recv_bucket_offset, recv_data_offset, recv_ivmax_pre);

  int ** send_bucket_offset; alloc_ptr(sizeof(int*)*order, (void**)&send_bucket_offset);
  int ** send_pe_offset; alloc_ptr(sizeof(int*)*order, (void**)&send_pe_offset);
  int ** send_ivmax_pre; alloc_ptr(sizeof(int*)*order, (void**)&send_ivmax_pre);
  int64_t ** send_data_offset; alloc_ptr(sizeof(int64_t*)*order, (void**)&send_data_offset);

  precompute_offsets(old_dist, new_dist, sym, edge_len, old_rep_phase, old_phys_edge_len, old_virt_edge_len, old_dist.virt_phase, old_virt_lda, old_virt_nelem, send_pe_offset, send_bucket_offset, send_data_offset, send_ivmax_pre);

#if !defined(IREDIST) && !defined(PUTREDIST)
  int64_t * send_displs = (int64_t*)alloc(sizeof(int64_t)*nold_rep);
  send_displs[0] = 0;
  for (int i=1; i<nold_rep; i++){
    send_displs[i] = send_displs[i-1] + send_counts[i-1];
  }
#elif defined(PUTREDIST)
  int64_t * all_recv_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
  SWITCH_ORD_CALL(CTF_int::calc_cnt_from_rep_cnt, order-1, new_rep_phase, recv_pe_offset, recv_bucket_offset, recv_displs, all_recv_displs, 0, 0, 1);

  int64_t * all_put_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
  MPI_Alltoall(all_recv_displs, 1, MPI_INT64_T, all_put_displs, 1, MPI_INT64_T, ord_glb_comm.cm);
  CTF_int::cdealloc(all_recv_displs);

  int64_t * put_displs = (int64_t*)alloc(sizeof(int64_t)*nold_rep);
  SWITCH_ORD_CALL(CTF_int::calc_cnt_from_rep_cnt, order-1, old_rep_phase, send_pe_offset, send_bucket_offset, all_put_displs, put_displs, 0, 0, 0);

  CTF_int::cdealloc(all_put_displs);

  char * recv_buffer;
  //mst_alloc_ptr(new_dist.size*sr->el_size, (void**)&recv_buffer);
  recv_buffer = sr->alloc(new_dist.size);

  CTF_Win win;
  int suc = MPI_Win_create(recv_buffer, new_dist.size*sr->el_size, sr->el_size, MPI_INFO_NULL, ord_glb_comm.cm, &win);
  ASSERT(suc == MPI_SUCCESS);
#ifndef USE_FOMPI
  MPI_Win_fence(0, win);
#endif
#endif

#ifdef IREDIST
#ifdef PUTREDIST
  if (new_idx_lyr == 0)
    SWITCH_ORD_CALL(isendrecv, order-1, recv_pe_offset, recv_bucket_offset, new_rep_phase, recv_counts, recv_displs, recv_reqs, win, recv_buffer, sr, 0, 0, 1);
#else
  char * recv_buffer;
  recv_buffer = sr->alloc(new_dist.size);
  if (new_idx_lyr == 0)
    SWITCH_ORD_CALL(isendrecv, order-1, recv_pe_offset, recv_bucket_offset, new_rep_phase, recv_counts, recv_displs, recv_reqs, ord_glb_comm.cm, recv_buffer, sr, 0, 0, 1);
#endif
#endif


  if (old_idx_lyr == 0){
    char * aux_buf = sr->alloc(old_dist.size);
    char * tmp = aux_buf;
    aux_buf = tsr_data;
    tsr_data = tmp;
    char ** buckets = (char**)alloc(sizeof(char**)*nold_rep);

    buckets[0] = tsr_data;
    for (int i=1; i<nold_rep; i++){
      buckets[i] = buckets[i-1] + sr->el_size*send_counts[i-1];
    }
#if DEBUG >= 1
    int64_t save_counts[nold_rep];
    memcpy(save_counts, send_counts, sizeof(int64_t)*nold_rep);
#endif
    std::fill(send_counts, send_counts+nold_rep, 0);
    TAU_FSTART(redist_bucket);
#ifdef ROR
    int * old_rep_idx; alloc_ptr(sizeof(int)*order, (void**)&old_rep_idx);
    memset(old_rep_idx, 0, sizeof(int)*order);
    old_rep_idx[0]=-1;
    SWITCH_ORD_CALL(redist_bucket_isr, order-1, order, send_pe_offset, send_bucket_offset, send_data_offset,
                    send_ivmax_pre, old_rep_phase, old_rep_idx, old_dist.virt_phase[0],
#ifdef IREDIST
                    send_reqs, ord_glb_comm.cm,
#endif
#ifdef PUTREDIST
                    put_displs, win,
#endif
                    1, aux_buf, buckets, send_counts, sr);
    CTF_int::cdealloc(old_rep_idx);
#else
    SWITCH_ORD_CALL(redist_bucket, order-1, send_bucket_offset, send_data_offset,
                    send_ivmax_pre, old_rep_phase[0], old_dist.virt_phase[0], 1, aux_buf, buckets, send_counts, sr);
#endif
    TAU_FSTOP(redist_bucket);
    CTF_int::cdealloc(buckets);

#if DEBUG>= 1
    bool pass = true;
    for (int i=0; i<nold_rep; i++){
      if (save_counts[i] != send_counts[i]) pass = false;
    }
    if (!pass){
      for (int i=0; i<nold_rep; i++){
        printf("[%d] send_counts[%d] = %ld, redist_bucket counts[%d] = %ld\n", ord_glb_comm.rank, i, save_counts[i], i, send_counts[i]);
      }
    }
    ASSERT(pass);
#endif
    sr->dealloc(aux_buf);
  }
#ifndef WAITANY
#ifndef IREDIST
#ifndef PUTREDIST
  char * recv_buffer = sr->alloc(new_dist.size);

  /* Communicate data */
  TAU_FSTART(COMM_RESHUFFLE);

  CTF_Request * reqs = (CTF_Request*)alloc(sizeof(CTF_Request)*(nnew_rep+nold_rep));
  int nrecv = 0;
  if (new_idx_lyr == 0){
    nrecv = nnew_rep;
    SWITCH_ORD_CALL(isendrecv, order-1, recv_pe_offset, recv_bucket_offset, new_rep_phase, recv_counts, recv_displs, reqs, ord_glb_comm.cm, recv_buffer, sr, 0, 0, 1);
  }
  int nsent = 0;
  if (old_idx_lyr == 0){
    nsent = nold_rep;
    SWITCH_ORD_CALL(isendrecv, order-1, send_pe_offset, send_bucket_offset, old_rep_phase, send_counts, send_displs, reqs+nrecv, ord_glb_comm.cm, tsr_data, sr, 0, 0, 0);
  }
  if (nrecv+nsent > 0){
//      MPI_Status * stat = (MPI_Status*)alloc(sizeof(MPI_Status)*(nrecv+nsent));
    MPI_Waitall(nrecv+nsent, reqs, MPI_STATUSES_IGNORE);
  }
  cdealloc(reqs);
  //ord_glb_comm.all_to_allv(tsr_data, send_counts, send_displs, sr->el_size,
  //                         recv_buffer, recv_counts, recv_displs);
  TAU_FSTOP(COMM_RESHUFFLE);
  CTF_int::cdealloc(send_displs);
#else
  CTF_int::cdealloc(put_displs);
  TAU_FSTART(redist_fence);
  MPI_Win_fence(0, win);
  TAU_FSTOP(redist_fence);
  MPI_Win_free(&win);
#endif
  sr->dealloc(tsr_data);
#endif
#endif
  CTF_int::cdealloc(send_counts);

  if (new_idx_lyr == 0){
    char * aux_buf = sr->alloc(new_dist.size);
    sr->init(new_dist.size, aux_buf);

    char ** buckets = (char**)alloc(sizeof(char**)*nnew_rep);

    buckets[0] = recv_buffer;
    //printf("[%d] size of %dth bucket is %ld\n", ord_glb_comm.rank, 0, send_counts[0]);
    for (int i=1; i<nnew_rep; i++){
      buckets[i] = buckets[i-1] + sr->el_size*recv_counts[i-1];
      //printf("[%d] size of %dth bucket is %ld\n", ord_glb_comm.rank, i, send_counts[i]);
    }

#if DEBUG >= 1
    int64_t save_counts[nnew_rep];
    memcpy(save_counts, recv_counts, sizeof(int64_t)*nnew_rep);
#endif
    std::fill(recv_counts, recv_counts+nnew_rep, 0);

    TAU_FSTART(redist_debucket);

#ifdef WAITANY
    for (int nb=0; nb<nnew_rep; nb++){
      //int source;
/*#ifdef PUT_NOTIFY
      foMPI_Request req;
      foMPI_Wait(&req, MPY_ANY_TAG);
      source = req.source;
#else*/
      MPI_Status stat;
      int bucket_off;
      MPI_Waitany(nnew_rep, recv_reqs, &bucket_off, &stat);
//        foMPI_Start(recv_reqs+bucket_off);
//      ASSERT(ret== MPI_SUCCESS);
      ASSERT(bucket_off != MPI_UNDEFINED);
      ASSERT(bucket_off >= 0 && bucket_off <nnew_rep);
      ASSERT(recv_counts[bucket_off] == 0);
      //source = stat.source;
//#endif
      int rep_idx[order];
      int iboff=bucket_off;
      for (int i=0; i<order; i++){
/*          //FIXME: lame
        int pe_offi = ((source/MAX(1,old_dist.pe_lda[i]))%old_dist.phys_phase[i])*MAX(1,old_dist.pe_lda[i]);
        int pidx = -1;
        for (int j=0; j<new_rep_phase[i]; j++){
          printf("source = %d i = %d pe_offi=%d recv_pe_offset[i][%d]=%d\n",source,i,pe_offi,j,recv_pe_offset[i][j]);
          if (pe_offi == recv_pe_offset[i][j]) pidx=j;
        }
        ASSERT(pidx!=-1);
        rep_idx[i] = pidx;
        bucket_off += recv_bucket_offset[i][pidx];*/
        rep_idx[i] = iboff%new_rep_phase[i];
        iboff = iboff/new_rep_phase[i];
      }

      SWITCH_ORD_CALL(redist_bucket_ror, order-1, recv_bucket_offset, recv_data_offset, recv_ivmax_pre, new_rep_phase, rep_idx, new_dist.virt_phase[0], 0, aux_buf, buckets, recv_counts, sr, 0, bucket_off, 0)
      //printf("recv_counts[%d]=%d, saved_counts[%d]=%d\n",bucket_off,recv_counts[bucket_off],bucket_off,save_counts[bucket_off]);

    }
#ifdef PUT_NOTIFY
    for (int nb=0; nb<nnew_rep; nb++){
      foMPI_Request_free(recv_reqs+nb);
    }
#endif
#else
#ifdef ROR
    int * new_rep_idx; alloc_ptr(sizeof(int)*order, (void**)&new_rep_idx);
    memset(new_rep_idx, 0, sizeof(int)*order);
    new_rep_idx[0] = -1;
    SWITCH_ORD_CALL(redist_bucket_isr, order-1, order, recv_pe_offset, recv_bucket_offset, recv_data_offset,
                    recv_ivmax_pre, new_rep_phase, new_rep_idx, new_dist.virt_phase[0],
#ifdef IREDIST
                    recv_reqs, ord_glb_comm.cm,
#endif
#ifdef  PUTREDIST
                    NULL, win,
#endif
                    0, aux_buf, buckets, recv_counts, sr);
    CTF_int::cdealloc(new_rep_idx);
#else
    SWITCH_ORD_CALL(redist_bucket, order-1,
                    recv_bucket_offset, recv_data_offset, recv_ivmax_pre,
                    new_rep_phase[0], new_dist.virt_phase[0], 0, aux_buf, buckets, recv_counts, sr);
#endif
#endif
    TAU_FSTOP(redist_debucket);
    CTF_int::cdealloc(buckets);
#if DEBUG >= 1
    bool pass = true;
    for (int i=0; i<nnew_rep; i++){
      if (save_counts[i] != recv_counts[i]) pass = false;
    }
    if (!pass){
      for (int i=0; i<nnew_rep; i++){
        printf("[%d] recv_counts[%d] = %ld, redist_bucket counts[%d] = %ld\n", ord_glb_comm.rank, i, save_counts[i], i, recv_counts[i]);
      }
    }
    ASSERT(pass);
#endif
    *ptr_tsr_new_data = aux_buf;
    sr->dealloc(recv_buffer);
  } else {
    if (sr->addid() != NULL)
      sr->set(recv_buffer, sr->addid(), new_dist.size);
    *ptr_tsr_new_data = recv_buffer;
  }
  //printf("[%d] reached final barrier %d\n",ord_glb_comm.rank, MTAG);
#ifdef IREDIST

  CTF_int::cdealloc(recv_reqs);
  CTF_int::cdealloc(send_reqs);
#endif
  for (int i=0; i<order; i++){
    CTF_int::cdealloc(recv_pe_offset[i]);
    CTF_int::cdealloc(recv_bucket_offset[i]);
    CTF_int::cdealloc(recv_data_offset[i]);
    CTF_int::cdealloc(recv_ivmax_pre[i]);
  }
  CTF_int::cdealloc(recv_pe_offset);
  CTF_int::cdealloc(recv_bucket_offset);
  CTF_int::cdealloc(recv_data_offset);
  CTF_int::cdealloc(recv_ivmax_pre);

  for (int i=0; i<order; i++){
    CTF_int::cdealloc(send_pe_offset[i]);
    CTF_int::cdealloc(send_bucket_offset[i]);
    CTF_int::cdealloc(send_data_offset[i]);
    CTF_int::cdealloc(send_ivmax_pre[i]);
  }
  CTF_int::cdealloc(send_pe_offset);
  CTF_int::cdealloc(send_bucket_offset);
  CTF_int::cdealloc(send_data_offset);
  CTF_int::cdealloc(send_ivmax_pre);

  CTF_int::cdealloc(old_virt_lda);
  CTF_int::cdealloc(new_virt_lda);
  CTF_int::cdealloc(recv_counts);
  CTF_int::cdealloc(recv_displs);
  CTF_int::cdealloc(old_phys_edge_len);
  CTF_int::cdealloc(new_phys_edge_len);
  CTF_int::cdealloc(old_virt_edge_len);
  CTF_int::cdealloc(new_virt_edge_len);
  CTF_int::cdealloc(old_rep_phase);
  CTF_int::cdealloc(new_rep_phase);
#ifdef IREDIST
#ifdef PUT_NOTIFY
  foMPI_Win_flush_all(win);
  foMPI_Win_free(&win);
#else
  TAU_FSTART(barrier_after_dgtog_reshuffle);
  MPI_Barrier(ord_glb_comm.cm);
  TAU_FSTOP(barrier_after_dgtog_reshuffle);
#endif
  sr->dealloc(tsr_data);
#endif
#ifdef TUNE
  MPI_Barrier(ord_glb_comm.cm);
#endif
  double exe_time = MPI_Wtime()-st_time;
  double tps[] = {exe_time, 1.0, (double)log2(ord_glb_comm.np), (double)std::max(old_dist.size, new_dist.size)*log2(ord_glb_comm.np)*sr->el_size};

  // double-check
   dgtog_res_mdl.observe(tps);
  TAU_FSTOP(dgtog_reshuffle);
}
