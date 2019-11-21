/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "cyclic_reshuffle.h"
#include "../shared/util.h"

namespace CTF_int {

  void pad_cyclic_pup_virt_buff(int const *          sym,
                                distribution const & old_dist,
                                distribution const & new_dist,
                                int64_t const *      len,
                                int const *          old_phys_dim,
                                int64_t const *      old_phys_edge_len,
                                int64_t const *      old_virt_edge_len,
                                int64_t              old_virt_nelem,
                                int64_t const *      old_offsets,
                                int * const *        old_permutation,
                                int                  total_np,
                                int const *          new_phys_dim,
                                int64_t const *      new_phys_edge_len,
                                int64_t const *      new_virt_edge_len,
                                int64_t              new_virt_nelem,
                                char *               old_data,
                                char **              new_data,
                                int                  forward,
                                int * const *        bucket_offset,
                                char const *         alpha,
                                char const *         beta,
                                algstrct const *     sr){
    bool is_copy = false;
    if (sr->isequal(sr->mulid(), alpha) && sr->isequal(sr->addid(), beta)) is_copy = true;
    if (old_dist.order == 0){
      if (forward)
        sr->copy(new_data[0], old_data);
      else {
        if (is_copy)
          sr->copy(old_data, new_data[0]);
        else
          sr->acc(old_data, beta, new_data[0], alpha);
      }
      return;
    }

    int old_virt_np = 1;
    for (int dim = 0;dim < old_dist.order;dim++) old_virt_np *= old_dist.virt_phase[dim];

    int new_virt_np = 1;
    for (int dim = 0;dim < old_dist.order;dim++) new_virt_np *= new_dist.virt_phase[dim];
    
    int nbucket = total_np; //*(forward ? new_virt_np : old_virt_np);

  #if DEBUG >= 1
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  #endif

    TAU_FSTART(cyclic_pup_bucket);
  #ifdef USE_OMP
    int max_ntd = omp_get_max_threads();
    max_ntd = MAX(1,MIN(max_ntd,new_virt_nelem/nbucket));

    int64_t old_size, new_size;
    old_size = sy_packed_size(old_dist.order, old_virt_edge_len, sym)*old_virt_np;
    new_size = sy_packed_size(old_dist.order, new_virt_edge_len, sym)*new_virt_np;
    /*if (forward){
    } else {
      old_size = sy_packed_size(old_dist.order, old_virt_edge_len, sym)*new_virt_np;
      new_size = sy_packed_size(old_dist.order, new_virt_edge_len, sym)*old_virt_np;
    }*/
    /*printf("old_size=%d, new_size=%d,old_virt_np=%d,new_virt_np=%d\n",
            old_size,new_size,old_virt_np,new_virt_np);
  */
    int64_t * bucket_store;  
    int64_t * count_store;  
    int64_t * thread_store;  
    alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&bucket_store);
    alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&count_store);
    alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&thread_store);
    std::fill(bucket_store, bucket_store+MAX(old_size,new_size), -1);

    int64_t ** par_virt_counts;
    alloc_ptr(sizeof(int64_t*)*max_ntd, (void**)&par_virt_counts);
    for (int t=0; t<max_ntd; t++){
      alloc_ptr(sizeof(int64_t)*nbucket, (void**)&par_virt_counts[t]);
      std::fill(par_virt_counts[t], par_virt_counts[t]+nbucket, 0);
    }
    #pragma omp parallel num_threads(max_ntd)
    {
  #endif

    int64_t *offs; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&offs);
    if (old_offsets == NULL)
      for (int dim = 0;dim < old_dist.order;dim++) offs[dim] = 0;
    else 
      for (int dim = 0;dim < old_dist.order;dim++) offs[dim] = old_offsets[dim];

    int64_t *ends; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&ends);
    for (int dim = 0;dim < old_dist.order;dim++) ends[dim] = len[dim];

  #ifdef USE_OMP
    int tid = omp_get_thread_num();
    int ntd = omp_get_num_threads();
    //partition the global tensor among threads, to preserve 
    //global ordering and load balance in partitioning
    int64_t gidx_st[old_dist.order];
    int64_t gidx_end[old_dist.order];
    if (old_dist.order > 1){
      int64_t all_size = packed_size(old_dist.order, len, sym);
      int64_t chnk = all_size/ntd;
      int64_t glb_idx_st = chnk*tid + MIN(tid,all_size%ntd);
      int64_t glb_idx_end = glb_idx_st+chnk+(tid<(all_size%ntd));
      //calculate global indices along each dimension corresponding to partition
//      printf("glb_idx_st = %ld, glb_idx_end = %ld\n",glb_idx_st,glb_idx_end);
      calc_idx_arr(old_dist.order, len, sym, glb_idx_st,  gidx_st);
      calc_idx_arr(old_dist.order, len, sym, glb_idx_end, gidx_end);
      gidx_st[0] = 0;
      //FIXME: wrong but evidently not used
      gidx_end[0] = 0;
  #if DEBUG >= 1
      if (ntd == 1){
        if (gidx_end[old_dist.order-1] != len[old_dist.order-1]){
          for (int dim=0; dim<old_dist.order; dim++){
            printf("glb_idx_end = %ld, gidx_end[%d]= %ld, len[%d] = %ld\n", 
                   glb_idx_end, dim, gidx_end[dim], dim, len[dim]);
          }
          ABORT;
        }
        ASSERT(gidx_end[old_dist.order-1] <= ends[old_dist.order-1]);
      } 
  #endif
    } else {
      //FIXME the below means redistribution of a vector is non-threaded
      if (tid == 0){
        gidx_st[0] = 0;
        gidx_end[0] = ends[0];
      } else {
        gidx_st[0] = 0;
        gidx_end[0] = 0;
      }

    }
    //clip global indices to my physical cyclic phase (local tensor data)

  #endif
    // FIXME: may be better to alloc, but this should ensure the 
    //        compiler knows there are no write conflicts
  #ifdef USE_OMP
    int64_t * count = par_virt_counts[tid];
  #else
    int64_t *count; alloc_ptr(sizeof(int64_t)*nbucket, (void**)&count);
    memset(count, 0, sizeof(int64_t)*nbucket);
  #endif

    int64_t *gidx; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&gidx);
    memset(gidx, 0, sizeof(int64_t)*old_dist.order);
    for (int dim = 0;dim < old_dist.order;dim++){
      gidx[dim] = old_dist.perank[dim];
    }

    int64_t *virt_offset; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&virt_offset);
    memset(virt_offset, 0, sizeof(int64_t)*old_dist.order);

    int64_t *idx; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&idx);
    memset(idx, 0, sizeof(int64_t)*old_dist.order);

    int64_t *virt_acc; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&virt_acc);
    memset(virt_acc, 0, sizeof(int64_t)*old_dist.order);

    int64_t *idx_acc; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&idx_acc);
    memset(idx_acc, 0, sizeof(int64_t)*old_dist.order);
    
    int64_t *old_virt_lda; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&old_virt_lda);
    old_virt_lda[0] = old_virt_nelem;
    for (int dim=1; dim<old_dist.order; dim++){
      old_virt_lda[dim] = old_virt_lda[dim-1]*old_dist.virt_phase[dim-1];
    }

    int64_t offset = 0;

    int64_t zero_len_toff = 0;

  #ifdef USE_OMP
    for (int dim=old_dist.order-1; dim>=0; dim--){
      int64_t iist = MAX(0,(gidx_st[dim]-old_dist.perank[dim]));
      int64_t ist = iist/old_dist.phase[dim];//(old_phys_dim[dim]*old_dist.virt_phase[dim]);
      if (sym[dim] != NS) ist = MIN(ist,idx[dim+1]);
      int64_t plen[old_dist.order];
      memcpy(plen,old_virt_edge_len,old_dist.order*sizeof(int64_t));
      int idim = dim;
      do {
        plen[idim] = ist;
        idim--;
      } while (idim >= 0 && sym[idim] != NS);
      //gidx[dim] += ist*old_phys_dim[dim]*old_dist.virt_phase[dim];
      gidx[dim] += ist*old_dist.phase[dim];//old_phys_dim[dim]*old_dist.virt_phase[dim];
      idx[dim] = ist;
      idx_acc[dim] = sy_packed_size(dim+1, plen, sym);
      offset += idx_acc[dim]; 

      ASSERT(ist == 0 || gidx[dim] <= gidx_st[dim]);
  //    ASSERT(ist < old_virt_edge_len[dim]);

      if (gidx[dim] > gidx_st[dim]) break;

      int64_t vst = iist-ist*old_dist.phase[dim];//*old_phys_dim[dim]*old_dist.virt_phase[dim];
      if (vst > 0 ){
        vst = MIN(old_dist.virt_phase[dim]-1,vst/old_dist.phys_phase[dim]);
        gidx[dim] += vst*old_dist.phys_phase[dim];
        virt_offset[dim] = vst;
        offset += vst*old_virt_lda[dim];
      } else vst = 0;
      if (gidx[dim] > gidx_st[dim]) break;
    }
  #endif

    bool done = false;
    for (;!done;){
      int64_t bucket0 = 0;
      bool outside0 = false;
      int64_t len_zero_max = ends[0];
  #ifdef USE_OMP
      bool is_at_end = true;
      bool is_at_start = true;
      for (int dim = old_dist.order-1;dim >0;dim--){
        if (gidx[dim] > gidx_st[dim]){
          is_at_start = false;
          break;
        }
        if (gidx[dim] < gidx_st[dim]){
          outside0 = true;
          break;
        }
      }
      if (is_at_start){
        zero_len_toff = gidx_st[0];
      }
      for (int dim = old_dist.order-1;dim >0;dim--){
        if (gidx_end[dim] < gidx[dim]){
          outside0 = true;
          done = true;
          break;
        }
        if (gidx_end[dim] > gidx[dim]){
          is_at_end = false;
          break;
        }
      }
      if (is_at_end){
        len_zero_max = MIN(ends[0],gidx_end[0]);
        done = true;
      }
  #endif

      if (!outside0){
        for (int dim = 1;dim < old_dist.order;dim++){
          if (bucket_offset[dim][virt_offset[dim]+idx[dim]*old_dist.virt_phase[dim]] == -1) outside0 = true;
          bucket0 += bucket_offset[dim][virt_offset[dim]+idx[dim]*old_dist.virt_phase[dim]];
        }
      }

      if (!outside0){
        for (int dim = 1;dim < old_dist.order;dim++){
          if (gidx[dim] >= (sym[dim] == NS ? ends[dim] :
                           (sym[dim] == SY ? gidx[dim+1]+1 :
                                             gidx[dim+1])) ||
              gidx[dim] < offs[dim]){
            outside0 = true;
            break;
          }
        }
      }

      int64_t idx_max = (sym[0] == NS ? old_virt_edge_len[0] : idx[1]+1);
      int64_t idx_st = 0;

      if (!outside0){
        int64_t gidx_min = MAX(zero_len_toff,offs[0]);
        int64_t gidx_max = (sym[0] == NS ? ends[0] : (sym[0] == SY ? gidx[1]+1 : gidx[1]));
        gidx_max = MIN(gidx_max, len_zero_max);
        for (idx[0] = idx_st;idx[0] < idx_max;idx[0]++){
          int virt_min = MAX(0,MIN(old_dist.virt_phase[0],(gidx_min-gidx[0])/old_dist.phys_phase[0]));
          //int virt_min = MAX(0,MIN(old_dist.virt_phase[0],(gidx_min-gidx[0]+old_dist.phys_phase[0]-1)/old_dist.phys_phase[0]));
          int virt_max = MAX(0,MIN(old_dist.virt_phase[0],(gidx_max-gidx[0]+old_dist.phys_phase[0]-1)/old_dist.phys_phase[0]));

          offset += old_virt_nelem*virt_min;
          if (forward){
            ASSERT(is_copy);
            for (virt_offset[0] = virt_min;
                 virt_offset[0] < virt_max;
                 virt_offset[0] ++)
            {
              int bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]*old_dist.virt_phase[0]];
  #ifdef USE_OMP
              bucket_store[offset] = bucket;
              count_store[offset]  = count[bucket]++;
              thread_store[offset] = tid;
  #else
/*              printf("[%d] bucket = %d offset = %ld\n", rank, bucket, offset);
              printf("[%d] count[bucket] = %d, nbucket = %d\n", rank, count[bucket]+1, nbucket);
              std::cout << "old_data[offset]=";
              sr->print(old_data+ sr->el_size*offset);*/
              sr->copy(new_data[bucket]+sr->el_size*(count[bucket]++), old_data+ sr->el_size*offset);
/*              std::cout << "\nnew_data[bucket][count[bucket]++]=";
              sr->print(new_data[bucket]+sr->el_size*(count[bucket]-1));
              std::cout << "\n";*/
  #endif
              offset += old_virt_nelem;
            }
          }
          else{
            for (virt_offset[0] = virt_min;
                 virt_offset[0] < virt_max;
                 virt_offset[0] ++)
            {
              int bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]*old_dist.virt_phase[0]];
  #ifdef USE_OMP
              bucket_store[offset] = bucket;
              count_store[offset]  = count[bucket]++;
              thread_store[offset] = tid;
  #else
              if (is_copy) 
                sr->copy(old_data+sr->el_size*offset,       new_data[bucket]+sr->el_size*(count[bucket]++));
              else 
                sr->acc( old_data+sr->el_size*offset, beta, new_data[bucket]+sr->el_size*(count[bucket]++), alpha);
//              old_data[offset] = beta*old_data[offset] + alpha*new_data[bucket][count[bucket]++];
  #endif
              offset += old_virt_nelem;
            }
          }

          offset++;
          offset -= old_virt_nelem*virt_max;
          gidx[0] += old_dist.phase[0];//old_phys_dim[0]*old_dist.virt_phase[0];
        }

        offset -= idx_max;
        gidx[0] -= idx_max*old_dist.phase[0];//old_phys_dim[0]*old_dist.virt_phase[0];
      }
       
      idx_acc[0] = idx_max;

      idx[0] = 0;

      zero_len_toff = 0;

      /* Adjust outer indices */
      if (!done){
        for (int dim = 1;dim < old_dist.order;dim++){
          offset += old_virt_lda[dim];
    
          virt_offset[dim] ++;//= old_virt_edge_len[dim];
          gidx[dim]+=old_dist.phys_phase[dim];
          if (virt_offset[dim] == old_dist.virt_phase[dim]){
            offset -= old_virt_lda[dim]*old_dist.virt_phase[dim];
            gidx[dim] -= old_dist.phase[dim];
            virt_offset[dim] = 0;

            offset += idx_acc[dim-1];
            idx_acc[dim] += idx_acc[dim-1];
            idx_acc[dim-1] = 0;

            gidx[dim] -= idx[dim]*old_dist.phase[dim];//phys_dim[dim]*old_dist.virt_phase[dim];
            idx[dim]++;

            if (idx[dim] == (sym[dim] == NS ? old_virt_edge_len[dim] : idx[dim+1]+1)){
              offset -= idx_acc[dim];
              //index should always be zero here sicne everything is SY and not SH
              idx[dim] = 0;//(dim == 0 || sym[dim-1] == NS ? 0 : idx[dim-1]);
              //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];

              if (dim == old_dist.order-1) done = true;
            }
            else{
              //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];
              gidx[dim] += idx[dim]*old_dist.phase[dim];//old_phys_dim[dim]*old_dist.virt_phase[dim];
              break;
            }
          }
          else{
            idx_acc[dim-1] = 0;
            break;
          }
        }
        if (old_dist.order <= 1) done = true;
      }
    }
    cdealloc(gidx);
    cdealloc(idx_acc);
    cdealloc(virt_acc);
    cdealloc(idx);
    cdealloc(virt_offset);
    cdealloc(old_virt_lda);

  #ifndef USE_OMP
  #if DEBUG >= 1
    bool pass = true;
    for (int i = 0;i < nbucket-1;i++){
      if (count[i] != (int64_t)((new_data[i+1]-new_data[i])/sr->el_size)){
        printf("rank = %d count %d should have been %d is %ld\n", rank, i, (int)((new_data[i+1]-new_data[i])/sr->el_size), count[i]);
        pass = false;
      }
    }
    if (!pass) ABORT;
  #endif
  #endif
    cdealloc(offs);
    cdealloc(ends);
   
  #ifndef USE_OMP
    cdealloc(count);
    TAU_FSTOP(cyclic_pup_bucket);
  #else
    par_virt_counts[tid] = count;
    } //#pragma omp endfor
    for (int bckt=0; bckt<nbucket; bckt++){
      int par_tmp = 0;
      for (int thread=0; thread<max_ntd; thread++){
        par_tmp += par_virt_counts[thread][bckt];
        par_virt_counts[thread][bckt] = par_tmp - par_virt_counts[thread][bckt];
      }
  #if DEBUG >= 1
      if (bckt < nbucket-1 && par_tmp != (new_data[bckt+1]-new_data[bckt])/sr->el_size){
        printf("rank = %d count for bucket %d is %d should have been %ld\n",rank,bckt,par_tmp,(int64_t)(new_data[bckt+1]-new_data[bckt])/sr->el_size);
        ABORT;
      }
  #endif
    }
    TAU_FSTOP(cyclic_pup_bucket);
    TAU_FSTART(cyclic_pup_move);
    {
      int64_t tot_sz = MAX(old_size, new_size);
      int64_t i;
      if (forward){
        ASSERT(is_copy);
        #pragma omp parallel for private(i)
        for (i=0; i<tot_sz; i++){
          if (bucket_store[i] != -1){
            int64_t pc = par_virt_counts[thread_store[i]][bucket_store[i]];
            int64_t ct = count_store[i]+pc;
            sr->copy(new_data[bucket_store[i]]+ct*sr->el_size, old_data+i*sr->el_size);
          }
        }
      } else {
        if (is_copy){// alpha == 1.0 && beta == 0.0){
          #pragma omp parallel for private(i)
          for (i=0; i<tot_sz; i++){
            if (bucket_store[i] != -1){
              int64_t pc = par_virt_counts[thread_store[i]][bucket_store[i]];
              int64_t ct = count_store[i]+pc;
              sr->copy(old_data+i*sr->el_size, new_data[bucket_store[i]]+ct*sr->el_size);
            }
          }
        } else {
          #pragma omp parallel for private(i)
          for (i=0; i<tot_sz; i++){
            if (bucket_store[i] != -1){
              int64_t pc = par_virt_counts[thread_store[i]][bucket_store[i]];
              int64_t ct = count_store[i]+pc;
              sr->acc(old_data+i*sr->el_size, beta, new_data[bucket_store[i]]+ct*sr->el_size, alpha);
            }
          }
        }
      }
    }
    TAU_FSTOP(cyclic_pup_move);
    for (int t=0; t<max_ntd; t++){
      cdealloc(par_virt_counts[t]);
    }
    cdealloc(par_virt_counts);
    cdealloc(count_store);
    cdealloc(bucket_store);
    cdealloc(thread_store);
  #endif

  }

  void cyclic_reshuffle(int const *          sym,
                        distribution const & old_dist,
                        int64_t const *      old_offsets,
                        int * const *        old_permutation,
                        distribution const & new_dist,
                        int64_t const *      new_offsets,
                        int * const *        new_permutation,
                        char **              ptr_tsr_data,
                        char **              ptr_tsr_cyclic_data,
                        algstrct const *     sr,
                        CommData             ord_glb_comm,
                        bool                 reuse_buffers,
                        char const *         alpha,
                        char const *         beta){
    int i, np, old_nvirt, new_nvirt, old_np, new_np, idx_lyr;
    int64_t vbs_old, vbs_new;
    int64_t swp_nval;
    int * hsym;
    int64_t * send_counts, * recv_counts;
    int64_t * idx;
    int64_t * idx_offs;
    int64_t  * send_displs;
    int64_t * recv_displs;
    int * new_virt_lda, * old_virt_lda;
    int64_t * old_sub_edge_len, * new_sub_edge_len;
    int order = old_dist.order; 

    char * tsr_data = *ptr_tsr_data;
    char * tsr_cyclic_data = *ptr_tsr_cyclic_data;
    if (order == 0){
      bool is_copy = false;
      if (sr->isequal(sr->mulid(), alpha) && sr->isequal(sr->addid(), beta)) is_copy = true;
      alloc_ptr(sr->el_size, (void**)&tsr_cyclic_data);
      if (ord_glb_comm.rank == 0){
        if (is_copy)
          sr->copy(tsr_cyclic_data, tsr_data);
        else
          sr->acc(tsr_cyclic_data, beta, tsr_data, alpha);
      } else {
        sr->copy(tsr_cyclic_data, sr->addid());
      }
      *ptr_tsr_cyclic_data = tsr_cyclic_data;
      return;
    }
    
    ASSERT(!reuse_buffers || sr->isequal(beta, sr->addid()));
    ASSERT(old_dist.is_cyclic&&new_dist.is_cyclic);

    TAU_FSTART(cyclic_reshuffle);
      np = ord_glb_comm.np;

    alloc_ptr(order*sizeof(int),     (void**)&hsym);
    alloc_ptr(order*sizeof(int64_t), (void**)&idx);
    alloc_ptr(order*sizeof(int64_t), (void**)&idx_offs);
    alloc_ptr(order*sizeof(int),     (void**)&old_virt_lda);
    alloc_ptr(order*sizeof(int),     (void**)&new_virt_lda);

    new_nvirt = 1;
    old_nvirt = 1;
    old_np = 1;
    new_np = 1;
    idx_lyr = ord_glb_comm.rank;
    for (i=0; i<order; i++) {
      new_virt_lda[i] = new_nvirt;
      old_virt_lda[i] = old_nvirt;
   //   nbuf = nbuf*new_dist.phase[i];
      /*printf("is_new_pad = %d\n", is_new_pad);
      if (is_new_pad)
        printf("new_dist.padding[%d] = %d\n", i, new_dist.padding[i]);
      printf("is_old_pad = %d\n", is_old_pad);
      if (is_old_pad)
        printf("old_dist.padding[%d] = %d\n", i, old_dist.padding[i]);*/
      old_nvirt = old_nvirt*old_dist.virt_phase[i];
      new_nvirt = new_nvirt*new_dist.virt_phase[i];
      new_np = new_np*new_dist.phase[i]/new_dist.virt_phase[i];
      old_np = old_np*old_dist.phase[i]/old_dist.virt_phase[i];
      idx_lyr -= old_dist.perank[i]*old_dist.pe_lda[i];
    }
    vbs_old = old_dist.size/old_nvirt;

    alloc_ptr(np*sizeof(int64_t),   (void**)&recv_counts);
    alloc_ptr(np*sizeof(int64_t),   (void**)&send_counts);
    alloc_ptr(np*sizeof(int64_t),   (void**)&send_displs);
    alloc_ptr(np*sizeof(int64_t),   (void**)&recv_displs);
    alloc_ptr(order*sizeof(int64_t), (void**)&old_sub_edge_len);
    alloc_ptr(order*sizeof(int64_t), (void**)&new_sub_edge_len);
    int ** bucket_offset;
    
    int64_t *real_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&real_edge_len);
    for (i=0; i<order; i++) real_edge_len[i] = old_dist.pad_edge_len[i]-old_dist.padding[i];
    
    int *old_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&old_phys_dim);
    for (i=0; i<order; i++) old_phys_dim[i] = old_dist.phase[i]/old_dist.virt_phase[i];

    int *new_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&new_phys_dim);
    for (i=0; i<order; i++) new_phys_dim[i] = new_dist.phase[i]/new_dist.virt_phase[i];
    
    int64_t *old_phys_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&old_phys_edge_len);
    for (int dim = 0;dim < order;dim++) old_phys_edge_len[dim] = (real_edge_len[dim]+old_dist.padding[dim])/old_phys_dim[dim];

    int64_t *new_phys_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&new_phys_edge_len);
    for (int dim = 0;dim < order;dim++) new_phys_edge_len[dim] = (real_edge_len[dim]+new_dist.padding[dim])/new_phys_dim[dim];

    int64_t *old_virt_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&old_virt_edge_len);
    for (int dim = 0;dim < order;dim++) old_virt_edge_len[dim] = old_phys_edge_len[dim]/old_dist.virt_phase[dim];

    int64_t *new_virt_edge_len; alloc_ptr(sizeof(int64_t)*order, (void**)&new_virt_edge_len);
    for (int dim = 0;dim < order;dim++) new_virt_edge_len[dim] = new_phys_edge_len[dim]/new_dist.virt_phase[dim];
    


    bucket_offset = 
      compute_bucket_offsets( old_dist,
                              new_dist,
                              real_edge_len,
                              old_phys_edge_len,
                              old_virt_lda,
                              old_offsets,
                              old_permutation,
                              new_phys_edge_len,
                              new_virt_lda,
                              1,
                              old_nvirt,
                              new_nvirt,
                              old_virt_edge_len);



    TAU_FSTART(calc_cnt_displs);
    /* Calculate bucket counts to begin exchange */
    calc_cnt_displs(sym,
                    old_dist,
                    new_dist,
                    new_nvirt,
                    np,
                    old_virt_edge_len,
                    new_virt_lda,
                    send_counts,    
                    recv_counts,
                    send_displs,
                    recv_displs,    
                    ord_glb_comm, 
                    idx_lyr,
                    bucket_offset);
    
    TAU_FSTOP(calc_cnt_displs);
    /*for (i=0; i<np; i++){
      printf("[%d] send_counts[%d] = %d recv_counts[%d] = %d\n", ord_glb_comm.rank, i, send_counts[i], i, recv_counts[i]);
    }
    for (i=0; i<nbuf; i++){
      printf("[%d] svirt_displs[%d] = %d rvirt_displs[%d] = %d\n", ord_glb_comm.rank, i, svirt_displs[i], i, rvirt_displs[i]);
    }*/

  //  }
    for (i=0; i<order; i++){
      new_sub_edge_len[i] = new_dist.pad_edge_len[i];
      old_sub_edge_len[i] = old_dist.pad_edge_len[i];
    }
    for (i=0; i<order; i++){
      new_sub_edge_len[i] = new_sub_edge_len[i] / new_dist.phase[i];
      old_sub_edge_len[i] = old_sub_edge_len[i] / old_dist.phase[i];
    }
    for (i=1; i<order; i++){
      hsym[i-1] = sym[i];
    }
    swp_nval = new_nvirt*sy_packed_size(order, new_sub_edge_len, sym);
    vbs_new = swp_nval/new_nvirt;

    char * send_buffer, * recv_buffer;
    if (reuse_buffers){
      alloc_ptr(MAX(old_dist.size,swp_nval)*sr->el_size, (void**)&tsr_cyclic_data);
    } else {
      alloc_ptr(old_dist.size*sr->el_size, (void**)&send_buffer);
      alloc_ptr(swp_nval*sr->el_size, (void**)&recv_buffer);
    }

    TAU_FSTART(pack_virt_buf);
    if (idx_lyr == 0){
      /*char new1[old_dist.size*sr->el_size];
      char new2[old_dist.size*sr->el_size];
      sr->set(new1, sr->addid(), old_dist.size);
      sr->set(new2, sr->addid(), old_dist.size);
      //if (ord_glb_comm.rank == 0)
        //printf("old_dist.size = %ld\n",old_dist.size);
      //std::fill((double*)new1, ((double*)new1)+old_dist.size, 0.0);
      //std::fill((double*)new2, ((double*)new2)+old_dist.size, 0.0);
      order_globally(sym, old_dist, old_virt_edge_len, old_virt_lda, vbs_old, 1, tsr_data, new1, sr);
      order_globally(sym, old_dist, old_virt_edge_len, old_virt_lda, vbs_old, 0, new1, new2, sr);
      
      if (ord_glb_comm.rank == 0){
        for (int64_t i=0; i<old_dist.size; i++){
          if (!sr->isequal(new2+i*sr->el_size, tsr_data +i*sr->el_size)){
            printf("tsr_data[%ld] was ",i);
            sr->print(tsr_data +i*sr->el_size);
            printf(" became ");
            sr->print(new2+i*sr->el_size);
            printf("\n");
            ASSERT(0);
          }
        }
      }*/
      

      char **new_data; alloc_ptr(sizeof(char*)*np, (void**)&new_data);
      if (reuse_buffers){
        for (int64_t p = 0;p < np;p++){
          new_data[p] = tsr_cyclic_data+sr->el_size*send_displs[p];
        }
      } else {
        for (int64_t p = 0;p < np;p++){
          new_data[p] = send_buffer+sr->el_size*send_displs[p];
        }
      }

      pad_cyclic_pup_virt_buff(sym,
                               old_dist, 
                               new_dist, 
                               real_edge_len,
                               old_phys_dim,
                               old_phys_edge_len,
                               old_virt_edge_len,
                               vbs_old,
                               old_offsets,
                               old_permutation,
                               np,
                               new_phys_dim,
                               new_phys_edge_len,
                               new_virt_edge_len,
                               vbs_new,  
                               tsr_data,
                               new_data,
                               1,
                               bucket_offset, 
                               sr->mulid(),
                               sr->addid(),
                               sr);
      cdealloc(new_data);
    }
    for (int dim = 0;dim < order;dim++){
      cdealloc(bucket_offset[dim]);
    }
    cdealloc(bucket_offset);

    TAU_FSTOP(pack_virt_buf);

    if (reuse_buffers){
      if (swp_nval > old_dist.size){
        cdealloc(tsr_data);
        alloc_ptr(swp_nval*sr->el_size, (void**)&tsr_data);
      }
      send_buffer = tsr_cyclic_data;
      recv_buffer = tsr_data;
    }

    /* Communicate data */
    TAU_FSTART(ALL_TO_ALL_V);
    ord_glb_comm.all_to_allv(send_buffer, send_counts, send_displs, sr->el_size,
                             recv_buffer, recv_counts, recv_displs);
    TAU_FSTOP(ALL_TO_ALL_V);

    if (reuse_buffers)
      sr->set(tsr_cyclic_data, sr->addid(), swp_nval);
    else
      cdealloc(send_buffer);
    TAU_FSTART(unpack_virt_buf);
    /* Deserialize data into correctly ordered virtual sub blocks */
    if (recv_displs[ord_glb_comm.np-1] + recv_counts[ord_glb_comm.np-1] > 0){
      char **new_data; alloc_ptr(sizeof(char*)*np, (void**)&new_data);
      for (int64_t p = 0;p < np;p++){
        new_data[p] = recv_buffer+recv_displs[p]*sr->el_size;
      }
      bucket_offset = 
        compute_bucket_offsets( new_dist,
                                old_dist,
                                real_edge_len,
                                new_phys_edge_len,
                                new_virt_lda,
                                new_offsets,
                                new_permutation,
                                old_phys_edge_len,
                                old_virt_lda,
                                0,
                                new_nvirt,
                                old_nvirt,
                                new_virt_edge_len);

      pad_cyclic_pup_virt_buff(sym,
                               new_dist, 
                               old_dist, 
                               real_edge_len,
                               new_phys_dim,
                               new_phys_edge_len,
                               new_virt_edge_len,
                               vbs_new,
                               new_offsets,
                               new_permutation,
                               np,
                               old_phys_dim,
                               old_phys_edge_len,
                               old_virt_edge_len,
                               vbs_old,  
                               tsr_cyclic_data,
                               new_data,
                               0,
                               bucket_offset, 
                               alpha,
                               beta,
                               sr);
      for (int dim = 0;dim < order;dim++){
        cdealloc(bucket_offset[dim]);
      }
      cdealloc(bucket_offset);
      cdealloc(new_data);
    }
    TAU_FSTOP(unpack_virt_buf);

    if (!reuse_buffers) cdealloc(recv_buffer);
    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    *ptr_tsr_data = tsr_data;

    cdealloc(real_edge_len);
    cdealloc(old_phys_dim);
    cdealloc(new_phys_dim);
    cdealloc(hsym);
    cdealloc(idx);
    cdealloc(idx_offs);
    cdealloc(old_virt_lda);
    cdealloc(new_virt_lda);
    cdealloc(recv_counts);
    cdealloc(send_counts);
    cdealloc(send_displs);
    cdealloc(recv_displs);
    cdealloc(old_sub_edge_len);
    cdealloc(new_sub_edge_len);
    cdealloc(new_virt_edge_len);
    cdealloc(old_virt_edge_len);
    cdealloc(new_phys_edge_len);
    cdealloc(old_phys_edge_len);

    TAU_FSTOP(cyclic_reshuffle);

  }
}
