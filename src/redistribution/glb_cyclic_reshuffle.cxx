/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "glb_cyclic_reshuffle.h"
#include "../shared/util.h"


namespace CTF_int {
  void glb_ord_pup(int const *          sym,
                   distribution const & old_dist,
                   distribution const & new_dist,
                   int const *          len,
                   int const *          old_phys_dim,
                   int const *          old_phys_edge_len,
                   int const *          old_virt_edge_len,
                   int64_t              old_virt_nelem,
                   int const *          old_offsets,
                   int * const *        old_permutation,
                   int                  total_np,
                   int const *          new_phys_dim,
                   int const *          new_phys_edge_len,
                   int const *          new_virt_edge_len,
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

#ifdef USE_OMP
//    ASSERT(0);
  //  assert(0);
#endif
    TAU_FSTART(cyclic_pup_bucket);
  #ifdef USSSE_OMP
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

    int *offs; alloc_ptr(sizeof(int)*old_dist.order, (void**)&offs);
    if (old_offsets == NULL)
      for (int dim = 0;dim < old_dist.order;dim++) offs[dim] = 0;
    else 
      for (int dim = 0;dim < old_dist.order;dim++) offs[dim] = old_offsets[dim];

    int *ends; alloc_ptr(sizeof(int)*old_dist.order, (void**)&ends);
    for (int dim = 0;dim < old_dist.order;dim++) ends[dim] = len[dim];

  #ifdef USSSE_OMP
    int tid = omp_get_thread_num();
    int ntd = omp_get_num_threads();
    //partition the global tensor among threads, to preserve 
    //global ordering and load balance in partitioning
    int gidx_st[old_dist.order];
    int gidx_end[old_dist.order];
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
            printf("glb_idx_end = %ld, gidx_end[%d]= %d, len[%d] = %d\n", 
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
  #ifdef USSSE_OMP
    int64_t * count = par_virt_counts[tid];
  #else
    int64_t *count; alloc_ptr(sizeof(int64_t)*nbucket, (void**)&count);
    memset(count, 0, sizeof(int64_t)*nbucket);
  #endif

    int *gidx; alloc_ptr(sizeof(int)*old_dist.order, (void**)&gidx);
    memset(gidx, 0, sizeof(int)*old_dist.order);
    for (int dim = 0;dim < old_dist.order;dim++){
      gidx[dim] = old_dist.perank[dim];
    }

    int64_t *virt_offset; alloc_ptr(sizeof(int64_t)*old_dist.order, (void**)&virt_offset);
    memset(virt_offset, 0, sizeof(int64_t)*old_dist.order);

    int *idx; alloc_ptr(sizeof(int)*old_dist.order, (void**)&idx);
    memset(idx, 0, sizeof(int)*old_dist.order);

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

  #ifdef USSSE_OMP
    for (int dim=old_dist.order-1; dim>=0; dim--){
      int64_t iist = MAX(0,(gidx_st[dim]-old_dist.perank[dim]));
      int64_t ist = iist/old_dist.phase[dim];//(old_phys_dim[dim]*old_dist.virt_phase[dim]);
      if (sym[dim] != NS) ist = MIN(ist,idx[dim+1]);
      int plen[old_dist.order];
      memcpy(plen,old_virt_edge_len,old_dist.order*sizeof(int));
      int idim = dim;
      do {
        plen[idim] = ist;
        idim--;
      } while (idim >= 0 && sym[idim] != NS);
      gidx[dim] += ist*old_dist.phase[dim];//old_phys_dim[dim]*old_dist.virt_phase[dim];
      idx[dim] = ist;
      idx_acc[dim] = sy_packed_size(dim+1, plen, sym);
      offset += idx_acc[dim]; 

      ASSERT(ist == 0 || gidx[dim] <= gidx_st[dim]);
  //    ASSERT(ist < old_virt_edge_len[dim]);

      if (gidx[dim] > gidx_st[dim]) break;

      int64_t vst = iist-ist*old_dist.phase[dim];//*old_phys_dim[dim]*old_dist.virt_phase[dim];
      if (vst > 0 ){
        vst = MIN(old_dist.virt_phase[dim]-1,vst);
        gidx[dim] += vst*old_dist.phys_phase[dim];
        virt_offset[dim] = vst;
        offset += vst*old_virt_lda[dim];
      } else vst = 0;
      if (gidx[dim] > gidx_st[dim]) break;
    }
  #endif

    ASSERT(old_permutation == NULL);
    int rep_phase0 = lcm(old_phys_dim[0],new_phys_dim[0])/old_phys_dim[0];
  #if DEBUG >=2
    if (rank == 0)
      printf("rep_phase0 = %d\n",rep_phase0);
    for (int id=0; id<rep_phase0; id++){
      for (int jd=0; jd<(old_phys_edge_len[0]-id)/rep_phase0; jd++){
        printf("bucket_offset[%d] = %d\n",id+jd*rep_phase0,bucket_offset[0][id+jd*rep_phase0]);
        ASSERT(bucket_offset[0][id+jd*rep_phase0] == bucket_offset[0][id] || bucket_offset[0][id+jd*rep_phase0] == -1);
      }      
    }
  #endif

    bool done = false;
    for (;!done;){
      int64_t bucket0 = 0;
      bool outside0 = false;
      int len_zero_max = ends[0];
  #ifdef USSSE_OMP
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

      int idx_max = (sym[0] == NS ? old_virt_edge_len[0] : idx[1]+1);

      if (!outside0){
        int gidx_min = MAX(zero_len_toff,offs[0]);
        int gidx_max = (sym[0] == NS ? ends[0] : (sym[0] == SY ? gidx[1]+1 : gidx[1]));
        gidx_max = MIN(gidx_max, len_zero_max);

        int idx0 = MAX(0,(gidx_min-gidx[0])/old_phys_dim[0]);
        //int vidx0 = idx0%old_dist.virt_phase[0];
        int idx1 = MAX(0,(gidx_max-gidx[0]+old_phys_dim[0]-1)/old_phys_dim[0]);
        int lencp = MIN(rep_phase0,idx1-idx0);
        ASSERT(is_copy);
        if (forward){
          for (int ia=0; ia<lencp; ia++){
            int64_t bucket = bucket0+bucket_offset[0][idx0];//((vidx0+ia)%old_dist.virt_phase[0])*old_virt_edge_len[0]+idx0/old_dist.virt_phase[0]];
            sr->copy((idx1-idx0+rep_phase0-1)/rep_phase0, 
                     old_data+ sr->el_size*(offset+idx0), rep_phase0,
                     new_data[bucket]+sr->el_size*count[bucket], 1);
            count[bucket]+=(idx1-idx0+rep_phase0-1)/rep_phase0;
#if DEBUG>=1
          //  printf("[%d] gidx[0]=%d,gidx_min=%d,idx0=%d,gidx_max=%d,idx1=%d,bucket=%d,len=%d\n",rank,gidx[0],gidx_min,idx0,gidx_max,idx1,bucket,(idx1-idx0+rep_phase0-1)/rep_phase0);
#endif
            idx0++;
          }
        } else {
          for (int ia=0; ia<lencp; ia++){
            int64_t bucket = bucket0+bucket_offset[0][idx0];//((vidx0+ia)%old_dist.virt_phase[0])*old_virt_edge_len[0]+idx0/old_dist.virt_phase[0]];
            sr->copy((idx1-idx0+rep_phase0-1)/rep_phase0, 
                     new_data[bucket]+sr->el_size*count[bucket], 1,
                     old_data+ sr->el_size*(offset+idx0), rep_phase0);
            count[bucket]+=(idx1-idx0+rep_phase0-1)/rep_phase0;
            //printf("-r- gidx[0]=%d,gidx_min=%d,idx0=%d,gidx_max=%d,idx1=%d,bucket=%d,len=%d\n",gidx[0],gidx_min,idx0,gidx_max,idx1,bucket,(idx1-idx0+rep_phase0-1)/rep_phase0);
            idx0++;
          }
        }
/*
        gidx_max = MIN(gidx_max, len_zero_max);
        int64_t moffset = offset;
        for (idx[0] = idx_st;idx[0] < idx_max;idx[0]++){
          int virt_min = MAX(0,MIN(old_dist.virt_phase[0],gidx_min-gidx[0]));
          int virt_max = MAX(0,MIN(old_dist.virt_phase[0],gidx_max-gidx[0]));

          moffset += virt_min;
          if (forward){
            ASSERT(is_copy);
            for (virt_offset[0] = virt_min*old_virt_edge_len[0];
                 virt_offset[0] < virt_max*old_virt_edge_len[0];
                 virt_offset[0] += old_virt_edge_len[0])
            {
              int64_t bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
  #ifdef USSSE_OMP
              bucket_store[moffset] = bucket;
              count_store[moffset]  = count[bucket]++;
              thread_store[moffset] = tid;
  #else
//              printf("[%d] bucket = %d offset = %ld\n", rank, bucket, offset);
  //            printf("[%d] count[bucket] = %d, nbucket = %d\n", rank, count[bucket]+1, nbucket);
    //          std::cout << "old_data[offset]=";
      //        sr->print(old_data+ sr->el_size*offset);
              sr->copy(new_data[bucket]+sr->el_size*(count[bucket]++), old_data+ sr->el_size*moffset);
//              std::cout << "\nnew_data[bucket][count[bucket]++]=";
  //            sr->print(new_data[bucket]+sr->el_size*(count[bucket]-1));
    //          std::cout << "\n";
  #endif
              moffset++;
            }
          }
          else{
            for (virt_offset[0] = virt_min*old_virt_edge_len[0];
                 virt_offset[0] < virt_max*old_virt_edge_len[0];
                 virt_offset[0] += old_virt_edge_len[0])
            {
              int64_t bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
  #ifdef USSSE_OMP
              bucket_store[moffset] = bucket;
              count_store[moffset]  = count[bucket]++;
              thread_store[moffset] = tid;
  #else
              if (is_copy) 
                sr->copy(old_data+sr->el_size*moffset,       new_data[bucket]+sr->el_size*(count[bucket]++));
              else 
                sr->acc( old_data+sr->el_size*moffset, beta, new_data[bucket]+sr->el_size*(count[bucket]++), alpha);
//              old_data[moffset] = beta*old_data[moffset] + alpha*new_data[bucket][count[bucket]++];
  #endif
              moffset ++;
            }
          }
          gidx[0] += old_phys_dim[0]*old_dist.virt_phase[0];
        }
        gidx[0] -= idx_max*old_phys_dim[0]*old_dist.virt_phase[0];
      */
      }
      offset += idx_max*old_dist.virt_phase[0];
       
      idx[0] = 0;

      zero_len_toff = 0;

      /* Adjust outer indices */
      if (!done){
        for (int dim = 1;dim < old_dist.order;dim++){
          virt_offset[dim] += 1; //old_virt_edge_len[dim];
          gidx[dim]+=old_dist.phys_phase[dim];

          if (virt_offset[dim] == old_dist.virt_phase[dim]){
            gidx[dim] -= old_dist.phase[dim];
            virt_offset[dim] = 0;

            gidx[dim] -= idx[dim]*old_dist.phase[dim];//phys_dim[dim]*old_dist.virt_phase[dim];
            idx[dim]++;

            if (idx[dim] == (sym[dim] == NS ? old_virt_edge_len[dim] : idx[dim+1]+1)){
              //index should always be zero here sicne everything is SY and not SH
              idx[dim] = 0;//(dim == 0 || sym[dim-1] == NS ? 0 : idx[dim-1]);
              //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];

              if (dim == old_dist.order-1) done = true;
            }
            else{
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

  #ifndef USSSE_OMP
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
   
  #ifndef USSSE_OMP
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


  static inline
  int64_t sy_packed_offset(int dim, int const * len, int idx, int const * sym){
    if (idx == 0) return 0;
    if (sym[dim-1] == NS){
      return sy_packed_size(dim, len, sym)*idx;
    } else {
      int i=1;
      int ii=1;
      int iidx = idx;
      int64_t offset = iidx;
      do {
        i++;
        ii*=i;
        iidx++;
        offset *= iidx;
      } while (i<=dim && sym[dim-i] != NS);
      return (offset/ii)*sy_packed_size(dim-i+1,len,sym);

    }
  }

  template <int idim>
  void ord_glb(int const *          sym,
               distribution const & dist,
               int const *          virt_edge_len,
               int const *          virt_phase_lda,
               int64_t              vbs,
               bool                 dir,
               char const *         tsr_data_in,
               char *               tsr_data_out,
               algstrct const *     sr,
               int                  prev_idx=0,
               int64_t              glb_ord_offset=0,
               int64_t              blk_ord_offset=0){
   
    int imax=virt_edge_len[idim];
    if (sym[idim] != NS) imax = prev_idx+1;
    int vp_stride = virt_phase_lda[idim]*dist.virt_phase[idim];
    for (int i=0; i<imax; i++){
      int64_t dim_offset = sy_packed_offset(idim, virt_edge_len, i, sym);
      int64_t i_blk_ord_offset = blk_ord_offset + dim_offset;
      int64_t i_glb_ord_offset = glb_ord_offset + dim_offset*vp_stride;
      for (int v=0; v<dist.virt_phase[idim]; v++){
        int64_t iv_blk_ord_offset = i_blk_ord_offset + v*virt_phase_lda[idim]*vbs;
        int64_t iv_glb_ord_offset = i_glb_ord_offset;
        if (v>0){
          int64_t glb_vrt_offset = sy_packed_offset(idim, virt_edge_len, i+1, sym);
          iv_glb_ord_offset += (glb_vrt_offset-dim_offset)*virt_phase_lda[idim]*v;
        }
        ord_glb<idim-1>(sym, dist, virt_edge_len, virt_phase_lda, vbs, dir, tsr_data_in, tsr_data_out, sr, i, iv_glb_ord_offset, iv_blk_ord_offset);
      }
    }
  }

  template <>
  inline void ord_glb<0>(int const *          sym,
                         distribution const & dist,
                         int const *          virt_edge_len,
                         int const *          virt_phase_lda,
                         int64_t              vbs,
                         bool                 dir,
                         char const *         tsr_data_in,
                         char *               tsr_data_out,
                         algstrct const *     sr,
                         int                  prev_idx,
                         int64_t              glb_ord_offset,
                         int64_t              blk_ord_offset){
    int imax=virt_edge_len[0];
    if (sym[0] != NS) imax = prev_idx+1;
    for (int v=0; v<dist.virt_phase[0]; v++){
      if (dir){
        sr->copy(imax, tsr_data_in  + sr->el_size*(blk_ord_offset+v*vbs), 1, 
                       tsr_data_out + sr->el_size*(glb_ord_offset+v), dist.virt_phase[0]);
      } else {
        sr->copy(imax, tsr_data_in  + sr->el_size*(glb_ord_offset+v), dist.virt_phase[0], 
                       tsr_data_out + sr->el_size*(blk_ord_offset+v*vbs), 1);
      }
    }
  }

  template 
  void ord_glb<7>(int const *          sym,
                  distribution const & dist,
                  int const *          virt_edge_len,
                  int const *          virt_phase_lda,
                  int64_t              vbs,
                  bool                 dir,
                  char const *         tsr_data_in,
                  char *               tsr_data_out,
                  algstrct const *     sr,
                  int                  prev_idx,
                  int64_t              glb_ord_offset,
                  int64_t              blk_ord_offset);

  template <int idim>
  void ord_glb_omp(int const *          sym,
                   distribution const & dist,
                   int const *          virt_edge_len,
                   int const *          virt_phase_lda,
                   int64_t              vbs,
                   bool                 dir,
                   char const *         tsr_data_in,
                   char *               tsr_data_out,
                   algstrct const *     sr,
                   int const *          idx_st,
                   int const *          idx_end,
                   int                  prev_idx=0,
                   int64_t              glb_ord_offset=0,
                   int64_t              blk_ord_offset=0){
    int imax=virt_edge_len[idim];
    if (sym[idim] != NS) imax = prev_idx+1;
    if (idx_end != NULL)
      imax = MIN(imax,idx_end[idim]+1);
    int ist;
    if (idx_st != NULL) 
      ist = idx_st[idim];
    else 
      ist = 0;
    int vp_stride = virt_phase_lda[idim]*dist.virt_phase[idim];
    // first iteration
    for (int i=ist; i<imax; i++){
      int64_t dim_offset = sy_packed_offset(idim, virt_edge_len, i, sym);
      int64_t i_blk_ord_offset = blk_ord_offset + dim_offset;
      int64_t i_glb_ord_offset = glb_ord_offset + dim_offset*vp_stride;
      for (int v=0; v<dist.virt_phase[idim]; v++){
        int64_t iv_blk_ord_offset = i_blk_ord_offset + v*virt_phase_lda[idim]*vbs;
        int64_t iv_glb_ord_offset = i_glb_ord_offset;
        if (v>0){
          int64_t glb_vrt_offset = sy_packed_offset(idim, virt_edge_len, i+1, sym);
          iv_glb_ord_offset += (glb_vrt_offset-dim_offset)*virt_phase_lda[idim]*v;
        }
        if (i==ist && i==imax-1)
          ord_glb_omp<idim-1>(sym, dist, virt_edge_len, virt_phase_lda, vbs, dir, tsr_data_in, tsr_data_out, sr, idx_st, idx_end, i, iv_glb_ord_offset, iv_blk_ord_offset);
        else if (i==ist)
          ord_glb_omp<idim-1>(sym, dist, virt_edge_len, virt_phase_lda, vbs, dir, tsr_data_in, tsr_data_out, sr, idx_st, NULL,    i, iv_glb_ord_offset, iv_blk_ord_offset);
        else if (i==imax-1)
          ord_glb_omp<idim-1>(sym, dist, virt_edge_len, virt_phase_lda, vbs, dir, tsr_data_in, tsr_data_out, sr, NULL,   idx_end, i, iv_glb_ord_offset, iv_blk_ord_offset);
        else
          ord_glb<idim-1>(sym, dist, virt_edge_len, virt_phase_lda, vbs, dir, tsr_data_in, tsr_data_out, sr, i, iv_glb_ord_offset, iv_blk_ord_offset);
      }
    }
  }

  template <>
  void ord_glb_omp<0>(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      bool                 dir,
                      char const *         tsr_data_in,
                      char *               tsr_data_out,
                      algstrct const *     sr,
                      int const *          idx_st,
                      int const *          idx_end,
                      int                  prev_idx,
                      int64_t              glb_ord_offset,
                      int64_t              blk_ord_offset){
    ord_glb<0>(sym,dist,virt_edge_len,virt_phase_lda,vbs,dir,tsr_data_in,tsr_data_out,sr,prev_idx,glb_ord_offset,blk_ord_offset);
  }

  template
  void ord_glb_omp<7>(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      bool                 dir,
                      char const *         tsr_data_in,
                      char *               tsr_data_out,
                      algstrct const *     sr,
                      int const *          idx_st,
                      int const *          idx_end,
                      int                  prev_idx,
                      int64_t              glb_ord_offset,
                      int64_t              blk_ord_offset);


  void order_globally(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      bool                 dir,
                      char const *         tsr_data_in,
                      char *               tsr_data_out,
                      algstrct const *     sr){
    TAU_FSTART(order_globally);
    ASSERT(dist.order != 0);
    if (dist.order == 1){
      return ord_glb<0>(sym,dist,virt_edge_len,virt_phase_lda,vbs,dir,tsr_data_in,tsr_data_out,sr);
    }
    if (dist.order <= 8){
    //int rank;
    //MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifdef USE_OMP
    #pragma omp parallel
    {
    int tid = omp_get_thread_num();
    int ntd = omp_get_num_threads();
    int64_t vbs_chunk = vbs/ntd;
    int64_t fidx_st = vbs_chunk*tid + MIN(tid,vbs%ntd);
    //limit (index of next thread)
    int64_t fidx_end = fidx_st + vbs_chunk;
    if (tid < vbs%ntd) fidx_end++;
    int * idx_st = (int*)alloc(dist.order*sizeof(int));
    int * idx_end = (int*)alloc(dist.order*sizeof(int));
    sy_calc_idx_arr(dist.order, virt_edge_len, sym, fidx_st, idx_st);
    sy_calc_idx_arr(dist.order, virt_edge_len, sym, fidx_end, idx_end);
    int idim=1;
    bool cont=true;
    do {
      ASSERT(idim<dist.order);
      idx_end[idim]--;
      if (idx_end[idim] < 0 && idim+1<dist.order){ 
        idx_end[idim] = virt_edge_len[idim]-1;
        idim++;
      } else cont=false;
    } while (cont);
    /*if (rank == 0){
      for (int itid =0; itid< ntd; itid++){
        #pragma omp barrier
        if (itid==tid){
          for (int i=0; i<dist.order; i++){
            printf("[%d] idx_st[%d] = %d, idx_end[%d] = %d, pad_edge_len[%d] = %d\n", tid, i, idx_st[i], i, idx_end[i], i, virt_edge_len[i]);
          }
        }
      } 
    }*/
  #define CASE_ORD_GLB(n)                                                                                         \
    case n:                                                                                                       \
      ord_glb_omp<n-1>(sym,dist,virt_edge_len,virt_phase_lda,vbs,dir,tsr_data_in,tsr_data_out,sr,idx_st,idx_end); \
      break;                                                   
#else
  #define CASE_ORD_GLB(n)                                                                      \
    case n:                                                                                    \
      ord_glb<n-1>(sym,dist,virt_edge_len,virt_phase_lda,vbs,dir,tsr_data_in,tsr_data_out,sr); \
      break;                                                   
#endif
      switch (dist.order){
        CASE_ORD_GLB(1)
        CASE_ORD_GLB(2)
        CASE_ORD_GLB(3)
        CASE_ORD_GLB(4)
        CASE_ORD_GLB(5)
        CASE_ORD_GLB(6)
        CASE_ORD_GLB(7)
        CASE_ORD_GLB(8)
        default:
          assert(0);
          break;
      }
  #undef CASE_ORD_GLB
#ifdef USE_OMP
    }
#endif
      TAU_FSTOP(order_globally);
      return;
    }
    int order = dist.order;
    int * virt_idx = (int*)alloc(order*sizeof(int));
    int * idx = (int*)alloc(order*sizeof(int));
    
    std::fill(virt_idx, virt_idx+order, 0);
    for (;;){
      std::fill(idx, idx+order, 0);

      //int _rank;
      //MPI_Comm_rank(MPI_COMM_WORLD, &_rank);
      for (;;){
        int64_t glb_ord_offset = virt_idx[0];
        int64_t blk_ord_offset = virt_idx[0]*vbs;
        for (int idim=1; idim<order; idim++){
          //calculate offset within virtual block
          int64_t dim_offset = sy_packed_offset(idim, virt_edge_len, idx[idim], sym);
          //when each virtual block is stored contiguously, this is the offset within the glock
          blk_ord_offset += dim_offset;
          blk_ord_offset += virt_idx[idim]*virt_phase_lda[idim]*vbs;
          //when the virtual blocks are interleaved according to global order, this is a part of the
          // offset and needs to be scaled by all smaller virtualization factors
          glb_ord_offset += dim_offset*virt_phase_lda[idim]*dist.virt_phase[idim];
          //an dditional offset is needed for the global ordering, if this is not the first virtual 
          // block along this dimension, in which case we must offset according to all elements in
          // smaller virtual block with the same idim and greater indices
        //if (_rank == 0)
        //  printf("idim = %d, idx[idim] = %d blk_ord_offset = %ld glb_ord_offset = %ld\n",idim,idx[idim],blk_ord_offset,glb_ord_offset);
          if (virt_idx[idim] > 0){
            int64_t glb_vrt_offset = sy_packed_offset(idim, virt_edge_len, idx[idim]+1, sym);
            glb_ord_offset += (glb_vrt_offset-dim_offset)*virt_phase_lda[idim]*virt_idx[idim];
       // if (_rank == 0)
        //  printf("idim = %d virt add glb_ord_offset = %ld\n",idim,glb_ord_offset);
          }
        }
        
        int n = virt_edge_len[0];
        if (sym[0] != NS) n = idx[1]+1;
        /*if (_rank == 0){
          printf("blk_ord_offset = %ld, glb_ord_offset = %ld\n",blk_ord_offset,glb_ord_offset);
          for (int _i=0; _i<order; _i++){
            printf("idx[%d] = %d virt_idx[%d]=%d\n",_i,idx[_i],_i,virt_idx[_i]);
          }
          for (int _i=0; _i<n; _i++){
            if (dir){
              printf("Writing [%ld] ",blk_ord_offset+_i);
              sr->print(tsr_data_in+sr->el_size*(blk_ord_offset+_i));
              printf(" to [%ld] ", glb_ord_offset+_i*dist.virt_phase[0]);
              sr->print(tsr_data_out+sr->el_size*(glb_ord_offset+_i*dist.virt_phase[0]));
              printf("\n");
            } else {
              printf("Writing [%ld] ", glb_ord_offset+_i*dist.virt_phase[0]);
              sr->print(tsr_data_in+sr->el_size*(glb_ord_offset+_i*dist.virt_phase[0]));
              printf(" to [%ld] ",blk_ord_offset+_i);
              sr->print(tsr_data_out+sr->el_size*(blk_ord_offset+_i));
              printf("\n");
            }
          }}*/
        if (dir){
          sr->copy(n, tsr_data_in+sr->el_size*blk_ord_offset, 1, tsr_data_out+sr->el_size*glb_ord_offset, dist.virt_phase[0]);
        } else {
          sr->copy(n, tsr_data_in+sr->el_size*glb_ord_offset, dist.virt_phase[0], tsr_data_out+sr->el_size*blk_ord_offset, 1);
        }

        int dim=1;
        bool exit, finish=false;
        do {
          if (dim==order){
            exit = true;
            finish = true;
          } else {
            if (idx[dim] == virt_edge_len[dim]-1 || (sym[dim] != NS && idx[dim] == idx[dim+1])){
              idx[dim] = 0;
              dim++;
              exit = false;
            } else {
              idx[dim]++;
              exit = true;
            }
          }
        } while (!exit);
        if (finish) break;
      }

      int dim=0;
      bool exit, finish=false;
      do {
        if (dim==order){
          exit = true;
          finish = true;
        } else {
          if (virt_idx[dim] == dist.virt_phase[dim]-1){
            virt_idx[dim] = 0;
            dim++;
            exit = false;
          } else {
            virt_idx[dim]++;
            exit = true;
          }
        }
      } while (!exit);
      if (finish) break;
    }
    cdealloc(idx);
    cdealloc(virt_idx);
    TAU_FSTOP(order_globally);
  }

//  void
  char *
       glb_cyclic_reshuffle(int const *          sym,
                            distribution const & old_dist,
                            int const *          old_offsets,
                            int * const *        old_permutation,
                            distribution const & new_dist,
                            int const *          new_offsets,
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
    int * idx;
    int64_t * idx_offs;
    int64_t  * send_displs;
    int64_t * recv_displs;
    int * new_virt_lda, * old_virt_lda;
    int * old_sub_edge_len, * new_sub_edge_len;
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
      return tsr_cyclic_data;
    }
    
    ASSERT(!reuse_buffers || sr->isequal(beta, sr->addid()));
    ASSERT(old_dist.is_cyclic&&new_dist.is_cyclic);

    TAU_FSTART(cyclic_reshuffle);
      np = ord_glb_comm.np;

    alloc_ptr(order*sizeof(int),     (void**)&hsym);
    alloc_ptr(order*sizeof(int),     (void**)&idx);
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
    alloc_ptr(order*sizeof(int), (void**)&old_sub_edge_len);
    alloc_ptr(order*sizeof(int), (void**)&new_sub_edge_len);
    int ** bucket_offset;
    
    int *real_edge_len; alloc_ptr(sizeof(int)*order, (void**)&real_edge_len);
    for (i=0; i<order; i++) real_edge_len[i] = old_dist.pad_edge_len[i]-old_dist.padding[i];
   
    int *old_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_phys_edge_len);
    for (int dim = 0;dim < order;dim++) old_phys_edge_len[dim] = (real_edge_len[dim]+old_dist.padding[dim])/old_dist.phys_phase[dim];

    int *new_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_phys_edge_len);
    for (int dim = 0;dim < order;dim++) new_phys_edge_len[dim] = (real_edge_len[dim]+new_dist.padding[dim])/new_dist.phys_phase[dim];

    int *old_virt_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_virt_edge_len);
    for (int dim = 0;dim < order;dim++) old_virt_edge_len[dim] = old_phys_edge_len[dim]/old_dist.virt_phase[dim];

    int *new_virt_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_virt_edge_len);
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

/*    bool is_AS = false;
    for (int asd=0; asd<order; asd++){
      if (sym[asd] == AS || sym[asd] == SH){
        is_AS =true;
      }
    }
    if (!is_AS){
      int64_t send_counts2[np];
      calc_drv_displs(sym, real_edge_len, old_phys_edge_len, old_dist, new_dist, send_counts2, ord_glb_comm, idx_lyr);
      
      TAU_FSTOP(calc_cnt_displs);
      bool is_same = true;
      for (i=0; i<np; i++){
        //printf("[%d] send_counts[%d] = %ld send_counts2[%d] = %ld\n", ord_glb_comm.rank, i, send_counts[i], i, send_counts2[i]);
        if (send_counts[i] != send_counts2[i]) is_same = false;
      }
      assert(is_same);
    }*/
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
      recv_buffer = tsr_cyclic_data;
    } else {
      alloc_ptr(old_dist.size*sr->el_size, (void**)&send_buffer);
      alloc_ptr(swp_nval*sr->el_size, (void**)&recv_buffer);
    }
    ASSERT(reuse_buffers);
    TAU_FSTART(pack_virt_buf);
    if (idx_lyr == 0){
      order_globally(sym, old_dist, old_virt_edge_len, old_virt_lda, vbs_old, 1, tsr_data, tsr_cyclic_data, sr);
      char **new_data; alloc_ptr(sizeof(char*)*np, (void**)&new_data);
      for (int64_t p = 0;p < np;p++){
        new_data[p] = tsr_data+sr->el_size*send_displs[p];
      }

      glb_ord_pup(sym,
                  old_dist, 
                  new_dist, 
                  real_edge_len,
                  old_dist.phys_phase,
                  old_phys_edge_len,
                  old_virt_edge_len,
                  vbs_old,
                  old_offsets,
                  old_permutation,
                  np,
                  new_dist.phys_phase,
                  new_phys_edge_len,
                  new_virt_edge_len,
                  vbs_new,  
                  tsr_cyclic_data,
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
      recv_buffer = tsr_cyclic_data;
      send_buffer = tsr_data;
    }
    return tsr_data;
  }
#if 0

    /* Communicate data */
    TAU_FSTART(ALL_TO_ALL_V);
    ord_glb_comm.all_to_allv(send_buffer, send_counts, send_displs, sr->el_size,
                             recv_buffer, recv_counts, recv_displs);
    TAU_FSTOP(ALL_TO_ALL_V);

    if (reuse_buffers){
      if (swp_nval > old_dist.size){
        cdealloc(tsr_data);
        alloc_ptr(swp_nval*sr->el_size, (void**)&tsr_data);
      }
    }
    TAU_FSTART(unpack_virt_buf);
    /* Deserialize data into correctly ordered virtual sub blocks */
    if (recv_displs[ord_glb_comm.np-1] + recv_counts[ord_glb_comm.np-1] > 0){
      sr->set(tsr_data, sr->addid(), swp_nval);
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


      glb_ord_pup(sym,
                  new_dist, 
                  old_dist, 
                  real_edge_len,
                  new_dist.phys_phase,
                  new_phys_edge_len,
                  new_virt_edge_len,
                  vbs_new,
                  new_offsets,
                  new_permutation,
                  np,
                  old_dist.phys_phase,
                  old_phys_edge_len,
                  old_virt_edge_len,
                  vbs_old,  
                  tsr_data,
                  new_data,
                  0,
                  bucket_offset, 
                  alpha,
                  beta,
                  sr);


      order_globally(sym, new_dist, new_virt_edge_len, new_virt_lda, vbs_new, 0, tsr_data, tsr_cyclic_data, sr);
      for (int dim = 0;dim < order;dim++){
        cdealloc(bucket_offset[dim]);
      }
      cdealloc(bucket_offset);
      cdealloc(new_data);
    } else {
      sr->set(tsr_cyclic_data, sr->addid(), swp_nval);
    }
    TAU_FSTOP(unpack_virt_buf);

    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    *ptr_tsr_data = tsr_data;

    cdealloc(real_edge_len);
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
#endif
}
