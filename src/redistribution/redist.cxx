/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "redist.h"
#include "../shared/util.h"
#include "sparse_rw.h"

namespace CTF_int {
  void padded_reshuffle(int const *          sym,
                        distribution const & old_dist,
                        distribution const & new_dist,
                        char *               tsr_data,
                        char **              tsr_cyclic_data,
                        algstrct const *     sr,
                        CommData             ord_glb_comm){
    int old_num_virt, new_num_virt, numPes;
    int64_t new_nval, swp_nval;
    int idx_lyr;
    int * virt_phase_rank, * old_virt_phase_rank, * sub_edge_len;
    char * pairs, * tsr_new_data;
    DEBUG_PRINTF("Performing padded reshuffle\n");

    TAU_FSTART(padded_reshuffle);

    numPes = ord_glb_comm.np;

    alloc_ptr(old_dist.order*sizeof(int), (void**)&virt_phase_rank);
    alloc_ptr(old_dist.order*sizeof(int), (void**)&old_virt_phase_rank);
    alloc_ptr(old_dist.order*sizeof(int), (void**)&sub_edge_len);

    new_num_virt = 1;
    old_num_virt = 1;
    idx_lyr = ord_glb_comm.rank;
    for (int i=0; i<old_dist.order; i++){
      old_num_virt = old_num_virt*old_dist.virt_phase[i];
      new_num_virt = new_num_virt*new_dist.virt_phase[i];
      virt_phase_rank[i] = new_dist.perank[i]*new_dist.virt_phase[i];
      old_virt_phase_rank[i] = old_dist.perank[i]*old_dist.virt_phase[i];
      idx_lyr -= old_dist.perank[i]*old_dist.pe_lda[i];
    }
    if (idx_lyr == 0 ){
      read_loc_pairs(old_dist.order, old_dist.size, old_num_virt, sym,
                     old_dist.pad_edge_len, old_dist.padding, old_dist.virt_phase,
                     old_dist.phase, old_virt_phase_rank, &new_nval, tsr_data,
                     &pairs, sr);
    } else {
      new_nval = 0;
      pairs = NULL;
    }

  #if DEBUG >= 1
    int64_t old_size = sy_packed_size(old_dist.order, new_dist.pad_edge_len, sym);
  #endif

    for (int i=0; i<old_dist.order; i++){
      sub_edge_len[i] = new_dist.pad_edge_len[i] / new_dist.phase[i];
    }
    if (ord_glb_comm.rank == 0){
      DPRINTF(1,"Tensor now has virtualization factor of %d\n",new_num_virt);
    }
    swp_nval = new_num_virt*sy_packed_size(old_dist.order, sub_edge_len, sym);
    if (ord_glb_comm.rank == 0){
      DPRINTF(1,"Tensor is of size %ld, has factor of %lf growth due to padding\n", 
            swp_nval,
            ord_glb_comm.np*(swp_nval/(double)old_dist.size));
    }

    alloc_ptr(swp_nval*sr->el_size, (void**)&tsr_new_data);

    sr->set(tsr_new_data, sr->addid(), swp_nval);


    wr_pairs_layout(old_dist.order,
                    numPes,
                    new_nval,
                    sr->mulid(),
                    sr->addid(),
                    'w',
                    new_num_virt,
                    sym,
                    new_dist.pad_edge_len,
                    new_dist.padding,
                    new_dist.phase,
                    new_dist.virt_phase,
                    virt_phase_rank,
                    new_dist.pe_lda,
                    pairs,
                    tsr_new_data,
                    ord_glb_comm,
                    sr);
                  
    *tsr_cyclic_data = tsr_new_data;

    cfree(old_virt_phase_rank);
    if (pairs != NULL)
      cfree(pairs);
    cfree(virt_phase_rank);
    cfree(sub_edge_len);
    TAU_FSTOP(padded_reshuffle);
  }


  int ** compute_bucket_offsets(distribution const & old_dist,
                                distribution const & new_dist,
                                int const *          len,
                                int const *          old_phys_edge_len,
                                int const *          old_virt_lda,
                                int const *          old_offsets,
                                int * const *        old_permutation,
                                int const *          new_phys_edge_len,
                                int const *          new_virt_lda,
                                int                  forward,
                                int                  old_virt_np,
                                int                  new_virt_np,
                                int const *          old_virt_edge_len){
    TAU_FSTART(compute_bucket_offsets);
    
    int **bucket_offset; alloc_ptr(sizeof(int*)*old_dist.order, (void**)&bucket_offset);
    
    for (int dim = 0;dim < old_dist.order;dim++){
      alloc_ptr(sizeof(int)*old_phys_edge_len[dim], (void**)&bucket_offset[dim]);
      int pidx = 0;
      for (int vr = 0;vr < old_dist.virt_phase[dim];vr++){
        for (int vidx = 0;vidx < old_virt_edge_len[dim];vidx++,pidx++){
          int64_t _gidx = (int64_t)vidx*old_dist.phase[dim]+old_dist.perank[dim]*old_dist.virt_phase[dim]+(int64_t)vr;
          //int64_t _gidx = (vr*old_virt_edge_len[dim]+(int64_t)vidx)*old_dist.phase[dim]/old_dist.virt_phase[dim]+old_dist.perank[dim];
          int64_t gidx;
          if (_gidx > len[dim] || (old_offsets != NULL && _gidx < old_offsets[dim])){
            gidx = -1;
//            printf("_gidx=%ld, len[%d]=%d, vidx=%d, vr=%d, old_phase=%d, old_perank =%d, old_virt_phase=%d\n",_gidx,dim,len[dim],vidx,vr,old_dist.phase[dim], old_dist.perank[dim[,old_dist.virt_phase[dim]);
          } else {
            if (old_permutation == NULL || old_permutation[dim] == NULL){
              gidx = _gidx;
            } else {
              gidx = old_permutation[dim][_gidx];
            }
          }
          if (gidx != -1){
            //int phys_rank = gidx%(new_dist.phase[dim]/new_dist.virt_phase[dim]);
            int total_rank = gidx%new_dist.phase[dim];
            int phys_rank = total_rank/new_dist.virt_phase[dim];
            if (forward){
              //int virt_rank = total_rank%new_dist.virt_phase[dim];
              //int virt_rank = gidx/(new_phys_edge_len[dim]*new_dist.phase[dim]/new_dist.virt_phase[dim]);
              //bucket_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim])*new_virt_np+
              //               virt_rank*new_virt_lda[dim];
              bucket_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim]);
              //printf("f %d - %d %d %d - %d - %d %d %d - %d\n", dim, vr, vidx, pidx, gidx, total_rank,
              //    phys_rank, virt_rank, bucket_offset[dim][pidx]);
            }
            else{
              //bucket_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim])*old_virt_np+
              //               vr*old_virt_lda[dim];
              bucket_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim]);
              //printf("r %d - %d %d %d - %d - %d %d - %d\n", dim, vr, vidx, pidx, gidx, total_rank,
              //    phys_rank, bucket_offset[dim][pidx]);
            }
          } else {
            bucket_offset[dim][pidx] = -1;
          }
        }
      }
    }

    TAU_FSTOP(compute_bucket_offsets);

    return bucket_offset;
  }




  void calc_cnt_displs(int const *          sym,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int                  new_nvirt,
                       int                  np,
                       int const *          old_virt_edge_len,
                       int const *          new_virt_lda,
                       int64_t *            send_counts,
                       int64_t *            recv_counts,
                       int64_t *            send_displs,
                       int64_t *            recv_displs,
                       CommData             ord_glb_comm,
                       int                  idx_lyr,
                       int * const *        bucket_offset){
    int64_t *  all_virt_counts;

  #ifdef USE_OMP
    int act_omp_ntd;
    int64_t vbs = sy_packed_size(old_dist.order, old_virt_edge_len, sym);
    int max_ntd = omp_get_max_threads();
    max_ntd = MAX(1,MIN(max_ntd,vbs/np));
  #else
    int max_ntd = 1;
  #endif
    
    mst_alloc_ptr(np*sizeof(int64_t)*max_ntd, (void**)&all_virt_counts);


    /* Count how many elements need to go to each new virtual bucket */
    if (idx_lyr==0){
      if (old_dist.order == 0){
        memset(all_virt_counts, 0, np*sizeof(int64_t));
        all_virt_counts[0]++;
      } else {
  #ifdef USE_OMP
  #pragma omp parallel num_threads(max_ntd)
        {
        int imax, act_max, skip;
        int start_ldim, end_ldim;
        int i_st, vc, dim;
        int64_t *  virt_counts;
        int * old_virt_idx, * virt_rank;
        int * idx;
        int64_t idx_offset;
        int64_t *  idx_offs;
        int * spad;
        int last_len = old_dist.pad_edge_len[old_dist.order-1]/old_dist.phase[old_dist.order-1]+1;
        int ntd, tid;
        ntd = omp_get_num_threads();
        tid = omp_get_thread_num();
/*
        int lidx_st[old_dist.order];
        int lidx_end[old_dist.order];
        if (old_dist.order > 1){
          int64_t loc_upsize = packed_size(old_dist.order-1, old_virt_edge_len+1, sym+1);
          int64_t chnk = loc_upsize/ntd;
          int64_t loc_idx_st = chnk*tid + MIN(tid,loc_upsize%ntd);
          int64_t loc_idx_end = loc_idx_st+chnk+(tid<(loc_upsize%ntd));
          //calculate global indices along each dimension corresponding to partition
      //    printf("loc_idx_st = %ld, loc_idx_end = %ld\n",loc_idx_st,loc_idx_end);
          calc_idx_arr(old_dist.order-1, len+1, sym+1, loc_idx_st, lidx_st+1);
          calc_idx_arr(old_dist.order-1, len+1, sym+1, loc_idx_end, lidx_end+1);
          lidx_st[0] = 0;
          //FIXME: wrong but evidently not used
          lidx_end[0] = 0
        } else {
          //FIXME the below means redistribution of a vector is non-threaded
          if (tid == 0){
            lidx_st[0] = 0;
            lidx_end[0] = ends[0];
          } else {
            lidx_st[0] = 0;
            lidx_end[0] = 0;
          }
        }*/

        virt_counts = all_virt_counts+np*tid;
        start_ldim = (last_len/ntd)*tid;
        start_ldim += MIN(tid,last_len%ntd);
        end_ldim = (last_len/ntd)*(tid+1);
        end_ldim += MIN(tid+1,last_len%ntd);
  #else
        {
        int imax, act_max, skip;
        int start_ldim, end_ldim, i_st, vc, dim;
        int64_t *  virt_counts;
        int64_t *  old_virt_idx, * virt_rank;
        int64_t *  idx;
        int64_t idx_offset;
        int64_t *  idx_offs;
        int * spad;
        int last_len = old_dist.pad_edge_len[old_dist.order-1]/old_dist.phase[old_dist.order-1]+1;
        virt_counts = all_virt_counts;
        start_ldim = 0;
        end_ldim = last_len;
  #endif
        alloc_ptr(old_dist.order*sizeof(int64_t), (void**)&idx);
        alloc_ptr(old_dist.order*sizeof(int64_t), (void**)&idx_offs);
        alloc_ptr(old_dist.order*sizeof(int64_t), (void**)&old_virt_idx);
        alloc_ptr(old_dist.order*sizeof(int64_t), (void**)&virt_rank);
        alloc_ptr(old_dist.order*sizeof(int), (void**)&spad);
        memset(virt_counts, 0, np*sizeof(int64_t));
        memset(old_virt_idx, 0, old_dist.order*sizeof(int64_t));
        /* virt_rank = physical_rank*num_virtual_ranks + virtual_rank */
        for (int i=0; i<old_dist.order; i++){ 
          virt_rank[i] = old_dist.perank[i]*old_dist.virt_phase[i]; 
        }
        for (;;){
          memset(idx, 0, old_dist.order*sizeof(int64_t));
          memset(idx_offs, 0, old_dist.order*sizeof(int64_t));
          idx_offset = 0; 
          skip = 0;
          idx[old_dist.order-1] = MAX(idx[old_dist.order-1],start_ldim);
          for (dim=0; dim<old_dist.order; dim++) {
            /* Warning: This next if block has a history of bugs */
            //spad[dim] = old_dist.padding[dim];
            if (sym[dim] != NS){
              ASSERT(old_dist.padding[dim] < old_dist.phase[dim]);
              spad[dim] = 1;
              if (sym[dim] != SY && virt_rank[dim] < virt_rank[dim+1])
                spad[dim]--;
              if (sym[dim] == SY && virt_rank[dim] <= virt_rank[dim+1])
                spad[dim]--;
            }
            if (sym[dim] != NS && idx[dim] >= idx[dim+1]-spad[dim]){
              idx[dim+1] = idx[dim]+spad[dim];
    //        if (virt_rank[sym[dim]] + (sym[dim]==SY) <= virt_rank[dim])
    //          idx[sym[dim]]++;
            }
            if (dim > 0){
              imax = (old_dist.pad_edge_len[dim]-old_dist.padding[dim])/old_dist.phase[dim];
              if (virt_rank[dim] < (old_dist.pad_edge_len[dim]-old_dist.padding[dim])%old_dist.phase[dim])
                imax++;
              if (dim == old_dist.order - 1)
                imax = MIN(imax, end_ldim);
              if (idx[dim] >= imax)
                skip = 1;
              else  {
                idx_offs[dim] = bucket_offset[dim][old_virt_idx[dim]*old_virt_edge_len[dim]+idx[dim]];
                idx_offset += idx_offs[dim];
              }
            }
          }
          /* determine how many elements belong to each processor */
          /* (INNER LOOP) */
          if (!skip){
            for (;;){
              imax = (old_dist.pad_edge_len[0]-old_dist.padding[0])/old_dist.phase[0];
              if (virt_rank[0] < (old_dist.pad_edge_len[0]-old_dist.padding[0])%old_dist.phase[0])
                imax++;
              if (sym[0] != NS) {
                imax = MIN(imax,idx[1]+1-spad[0]);
              }
              if (old_dist.order == 1){
                imax = MIN(imax, end_ldim);
                i_st = start_ldim;
              } else
                i_st = 0;
              
              /* Increment virtual bucket */
              for (int i=i_st; i<imax; i++){
                vc = bucket_offset[0][old_virt_idx[0]*old_virt_edge_len[0]+i];
                virt_counts[idx_offset+vc]++;
              }
              /* Increment indices and set up offsets */
              for (dim=1; dim < old_dist.order; dim++){
                idx[dim]++;
                act_max = (old_dist.pad_edge_len[dim]-old_dist.padding[dim])/old_dist.phase[dim];
                if (virt_rank[dim] <
                    (old_dist.pad_edge_len[dim]-old_dist.padding[dim])%old_dist.phase[dim])
                  act_max++;
                if (dim == old_dist.order - 1)
                  act_max = MIN(act_max, end_ldim);
                if (sym[dim] != NS) 
                  act_max = MIN(act_max,idx[dim+1]+1-spad[dim]);
                bool ended = true;
                if (idx[dim] >= act_max){
                  ended = false;
                  idx[dim] = 0;
                  if (sym[dim-1] != NS) idx[dim] = idx[dim-1]+spad[dim-1];
                }
                idx_offset -= idx_offs[dim];
                idx_offs[dim] = bucket_offset[dim][old_virt_idx[dim]*old_virt_edge_len[dim]+idx[dim]];
                idx_offset += idx_offs[dim];
                if (ended)
                  break;
              }
              if (dim == old_dist.order) break;
            }
          }
          /* (OUTER LOOP) Iterate over virtual ranks on this pe */
          for (dim=0; dim<old_dist.order; dim++){
            old_virt_idx[dim]++;
            if (old_virt_idx[dim] >= old_dist.virt_phase[dim])
              old_virt_idx[dim] = 0;

            virt_rank[dim] = old_dist.perank[dim]*old_dist.virt_phase[dim]
                                 +old_virt_idx[dim];
    
            if (old_virt_idx[dim] > 0)
              break;  
          }
          if (dim == old_dist.order) break;
        }
        cfree(idx);
        cfree(idx_offs);
        cfree(old_virt_idx);
        cfree(virt_rank);
        cfree(spad);
#ifdef USE_OMP
#pragma omp master
        {
          act_omp_ntd = ntd;
        }
#endif
        }
  #ifdef USE_OMP
        for (int j=1; j<act_omp_ntd; j++){
          for (int64_t i=0; i<np; i++){
            all_virt_counts[i] += all_virt_counts[i+np*j];
          }
        }
  #endif
      }
    } else
      memset(all_virt_counts, 0, sizeof(int64_t)*np);

    //int tmp1, tmp2;
    //int64_t *  virt_counts = all_virt_counts;

    memcpy(send_counts, all_virt_counts, np*sizeof(int64_t));
//    memset(send_counts, 0, np*sizeof(int64_t));

    /* Calculate All-to-all processor grid offsets from the virtual bucket counts */
/*    if (old_dist.order == 0){
   //   send_counts[0] = virt_counts[0];
      memset(svirt_displs, 0, np*sizeof(int64_t));
    } else {
      for (int i=0; i<np; i++){
        for (int j=0; j<new_nvirt; j++){
          //printf("virt_count[%d][%d]=%d\n",i,j,virt_counts[i*new_nvirt+j]);
          send_counts[i] += virt_counts[i*new_nvirt+j];
          svirt_displs[i*new_nvirt+j] = virt_counts[i*new_nvirt+j];
        }
      }
    }*/

    /* Exchange counts */
    MPI_Alltoall(send_counts, 1, MPI_INT64_T, 
                 recv_counts, 1, MPI_INT64_T, ord_glb_comm.cm);
    
    /* Calculate displacements out of the count arrays */
    send_displs[0] = 0;
    recv_displs[0] = 0;
    for (int i=1; i<np; i++){
      send_displs[i] = send_displs[i-1] + send_counts[i-1];
      recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }

    /* Calculate displacements for virt buckets in each message */
/*    for (int i=0; i<np; i++){
      tmp2 = svirt_displs[i*new_nvirt];
      svirt_displs[i*new_nvirt] = 0;
      for (int j=1; j<new_nvirt; j++){
        tmp1 = svirt_displs[i*new_nvirt+j];
        svirt_displs[i*new_nvirt+j] = svirt_displs[i*new_nvirt+j-1]+tmp2;
        tmp2 = tmp1;
      }
    }*/

    /* Exchange displacements for virt buckets */
/*    MPI_Alltoall(svirt_displs, new_nvirt, MPI_INT64_T, 
                 rvirt_displs, new_nvirt, MPI_INT64_T, ord_glb_comm.cm);*/
    
    cfree(all_virt_counts);
  }


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
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&bucket_store);
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&count_store);
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&thread_store);
    std::fill(bucket_store, bucket_store+MAX(old_size,new_size), -1);

    int64_t ** par_virt_counts;
    alloc_ptr(sizeof(int64_t*)*max_ntd, (void**)&par_virt_counts);
    for (int t=0; t<max_ntd; t++){
      mst_alloc_ptr(sizeof(int64_t)*nbucket, (void**)&par_virt_counts[t]);
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

  #ifdef USE_OMP
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
    // FIXME: may be better to mst_alloc, but this should ensure the 
    //        compiler knows there are no write conflicts
  #ifdef USE_OMP
    int64_t * count = par_virt_counts[tid];
  #else
    int64_t *count; alloc_ptr(sizeof(int64_t)*nbucket, (void**)&count);
    memset(count, 0, sizeof(int64_t)*nbucket);
  #endif

    int *gidx; alloc_ptr(sizeof(int)*old_dist.order, (void**)&gidx);
    memset(gidx, 0, sizeof(int)*old_dist.order);
    for (int dim = 0;dim < old_dist.order;dim++){
      gidx[dim] = old_dist.perank[dim]*old_dist.virt_phase[dim];
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

  #ifdef USE_OMP
    for (int dim=old_dist.order-1; dim>=0; dim--){
      int64_t iist = MAX(0,(gidx_st[dim]-old_dist.perank[dim]*old_dist.virt_phase[dim]));
      int64_t ist = iist/(old_phys_dim[dim]*old_dist.virt_phase[dim]);
      if (sym[dim] != NS) ist = MIN(ist,idx[dim+1]);
      int plen[old_dist.order];
      memcpy(plen,old_virt_edge_len,old_dist.order*sizeof(int));
      int idim = dim;
      do {
        plen[idim] = ist;
        idim--;
      } while (idim >= 0 && sym[idim] != NS);
      gidx[dim] += ist*old_phys_dim[dim]*old_dist.virt_phase[dim];
      idx[dim] = ist;
      idx_acc[dim] = sy_packed_size(dim+1, plen, sym);
      offset += idx_acc[dim]; 

      ASSERT(ist == 0 || gidx[dim] <= gidx_st[dim]);
  //    ASSERT(ist < old_virt_edge_len[dim]);

      if (gidx[dim] > gidx_st[dim]) break;

      int64_t vst = iist-ist*old_phys_dim[dim]*old_dist.virt_phase[dim];
      if (vst > 0 ){
        vst = MIN(old_dist.virt_phase[dim]-1,vst);
        gidx[dim] += vst;
        virt_offset[dim] = vst*old_virt_edge_len[dim];
        offset += vst*old_virt_lda[dim];
      } else vst = 0;
      if (gidx[dim] > gidx_st[dim]) break;
    }
  #endif

    ASSERT(old_permutation == NULL);
    int rep_phase0 = lcm(old_phys_dim[0],new_phys_dim[0])/old_phys_dim[0];

    bool done = false;
    for (;!done;){
      int64_t bucket0 = 0;
      bool outside0 = false;
      int len_zero_max = ends[0];
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
          if (bucket_offset[dim][virt_offset[dim]+idx[dim]] == -1) outside0 = true;
          bucket0 += bucket_offset[dim][virt_offset[dim]+idx[dim]];
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
      int idx_st = 0;

      if (!outside0){
        int gidx_min = MAX(zero_len_toff,offs[0]);
        int gidx_max = (sym[0] == NS ? ends[0] : (sym[0] == SY ? gidx[1]+1 : gidx[1]));

        int idx0 = MAX(0,(gidx_min-gidx[0])/old_phys_dim[0]);
        int vidx0 = idx0%old_dist.virt_phase[0];
        int idx1 = MAX(0,(gidx_max-gidx[0])/old_phys_dim[0]);
        int lencp = MIN(rep_phase0,idx1-idx0);
        ASSERT(is_copy);
        if (forward){
          for (int ia=0; ia<lencp; ia++){
            int64_t bucket = bucket0+bucket_offset[0][((vidx0+ia)%old_dist.virt_phase[0])+idx[0]];
            sr->copy((idx1-idx0)/rep_phase0, 
                     new_data[bucket]+sr->el_size*count[bucket], 1,
                     old_data+ sr->el_size*idx0, rep_phase0);
            count[bucket]+=(idx1-idx0)/rep_phase0;
            idx0++;
          }
        } else {
          for (int ia=0; ia<lencp; ia++){
            int64_t bucket = bucket0+bucket_offset[0][((vidx0+ia)%old_dist.virt_phase[0])+idx[0]];
            sr->copy((idx1-idx0)/rep_phase0, 
                     old_data+ sr->el_size*idx0, rep_phase0,
                     new_data[bucket]+sr->el_size*count[bucket], 1);
            count[bucket]+=(idx1-idx0)/rep_phase0;
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
  #ifdef USE_OMP
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
  #ifdef USE_OMP
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
          virt_offset[dim] += old_virt_edge_len[dim];
          gidx[dim]++;

          if (virt_offset[dim] == old_dist.virt_phase[dim]*old_virt_edge_len[dim]){
            gidx[dim] -= old_dist.virt_phase[dim];
            virt_offset[dim] = 0;

            gidx[dim] -= idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];
            idx[dim]++;

            if (idx[dim] == (sym[dim] == NS ? old_virt_edge_len[dim] : idx[dim+1]+1)){
              //index should always be zero here sicne everything is SY and not SH
              idx[dim] = 0;//(dim == 0 || sym[dim-1] == NS ? 0 : idx[dim-1]);
              //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];

              if (dim == old_dist.order-1) done = true;
            }
            else{
              gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];
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
    cfree(gidx);
    cfree(idx_acc);
    cfree(virt_acc);
    cfree(idx);
    cfree(virt_offset);
    cfree(old_virt_lda);

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
    cfree(offs);
    cfree(ends);
   
  #ifndef USE_OMP
    cfree(count);
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
      cfree(par_virt_counts[t]);
    }
    cfree(par_virt_counts);
    cfree(count_store);
    cfree(bucket_store);
    cfree(thread_store);
  #endif

  }



  void pad_cyclic_pup_virt_buff(int const *          sym,
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
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&bucket_store);
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&count_store);
    mst_alloc_ptr(sizeof(int64_t)*MAX(old_size,new_size), (void**)&thread_store);
    std::fill(bucket_store, bucket_store+MAX(old_size,new_size), -1);

    int64_t ** par_virt_counts;
    alloc_ptr(sizeof(int64_t*)*max_ntd, (void**)&par_virt_counts);
    for (int t=0; t<max_ntd; t++){
      mst_alloc_ptr(sizeof(int64_t)*nbucket, (void**)&par_virt_counts[t]);
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

  #ifdef USE_OMP
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
    // FIXME: may be better to mst_alloc, but this should ensure the 
    //        compiler knows there are no write conflicts
  #ifdef USE_OMP
    int64_t * count = par_virt_counts[tid];
  #else
    int64_t *count; alloc_ptr(sizeof(int64_t)*nbucket, (void**)&count);
    memset(count, 0, sizeof(int64_t)*nbucket);
  #endif

    int *gidx; alloc_ptr(sizeof(int)*old_dist.order, (void**)&gidx);
    memset(gidx, 0, sizeof(int)*old_dist.order);
    for (int dim = 0;dim < old_dist.order;dim++){
      gidx[dim] = old_dist.perank[dim]*old_dist.virt_phase[dim];
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

  #ifdef USE_OMP
    for (int dim=old_dist.order-1; dim>=0; dim--){
      int64_t iist = MAX(0,(gidx_st[dim]-old_dist.perank[dim]*old_dist.virt_phase[dim]));
      int64_t ist = iist/(old_phys_dim[dim]*old_dist.virt_phase[dim]);
      if (sym[dim] != NS) ist = MIN(ist,idx[dim+1]);
      int plen[old_dist.order];
      memcpy(plen,old_virt_edge_len,old_dist.order*sizeof(int));
      int idim = dim;
      do {
        plen[idim] = ist;
        idim--;
      } while (idim >= 0 && sym[idim] != NS);
      gidx[dim] += ist*old_phys_dim[dim]*old_dist.virt_phase[dim];
      idx[dim] = ist;
      idx_acc[dim] = sy_packed_size(dim+1, plen, sym);
      offset += idx_acc[dim]; 

      ASSERT(ist == 0 || gidx[dim] <= gidx_st[dim]);
  //    ASSERT(ist < old_virt_edge_len[dim]);

      if (gidx[dim] > gidx_st[dim]) break;

      int64_t vst = iist-ist*old_phys_dim[dim]*old_dist.virt_phase[dim];
      if (vst > 0 ){
        vst = MIN(old_dist.virt_phase[dim]-1,vst);
        gidx[dim] += vst;
        virt_offset[dim] = vst*old_virt_edge_len[dim];
        offset += vst*old_virt_lda[dim];
      } else vst = 0;
      if (gidx[dim] > gidx_st[dim]) break;
    }
  #endif

    bool done = false;
    for (;!done;){
      int64_t bucket0 = 0;
      bool outside0 = false;
      int len_zero_max = ends[0];
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
          if (bucket_offset[dim][virt_offset[dim]+idx[dim]] == -1) outside0 = true;
          bucket0 += bucket_offset[dim][virt_offset[dim]+idx[dim]];
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
      int idx_st = 0;

      if (!outside0){
        int gidx_min = MAX(zero_len_toff,offs[0]);
        int gidx_max = (sym[0] == NS ? ends[0] : (sym[0] == SY ? gidx[1]+1 : gidx[1]));
        gidx_max = MIN(gidx_max, len_zero_max);
        for (idx[0] = idx_st;idx[0] < idx_max;idx[0]++){
          int virt_min = MAX(0,MIN(old_dist.virt_phase[0],gidx_min-gidx[0]));
          int virt_max = MAX(0,MIN(old_dist.virt_phase[0],gidx_max-gidx[0]));

          offset += old_virt_nelem*virt_min;
          if (forward){
            ASSERT(is_copy);
            for (virt_offset[0] = virt_min*old_virt_edge_len[0];
                 virt_offset[0] < virt_max*old_virt_edge_len[0];
                 virt_offset[0] += old_virt_edge_len[0])
            {
              int64_t bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
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
            for (virt_offset[0] = virt_min*old_virt_edge_len[0];
                 virt_offset[0] < virt_max*old_virt_edge_len[0];
                 virt_offset[0] += old_virt_edge_len[0])
            {
              int64_t bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
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
          gidx[0] += old_phys_dim[0]*old_dist.virt_phase[0];
        }

        offset -= idx_max;
        gidx[0] -= idx_max*old_phys_dim[0]*old_dist.virt_phase[0];
      }
       
      idx_acc[0] = idx_max;

      idx[0] = 0;

      zero_len_toff = 0;

      /* Adjust outer indices */
      if (!done){
        for (int dim = 1;dim < old_dist.order;dim++){
          offset += old_virt_lda[dim];
    
          virt_offset[dim] += old_virt_edge_len[dim];
          gidx[dim]++;

          if (virt_offset[dim] == old_dist.virt_phase[dim]*old_virt_edge_len[dim]){
            offset -= old_virt_lda[dim]*old_dist.virt_phase[dim];
            gidx[dim] -= old_dist.virt_phase[dim];
            virt_offset[dim] = 0;

            offset += idx_acc[dim-1];
            idx_acc[dim] += idx_acc[dim-1];
            idx_acc[dim-1] = 0;

            gidx[dim] -= idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];
            idx[dim]++;

            if (idx[dim] == (sym[dim] == NS ? old_virt_edge_len[dim] : idx[dim+1]+1)){
              offset -= idx_acc[dim];
              //index should always be zero here sicne everything is SY and not SH
              idx[dim] = 0;//(dim == 0 || sym[dim-1] == NS ? 0 : idx[dim-1]);
              //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];

              if (dim == old_dist.order-1) done = true;
            }
            else{
              gidx[dim] += idx[dim]*old_phys_dim[dim]*old_dist.virt_phase[dim];
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
    cfree(gidx);
    cfree(idx_acc);
    cfree(virt_acc);
    cfree(idx);
    cfree(virt_offset);
    cfree(old_virt_lda);

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
    cfree(offs);
    cfree(ends);
   
  #ifndef USE_OMP
    cfree(count);
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
      cfree(par_virt_counts[t]);
    }
    cfree(par_virt_counts);
    cfree(count_store);
    cfree(bucket_store);
    cfree(thread_store);
  #endif

  }

  static inline
  int64_t sy_packed_offset(int dim, int const * len, int const * idx, int const * sym){
    if (idx[dim] == 0) return 0;
    if (sym[dim-1] == NS){
      return sy_packed_size(dim, len, sym)*idx[dim];
    } else {
      int i=1;
      int ii=1;
      int iidx = idx[dim];
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

  void order_globally(int const *          sym,
                      distribution const & dist,
                      int const *          virt_edge_len,
                      int const *          virt_phase_lda,
                      int64_t              vbs,
                      int                  dir,
                      char const *         tsr_data_in,
                      char *               tsr_data_out,
                      algstrct const *     sr){
    TAU_FSTART(order_globally);
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
          int64_t dim_offset = sy_packed_offset(idim, virt_edge_len, idx, sym);
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
            idx[idim]++;
            int64_t glb_vrt_offset = sy_packed_offset(idim, virt_edge_len, idx, sym);
            glb_ord_offset += (glb_vrt_offset-dim_offset)*virt_phase_lda[idim]*virt_idx[idim];
            idx[idim]--;
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
    TAU_FSTOP(order_globally);
  }

  void glb_cyclic_reshuffle(int const *          sym,
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
      return;
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

    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&recv_counts);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&send_counts);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&send_displs);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&recv_displs);
    alloc_ptr(order*sizeof(int), (void**)&old_sub_edge_len);
    alloc_ptr(order*sizeof(int), (void**)&new_sub_edge_len);
    int ** bucket_offset;
    
    int *real_edge_len; alloc_ptr(sizeof(int)*order, (void**)&real_edge_len);
    for (i=0; i<order; i++) real_edge_len[i] = old_dist.pad_edge_len[i]-old_dist.padding[i];
    
    int *old_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&old_phys_dim);
    for (i=0; i<order; i++) old_phys_dim[i] = old_dist.phase[i]/old_dist.virt_phase[i];

    int *new_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&new_phys_dim);
    for (i=0; i<order; i++) new_phys_dim[i] = new_dist.phase[i]/new_dist.virt_phase[i];
    
    int *old_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_phys_edge_len);
    for (int dim = 0;dim < order;dim++) old_phys_edge_len[dim] = (real_edge_len[dim]+old_dist.padding[dim])/old_phys_dim[dim];

    int *new_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_phys_edge_len);
    for (int dim = 0;dim < order;dim++) new_phys_edge_len[dim] = (real_edge_len[dim]+new_dist.padding[dim])/new_phys_dim[dim];

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
      mst_alloc_ptr(MAX(old_dist.size,swp_nval)*sr->el_size, (void**)&tsr_cyclic_data);
      recv_buffer = tsr_cyclic_data;
    } else {
      mst_alloc_ptr(old_dist.size*sr->el_size, (void**)&send_buffer);
      mst_alloc_ptr(swp_nval*sr->el_size, (void**)&recv_buffer);
    }
    ASSERT(reuse_buffers);
#ifdef USE_OMP
    ASSERT(0);
#endif
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
                  tsr_cyclic_data,
                  new_data,
                  1,
                  bucket_offset, 
                  sr->mulid(),
                  sr->addid(),
                  sr);
      cfree(new_data);
    }
    for (int dim = 0;dim < order;dim++){
      cfree(bucket_offset[dim]);
    }
    cfree(bucket_offset);

    TAU_FSTOP(pack_virt_buf);

    if (reuse_buffers){
      recv_buffer = tsr_cyclic_data;
      send_buffer = tsr_data;
    }

    /* Communicate data */
    TAU_FSTART(ALL_TO_ALL_V);
    ord_glb_comm.all_to_allv(send_buffer, send_counts, send_displs, sr->el_size,
                             recv_buffer, recv_counts, recv_displs);
    TAU_FSTOP(ALL_TO_ALL_V);

    if (reuse_buffers){
      if (swp_nval > old_dist.size){
        cfree(tsr_data);
        mst_alloc_ptr(swp_nval*sr->el_size, (void**)&tsr_data);
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
                  tsr_data,
                  new_data,
                  0,
                  bucket_offset, 
                  alpha,
                  beta,
                  sr);
      order_globally(sym, new_dist, new_virt_edge_len, new_virt_lda, vbs_new, 0, tsr_data, tsr_cyclic_data, sr);
      for (int dim = 0;dim < order;dim++){
        cfree(bucket_offset[dim]);
      }
      cfree(bucket_offset);
      cfree(new_data);
    } else {
      sr->set(tsr_cyclic_data, sr->addid(), swp_nval);
    }
    TAU_FSTOP(unpack_virt_buf);

    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    *ptr_tsr_data = tsr_data;

    cfree(real_edge_len);
    cfree(old_phys_dim);
    cfree(new_phys_dim);
    cfree(hsym);
    cfree(idx);
    cfree(idx_offs);
    cfree(old_virt_lda);
    cfree(new_virt_lda);
    cfree(recv_counts);
    cfree(send_counts);
    cfree(send_displs);
    cfree(recv_displs);
    cfree(old_sub_edge_len);
    cfree(new_sub_edge_len);
    cfree(new_virt_edge_len);
    cfree(old_virt_edge_len);
    cfree(new_phys_edge_len);
    cfree(old_phys_edge_len);

    TAU_FSTOP(cyclic_reshuffle);

  }

  void cyclic_reshuffle(int const *          sym,
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
      return;
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

    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&recv_counts);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&send_counts);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&send_displs);
    mst_alloc_ptr(np*sizeof(int64_t),   (void**)&recv_displs);
    alloc_ptr(order*sizeof(int), (void**)&old_sub_edge_len);
    alloc_ptr(order*sizeof(int), (void**)&new_sub_edge_len);
    int ** bucket_offset;
    
    int *real_edge_len; alloc_ptr(sizeof(int)*order, (void**)&real_edge_len);
    for (i=0; i<order; i++) real_edge_len[i] = old_dist.pad_edge_len[i]-old_dist.padding[i];
    
    int *old_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&old_phys_dim);
    for (i=0; i<order; i++) old_phys_dim[i] = old_dist.phase[i]/old_dist.virt_phase[i];

    int *new_phys_dim; alloc_ptr(sizeof(int)*order, (void**)&new_phys_dim);
    for (i=0; i<order; i++) new_phys_dim[i] = new_dist.phase[i]/new_dist.virt_phase[i];
    
    int *old_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_phys_edge_len);
    for (int dim = 0;dim < order;dim++) old_phys_edge_len[dim] = (real_edge_len[dim]+old_dist.padding[dim])/old_phys_dim[dim];

    int *new_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_phys_edge_len);
    for (int dim = 0;dim < order;dim++) new_phys_edge_len[dim] = (real_edge_len[dim]+new_dist.padding[dim])/new_phys_dim[dim];

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
      mst_alloc_ptr(MAX(old_dist.size,swp_nval)*sr->el_size, (void**)&tsr_cyclic_data);
    } else {
      mst_alloc_ptr(old_dist.size*sr->el_size, (void**)&send_buffer);
      mst_alloc_ptr(swp_nval*sr->el_size, (void**)&recv_buffer);
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
      cfree(new_data);
    }
    for (int dim = 0;dim < order;dim++){
      cfree(bucket_offset[dim]);
    }
    cfree(bucket_offset);

    TAU_FSTOP(pack_virt_buf);

    if (reuse_buffers){
      if (swp_nval > old_dist.size){
        cfree(tsr_data);
        mst_alloc_ptr(swp_nval*sr->el_size, (void**)&tsr_data);
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
        cfree(bucket_offset[dim]);
      }
      cfree(bucket_offset);
      cfree(new_data);
    }
    TAU_FSTOP(unpack_virt_buf);

    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    *ptr_tsr_data = tsr_data;

    cfree(real_edge_len);
    cfree(old_phys_dim);
    cfree(new_phys_dim);
    cfree(hsym);
    cfree(idx);
    cfree(idx_offs);
    cfree(old_virt_lda);
    cfree(new_virt_lda);
    cfree(recv_counts);
    cfree(send_counts);
    cfree(send_displs);
    cfree(recv_displs);
    cfree(old_sub_edge_len);
    cfree(new_sub_edge_len);
    cfree(new_virt_edge_len);
    cfree(old_virt_edge_len);
    cfree(new_phys_edge_len);
    cfree(old_phys_edge_len);

    TAU_FSTOP(cyclic_reshuffle);

  }

  void block_reshuffle(distribution const & old_dist,
                       distribution const & new_dist,
                       char *               tsr_data,
                       char *&              tsr_cyclic_data,
                       algstrct const *     sr,
                       CommData             glb_comm){
    int i, idx_lyr_new, idx_lyr_old, blk_idx, prc_idx, loc_idx;
    int num_old_virt, num_new_virt;
    int * idx, * old_loc_lda, * new_loc_lda, * phase_lda;
    int64_t blk_sz;
    MPI_Request * reqs;
    int * phase = old_dist.phase;
    int order = old_dist.order; 

    if (order == 0){
      alloc_ptr(sr->el_size*new_dist.size, (void**)&tsr_cyclic_data);

      if (glb_comm.rank == 0){
        sr->copy(tsr_cyclic_data,  tsr_data);
      } else {
        sr->copy(tsr_cyclic_data, sr->addid());
      }
      return;
    }

    TAU_FSTART(block_reshuffle);

    mst_alloc_ptr(sr->el_size*new_dist.size, (void**)&tsr_cyclic_data);
    alloc_ptr(sizeof(int)*order, (void**)&idx);
    alloc_ptr(sizeof(int)*order, (void**)&old_loc_lda);
    alloc_ptr(sizeof(int)*order, (void**)&new_loc_lda);
    alloc_ptr(sizeof(int)*order, (void**)&phase_lda);

    blk_sz = old_dist.size;
    old_loc_lda[0] = 1;
    new_loc_lda[0] = 1;
    phase_lda[0] = 1;
    num_old_virt = 1;
    num_new_virt = 1;
    idx_lyr_old = glb_comm.rank;
    idx_lyr_new = glb_comm.rank;

    for (i=0; i<order; i++){
      num_old_virt *= old_dist.virt_phase[i];
      num_new_virt *= new_dist.virt_phase[i];
      blk_sz = blk_sz/old_dist.virt_phase[i];
      idx_lyr_old -= old_dist.perank[i]*old_dist.pe_lda[i];
      idx_lyr_new -= new_dist.perank[i]*new_dist.pe_lda[i];
      if (i>0){
        old_loc_lda[i] = old_loc_lda[i-1]*old_dist.virt_phase[i-1];
        new_loc_lda[i] = new_loc_lda[i-1]*new_dist.virt_phase[i-1];
        phase_lda[i] = phase_lda[i-1]*phase[i-1];
      }
    }
    
    alloc_ptr(sizeof(MPI_Request)*(num_old_virt+num_new_virt), (void**)&reqs);

    if (idx_lyr_new == 0){
      memset(idx, 0, sizeof(int)*order);

      for (;;){
        loc_idx = 0;
        blk_idx = 0;
        prc_idx = 0;
        for (i=0; i<order; i++){
          loc_idx += idx[i]*new_loc_lda[i];
          blk_idx += ( idx[i] + new_dist.perank[i]*new_dist.virt_phase[i])                 *phase_lda[i];
          prc_idx += ((idx[i] + new_dist.perank[i]*new_dist.virt_phase[i])/old_dist.virt_phase[i])*old_dist.pe_lda[i];
        }
        DPRINTF(3,"proc %d receiving blk %d (loc %d, size %ld) from proc %d\n", 
                glb_comm.rank, blk_idx, loc_idx, blk_sz, prc_idx);
        MPI_Irecv(tsr_cyclic_data+sr->el_size*loc_idx*blk_sz, blk_sz*sr->el_size, 
                  MPI_CHAR, prc_idx, blk_idx, glb_comm.cm, reqs+loc_idx);
        for (i=0; i<order; i++){
          idx[i]++;
          if (idx[i] >= new_dist.virt_phase[i])
            idx[i] = 0;
          else 
            break;
        }
        if (i==order) break;
      }
    }

    if (idx_lyr_old == 0){
      memset(idx, 0, sizeof(int)*order);

      for (;;){
        loc_idx = 0;
        blk_idx = 0;
        prc_idx = 0;
        for (i=0; i<order; i++){
          loc_idx += idx[i]*old_loc_lda[i];
          blk_idx += ( idx[i] + old_dist.perank[i]*old_dist.virt_phase[i])                 *phase_lda[i];
          prc_idx += ((idx[i] + old_dist.perank[i]*old_dist.virt_phase[i])/new_dist.virt_phase[i])*new_dist.pe_lda[i];
        }
        DPRINTF(3,"proc %d sending blk %d (loc %d size %ld) to proc %d el_size = %d\n", 
                glb_comm.rank, blk_idx, loc_idx, blk_sz, prc_idx, sr->el_size);
        MPI_Isend(tsr_data+sr->el_size*loc_idx*blk_sz, blk_sz*sr->el_size, 
                  MPI_CHAR, prc_idx, blk_idx, glb_comm.cm, reqs+num_new_virt+loc_idx);
        for (i=0; i<order; i++){
          idx[i]++;
          if (idx[i] >= old_dist.virt_phase[i])
            idx[i] = 0;
          else 
            break;
        }
        if (i==order) break;
      }
    }

    if (idx_lyr_new == 0 && idx_lyr_old == 0){
      MPI_Waitall(num_new_virt+num_old_virt, reqs, MPI_STATUSES_IGNORE);
    } else if (idx_lyr_new == 0){
      MPI_Waitall(num_new_virt, reqs, MPI_STATUSES_IGNORE);
    } else if (idx_lyr_old == 0){
      MPI_Waitall(num_old_virt, reqs+num_new_virt, MPI_STATUSES_IGNORE);
      sr->set(tsr_cyclic_data, sr->addid(), new_dist.size);
    } else {
      sr->set(tsr_cyclic_data, sr->addid(), new_dist.size);
    }

    cfree(idx);
    cfree(old_loc_lda);
    cfree(new_loc_lda);
    cfree(phase_lda);
    cfree(reqs);

    TAU_FSTOP(block_reshuffle);

  }

  int can_block_reshuffle(int             order,
                          int const *     old_phase,
                          mapping const * map){
    int new_phase, j;
    int can_block_resh = 1;
    for (j=0; j<order; j++){
      new_phase  = map[j].calc_phase();
      if (new_phase != old_phase[j]) can_block_resh = 0;
    }
    return can_block_resh;
  }


}
