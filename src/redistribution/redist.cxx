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
    int * virt_phase_rank, * old_virt_phase_rank;
    int64_t * sub_edge_len;
    char * pairs, * tsr_new_data;
    DEBUG_PRINTF("Performing padded reshuffle\n");

    TAU_FSTART(padded_reshuffle);

    numPes = ord_glb_comm.np;

    alloc_ptr(old_dist.order*sizeof(int), (void**)&virt_phase_rank);
    alloc_ptr(old_dist.order*sizeof(int), (void**)&old_virt_phase_rank);
    alloc_ptr(old_dist.order*sizeof(int64_t), (void**)&sub_edge_len);

    new_num_virt = 1;
    old_num_virt = 1;
    idx_lyr = ord_glb_comm.rank;
    for (int i=0; i<old_dist.order; i++){
      old_num_virt = old_num_virt*old_dist.virt_phase[i];
      new_num_virt = new_num_virt*new_dist.virt_phase[i];
      virt_phase_rank[i] = new_dist.perank[i];//*new_dist.virt_phase[i];
      old_virt_phase_rank[i] = old_dist.perank[i];//*old_dist.virt_phase[i];
      idx_lyr -= old_dist.perank[i]*old_dist.pe_lda[i];
    }
    if (idx_lyr == 0 ){
      read_loc_pairs(old_dist.order, old_dist.size, old_num_virt, sym,
                     old_dist.pad_edge_len, old_dist.padding, old_dist.phase, old_dist.phys_phase,
                     old_dist.virt_phase, old_virt_phase_rank, &new_nval, tsr_data,
                     &pairs, sr);
    } else {
      new_nval = 0;
      pairs = NULL;
    }

  /*#if DEBUG >= 1
    int64_t old_size = sy_packed_size(old_dist.order, new_dist.pad_edge_len, sym);
  #endif*/

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

    if (sr->addid() != NULL)
      sr->set(tsr_new_data, sr->addid(), swp_nval);


    int64_t ignrd;
    char * aignrd;
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
                    new_dist.phys_phase,
                    new_dist.virt_phase,
                    virt_phase_rank,
                    new_dist.pe_lda,
                    pairs,
                    tsr_new_data,
                    ord_glb_comm,
                    sr,
                    false,
                    0,
                    NULL,
                    aignrd,
                    ignrd);

    *tsr_cyclic_data = tsr_new_data;

    cdealloc(old_virt_phase_rank);
    if (pairs != NULL)
      cdealloc(pairs);
    cdealloc(virt_phase_rank);
    cdealloc(sub_edge_len);
    TAU_FSTOP(padded_reshuffle);
  }


  int ** compute_bucket_offsets(distribution const & old_dist,
                                distribution const & new_dist,
                                int64_t const *      len,
                                int64_t const *      old_phys_edge_len,
                                int const *          old_virt_lda,
                                int64_t const *      old_offsets,
                                int * const *        old_permutation,
                                int64_t const *      new_phys_edge_len,
                                int const *          new_virt_lda,
                                int                  forward,
                                int                  old_virt_np,
                                int                  new_virt_np,
                                int64_t const *      old_virt_edge_len){
    TAU_FSTART(compute_bucket_offsets);

    int **bucket_offset; alloc_ptr(sizeof(int*)*old_dist.order, (void**)&bucket_offset);

    for (int dim = 0;dim < old_dist.order;dim++){
      alloc_ptr(sizeof(int)*old_phys_edge_len[dim], (void**)&bucket_offset[dim]);
      int pidx = 0;
      for (int64_t vidx = 0;vidx < old_virt_edge_len[dim];vidx++){
        for (int vr = 0;vr < old_dist.virt_phase[dim];vr++,pidx++){
          int64_t _gidx = (int64_t)vidx*old_dist.phase[dim]+old_dist.perank[dim]+(int64_t)vr*old_dist.phys_phase[dim];
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
            int phys_rank = gidx%new_dist.phys_phase[dim];
            if (forward){
              bucket_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim]);
              //printf("f %d - %d %d %d - %d - %d %d %d - %d\n", dim, vr, vidx, pidx, gidx, total_rank,
              //    phys_rank, virt_rank, bucket_offset[dim][pidx]);
            }
            else{
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
                       int64_t const *      old_virt_edge_len,
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

    alloc_ptr(np*sizeof(int64_t)*max_ntd, (void**)&all_virt_counts);


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
        int64_t * idx;
        int64_t idx_offset;
        int64_t *  idx_offs;
        int * spad;
        int64_t last_len = old_dist.pad_edge_len[old_dist.order-1]/old_dist.phase[old_dist.order-1]+1;
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
        /* virt_rank = physical_rank + virtual_rank*num_phys_ranks */
        for (int i=0; i<old_dist.order; i++){
          virt_rank[i] = old_dist.perank[i];
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
                idx_offs[dim] = bucket_offset[dim][old_virt_idx[dim]+idx[dim]*old_dist.virt_phase[dim]];
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
                vc = bucket_offset[0][old_virt_idx[0]+i*old_dist.virt_phase[0]];
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
                idx_offs[dim] = bucket_offset[dim][old_virt_idx[dim]+idx[dim]*old_dist.virt_phase[dim]];
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

            virt_rank[dim] = old_dist.perank[dim]+old_virt_idx[dim]*old_dist.phys_phase[dim];

            if (old_virt_idx[dim] > 0)
              break;
          }
          if (dim == old_dist.order) break;
        }
        cdealloc(idx);
        cdealloc(idx_offs);
        cdealloc(old_virt_idx);
        cdealloc(virt_rank);
        cdealloc(spad);
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

    cdealloc(all_virt_counts);
  }

  //static double init_mdl[] = {COST_LATENCY, COST_LATENCY, COST_NETWBW};
  LinModel<2> blres_mdl(blres_mdl_init,"blres_mdl");

  double blres_est_time(int64_t tot_sz, int nv0, int nv1){
    double ps[] = {(double)nv0+nv1, (double)tot_sz};
    return blres_mdl.est_time(ps);
  }

  void block_reshuffle(distribution const & old_dist,
                       distribution const & new_dist,
                       char *               tsr_data,
                       char *&              tsr_cyclic_data,
                       algstrct const *     sr,
                       CommData             glb_comm){

    int i, idx_lyr_new, idx_lyr_old, rem_idx, prc_idx, loc_idx;
    int num_old_virt, num_new_virt;
    int * idx, * old_loc_lda, * new_loc_lda, * phase_lda;
    int64_t blk_sz;
    MPI_Request * reqs;
    int * phase = old_dist.phase;
    int order = old_dist.order;


    if (order == 0){
      tsr_cyclic_data = sr->alloc(new_dist.size);

      if (glb_comm.rank == 0){
        sr->copy(tsr_cyclic_data,  tsr_data);
      } else {
        sr->copy(tsr_cyclic_data, sr->addid());
      }
      return;
    }

    TAU_FSTART(block_reshuffle);
#ifdef TUNE
    MPI_Barrier(glb_comm.cm);
    double st_time = MPI_Wtime();
#endif

    tsr_cyclic_data = sr->alloc(new_dist.size);
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

#ifdef TUNE
    double * tps = (double*)malloc(3*sizeof(double));
    tps[0] = 0;
    tps[1] = (double)num_old_virt+num_new_virt;
    tps[2] = (double)std::max(new_dist.size, new_dist.size);

    if (!(blres_mdl.should_observe(tps))){
      cdealloc(idx);
      cdealloc(old_loc_lda);
      cdealloc(new_loc_lda);
      cdealloc(phase_lda);
      return;
    }
    free(tps);
#endif
    alloc_ptr(sizeof(MPI_Request)*(num_old_virt+num_new_virt), (void**)&reqs);

    if (idx_lyr_new == 0){
      memset(idx, 0, sizeof(int)*order);

      for (;;){
        loc_idx = 0;
        rem_idx = 0;
        prc_idx = 0;
        for (i=0; i<order; i++){
          loc_idx += idx[i]*new_loc_lda[i];
          rem_idx += ((idx[i]*new_dist.phys_phase[i] + new_dist.perank[i])/old_dist.phys_phase[i])*old_loc_lda[i];
          prc_idx += ((idx[i]*new_dist.phys_phase[i] + new_dist.perank[i])%old_dist.phys_phase[i])*old_dist.pe_lda[i];
        }
        DPRINTF(3,"proc %d receiving blk %d (loc %d, size %ld) from proc %d\n",
                glb_comm.rank, rem_idx, loc_idx, blk_sz, prc_idx);
        MPI_Irecv(tsr_cyclic_data+sr->el_size*loc_idx*blk_sz, blk_sz,
                  sr->mdtype(), prc_idx, rem_idx, glb_comm.cm, reqs+loc_idx);
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
        prc_idx = 0;
        for (i=0; i<order; i++){
          loc_idx += idx[i]*old_loc_lda[i];
          prc_idx += ((idx[i]*old_dist.phys_phase[i] + old_dist.perank[i])%new_dist.phys_phase[i])*new_dist.pe_lda[i];
        }
        DPRINTF(3,"proc %d sending blk %d (loc %d size %ld) to proc %d el_size = %d %p %p\n",
                glb_comm.rank, loc_idx, loc_idx, blk_sz, prc_idx, sr->el_size, tsr_data, reqs+num_new_virt+loc_idx);
        MPI_Isend(tsr_data+sr->el_size*loc_idx*blk_sz, blk_sz,
                  sr->mdtype(), prc_idx, loc_idx, glb_comm.cm, reqs+num_new_virt+loc_idx);
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
      if (sr->addid() != NULL)
        sr->set(tsr_cyclic_data, sr->addid(), new_dist.size);
    } else {
      if (sr->addid() != NULL)
        sr->set(tsr_cyclic_data, sr->addid(), new_dist.size);
    }

    cdealloc(idx);
    cdealloc(old_loc_lda);
    cdealloc(new_loc_lda);
    cdealloc(phase_lda);
    cdealloc(reqs);

#ifdef TUNE
    MPI_Barrier(glb_comm.cm);
    double exe_time = MPI_Wtime()-st_time;
    tps = (double*)malloc(3*sizeof(double));
    tps[0] = exe_time;
    tps[1] = (double)num_old_virt+num_new_virt;
    tps[2] = (double)std::max(new_dist.size, new_dist.size);
    blres_mdl.observe(tps);
    free(tps);
#endif

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
