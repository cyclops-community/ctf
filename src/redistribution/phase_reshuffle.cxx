/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "phase_reshuffle.h"
#include "glb_cyclic_reshuffle.h"
#include "../shared/util.h"
#include "nosym_transp.h"

namespace CTF_int {
  //correct for SY
  inline int get_glb(int i, int s, int t){
    return i*s+t;
  }
  //correct for SH/AS, but can treat everything as SY
  /*inline int get_glb(int i, int s, int t){
    return i*s+t-1;
  }*/
  inline int get_loc(int g, int s, int t){
    //round down, dowwwwwn
    if (t>g) return -1;
    else return (g-t)/s;
  }
   
  template <int idim>
  int64_t calc_cnt(int const * sym,
                   int const * rep_phase,
                   int const * sphase,
                   int const * gidx_off,
                   int const * edge_len,
                   int const * loc_edge_len){
    ASSERT(sym[idim] == NS); //otherwise should be in calc_sy_pfx
    if (sym[idim-1] == NS){
      return (get_loc(edge_len[idim]-1,sphase[idim],gidx_off[idim])+1)*calc_cnt<idim-1>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
    } else {
      int64_t * pfx = calc_sy_pfx<idim>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
      int64_t cnt = 0;
      for (int i=0; i<=get_loc(edge_len[idim]-1,sphase[idim],gidx_off[idim]); i++){
        cnt += pfx[i];
      }
      cfree(pfx);
      return cnt;
    }
  }
 
  template <>
  int64_t calc_cnt<0>(int const * sym,
                      int const * rep_phase,
                      int const * sphase,
                      int const * gidx_off,
                      int const * edge_len,
                      int const * loc_edge_len){
    ASSERT(sym[0] == NS);
    return get_loc(edge_len[0]-1, sphase[0], gidx_off[0])+1;
  }

  template <int idim>
  int64_t * calc_sy_pfx(int const * sym,
                        int const * rep_phase,
                        int const * sphase,
                        int const * gidx_off,
                        int const * edge_len,
                        int const * loc_edge_len){
    int64_t * pfx = (int64_t*)alloc(sizeof(int64_t)*loc_edge_len[idim]);
    if (sym[idim-1] == NS){
      int64_t ns_size = calc_cnt<idim-1>(sym,rep_phase,sphase,gidx_off,edge_len,loc_edge_len);
      for (int i=0; i<loc_edge_len[idim]; i++){
        pfx[i] = ns_size;
      }
    } else {
      int64_t * pfx_m1 = calc_sy_pfx<idim-1>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
      for (int i=0; i<loc_edge_len[idim]; i++){
        int jst;
        if (i>0){
          pfx[i] = pfx[i-1];
          jst = get_loc(get_glb(i-1,sphase[idim],gidx_off[idim]),sphase[idim-1],gidx_off[idim-1])+1;
        } else {
          pfx[i] = 0;
          jst = 0;
        }
        int jed = get_loc(get_glb(i,sphase[idim],gidx_off[idim]),sphase[idim-1],gidx_off[idim-1]);
        for (int j=jst; j<=jed; j++){
          //printf("idim = %d j=%d loc_edge[idim] = %d loc_Edge[idim-1]=%d\n",idim,j,loc_edge_len[idim],loc_edge_len[idim-1]);
          pfx[i] += pfx_m1[j];
        }
      }
      cfree(pfx_m1);
    }
    return pfx;
  }
 
  template <>
  int64_t * calc_sy_pfx<1>(int const * sym,
                           int const * rep_phase,
                           int const * sphase,
                           int const * gidx_off,
                           int const * edge_len,
                           int const * loc_edge_len){
    int64_t * pfx= (int64_t*)alloc(sizeof(int64_t)*loc_edge_len[1]);
    for (int i=0; i<loc_edge_len[1]; i++){
      pfx[i] = get_loc(get_glb(i,sphase[1],gidx_off[1]),sphase[0],gidx_off[0])+1;
    }
    return pfx;
  }

  template <int idim>
  void calc_drv_cnts(int         order,
                     int const * sym,
                     int64_t *   counts,
                     int const * rep_phase,
                     int const * rep_phase_lda,
                     int const * sphase,
                     int const * phys_phase,
                     int       * gidx_off,
                     int const * edge_len,
                     int const * loc_edge_len){
    for (int i=0; i<rep_phase[idim]; i++, gidx_off[idim]+=phys_phase[idim]){
       calc_drv_cnts<idim-1>(order, sym, counts+i*rep_phase_lda[idim], rep_phase, rep_phase_lda, sphase, phys_phase,
                             gidx_off, edge_len, loc_edge_len);
    }
    gidx_off[idim] -= phys_phase[idim]*rep_phase[idim];
  }
  
  template <>
  void calc_drv_cnts<0>(int         order,
                        int const * sym,
                        int64_t *   counts,
                        int const * rep_phase,
                        int const * rep_phase_lda,
                        int const * sphase,
                        int const * phys_phase,
                        int       * gidx_off,
                        int const * edge_len,
                        int const * loc_edge_len){
    for (int i=0; i<rep_phase[0]; i++, gidx_off[0]+=phys_phase[0]){
      SWITCH_ORD_CALL_RET(counts[i], calc_cnt, order-1, sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len)
    }
    gidx_off[0] -= phys_phase[0]*rep_phase[0];
  }

  template <int idim>
  void calc_cnt_from_rep_cnt(int const *     rep_phase,
                             int const *     rep_phase_lda,
                             int const *     rank,
                             int const *     new_pe_lda,
                             int const *     old_phys_phase,
                             int const *     new_phys_phase,
                             int64_t const * rep_counts,
                             int64_t *       counts,
                             int             coff,
                             int             roff){
    for (int i=0; i<rep_phase[idim]; i++){
      calc_cnt_from_rep_cnt<idim-1>(rep_phase, rep_phase_lda, rank, new_pe_lda, old_phys_phase, new_phys_phase, rep_counts, counts,
                                    coff+new_pe_lda[idim]*((rank[idim]+i*old_phys_phase[idim])%new_phys_phase[idim]),
                                    roff+rep_phase_lda[idim]*i);
    }
  }

  template <>
  void calc_cnt_from_rep_cnt<0>(int const *     rep_phase,
                                int const *     rep_phase_lda,
                                int const *     rank,
                                int const *     new_pe_lda,
                                int const *     old_phys_phase,
                                int const *     new_phys_phase,
                                int64_t const * rep_counts,
                                int64_t *       counts,
                                int             coff,
                                int             roff){
    for (int i=0; i<rep_phase[0]; i++){
      counts[coff+new_pe_lda[0]*((rank[0]+i*old_phys_phase[0])%new_phys_phase[0])] = rep_counts[roff + i];
    }
  }

  void calc_drv_displs(int const *          sym,
                       int const *          edge_len,
                       int const *          loc_edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int64_t *            counts,
                       CommData             ord_glb_comm,
                       int                  idx_lyr){
    TAU_FSTART(calc_drv_displs);
    int * rep_phase, * gidx_off, * sphase;
    int * rep_phase_lda;
    int * new_loc_edge_len;
    std::fill(counts, counts+ord_glb_comm.np, 0);
    if (idx_lyr == 0){
      int order = old_dist.order;
      rep_phase     = (int*)alloc(order*sizeof(int));
      rep_phase_lda = (int*)alloc(order*sizeof(int));
      sphase        = (int*)alloc(order*sizeof(int));
      gidx_off      = (int*)alloc(order*sizeof(int));
      new_loc_edge_len      = (int*)alloc(order*sizeof(int));
      int nrep = 1;
      for (int i=0; i<order; i++){
        rep_phase_lda[i]  = nrep;
        sphase[i]         = lcm(old_dist.phys_phase[i],new_dist.phys_phase[i]);
        rep_phase[i]      = sphase[i] / old_dist.phys_phase[i];
        gidx_off[i]       = old_dist.perank[i];
        nrep             *= rep_phase[i];
        new_loc_edge_len[i] = (edge_len[i]+sphase[i]-1)/sphase[i];
        //printf("rep_phase[%d] = %d, sphase = %d lda = %d, gidx = %d\n",i,rep_phase[i], sphase[i], rep_phase_lda[i], gidx_off[i]);
      }
      int64_t * rep_counts = (int64_t*)alloc(nrep*sizeof(int64_t));
      ASSERT(order>0);
      SWITCH_ORD_CALL(calc_drv_cnts, order-1, order, sym, rep_counts, rep_phase, rep_phase_lda, sphase, old_dist.phys_phase, gidx_off, edge_len, new_loc_edge_len)
    
      SWITCH_ORD_CALL(calc_cnt_from_rep_cnt, order-1, rep_phase, rep_phase_lda, old_dist.perank, new_dist.pe_lda, old_dist.phys_phase, new_dist.phys_phase, rep_counts, counts);
      cfree(rep_phase);
      cfree(rep_phase_lda);
      cfree(sphase);
      cfree(gidx_off);
      cfree(rep_counts);
    }
    TAU_FSTOP(calc_drv_displs);
  }

  template <int idim>
  void redist_bucket(int const *          sym,
                     int const *          phys_phase,
                     int const *          perank,
                     int const *          edge_len,
                     int const *          virt_edge_len,
                     int const *          virt_dim,
                     int const *          virt_lda,
                     int64_t              virt_nelem,
                     int * const *        bucket_offset,
                     int                  rep_phase0,
                     bool                 data_to_buckets,
                     char * __restrict__  data,
                     char ** __restrict__ buckets,
                     int64_t *            counts,
                     algstrct const *     sr,
                     int64_t              data_off,
                     int                  bucket_off,
                     int                  prev_idx){
    int64_t data_stride, sub_data_stride;
    int nsym, ivmax;
    if (sym[idim] != NS){
      ivmax = get_loc(get_glb(prev_idx, phys_phase[idim+1], perank[idim+1]),
                                        phys_phase[idim  ], perank[idim  ]);
    } else
      ivmax = get_loc(edge_len[idim]-1, phys_phase[idim], perank[idim]);

    if (sym[idim-1] != NS){
      int iidim=idim-2;
      nsym = 1;
      while (iidim>=0 && sym[iidim] != NS){ nsym++; iidim--; }
      if (iidim+1 == 0){
        sub_data_stride = virt_dim[0];
      } else {
        sub_data_stride = sy_packed_size(iidim+1, virt_edge_len, sym)*virt_dim[0];
      }
      data_stride = 0;
      
    } else {
      nsym = 0;
      data_stride = sy_packed_size(idim, virt_edge_len, sym)*virt_dim[0];
    }
    
    for (int i=0, iv=0; iv <= ivmax; i++){
      int64_t rec_data_off = data_off;
      for (int v=0; v<virt_dim[idim] && iv <= ivmax; v++, iv++){
        int64_t rec_bucket_off = bucket_off + bucket_offset[idim][iv];
        //printf("idim = %d iv = %d/%d v=%d rec_bucket_off = %ld rec_data_off = %ld data_stride = %ld virt_stride=%ld\n", idim, iv, ivmax+1, v, rec_bucket_off, rec_data_off, data_stride, virt_lda[idim]*virt_nelem);
        redist_bucket<idim-1>(sym, phys_phase, perank, edge_len, virt_edge_len, virt_dim, virt_lda, virt_nelem, bucket_offset, rep_phase0, data_to_buckets, data, buckets, counts, sr, rec_data_off, rec_bucket_off, iv);
        rec_data_off += virt_lda[idim]*virt_nelem; 
      }
      //FIXME: we could potentially precompute these strides
      if (sym[idim-1] != NS){
        data_stride = (i+1)*sub_data_stride;
        for (int j=1; j<nsym; j++){
          data_stride = (data_stride*(i+j+1))/(j+1);
        }
        //printf("idim = %d nsym = %d, i = %d, incremeneted by %ld\n", idim, nsym, i, data_stride);
      }
      data_off += data_stride;
    }
  }


  template <>
  void redist_bucket<0>(int const *          sym,
                        int const *          phys_phase,
                        int const *          perank,
                        int const *          edge_len,
                        int const *          virt_edge_len,
                        int const *          virt_dim,
                        int const *          virt_lda,
                        int64_t              virt_nelem,
                        int * const *        bucket_offset,
                        int                  rep_phase0,
                        bool                 data_to_buckets,
                        char * __restrict__  data,
                        char ** __restrict__ buckets,
                        int64_t *            counts,
                        algstrct const *     sr,
                        int64_t              data_off,
                        int                  bucket_off,
                        int                  prev_idx){
    int ivmax;
    if (sym[0] != NS){
      ivmax = get_loc(get_glb(prev_idx, phys_phase[1], perank[1]),
                                        phys_phase[0], perank[0]) + 1;
    } else
      ivmax = get_loc(edge_len[0]-1, phys_phase[0], perank[0]) + 1;

    if (data_to_buckets){
      for (int i=0; i<rep_phase0; i++){
        int bucket = bucket_off + bucket_offset[0][i];
        int n = (ivmax-i+rep_phase0-1)/rep_phase0;
        //printf("ivmax = %d bucket_off = %d, bucket = %d, counts[bucket] = %ld, n= %d data_off = %ld, rep_phase=%d\n",
          //      ivmax, bucket_off, bucket, counts[bucket], n, data_off, rep_phase0);
        if (n>0){
          sr->copy(n,
                   data + sr->el_size*(data_off+i), rep_phase0, 
                   buckets[bucket] + sr->el_size*counts[bucket], 1);
          counts[bucket] += n;
        }
      }
    } else {
      for (int i=0; i<rep_phase0; i++){
        int bucket = bucket_off + bucket_offset[0][i];
        int n = (ivmax-i+rep_phase0-1)/rep_phase0;
        if (n>0){
          sr->copy(n,
                   buckets[bucket] + sr->el_size*counts[bucket], 1,
                   data + sr->el_size*(data_off+i), rep_phase0);
          counts[bucket] += n;
        }
      }
    }
  }


  void phase_reshuffle(int const *          sym,
                       int const *          edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       char **              ptr_tsr_data,
                       char **              ptr_tsr_new_data,
                       algstrct const *     sr,
                       CommData             ord_glb_comm){
    int order = old_dist.order;

    char * tsr_data = *ptr_tsr_data;
    char * tsr_new_data = *ptr_tsr_new_data;
#if 0
    char * tsr_data_cpy = (char*)alloc(sr->el_size*old_dist.size);
    char * tsr_data2;
    memcpy(tsr_data_cpy, tsr_data, sr->el_size*old_dist.size);
    int sysym[order];
    for (int i=0; i<order; i++){
      if (sym[i] == NS) sysym[i] = NS;
      else sysym[i] = SY;
    }
    char * corr_bucket_data = glb_cyclic_reshuffle(sysym, old_dist, NULL, NULL, new_dist, NULL, NULL, &tsr_data_cpy, &tsr_data2, sr, ord_glb_comm, 1, sr->mulid(), sr->addid());
#endif

    if (order == 0){
      alloc_ptr(sr->el_size, (void**)&tsr_new_data);
      if (ord_glb_comm.rank == 0){
        sr->copy(tsr_new_data, tsr_data);
      } else {
        sr->copy(tsr_new_data, sr->addid());
      }
      *ptr_tsr_new_data = tsr_new_data;
      return;
    }
    TAU_FSTART(phase_reshuffle);

    int * old_virt_lda, * new_virt_lda;
    alloc_ptr(order*sizeof(int),     (void**)&old_virt_lda);
    alloc_ptr(order*sizeof(int),     (void**)&new_virt_lda);

    int new_nvirt=1, old_nvirt=1;
    int old_idx_lyr = ord_glb_comm.rank - old_dist.perank[0]*old_dist.pe_lda[0];
    int new_idx_lyr = ord_glb_comm.rank - new_dist.perank[0]*new_dist.pe_lda[0];
    //Ignore virt phase along first dim
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

    int *old_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_phys_edge_len);
    for (int dim = 0;dim < order;dim++) 
      old_phys_edge_len[dim] = old_dist.pad_edge_len[dim]/old_dist.phys_phase[dim];

    int *new_phys_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_phys_edge_len);
    for (int dim = 0;dim < order;dim++)
      new_phys_edge_len[dim] = new_dist.pad_edge_len[dim]/new_dist.phys_phase[dim];

    int *old_virt_edge_len; alloc_ptr(sizeof(int)*order, (void**)&old_virt_edge_len);
    for (int dim = 0;dim < order;dim++) 
      old_virt_edge_len[dim] = old_phys_edge_len[dim]/old_dist.virt_phase[dim];

    int *new_virt_edge_len; alloc_ptr(sizeof(int)*order, (void**)&new_virt_edge_len);
    for (int dim = 0;dim < order;dim++) 
      new_virt_edge_len[dim] = new_phys_edge_len[dim]/new_dist.virt_phase[dim];
    
    int64_t * send_counts = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
    calc_drv_displs(sym, edge_len, old_phys_edge_len, old_dist, new_dist, send_counts, ord_glb_comm, old_idx_lyr);

    int64_t * send_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
    send_displs[0] = 0;
    for (int i=1; i<ord_glb_comm.np; i++){
      send_displs[i] = send_displs[i-1] + send_counts[i-1];
    }

    if (old_idx_lyr == 0){
      char * aux_buf; alloc_ptr(sr->el_size*old_dist.size, (void**)&aux_buf);

      if (old_dist.virt_phase[0] != 1){
        //transpose so that innermost dimension of all locally data is in contiguous global order
        TAU_FSTART(phreshuffle_pretranspose);
        for (int i=0; i<old_nvirt*old_dist.virt_phase[0]; ){
          for (int j=0; j<old_dist.virt_phase[0]; j++, i++){
            sr->copy(old_virt_nelem/old_dist.virt_phase[0], tsr_data+sr->el_size*i*old_virt_nelem/old_dist.virt_phase[0], 1,
                     aux_buf+sr->el_size*(j+(i/old_dist.virt_phase[0])*old_virt_nelem), old_dist.virt_phase[0]);
          }
        }
        TAU_FSTOP(phreshuffle_pretranspose);
        //FIXME don't need
#if 0
        char * auxx_buf; alloc_ptr(sr->el_size*old_dist.size, (void**)&auxx_buf);
        tsr_data = auxx_buf;
#endif
      } else {
        char * tmp = aux_buf;
        aux_buf = tsr_data;
        tsr_data = tmp;
      }

      int old_rep_phase0 = lcm(old_dist.phys_phase[0], new_dist.phys_phase[0])/old_dist.phys_phase[0];

      int ** bucket_offset = 
        compute_bucket_offsets(old_dist,
                               new_dist,
                               edge_len,
                               old_phys_edge_len,
                               old_virt_lda,
                               NULL,
                               NULL,
                               new_phys_edge_len,
                               new_virt_lda,
                               1,
                               old_nvirt,
                               new_nvirt,
                               old_virt_edge_len);
      char ** buckets = (char**)alloc(sizeof(char**)*ord_glb_comm.np);

      buckets[0] = tsr_data;
      for (int i=1; i<ord_glb_comm.np; i++){
        buckets[i] = buckets[i-1] + sr->el_size*send_counts[i-1];
      }
#if DEBUG >= 1
      int64_t save_counts[ord_glb_comm.np];
      memcpy(save_counts, send_counts, sizeof(int64_t)*ord_glb_comm.np); 
#endif
      std::fill(send_counts, send_counts+ord_glb_comm.np, 0);
      TAU_FSTART(redist_bucket);
      SWITCH_ORD_CALL(redist_bucket, order-1, sym, old_dist.phys_phase, old_dist.perank, edge_len, old_virt_edge_len,
                      old_dist.virt_phase, old_virt_lda, old_virt_nelem, bucket_offset, 
                      old_rep_phase0, 1, aux_buf, buckets, send_counts, sr);
      TAU_FSTOP(redist_bucket);
      cfree(buckets);

#if DEBUG>= 1
      bool pass = true;
      for (int i=0; i<ord_glb_comm.np; i++){
  //      if (ord_glb_comm.rank == 1)
          //printf("[%d] send_counts[%d] = %ld, redist_bucket counts[%d] = %ld\n", ord_glb_comm.rank, i, save_counts[i], i, send_counts[i]);
        if (save_counts[i] != send_counts[i]) pass = false;
      }
#endif
#if 0
      assert(pass);
      int off = 0;
      for (int i=0; i<ord_glb_comm.np; i++){
        for (int j=0; j<send_counts[i]; j++, off++){
          /*if (ord_glb_comm.rank == 0){
            printf("sending %dth element, ",j);
            sr->print(tsr_data+off*sr->el_size);
            printf(", to processor %d, should be sending ",i);
            sr->print(corr_bucket_data+off*sr->el_size);
            printf("\n");
          }*/
          if (!sr->isequal(tsr_data+off*sr->el_size,corr_bucket_data+off*sr->el_size))
            pass = false;
        } 
      }
      assert(pass);
#endif      
      if (aux_buf != *ptr_tsr_data)
        cfree(aux_buf);
    }
    int64_t * recv_counts = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
    calc_drv_displs(sym, edge_len, new_phys_edge_len, new_dist, old_dist, recv_counts, ord_glb_comm, new_idx_lyr);
    int64_t * recv_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);

    recv_displs[0] = 0;
    for (int i=1; i<ord_glb_comm.np; i++){
      recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }

    char * recv_buffer;
    mst_alloc_ptr(new_dist.size*sr->el_size, (void**)&recv_buffer);

    /* Communicate data */
    TAU_FSTART(ALL_TO_ALL_V);
    ord_glb_comm.all_to_allv(tsr_data, send_counts, send_displs, sr->el_size,
                             recv_buffer, recv_counts, recv_displs);
    TAU_FSTOP(ALL_TO_ALL_V);

    cfree(tsr_data);

    if (new_idx_lyr == 0){
      char * aux_buf; alloc_ptr(sr->el_size*new_dist.size, (void**)&aux_buf);
      sr->set(aux_buf, sr->addid(), new_dist.size);

      int new_rep_phase0 = lcm(new_dist.phys_phase[0], old_dist.phys_phase[0])/new_dist.phys_phase[0];

      int ** bucket_offset = 
        compute_bucket_offsets(new_dist,
                               old_dist,
                               edge_len,
                               new_phys_edge_len,
                               new_virt_lda,
                               NULL,
                               NULL,
                               old_phys_edge_len,
                               old_virt_lda,
                               1,
                               new_nvirt,
                               old_nvirt,
                               new_virt_edge_len);
      char ** buckets = (char**)alloc(sizeof(char**)*ord_glb_comm.np);

      buckets[0] = recv_buffer;
      //printf("[%d] size of %dth bucket is %ld\n", ord_glb_comm.rank, 0, send_counts[0]);
      for (int i=1; i<ord_glb_comm.np; i++){
        buckets[i] = buckets[i-1] + sr->el_size*recv_counts[i-1];
        //printf("[%d] size of %dth bucket is %ld\n", ord_glb_comm.rank, i, send_counts[i]);
      }

#if DEBUG >= 1
      int64_t save_counts[ord_glb_comm.np];
      memcpy(save_counts, recv_counts, sizeof(int64_t)*ord_glb_comm.np); 
#endif
      std::fill(recv_counts, recv_counts+ord_glb_comm.np, 0);

      TAU_FSTART(redist_debucket);
      SWITCH_ORD_CALL(redist_bucket, order-1, sym, new_dist.phys_phase, new_dist.perank, edge_len, new_virt_edge_len,
                      new_dist.virt_phase, new_virt_lda, new_virt_nelem, bucket_offset, 
                      new_rep_phase0, 0, aux_buf, buckets, recv_counts, sr);
      TAU_FSTOP(redist_debucket);

      cfree(buckets);
#if DEBUG >= 1
      bool pass = true;
      for (int i=0; i<ord_glb_comm.np; i++){
        if (save_counts[i] != recv_counts[i]) pass = false;
      }
      assert(pass);
#endif

      if (new_dist.virt_phase[0] != 1){
        //transpose so that innermost dimension of all locally data is in contiguous global order
        TAU_FSTART(phreshuffle_posttranspose);
        for (int i=0; i<new_nvirt*new_dist.virt_phase[0]; ){
          for (int j=0; j<new_dist.virt_phase[0]; j++, i++){
            sr->copy(new_virt_nelem/new_dist.virt_phase[0], 
                     aux_buf+sr->el_size*(j+(i/new_dist.virt_phase[0])*new_virt_nelem), new_dist.virt_phase[0],
                     recv_buffer+sr->el_size*i*new_virt_nelem/new_dist.virt_phase[0], 1);
          }
        }
        *ptr_tsr_new_data = recv_buffer;
        cfree(aux_buf);
        /*int tr_order=1;
        int * nontriv_dims = (int*)alloc(sizeof(int)*(order+1));
        int * transp_lens = (int*)alloc(sizeof(int)*(order+1));
        int * new_ordering = (int*)alloc(sizeof(int)*(order+1));
        transp_lens[0] = new_virt_nelem/new_dist.virt_phase[0];
        new_ordering[0] = 1;
        new_ordering[1] = 0;
        for (int i=0; i<order; i++){
          if (new_dist.virt_phase[i] > 0){
            transp_lens[tr_order] = new_dist.virt_phase[i];
            if (tr_order>1){
              new_ordering[tr_order]=tr_order;
            }
            tr_order++;
          }
        }
        nosym_transpose(tr_order, new_ordering, transp_lens, aux_buf, 0, sr);
        *ptr_tsr_new_data = aux_buf; cfree(recv_buffer);*/
        TAU_FSTOP(phreshuffle_posttranspose);
      } else {
        *ptr_tsr_new_data = aux_buf;
        cfree(recv_buffer);
      }
    } else {
      sr->set(recv_buffer, sr->addid(), new_dist.size);
      *ptr_tsr_new_data = recv_buffer;
    }
    TAU_FSTOP(phase_reshuffle);
  }
}

