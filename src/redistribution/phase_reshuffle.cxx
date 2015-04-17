/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "phase_reshuffle.h"
#include "glb_cyclic_reshuffle.h"
#include "../shared/util.h"
#include "nosym_transp.h"

#define MTAG 777
#define ROR
#ifdef ROR
  //#define IREDIST
  #define REDIST_PUT
  #ifndef ROR_MIN_LOOP
  #define ROR_MIN_LOOP 0
  #endif
#endif


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
        int jed = get_loc(std::min(edge_len[idim]-1,get_glb(i,sphase[idim],gidx_off[idim])),sphase[idim-1],gidx_off[idim-1]);
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
                             int * const *   pe_offset,
                             int * const *   bucket_offset,
                             int64_t const * old_counts,
                             int64_t *       counts,
                             int             bucket_off,
                             int             pe_off,
                             int             dir){
    for (int i=0; i<rep_phase[idim]; i++){
      int rec_bucket_off = bucket_off+bucket_offset[idim][i];
      int rec_pe_off = pe_off+pe_offset[idim][i];
      calc_cnt_from_rep_cnt<idim-1>(rep_phase, pe_offset, bucket_offset, old_counts, counts, rec_bucket_off, rec_pe_off, dir);
//                                    coff+new_pe_lda[idim]*((rank[idim]+i*old_phys_phase[idim])%new_phys_phase[idim]),
  //                                  roff+rep_phase_lda[idim]*i, dir);
    }
  }

  template <>
  void calc_cnt_from_rep_cnt<0>
                            (int const *     rep_phase,
                             int * const *   pe_offset,
                             int * const *   bucket_offset,
                             int64_t const * old_counts,
                             int64_t *       counts,
                             int             bucket_off,
                             int             pe_off,
                             int             dir){
    if (dir){
      for (int i=0; i<rep_phase[0]; i++){
        counts[pe_off+pe_offset[0][i]] = old_counts[bucket_off+i];
      }
    } else {
      for (int i=0; i<rep_phase[0]; i++){
        counts[bucket_off+i] = old_counts[pe_off+pe_offset[0][i]];
      }

    }
  }

  void calc_drv_displs(int const *          sym,
                       int const *          edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int64_t *            counts,
                       int                  idx_lyr){
    TAU_FSTART(calc_drv_displs);
    int * rep_phase, * gidx_off, * sphase;
    int * rep_phase_lda;
    int * new_loc_edge_len;
    if (idx_lyr == 0){
      int order = old_dist.order;
      rep_phase     = (int*)alloc(order*sizeof(int));
      rep_phase_lda = (int*)alloc(order*sizeof(int));
      sphase        = (int*)alloc(order*sizeof(int));
      gidx_off      = (int*)alloc(order*sizeof(int));
      new_loc_edge_len      = (int*)alloc(order*sizeof(int));
      int nrep = 1;
      for (int i=0; i<order; i++){
        //FIXME: computed elsewhere already
        rep_phase_lda[i]  = nrep;
        sphase[i]         = lcm(old_dist.phys_phase[i],new_dist.phys_phase[i]);
        rep_phase[i]      = sphase[i] / old_dist.phys_phase[i];
        gidx_off[i]       = old_dist.perank[i];
        nrep             *= rep_phase[i];
        new_loc_edge_len[i] = (edge_len[i]+sphase[i]-1)/sphase[i];
        //printf("rep_phase[%d] = %d, sphase = %d lda = %d, gidx = %d\n",i,rep_phase[i], sphase[i], rep_phase_lda[i], gidx_off[i]);
      }
      ASSERT(order>0);
      SWITCH_ORD_CALL(calc_drv_cnts, order-1, order, sym, counts, rep_phase, rep_phase_lda, sphase, old_dist.phys_phase, gidx_off, edge_len, new_loc_edge_len)
    
      cfree(rep_phase);
      cfree(rep_phase_lda);
      cfree(sphase);
      cfree(gidx_off);
      cfree(new_loc_edge_len);
    }
    TAU_FSTOP(calc_drv_displs);
  }


  void precompute_offsets(distribution const & old_dist,
                          distribution const & new_dist,
                          int const *          sym,
                          int const *          len,
                          int const *          rep_phase,
                          int const *          phys_edge_len,
                          int const *          virt_edge_len,
                          int const *          virt_dim,
                          int const *          virt_lda,
                          int64_t              virt_nelem,
                          int **               pe_offset,
                          int **               bucket_offset,
                          int64_t **           data_offset){
    TAU_FSTART(precompute_offsets);
   
    int rep_phase_lda = 1; 
    for (int dim = 0;dim < old_dist.order;dim++){
      alloc_ptr(sizeof(int)*std::max(rep_phase[dim],phys_edge_len[dim]), (void**)&pe_offset[dim]);
      alloc_ptr(sizeof(int)*std::max(rep_phase[dim],phys_edge_len[dim]), (void**)&bucket_offset[dim]);
      alloc_ptr(sizeof(int64_t)*std::max(rep_phase[dim],phys_edge_len[dim]), (void**)&data_offset[dim]);

      int nsym;
      int pidx = 0;
      int64_t data_stride, sub_data_stride;

      if (dim > 0 && sym[dim-1] != NS){
        int jdim=dim-2;
        nsym = 1;
        while (jdim>=0 && sym[jdim] != NS){ nsym++; jdim--; }
        if (jdim+1 == 0){
          sub_data_stride = 1;
        } else {
          sub_data_stride = sy_packed_size(jdim+1, virt_edge_len, sym);
        }
        data_stride = 0;
      } else {
        nsym = 0; //not used
        sub_data_stride = 0; //not used
        if (dim == 0) data_stride = 1;
        else {
          data_stride = sy_packed_size(dim, virt_edge_len, sym);
        }
      }

      int64_t data_off =0; 

      for (int vidx = 0;
           vidx < std::max((rep_phase[dim]+old_dist.virt_phase[dim]-1)/old_dist.virt_phase[dim],virt_edge_len[dim]);
           vidx++){

        int64_t rec_data_off = data_off;
        if (dim > 0 && sym[dim-1] != NS){
          data_stride = (vidx+1)*sub_data_stride;
          for (int j=1; j<nsym; j++){
            data_stride = (data_stride*(vidx+j+1))/(j+1);
          }
        }
        data_off += data_stride;
        for (int vr = 0;vr < old_dist.virt_phase[dim];vr++,pidx++){

          data_offset[dim][pidx] = rec_data_off;
          rec_data_off += virt_lda[dim]*virt_nelem; 

          int64_t gidx = (int64_t)vidx*old_dist.phase[dim]+old_dist.perank[dim]+(int64_t)vr*old_dist.phys_phase[dim];
          int phys_rank = gidx%new_dist.phys_phase[dim];
          pe_offset[dim][pidx] = phys_rank*MAX(1,new_dist.pe_lda[dim]);
          bucket_offset[dim][pidx] = (pidx%rep_phase[dim])*rep_phase_lda;
        }
      }
      rep_phase_lda *= rep_phase[dim];
    }


    TAU_FSTOP(precompute_offsets);

  }


  template <int idim>
  void redist_bucket(int const *          sym,
                     int const *          phys_phase,
                     int const *          perank,
                     int const *          edge_len,
                     int * const *        bucket_offset,
                     int64_t * const *    data_offset,
                     int                  rep_phase0,
                     int                  virt_dim0,
                     bool                 data_to_buckets,
                     char * __restrict__  data,
                     char ** __restrict__ buckets,
                     int64_t *            counts,
                     algstrct const *     sr,
                     int64_t              data_off,
                     int                  bucket_off,
                     int                  prev_idx){
    int ivmax;
    if (sym[idim] != NS){
      ivmax = get_loc(get_glb(prev_idx, phys_phase[idim+1], perank[idim+1]),
                                        phys_phase[idim  ], perank[idim  ]);
    } else
      ivmax = get_loc(edge_len[idim]-1, phys_phase[idim], perank[idim]);
    
    for (int iv=0; iv <= ivmax; iv++){
      int rec_bucket_off   = bucket_off + bucket_offset[idim][iv];
      int64_t rec_data_off = data_off   + data_offset[idim][iv];
      redist_bucket<idim-1>(sym, phys_phase, perank, edge_len, bucket_offset, data_offset, rep_phase0, virt_dim0, data_to_buckets, data, buckets, counts, sr, rec_data_off, rec_bucket_off, iv);
    }
  }


  template <>
  void redist_bucket<0>(int const *          sym,
                        int const *          phys_phase,
                        int const *          perank,
                        int const *          edge_len,
                        int * const *        bucket_offset,
                        int64_t * const *    data_offset,
                        int                  rep_phase0,
                        int                  virt_dim0,
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
    //printf("ivmax = %d, rep_phase0 = %d data_off = %ld\n",ivmax, rep_phase0, data_off);
    if (virt_dim0 == 1){
      if (data_to_buckets){
        for (int i=0; i<rep_phase0; i++){
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off + bucket_offset[0][i];
            //printf("ivmax = %d bucket_off = %d, bucket = %d, counts[bucket] = %ld, n= %d data_off = %ld, rep_phase=%d\n",
         //          ivmax, bucket_off, bucket, counts[bucket], n, data_off, rep_phase0);
            sr->copy(n,
                     data + sr->el_size*(data_off+i), rep_phase0, 
                     buckets[bucket] + sr->el_size*counts[bucket], 1);
            counts[bucket] += n;
          }
        }
      } else {
        for (int i=0; i<rep_phase0; i++){
          int n = (ivmax-i+rep_phase0-1)/rep_phase0;
          if (n>0){
            int bucket = bucket_off + bucket_offset[0][i];
            sr->copy(n,
                     buckets[bucket] + sr->el_size*counts[bucket], 1,
                     data + sr->el_size*(data_off+i), rep_phase0);
            counts[bucket] += n;
          }
        }
      }
    } else {
      if (data_to_buckets){
        for (int i=0, iv=0; iv < ivmax; i++){
          for (int v=0; v<virt_dim0 && iv < ivmax; v++, iv++){
            int bucket = bucket_off + bucket_offset[0][iv];
            sr->copy(buckets[bucket] + sr->el_size*counts[bucket],
                     data + sr->el_size*(data_off+data_offset[0][iv]));//i+v*virt_nelem)); 
            counts[bucket]++;
          }
        }
      } else {
        for (int i=0, iv=0; iv < ivmax; i++){
          for (int v=0; v<virt_dim0 && iv < ivmax; v++, iv++){
            int bucket = bucket_off + bucket_offset[0][iv];
            sr->copy(data + sr->el_size*(data_off+data_offset[0][iv]),//+i+v*virt_nelem), 
                     buckets[bucket] + sr->el_size*counts[bucket]);
            counts[bucket]++;
          }
        }
      }
    }
    /*else {
      for (int i=0; i<rep_phase0; i++){
        int n = (ivmax-i+rep_phase0-1)/rep_phase0;
        if (n>0){
          int bucket = bucket_off + bucket_offset[0][i];
          sr->copy(n,
                   buckets[bucket] + sr->el_size*counts[bucket], 1,
                   data + sr->el_size*(data_off+i), rep_phase0);
          counts[bucket] += n;
         }
       }
     }*/

  }

  template <int idim>
  void isendrecv(int * const *    pe_offset,
                 int * const *    bucket_offset,
                 int const *      rep_phase,
                 int64_t const *  counts,
                 int64_t const *  displs,
                 MPI_Request *    reqs,
                 MPI_Comm         cm,
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
                 MPI_Request *    reqs,
                 MPI_Comm         cm,
                 char *           buffer,
                 algstrct const * sr,
                 int              bucket_off,
                 int              pe_off,
                 int              dir){
    for (int r=0; r<rep_phase[0]; r++){
      int bucket = bucket_off+r;
      int pe = pe_off+pe_offset[0][r];
      if (dir)
        MPI_Irecv(buffer+displs[bucket]*sr->el_size, counts[bucket], sr->mdtype(), pe, MTAG, cm, reqs+bucket);
      else
        MPI_Isend(buffer+displs[bucket]*sr->el_size, counts[bucket], sr->mdtype(), pe, MTAG, cm, reqs+bucket);
    }
  }

#ifdef ROR
  template <int idim>
  void redist_bucket_ror(int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
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
    int ivmax;
    if (sym[idim] != NS){
      ivmax = get_loc(get_glb(prev_idx, phys_phase[idim+1], perank[idim+1]),
                                        phys_phase[idim  ], perank[idim  ]);
    } else
      ivmax = get_loc(edge_len[idim]-1, phys_phase[idim], perank[idim]);
  
    for (int iv=rep_idx[idim]; iv<=ivmax; iv+=rep_phase[idim]){
      int64_t rec_data_off = data_off + data_offset[idim][iv];
      redist_bucket_ror<idim-1>(sym, phys_phase, perank, edge_len, bucket_offset, data_offset, rep_phase, rep_idx, virt_dim0, data_to_buckets, data, buckets, counts, sr, rec_data_off, bucket_off, iv);
    }
  }

  template <>
  void redist_bucket_ror<ROR_MIN_LOOP>
                        (int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
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
    redist_bucket<ROR_MIN_LOOP>(sym, phys_phase, perank, edge_len, bucket_offset, data_offset, rep_phase[0], virt_dim0, data_to_buckets, data, buckets, counts, sr, data_off, bucket_off, prev_idx);
  }

#ifdef REDIST_PUT
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
      MPI_Put(buckets[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), rec_pe_off, put_displs[rec_bucket_off], counts[rec_bucket_off], sr->mdtype(), win);
    }
  }
/*
  template <int idim>
  void redist_bucket_ror_put
                        (int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        pe_offset,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
                         int const *          rep_phase,
                         int const *          rep_idx,
                         bool                 data_to_buckets,
                         char * __restrict__  data,
                         char ** __restrict__ buckets,
                         int64_t *            counts,
                         int64_t const *      put_displs,
                         CTF_Win &            win,
                         algstrct const *     sr,
                         int64_t              data_off=0,
                         int                  bucket_off=0,
                         int                  prev_idx=0){
    int ivmax;
    if (sym[idim] != NS){
      ivmax = get_loc(get_glb(prev_idx, phys_phase[idim+1], perank[idim+1]),
                                        phys_phase[idim  ], perank[idim  ]);
    } else
      ivmax = get_loc(edge_len[idim]-1, phys_phase[idim], perank[idim]);
   
    int rec_bucket_off = bucket_off + bucket_offset[idim][r];
    int rec_pe_off = pe_off + pe_offset[idim][r];
    for (int iv=rep_idx[idim]; iv<=ivmax; iv+=rep_phase[idim]){
      int64_t rec_data_off = data_off + data_offset[idim][iv];
      redist_bucket_ror<idim-1>(sym, phys_phase, perank, edge_len,bucket_offset, data_offset, rep_phase, rep_idx, data_to_buckets, data, buckets, counts, sr, rec_data_off, rec_bucket_off, iv);
    }
    put_buckets<idim-1>(rep_phase, bucket_offset, buckets, counts, sr, put_displs, win, rec_bucket_off, rec_pe_off);
  }

  template <>
  void redist_bucket_ror_put<ROR_MIN_LOOP>
                        (int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        pe_offset,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
                         int const *          rep_phase,
                         bool                 data_to_buckets,
                         char * __restrict__  data,
                         char ** __restrict__ buckets,
                         int64_t *            counts,
                         int64_t const *      put_displs,
                         CTF_Win &            win,
                         algstrct const *     sr,
                         int64_t              data_off,
                         int                  bucket_off,
                         int                  prev_idx){ }*/
#endif


  template <int idim>
  void redist_bucket_isr(int                  order,
                         int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        pe_offset,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
                         int const *          rep_phase,
                         int *                rep_idx,
                         int                  virt_dim0,
#ifdef IREDIST
                         MPI_Request *        rep_reqs,
                         MPI_Comm             cm,
#endif
#ifdef  REDIST_PUT
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
    if (rep_phase[idim] == 1){
      int rec_bucket_off = bucket_off + bucket_offset[idim][0];
      int rec_pe_off = pe_off + pe_offset[idim][0];
      redist_bucket_isr<idim-1>(order,sym, phys_phase, perank, edge_len, pe_offset, bucket_offset, data_offset, rep_phase, rep_idx, virt_dim0,
#ifdef IREDIST
                                rep_reqs, cm, 
#endif
#ifdef  REDIST_PUT
                                put_displs, win,
#endif
                                data_to_buckets, data, buckets, counts, sr, rec_bucket_off, rec_pe_off);
    } else {
  #ifdef USE_OMP
      #pragma omp parallel for
  #endif
      for (int r=0; r<rep_phase[idim]; r++){
        int rep_idx2[order];
        memcpy(rep_idx2, rep_idx, sizeof(int)*order);
        rep_idx2[idim] = r;
        int rec_bucket_off = bucket_off + bucket_offset[idim][r];
        int rec_pe_off = pe_off + pe_offset[idim][r];
        redist_bucket_isr<idim-1>(order,sym, phys_phase, perank, edge_len, pe_offset, bucket_offset, data_offset, rep_phase, rep_idx2, virt_dim0,
#ifdef IREDIST
                                  rep_reqs, cm, 
#endif
#ifdef  REDIST_PUT
                                  put_displs, win,
#endif
                                  data_to_buckets, data, buckets, counts, sr, rec_bucket_off, rec_pe_off);
      }    
    }
  }


  template <>
  void redist_bucket_isr<0>
                        (int                  order,
                         int const *          sym,
                         int const *          phys_phase,
                         int const *          perank,
                         int const *          edge_len,
                         int * const *        pe_offset,
                         int * const *        bucket_offset,
                         int64_t * const *    data_offset,
                         int const *          rep_phase,
                         int *                rep_idx,
                         int                  virt_dim0,
#ifdef IREDIST
                         MPI_Request *        rep_reqs,
                         MPI_Comm             cm,
#endif
#ifdef  REDIST_PUT
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
#ifdef IREDIST
    if (!data_to_buckets){
      MPI_Waitall(rep_phase[0], rep_reqs+bucket_off, MPI_STATUSES_IGNORE);
    }
#endif
    SWITCH_ORD_CALL(redist_bucket_ror, order-1, sym, phys_phase, perank, edge_len, bucket_offset, data_offset, rep_phase, rep_idx, virt_dim0, data_to_buckets, data, buckets, counts, sr, 0, bucket_off, 0)
    if (data_to_buckets){
#ifdef IREDIST
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
#ifdef  REDIST_PUT
      put_buckets<0>(rep_phase, pe_offset, bucket_offset, buckets, counts, sr, put_displs, win, bucket_off, pe_off);
#endif
    }
  }
#endif

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

    if (order == 0){
      alloc_ptr(sr->el_size, (void**)&tsr_new_data);
      if (ord_glb_comm.rank == 0){
        sr->copy(tsr_new_data, tsr_data);
      } else {
        sr->copy(tsr_new_data, sr->addid());
      }
      *ptr_tsr_new_data = tsr_new_data;
      cfree(tsr_data);
      return;
    }
    TAU_FSTART(phase_reshuffle);

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

    int nold_rep = 1;
    int * old_rep_phase; alloc_ptr(sizeof(int)*order, (void**)&old_rep_phase);
    int * old_rep_phase_lda; alloc_ptr(sizeof(int)*order, (void**)&old_rep_phase_lda);
    for (int i=0; i<order; i++){
      old_rep_phase[i] = lcm(old_dist.phys_phase[i], new_dist.phys_phase[i])/old_dist.phys_phase[i];
      old_rep_phase_lda[i] = nold_rep;
      nold_rep *= old_rep_phase[i];
    }

    int nnew_rep = 1;
    int * new_rep_phase; alloc_ptr(sizeof(int)*order, (void**)&new_rep_phase);
    int * new_rep_phase_lda; alloc_ptr(sizeof(int)*order, (void**)&new_rep_phase_lda);
    for (int i=0; i<order; i++){
      new_rep_phase[i] = lcm(new_dist.phys_phase[i], old_dist.phys_phase[i])/new_dist.phys_phase[i];
      new_rep_phase_lda[i] = nnew_rep;
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
    MPI_Request * recv_reqs = (MPI_Request*)alloc(sizeof(MPI_Request)*nnew_rep);
    MPI_Request * send_reqs = (MPI_Request*)alloc(sizeof(MPI_Request)*nold_rep);
    char * recv_buffer;
    mst_alloc_ptr(new_dist.size*sr->el_size, (void**)&recv_buffer);
#endif

    for (int i=0; i<nnew_rep; i++){
      if (i==0)
        recv_displs[0] = 0;
      else
        recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
    }

    int ** recv_bucket_offset; alloc_ptr(sizeof(int*)*order, (void**)&recv_bucket_offset);
    int ** recv_pe_offset; alloc_ptr(sizeof(int*)*order, (void**)&recv_pe_offset);
    int64_t ** recv_data_offset; alloc_ptr(sizeof(int64_t*)*order, (void**)&recv_data_offset);
    precompute_offsets(new_dist, old_dist, sym, edge_len, new_rep_phase, new_phys_edge_len, new_virt_edge_len, new_dist.virt_phase, new_virt_lda, new_virt_nelem, recv_pe_offset, recv_bucket_offset, recv_data_offset);

    int ** send_pe_offset; alloc_ptr(sizeof(int*)*order, (void**)&send_pe_offset);
    int ** send_bucket_offset; alloc_ptr(sizeof(int*)*order, (void**)&send_bucket_offset);
    int64_t ** send_data_offset; alloc_ptr(sizeof(int64_t*)*order, (void**)&send_data_offset);

    precompute_offsets(old_dist, new_dist, sym, edge_len, old_rep_phase, old_phys_edge_len, old_virt_edge_len, old_dist.virt_phase, old_virt_lda, old_virt_nelem, send_pe_offset, send_bucket_offset, send_data_offset);

#ifdef IREDIST
    if (new_idx_lyr == 0)
      SWITCH_ORD_CALL(isendrecv, order-1, recv_pe_offset, recv_bucket_offset, new_rep_phase, recv_counts, recv_displs, recv_reqs, ord_glb_comm.cm, recv_buffer, sr, 0, 0, 1);
#endif
#ifndef IREDIST
#ifndef REDIST_PUT
    int64_t * send_displs = (int64_t*)alloc(sizeof(int64_t)*nold_rep);
    send_displs[0] = 0;
    for (int i=1; i<nold_rep; i++){
      send_displs[i] = send_displs[i-1] + send_counts[i-1];
    }
#else
    int64_t * all_recv_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
    SWITCH_ORD_CALL(calc_cnt_from_rep_cnt, order-1, new_rep_phase, recv_pe_offset, recv_bucket_offset, recv_displs, all_recv_displs, 0, 0, 1);

    int64_t * all_put_displs = (int64_t*)alloc(sizeof(int64_t)*ord_glb_comm.np);
    MPI_Alltoall(all_recv_displs, 1, MPI_INT64_T, all_put_displs, 1, MPI_INT64_T, ord_glb_comm.cm);
    cfree(all_recv_displs);

    int64_t * put_displs = (int64_t*)alloc(sizeof(int64_t)*nold_rep);
    SWITCH_ORD_CALL(calc_cnt_from_rep_cnt, order-1, old_rep_phase, send_pe_offset, send_bucket_offset, all_put_displs, put_displs, 0, 0, 0);

    cfree(all_put_displs);

    char * recv_buffer;
    mst_alloc_ptr(new_dist.size*sr->el_size, (void**)&recv_buffer);

    CTF_Win win;
    int suc = MPI_Win_create(recv_buffer, new_dist.size*sr->el_size, sr->el_size, MPI_INFO_NULL, ord_glb_comm.cm, &win);
    assert(suc == MPI_SUCCESS);
    MPI_Win_fence(0, win);
#endif
#endif

    if (old_idx_lyr == 0){
      char * aux_buf; alloc_ptr(sr->el_size*old_dist.size, (void**)&aux_buf);
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
      SWITCH_ORD_CALL(redist_bucket_isr, order-1, order, sym, old_dist.phys_phase, old_dist.perank, edge_len, send_pe_offset, send_bucket_offset, send_data_offset,
                      old_rep_phase, old_rep_idx, old_dist.virt_phase[0],
#ifdef IREDIST
                      send_reqs, ord_glb_comm.cm,
#endif
#ifdef REDIST_PUT
                      put_displs, win,
#endif
                      1, aux_buf, buckets, send_counts, sr);
      cfree(old_rep_idx);
#else
      SWITCH_ORD_CALL(redist_bucket, order-1, sym, old_dist.phys_phase, old_dist.perank, edge_len, 
                      send_bucket_offset, send_data_offset,
                      old_rep_phase[0], old_dist.virt_phase[0], 1, aux_buf, buckets, send_counts, sr);
#endif
      TAU_FSTOP(redist_bucket);
      cfree(buckets);

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
      assert(pass);
#endif
      cfree(aux_buf);
    }
#ifndef IREDIST
#ifndef REDIST_PUT
    char * recv_buffer;
    mst_alloc_ptr(new_dist.size*sr->el_size, (void**)&recv_buffer);

    /* Communicate data */
    TAU_FSTART(COMM_RESHUFFLE);

    MPI_Request * reqs = (MPI_Request*)alloc(sizeof(MPI_Request)*(nnew_rep+nold_rep));
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
    //ord_glb_comm.all_to_allv(tsr_data, send_counts, send_displs, sr->el_size,
    //                         recv_buffer, recv_counts, recv_displs);
    TAU_FSTOP(COMM_RESHUFFLE);
    cfree(send_displs);
#else
    cfree(put_displs);
    TAU_FSTART(redist_fence);
    MPI_Win_fence(0, win);
    TAU_FSTOP(redist_fence);
    MPI_Win_free(&win);
#endif
    cfree(tsr_data);
#endif
    cfree(send_counts);

    if (new_idx_lyr == 0){
      char * aux_buf; alloc_ptr(sr->el_size*new_dist.size, (void**)&aux_buf);
      sr->set(aux_buf, sr->addid(), new_dist.size);

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
#ifdef ROR
      int * new_rep_idx; alloc_ptr(sizeof(int)*order, (void**)&new_rep_idx);
      memset(new_rep_idx, 0, sizeof(int)*order);
      SWITCH_ORD_CALL(redist_bucket_isr, order-1, order, sym, new_dist.phys_phase, new_dist.perank, edge_len, recv_pe_offset, recv_bucket_offset, recv_data_offset,
                      new_rep_phase, new_rep_idx, new_dist.virt_phase[0],
#ifdef IREDIST
                      recv_reqs, ord_glb_comm.cm,
#endif
#ifdef  REDIST_PUT
                      NULL, win,
#endif
                      0, aux_buf, buckets, recv_counts, sr);
      cfree(new_rep_idx);
#else
      SWITCH_ORD_CALL(redist_bucket, order-1, sym, new_dist.phys_phase, new_dist.perank, edge_len,
                      recv_bucket_offset, recv_data_offset,
                      new_rep_phase[0], new_dist.virt_phase[0], 0, aux_buf, buckets, recv_counts, sr);
#endif
      TAU_FSTOP(redist_debucket);
      cfree(buckets);
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
      assert(pass);
#endif
      *ptr_tsr_new_data = aux_buf;
      cfree(recv_buffer);
    } else {
      sr->set(recv_buffer, sr->addid(), new_dist.size);
      *ptr_tsr_new_data = recv_buffer;
    }
    //printf("[%d] reached final barrier %d\n",ord_glb_comm.rank, MTAG);
#ifdef IREDIST
    cfree(recv_reqs);
    cfree(send_reqs);
#endif
    for (int i=0; i<order; i++){
      cfree(recv_pe_offset[i]);
      cfree(recv_bucket_offset[i]);
      cfree(recv_data_offset[i]);
    }
    cfree(recv_pe_offset);
    cfree(recv_bucket_offset);
    cfree(recv_data_offset);

    for (int i=0; i<order; i++){
      cfree(send_pe_offset[i]);
      cfree(send_bucket_offset[i]);
      cfree(send_data_offset[i]);
    }
    cfree(send_pe_offset);
    cfree(send_bucket_offset);
    cfree(send_data_offset);

    cfree(old_virt_lda);
    cfree(new_virt_lda);
    cfree(recv_counts);
    cfree(recv_displs);
    cfree(old_phys_edge_len);
    cfree(new_phys_edge_len);
    cfree(old_virt_edge_len);
    cfree(new_virt_edge_len);
    cfree(old_rep_phase);
    cfree(new_rep_phase);
#ifdef IREDIST
    TAU_FSTART(barrier_after_phase_reshuffle);
    MPI_Barrier(ord_glb_comm.cm);
    TAU_FSTOP(barrier_after_phase_reshuffle);
    cfree(tsr_data);
#endif
    TAU_FSTOP(phase_reshuffle);
  }
}

