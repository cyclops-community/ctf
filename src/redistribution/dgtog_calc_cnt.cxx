/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dgtog_calc_cnt.h"
#include "../shared/util.h"
#include "../interface/common.h"

namespace CTF_int {
  //correct for SY
  inline int64_t get_glb(int64_t i, int s, int64_t t){
    return i*s+t;
  }
  //correct for SH/AS, but can treat everything as SY
  /*inline int get_glb(int i, int s, int t){
    return i*s+t-1;
  }*/
  inline int64_t get_loc(int64_t g, int s, int64_t t){
    //round down, dowwwwwn
    if (t>g) return -1;
    else return (g-t)/s;
  }
   
  template <int idim>
  int64_t calc_cnt(int const *     sym,
                   int const *     rep_phase,
                   int const *     sphase,
                   int64_t const * gidx_off,
                   int64_t const * edge_len,
                   int64_t const * loc_edge_len){
    assert(sym[idim] == NS); //otherwise should be in calc_sy_pfx
    if (sym[idim-1] == NS){
      return (get_loc(edge_len[idim]-1,sphase[idim],gidx_off[idim])+1)*calc_cnt<idim-1>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
    } else {
      int64_t * pfx = calc_sy_pfx<idim>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
      int64_t cnt = 0;
      for (int i=0; i<=get_loc(edge_len[idim]-1,sphase[idim],gidx_off[idim]); i++){
        cnt += pfx[i];
      }
      cdealloc(pfx);
      return cnt;
    }
  }
 
  template <>
  int64_t calc_cnt<0>(int const *     sym,
                      int const *     rep_phase,
                      int const *     sphase,
                      int64_t const * gidx_off,
                      int64_t const * edge_len,
                      int64_t const * loc_edge_len){
    assert(sym[0] == NS);
    return get_loc(edge_len[0]-1, sphase[0], gidx_off[0])+1;
  }

  template <int idim>
  int64_t * calc_sy_pfx(int const *     sym,
                        int const *     rep_phase,
                        int const *     sphase,
                        int64_t const * gidx_off,
                        int64_t const * edge_len,
                        int64_t const * loc_edge_len){
    int64_t * pfx = (int64_t*)alloc(sizeof(int64_t)*loc_edge_len[idim]);
    if (sym[idim-1] == NS){
      int64_t ns_size = calc_cnt<idim-1>(sym,rep_phase,sphase,gidx_off,edge_len,loc_edge_len);
      for (int64_t i=0; i<loc_edge_len[idim]; i++){
        pfx[i] = ns_size;
      }
    } else {
      int64_t * pfx_m1 = calc_sy_pfx<idim-1>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
      for (int64_t i=0; i<loc_edge_len[idim]; i++){
        int64_t jst;
        if (i>0){
          pfx[i] = pfx[i-1];
          if (sym[idim-1] == SY)
            jst = get_loc(get_glb(i-1,sphase[idim],gidx_off[idim]),sphase[idim-1],gidx_off[idim-1])+1;
          else
            jst = get_loc(get_glb(i-1,sphase[idim],gidx_off[idim])-1,sphase[idim-1],gidx_off[idim-1])+1;
        } else {
          pfx[i] = 0;
          jst = 0;
        }
        int64_t jed;
        if (sym[idim-1] == SY)
          jed = get_loc(std::min(edge_len[idim]-1,get_glb(i,sphase[idim],gidx_off[idim])),sphase[idim-1],gidx_off[idim-1]);
        else
          jed = get_loc(std::min(edge_len[idim]-1,get_glb(i,sphase[idim],gidx_off[idim]))-1,sphase[idim-1],gidx_off[idim-1]);
        for (int j=jst; j<=jed; j++){
          //printf("idim = %d j=%d loc_edge[idim] = %d loc_Edge[idim-1]=%d\n",idim,j,loc_edge_len[idim],loc_edge_len[idim-1]);
          pfx[i] += pfx_m1[j];
        }
      }
      cdealloc(pfx_m1);
    }
    return pfx;
  }
 
  template <>
  int64_t * calc_sy_pfx<1>(int const *     sym,
                           int const *     rep_phase,
                           int const *     sphase,
                           int64_t const * gidx_off,
                           int64_t const * edge_len,
                           int64_t const * loc_edge_len){
    int64_t * pfx= (int64_t*)alloc(sizeof(int64_t)*loc_edge_len[1]);
    if (sym[0] == NS){
      int64_t cnt = calc_cnt<0>(sym, rep_phase, sphase, gidx_off, edge_len, loc_edge_len);
      std::fill(pfx, pfx+loc_edge_len[1], cnt);
    } else if (sym[0] == SY){
      for (int i=0; i<loc_edge_len[1]; i++){
        pfx[i] = get_loc(get_glb(i,sphase[1],gidx_off[1]),sphase[0],gidx_off[0])+1;
      }
    } else {
      for (int i=0; i<loc_edge_len[1]; i++){
        pfx[i] = get_loc(get_glb(i,sphase[1],gidx_off[1])-1,sphase[0],gidx_off[0])+1;
      }
    }
    return pfx;
  }

  template <int idim>
  void calc_drv_cnts(int             order,
                     int const *     sym,
                     int64_t *       counts,
                     int const *     rep_phase,
                     int const *     rep_phase_lda,
                     int const *     sphase,
                     int const *     phys_phase,
                     int64_t   *     gidx_off,
                     int64_t const * edge_len,
                     int64_t const * loc_edge_len){
    for (int i=0; i<rep_phase[idim]; i++, gidx_off[idim]+=phys_phase[idim]){
       calc_drv_cnts<idim-1>(order, sym, counts+i*rep_phase_lda[idim], rep_phase, rep_phase_lda, sphase, phys_phase,
                             gidx_off, edge_len, loc_edge_len);
    }
    gidx_off[idim] -= phys_phase[idim]*rep_phase[idim];
  }
  
  template <>
  void calc_drv_cnts<0>(int             order,
                        int const *     sym,
                        int64_t *       counts,
                        int const *     rep_phase,
                        int const *     rep_phase_lda,
                        int const *     sphase,
                        int const *     phys_phase,
                        int64_t   *     gidx_off,
                        int64_t const * edge_len,
                        int64_t const * loc_edge_len){
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
 

#define INST_CALC_CNT_BEC_ICPC_SUCKS(X) \
  template  \
  void calc_cnt_from_rep_cnt<X> \
                            (int const *     rep_phase, \
                             int * const *   pe_offset, \
                             int * const *   bucket_offset, \
                             int64_t const * old_counts, \
                             int64_t *       counts, \
                             int             bucket_off, \
                             int             pe_off, \
                             int             dir);


  INST_CALC_CNT_BEC_ICPC_SUCKS(1)
  INST_CALC_CNT_BEC_ICPC_SUCKS(2)
  INST_CALC_CNT_BEC_ICPC_SUCKS(3)
  INST_CALC_CNT_BEC_ICPC_SUCKS(4)
  INST_CALC_CNT_BEC_ICPC_SUCKS(5)
  INST_CALC_CNT_BEC_ICPC_SUCKS(6)
  INST_CALC_CNT_BEC_ICPC_SUCKS(7)
  INST_CALC_CNT_BEC_ICPC_SUCKS(8)
  INST_CALC_CNT_BEC_ICPC_SUCKS(9)
  INST_CALC_CNT_BEC_ICPC_SUCKS(10)
  INST_CALC_CNT_BEC_ICPC_SUCKS(11)

  void calc_drv_displs(int const *          sym,
                       int64_t const *      edge_len,
                       distribution const & old_dist,
                       distribution const & new_dist,
                       int64_t *            counts,
                       int                  idx_lyr){
    TAU_FSTART(calc_drv_displs);
    int * rep_phase, * sphase;
    int64_t  * gidx_off;
    int * rep_phase_lda;
    int64_t * new_loc_edge_len;
    if (idx_lyr == 0){
      int order = old_dist.order;
      rep_phase     = (int*)alloc(order*sizeof(int));
      rep_phase_lda = (int*)alloc(order*sizeof(int));
      sphase        = (int*)alloc(order*sizeof(int));
      gidx_off      = (int64_t*)alloc(order*sizeof(int64_t));
      new_loc_edge_len = (int64_t*)alloc(order*sizeof(int64_t));
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
      assert(order>0);
      SWITCH_ORD_CALL(calc_drv_cnts, order-1, order, sym, counts, rep_phase, rep_phase_lda, sphase, old_dist.phys_phase, gidx_off, edge_len, new_loc_edge_len)
    
      cdealloc(rep_phase);
      cdealloc(rep_phase_lda);
      cdealloc(sphase);
      cdealloc(gidx_off);
      cdealloc(new_loc_edge_len);
    }
    TAU_FSTOP(calc_drv_displs);
  }


  void precompute_offsets(distribution const & old_dist,
                          distribution const & new_dist,
                          int const *          sym,
                          int64_t const *      len,
                          int const *          rep_phase,
                          int64_t const *      phys_edge_len,
                          int64_t const *      virt_edge_len,
                          int const *          virt_dim,
                          int const *          virt_lda,
                          int64_t              virt_nelem,
                          int **               pe_offset,
                          int **               bucket_offset,
                          int64_t **           data_offset,
                          int **               ivmax_pre){
    TAU_FSTART(precompute_offsets);
   
    int rep_phase_lda = 1; 
    alloc_ptr(sizeof(int64_t)*1, (void**)&ivmax_pre[old_dist.order-1]);
    ivmax_pre[old_dist.order-1][0] = get_loc(len[old_dist.order-1]-1, old_dist.phys_phase[old_dist.order-1], old_dist.perank[old_dist.order-1]);

    for (int dim = 0;dim < old_dist.order;dim++){
      alloc_ptr(sizeof(int)*std::max((int64_t)rep_phase[dim],phys_edge_len[dim]), (void**)&pe_offset[dim]);
      alloc_ptr(sizeof(int)*std::max((int64_t)rep_phase[dim],phys_edge_len[dim]), (void**)&bucket_offset[dim]);
      alloc_ptr(sizeof(int64_t)*std::max((int64_t)rep_phase[dim],phys_edge_len[dim]), (void**)&data_offset[dim]);
      if (dim > 0)
        alloc_ptr(sizeof(int64_t)*std::max((int64_t)rep_phase[dim],phys_edge_len[dim]), (void**)&ivmax_pre[dim-1]);

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
           vidx < std::max(((int64_t)rep_phase[dim]+old_dist.virt_phase[dim]-1)/old_dist.virt_phase[dim],virt_edge_len[dim]);
           vidx++){

        int64_t rec_data_off = data_off;
        if (dim > 0 && sym[dim-1] != NS){
          data_stride = (vidx+1)*sub_data_stride;
          for (int j=1; j<nsym; j++){
            data_stride = (data_stride*(vidx+j+1))/(j+1);
          }
        }
        data_off += data_stride;
        for (int vr = 0;vr < old_dist.virt_phase[dim] && pidx<std::max((int64_t)rep_phase[dim],phys_edge_len[dim]) ;vr++,pidx++){

          if (dim>0){ 
            if (sym[dim-1] == NS){
              ivmax_pre[dim-1][pidx] = get_loc(len[dim-1]-1, old_dist.phys_phase[dim-1], old_dist.perank[dim-1]);
            } else if (sym[dim-1] == SY){
              ivmax_pre[dim-1][pidx] = get_loc(get_glb(pidx, old_dist.phys_phase[dim  ], old_dist.perank[dim  ]),
                                                              old_dist.phys_phase[dim-1], old_dist.perank[dim-1]);
            } else {
              ivmax_pre[dim-1][pidx] = get_loc(get_glb(pidx, old_dist.phys_phase[dim  ], old_dist.perank[dim  ])-1,
                                                              old_dist.phys_phase[dim-1], old_dist.perank[dim-1]);
            }
          }

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

}

