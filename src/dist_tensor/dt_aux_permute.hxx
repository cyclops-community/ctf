/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __DIST_TENSOR_PERMUTE_HXX__
#define __DIST_TENSOR_PERMUTE_HXX__
#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include "../ctr_comm/strp_tsr.h"
#ifdef USE_OMP
#include "omp.h"
#endif

/*
 * \brief calculates dimensional indices corresponding to a symmetric-packed index
 *        For each symmetric (SH or AS) group of size sg we have
 *          idx = n*(n-1)*...*(n-sg) / d*(d-1)*...
 *        therefore (idx*sg!)^(1/sg) >= n-sg
 *        or similarly in the SY case ... >= n
 *
 * \param[in] ndim number of dimensions in the tensor 
 * \param[in] lens edge lengths 
 * \param[in] sym symmetry
 * \param[in] idx index in the global tensor, in packed format
 * \param[out] idx_arr preallocated to size ndim, computed to correspond to idx
 */
void calc_idx_arr(int         ndim,
                  int const * lens,
                  int const * sym,
                  long_int    idx,
                  int *       idx_arr){
  long_int idx_rem = idx;
  memset(idx_arr, 0, ndim*sizeof(int));
  for (int dim=ndim-1; dim>=0; dim--){
    if (idx_rem == 0) break;
    int idim;
    if (dim == 0 || sym[dim-1] == NS){
      long_int lda = packed_size(dim, lens, sym);
      idx_arr[dim] = idx_rem/lda;
      idx_rem -= idx_arr[dim]*lda;
    } else {
      int plen[dim];
      memcpy(plen, lens, dim*sizeof(int));
      int sg = 2;
      int fsg = 2;
      while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
      long_int lda = packed_size(dim-sg+1, lens, sym);
      double fsg_idx = (((double)idx_rem)*fsg)/lda;
      int kidx = (int)pow(fsg_idx,1./sg);
      //if (sym[dim-1] != SY) 
      kidx += sg+1;
      int mkidx = kidx;
#if DEBUG >= 1
      for (int idim=dim-sg+1; idim<=dim; idim++){
        plen[idim] = mkidx+1;
      }
      long_int smidx = packed_size(dim+1, plen, sym);
      LIBT_ASSERT(smidx > idx_rem);
#endif
      long_int midx = 0;
      for (; mkidx >= 0; mkidx--){
        for (int idim=dim-sg+1; idim<=dim; idim++){
          plen[idim] = mkidx;
        }
        midx = packed_size(dim+1, plen, sym);
        if (midx <= idx_rem) break;
      }
      if (midx == 0) mkidx = 0;
      idx_arr[dim] = mkidx;
      idx_rem -= midx;
    }
  }
  LIBT_ASSERT(idx_rem == 0);
}

/**
 * \brief ,calculate the block-sizes of a tensor
 * \param[in] ndim number of dimensions of this tensor
 * \param[in] size is the size of the local tensor stored
 * \param[in] edge_len edge lengths of global tensor
 * \param[in] edge_map mapping of each dimension
 * \param[out] vrt_sz size of virtual block
 * \param[out] vrt_edge_len edge lengths of virtual block
 * \param[out] blk_edge_len edge lengths of local block
 */
inline
void calc_dim(int const         ndim,
              long_int const    size,
              int const *       edge_len,
              mapping const *   edge_map,
              long_int *        vrt_sz,
              int *             vrt_edge_len,
              int *             blk_edge_len){
  long_int vsz, i, cont;
  mapping const * map;
  vsz = size;

  for (i=0; i<ndim; i++){
    if (blk_edge_len != NULL)
      blk_edge_len[i] = edge_len[i];
    vrt_edge_len[i] = edge_len[i];
    map = &edge_map[i];
    do {
      if (blk_edge_len != NULL){
        if (map->type == PHYSICAL_MAP)
          blk_edge_len[i] = blk_edge_len[i] / map->np;
      }
      vrt_edge_len[i] = vrt_edge_len[i] / map->np;
      if (vrt_sz != NULL){
        if (map->type == VIRTUAL_MAP)
          vsz = vsz / map->np;
      } 
      if (map->has_child){
        cont = 1;
        map = map->child;
      } else 
        cont = 0;
    } while (cont);
  }
  if (vrt_sz != NULL)
    *vrt_sz = vsz;
}

/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] edge_map_B mapping of each dimension of A
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 2*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             mapping const *    edge_map_A,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  dim_max++;

  *ndim_tot = dim_max;
  CTF_alloc_ptr(sizeof(int)*dim_max, (void**)idx_arr);
  std::fill((*idx_arr), (*idx_arr)+dim_max, -1);

  for (i=0; i<ndim_A; i++){
    if ((*idx_arr)[idx_A[i]] == -1)
      (*idx_arr)[idx_A[i]] = i;
    else {
//      if (edge_map_A == NULL || edge_map_A[i].type == PHYSICAL_MAP)
        (*idx_arr)[idx_A[i]] = i;
    }
  }
}

/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] edge_map_B mapping of each dimension of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[in] edge_map_B mapping of each dimension of B
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 2*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             mapping const *    edge_map_A,
             int const          ndim_B,
             int const *        idx_B,
             mapping const *    edge_map_B,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_B[i] > dim_max) dim_max = idx_B[i];
  }
  dim_max++;

  *ndim_tot = dim_max;
  CTF_alloc_ptr(sizeof(int)*2*dim_max, (void**)idx_arr);
  std::fill((*idx_arr), (*idx_arr)+2*dim_max, -1);

  for (i=0; i<ndim_A; i++){
    if ((*idx_arr)[2*idx_A[i]] == -1)
      (*idx_arr)[2*idx_A[i]] = i;
    else {
//      if (edge_map_A == NULL || edge_map_A[i].type == PHYSICAL_MAP)
        (*idx_arr)[2*idx_A[i]] = i;
    }
  }
  for (i=0; i<ndim_B; i++){
    if ((*idx_arr)[2*idx_B[i]+1] == -1)
      (*idx_arr)[2*idx_B[i]+1] = i;
    else {
//      if (edge_map_B == NULL || edge_map_B[i].type == PHYSICAL_MAP)
        (*idx_arr)[2*idx_B[i]+1] = i;
    }
  }
}

/**
 * \brief invert index map
 * \param[in] ndim_A number of dimensions of A
 * \param[in] idx_A index map of A
 * \param[in] edge_map_B mapping of each dimension of A
 * \param[in] ndim_B number of dimensions of B
 * \param[in] idx_B index map of B
 * \param[in] edge_map_B mapping of each dimension of B
 * \param[in] ndim_C number of dimensions of C
 * \param[in] idx_C index map of C
 * \param[in] edge_map_C mapping of each dimension of C
 * \param[out] ndim_tot number of total dimensions
 * \param[out] idx_arr 3*ndim_tot index array
 */
inline
void inv_idx(int const          ndim_A,
             int const *        idx_A,
             mapping const *    edge_map_A,
             int const          ndim_B,
             int const *        idx_B,
             mapping const *    edge_map_B,
             int const          ndim_C,
             int const *        idx_C,
             mapping const *    edge_map_C,
             int *              ndim_tot,
             int **             idx_arr){
  int i, dim_max;

  dim_max = -1;
  for (i=0; i<ndim_A; i++){
    if (idx_A[i] > dim_max) dim_max = idx_A[i];
  }
  for (i=0; i<ndim_B; i++){
    if (idx_B[i] > dim_max) dim_max = idx_B[i];
  }
  for (i=0; i<ndim_C; i++){
    if (idx_C[i] > dim_max) dim_max = idx_C[i];
  }
  dim_max++;
  *ndim_tot = dim_max;
  CTF_alloc_ptr(sizeof(int)*3*dim_max, (void**)idx_arr);
  std::fill((*idx_arr), (*idx_arr)+3*dim_max, -1);

  for (i=0; i<ndim_A; i++){
    if ((*idx_arr)[3*idx_A[i]] == -1)
      (*idx_arr)[3*idx_A[i]] = i;
    else {
//      if (edge_map_A == NULL || edge_map_A[i].type == PHYSICAL_MAP)
        (*idx_arr)[3*idx_A[i]] = i;
    }
  }
  for (i=0; i<ndim_B; i++){
    if ((*idx_arr)[3*idx_B[i]+1] == -1)
      (*idx_arr)[3*idx_B[i]+1] = i;
    else {
//      if (edge_map_B == NULL || edge_map_B[i].type == PHYSICAL_MAP)
        (*idx_arr)[3*idx_B[i]+1] = i;
    }
  }
  for (i=0; i<ndim_C; i++){
    if ((*idx_arr)[3*idx_C[i]+2] == -1)
      (*idx_arr)[3*idx_C[i]+2] = i;
    else {
//      if (edge_map_C == NULL || edge_map_C[i].type == PHYSICAL_MAP)
        (*idx_arr)[3*idx_C[i]+2] = i;
    }
  }
}


/**
 * \brief assigns keys to an array of values
 * \param[in] ndim tensor dimension
 * \param[in] nbuf number of global virtual buckets
 * \param[in] new_nvirt new total virtualization factor
 * \param[in] np number of processors
 * \param[in] old_edge_len old tensor edge lengths
 * \param[in] new_edge_len new tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] old_phase current physical*virtual phase
 * \param[in] old_rank current physical rank
 * \param[in] new_phase new physical*virtual phase
 * \param[in] old_virt_dim current virtualization dimensions on each process
 * \param[in] new_virt_dim new virtualization dimensions on each process
 * \param[in] new_virt_lda prefix sum of new_virt_dim
 * \param[in] buf_lda prefix sum of new_phase
 * \param[in] pe_lda processor grid lda
 * \param[in] is_pad whether tensor is padded
 * \param[in] padding padding of tensor
 * \param[out] send_counts outgoing counts of pairs by pe
 * \param[out] recv_counts incoming counts of pairs by pe
 * \param[out] send_displs outgoing displs of pairs by pe
 * \param[out] recv_displs incoming displs of pairs by pe
 * \param[out] svirt_displs outgoing displs of pairs by virt bucket
 * \param[out] rvirt_displs incoming displs of pairs by virt bucket
 * \param[in] ord_glb_comm the global communicator
 * \param[in] idx_lyr starting processor layer (2.5D)
 * \param[in] was_cyclic whether the old mapping was cyclic
 * \param[in] is_cyclic whether the new mapping is cyclic
 * \param[in] bucket_offset offsets for target index for each dimension
 */
template<typename dtype>
void calc_cnt_displs_old(int const          ndim, 
                     int const          nbuf,
                     int const          new_nvirt,
                     int const          np,
                     int const *        old_edge_len,
                     int const *        new_edge_len,
                     int const *        old_virt_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const *        new_phase,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     int const *        new_virt_lda,
                     int const *        buf_lda,
                     int const *        pe_lda,
                     int const          is_pad,
                     int const *        padding,
                     int *              send_counts,
                     int *              recv_counts,
                     int *              send_displs,
                     int *              recv_displs,
                     int *              svirt_displs,
                     int *              rvirt_displs,
                     CommData_t *       ord_glb_comm,
                     int const          idx_lyr,
                     int const          was_cyclic,
                     int const          is_cyclic,
                     int * const *      bucket_offset){
  int i, j, imax, act_lda, act_max;
  long_int buf_offset, idx_offset;
  int virt_offset, tmp1, tmp2, skip;
  int * all_virt_counts;

#ifdef USE_OMP
//  int max_ntd = MIN(omp_get_max_threads(),REDIST_MAX_THREADS);
  long_int vbs = sy_packed_size(ndim, old_virt_edge_len, sym);
  int max_ntd = omp_get_max_threads();
  max_ntd = MAX(1,MIN(max_ntd,vbs/nbuf));
  CTF_mst_alloc_ptr(nbuf*sizeof(int)*max_ntd, (void**)&all_virt_counts);
#else
  CTF_mst_alloc_ptr(nbuf*sizeof(int), (void**)&all_virt_counts);
#endif


  /* Count how many elements need to go to each new virtual bucket */
  if (idx_lyr==0){
    if (ndim == 0){
      memset(all_virt_counts, 0, nbuf*sizeof(int));
      all_virt_counts[0]++;
    } else {
#ifdef USE_OMP
#pragma omp parallel private (i,j,imax,act_lda,act_max,buf_offset,idx_offset,\
                              tmp1,tmp2,skip) num_threads (max_ntd)
      {
      int start_ldim, end_ldim, i_st, vc;
      int * virt_counts;
      int * old_virt_idx, * virt_rank;
      int * idx;
      long_int * idx_offs;
      int * spad;
      int last_len = old_edge_len[ndim-1]/old_phase[ndim-1]+1;
      int omp_ntd, omp_tid;
      omp_ntd = omp_get_num_threads();
      omp_tid = omp_get_thread_num();
//      if (omp_tid < omp_ntd){
      virt_counts = all_virt_counts+nbuf*omp_tid;
      start_ldim = (last_len/omp_ntd)*omp_tid;
      start_ldim += MIN(omp_tid,last_len%omp_ntd);
      end_ldim = (last_len/omp_ntd)*(omp_tid+1);
      end_ldim += MIN(omp_tid+1,last_len%omp_ntd);
#else
      {
      int start_ldim, end_ldim, i_st, vc;
      int * virt_counts;
      long_int * old_virt_idx, * virt_rank;
      long_int * idx;
      long_int * idx_offs;
      int * spad;
      int last_len = old_edge_len[ndim-1]/old_phase[ndim-1]+1;
      virt_counts = all_virt_counts;
      start_ldim = 0;
      end_ldim = last_len;
#endif
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&old_virt_idx);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&virt_rank);
      CTF_alloc_ptr(ndim*sizeof(int), (void**)&spad);
      memset(virt_counts, 0, nbuf*sizeof(int));
      memset(old_virt_idx, 0, ndim*sizeof(long_int));
      /* virt_rank = physical_rank*num_virtual_ranks + virtual_rank */
      for (i=0; i<ndim; i++){ 
        virt_rank[i] = old_rank[i]*old_virt_dim[i]; 
      }
      for (;;){
        memset(idx, 0, ndim*sizeof(long_int));
        memset(idx_offs, 0, ndim*sizeof(long_int));
        idx_offset = 0; 
        skip = 0;
        idx[ndim-1] = MAX(idx[ndim-1],start_ldim);
        for (i=0; i<ndim; i++) {
          /* Warning: This next if block has a history of bugs */
          if (is_pad){
            //spad[i] = padding[i];
            if (sym[i] != NS){
              LIBT_ASSERT(padding[i] < old_phase[i]);
              spad[i] = 1;
              if (sym[i] != SY && virt_rank[i] < virt_rank[i+1])
                spad[i]--;
              if (sym[i] == SY && virt_rank[i] <= virt_rank[i+1])
                spad[i]--;
            }
          } else {
            spad[i] = 0;
          }
          if (sym[i] != NS && idx[i] >= idx[i+1]-spad[i]){
            idx[i+1] = idx[i]+spad[i];
  //        if (virt_rank[sym[i]] + (sym[i]==SY) <= virt_rank[i])
  //          idx[sym[i]]++;
          }
          if (i > 0){
            imax = (old_edge_len[i]-padding[i])/old_phase[i];
            if (virt_rank[i] < (old_edge_len[i]-padding[i])%old_phase[i])
              imax++;
            if (i == ndim - 1)
              imax = MIN(imax, end_ldim);
            if (idx[i] >= imax)
              skip = 1;
            else  {
              if (was_cyclic)
                idx_offs[i] = idx[i]*old_phase[i]+virt_rank[i];
              else 
                idx_offs[i] = idx[i]+virt_rank[i]*old_edge_len[i]/old_phase[i];
              if (is_cyclic)
                idx_offs[i] = (idx_offs[i]%new_phase[i])*buf_lda[i];
              else
                idx_offs[i] = (idx_offs[i]/(new_edge_len[i]/new_phase[i]))*buf_lda[i];
            /*  if (idx_offs[i] != bucket_offset[i][old_virt_idx[i]*old_virt_edge_len[i]+idx[i]]){
                printf("%d %d\n",
                 idx_offs[i], bucket_offset[i][old_virt_idx[i]*old_virt_edge_len[i]+idx[i]]);
                ABORT;
              }*/
              idx_offset += idx_offs[i];
            }
          }
        }
        /* determine how many elements belong to each virtual processor */
        /* Iterate over a block corresponding to a virtual rank */
        /* (INNER LOOP) */
        if (!skip){
          for (;;){
            imax = (old_edge_len[0]-padding[0])/old_phase[0];
            if (virt_rank[0] < (old_edge_len[0]-padding[0])%old_phase[0])
              imax++;
            if (sym[0] != NS) {
              imax = MIN(imax,idx[1]+1-spad[0]);
            }
            if (ndim == 1){
              imax = MIN(imax, end_ldim);
              i_st = start_ldim;
            } else
              i_st = 0;
            
            /* Increment virtual bucket */
            for (i=i_st; i<imax; i++){
              //virt_counts[idx_offset+((i*old_phase[0]+virt_rank[0])%new_phase[0])]++;
              if (was_cyclic)
                vc = i*old_phase[0]+virt_rank[0];
              else 
                vc = i+virt_rank[0]*old_edge_len[0]/old_phase[0];
              if (is_cyclic)
                vc = (vc%new_phase[0])*buf_lda[0];
              else
                vc = (vc/(new_edge_len[0]/new_phase[0]))*buf_lda[0];
              virt_counts[idx_offset+vc]++;
//              vc = bucket_offset[0][old_virt_idx[0]*old_virt_edge_len[0]+i];
//              svirt_displs[idx_offset+vc]++;
            }
            /* Increment indices and set up offsets */
            for (act_lda=1; act_lda < ndim; act_lda++){
              idx[act_lda]++;
              if (is_pad){
                act_max = (old_edge_len[act_lda]-padding[act_lda])/old_phase[act_lda];
                if (virt_rank[act_lda] <
                    (old_edge_len[act_lda]-padding[act_lda])%old_phase[act_lda])
                  act_max++;
              } else
                act_max = old_edge_len[act_lda]/old_phase[act_lda];
              if (act_lda == ndim - 1)
                act_max = MIN(act_max, end_ldim);
              if (sym[act_lda] != NS) 
                act_max = MIN(act_max,idx[act_lda+1]+1-spad[act_lda]);
              bool ended = true;
              if (idx[act_lda] >= act_max){
                ended = false;
                idx[act_lda] = 0;
                if (sym[act_lda-1] != NS) idx[act_lda] = idx[act_lda-1]+spad[act_lda-1];
              }
              idx_offset -= idx_offs[act_lda];
              if (was_cyclic)
                idx_offs[act_lda] = idx[act_lda]*old_phase[act_lda]+virt_rank[act_lda];
              else 
                idx_offs[act_lda] = idx[act_lda]+virt_rank[act_lda]*old_edge_len[act_lda]/old_phase[act_lda];
              if (is_cyclic)
                idx_offs[act_lda] = (idx_offs[act_lda]%new_phase[act_lda])*buf_lda[act_lda];
              else
                idx_offs[act_lda] = (idx_offs[act_lda]/(new_edge_len[act_lda]/new_phase[act_lda]))*buf_lda[act_lda];
              idx_offset += idx_offs[act_lda];
              if (ended)//idx[act_lda] > 0)
                break;
            }
            if (act_lda == ndim) break;
          }
        }
        /* (OUTER LOOP) Iterate over virtual ranks on this pe */
        for (act_lda=0; act_lda<ndim; act_lda++){
          old_virt_idx[act_lda]++;
          if (old_virt_idx[act_lda] >= old_virt_dim[act_lda])
            old_virt_idx[act_lda] = 0;

          virt_rank[act_lda] = old_rank[act_lda]*old_virt_dim[act_lda]
                               +old_virt_idx[act_lda];
  
          if (old_virt_idx[act_lda] > 0)
            break;  
        }
        if (act_lda == ndim) break;
      }
      CTF_free(idx);
      CTF_free(idx_offs);
      CTF_free(old_virt_idx);
      CTF_free(virt_rank);
      CTF_free(spad);
      }
#ifdef USE_OMP
      for (j=1; j<max_ntd; j++){
        for (i=0; i<nbuf; i++){
          all_virt_counts[i] += all_virt_counts[i+nbuf*j];
        }
      }
#endif
    }
  } else
    memset(all_virt_counts, 0, sizeof(int)*nbuf);
  int * virt_counts = all_virt_counts;
  long_int * idx;
  long_int * idx_offs;

  /* reduce virtual counts to phyiscal processor counts */
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx);

  /* FIXME: unnecessary memset */
  memset(idx, 0, ndim*sizeof(long_int));
  memset(idx_offs, 0, ndim*sizeof(long_int));
  memset(send_counts, 0, np*sizeof(int));
  idx_offset = 0;
  buf_offset = 0;
  virt_offset = 0;

  /* Calculate All-to-all processor grid offsets from the virtual bucket counts */
  if (ndim == 0){
    send_counts[0] = virt_counts[0];
    memset(svirt_displs, 0, np*sizeof(int));
  } else {
     for (;;){
      for (i=0; i<new_phase[0]; i++){
        // Accumulate total pe-to-pe counts 
        send_counts[idx_offset + (i/new_virt_dim[0])*pe_lda[0]]
                     += virt_counts[buf_offset+i];
        // Calculate counts of virtual buckets going to each pe 
        svirt_displs[(idx_offset + (i/new_virt_dim[0])*pe_lda[0])*new_nvirt
                     +virt_offset+(i%new_virt_dim[0])] = virt_counts[buf_offset+i];
      }
      buf_offset += new_phase[0];//buf_lda[1];
      // Increment indices and set up offsets 
      for (act_lda=1; act_lda<ndim; act_lda++){
        idx[act_lda]++;
        if (idx[act_lda] >= new_phase[act_lda]){
          idx[act_lda] = 0;
        } 
        
        virt_offset -= (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
        idx_offset -= (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        idx_offs[act_lda] = idx[act_lda];
        virt_offset += (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
        idx_offset += (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        
        if (idx[act_lda] > 0)
          break;
      }
      if (act_lda == ndim) break;
    }    
  }

  /* Exchange counts */
  if (ord_glb_comm == NULL)
    recv_counts[0] = send_counts[0];
  else
    ALL_TO_ALL(send_counts, 1, COMM_INT_T, 
               recv_counts, 1, COMM_INT_T, ord_glb_comm);
  
  /* Calculate displacements out of the count arrays */
  send_displs[0] = 0;
  recv_displs[0] = 0;
  for (i=1; i<np; i++){
    send_displs[i] = send_displs[i-1] + send_counts[i-1];
    recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
  }

  /* Calculate displacements for virt buckets in each message */
  for (i=0; i<np; i++){
    tmp2 = svirt_displs[i*new_nvirt];
    svirt_displs[i*new_nvirt] = 0;
    for (j=1; j<new_nvirt; j++){
      tmp1 = svirt_displs[i*new_nvirt+j];
      svirt_displs[i*new_nvirt+j] = svirt_displs[i*new_nvirt+j-1]+tmp2;
      tmp2 = tmp1;
    }
  }

  /* Exchange displacements for virt buckets */
  if (ord_glb_comm == NULL)
    memcpy(rvirt_displs, svirt_displs, new_nvirt*sizeof(int));
  else
    ALL_TO_ALL(svirt_displs, new_nvirt, COMM_INT_T, 
               rvirt_displs, new_nvirt, COMM_INT_T, ord_glb_comm);
  
  CTF_free(all_virt_counts);
  CTF_free(idx);
  CTF_free(idx_offs);

}


/**
 * \brief assigns keys to an array of values
 * \param[in] ndim tensor dimension
 * \param[in] nbuf number of global virtual buckets
 * \param[in] new_nvirt new total virtualization factor
 * \param[in] np number of processors
 * \param[in] old_edge_len old tensor edge lengths
 * \param[in] new_edge_len new tensor edge lengths
 * \param[in] sym symmetries of tensor
 * \param[in] old_phase current physical*virtual phase
 * \param[in] old_rank current physical rank
 * \param[in] new_phase new physical*virtual phase
 * \param[in] old_virt_dim current virtualization dimensions on each process
 * \param[in] new_virt_dim new virtualization dimensions on each process
 * \param[in] new_virt_lda prefix sum of new_virt_dim
 * \param[in] buf_lda prefix sum of new_phase
 * \param[in] pe_lda processor grid lda
 * \param[in] is_pad whether tensor is padded
 * \param[in] padding padding of tensor
 * \param[out] send_counts outgoing counts of pairs by pe
 * \param[out] recv_counts incoming counts of pairs by pe
 * \param[out] send_displs outgoing displs of pairs by pe
 * \param[out] recv_displs incoming displs of pairs by pe
 * \param[out] svirt_displs outgoing displs of pairs by virt bucket
 * \param[out] rvirt_displs incoming displs of pairs by virt bucket
 * \param[in] ord_glb_comm the global communicator
 * \param[in] idx_lyr starting processor layer (2.5D)
 * \param[in] was_cyclic whether the old mapping was cyclic
 * \param[in] is_cyclic whether the new mapping is cyclic
 * \param[in] bucket_offset offsets for target index for each dimension
 */
template<typename dtype>
void calc_cnt_displs(int const          ndim, 
                     int const          nbuf,
                     int const          new_nvirt,
                     int const          np,
                     int const *        old_edge_len,
                     int const *        new_edge_len,
                     int const *        old_virt_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const *        new_phase,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     int const *        new_virt_lda,
                     int const *        buf_lda,
                     int const *        pe_lda,
                     int const          is_pad,
                     int const *        padding,
                     int *              send_counts,
                     int *              recv_counts,
                     int *              send_displs,
                     int *              recv_displs,
                     int *              svirt_displs,
                     int *              rvirt_displs,
                     CommData_t *       ord_glb_comm,
                     int const          idx_lyr,
                     int const          was_cyclic,
                     int const          is_cyclic,
                     int * const *      bucket_offset){
  int * all_virt_counts;

#ifdef USE_OMP
  long_int vbs = sy_packed_size(ndim, old_virt_edge_len, sym);
  int max_ntd = omp_get_max_threads();
  max_ntd = MAX(1,MIN(max_ntd,vbs/nbuf));
#else
  int max_ntd = 1;
#endif
  
  CTF_mst_alloc_ptr(nbuf*sizeof(int)*max_ntd, (void**)&all_virt_counts);


  /* Count how many elements need to go to each new virtual bucket */
  if (idx_lyr==0){
    if (ndim == 0){
      memset(all_virt_counts, 0, nbuf*sizeof(int));
      all_virt_counts[0]++;
    } else {
#ifdef USE_OMP
#pragma omp parallel num_threads(max_ntd)
      {
      int imax, act_max, skip;
      int start_ldim, end_ldim, i_st, vc, dim;
      int * virt_counts;
      int * old_virt_idx, * virt_rank;
      int * idx;
      long_int idx_offset;
      long_int * idx_offs;
      int * spad;
      int last_len = old_edge_len[ndim-1]/old_phase[ndim-1]+1;
      int omp_ntd, omp_tid;
      omp_ntd = omp_get_num_threads();
      omp_tid = omp_get_thread_num();
      virt_counts = all_virt_counts+nbuf*omp_tid;
      start_ldim = (last_len/omp_ntd)*omp_tid;
      start_ldim += MIN(omp_tid,last_len%omp_ntd);
      end_ldim = (last_len/omp_ntd)*(omp_tid+1);
      end_ldim += MIN(omp_tid+1,last_len%omp_ntd);
#else
      {
      int imax, act_max, skip;
      int start_ldim, end_ldim, i_st, vc, dim;
      int * virt_counts;
      long_int * old_virt_idx, * virt_rank;
      long_int * idx;
      long_int idx_offset;
      long_int * idx_offs;
      int * spad;
      int last_len = old_edge_len[ndim-1]/old_phase[ndim-1]+1;
      virt_counts = all_virt_counts;
      start_ldim = 0;
      end_ldim = last_len;
#endif
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&old_virt_idx);
      CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&virt_rank);
      CTF_alloc_ptr(ndim*sizeof(int), (void**)&spad);
      memset(virt_counts, 0, nbuf*sizeof(int));
      memset(old_virt_idx, 0, ndim*sizeof(long_int));
      /* virt_rank = physical_rank*num_virtual_ranks + virtual_rank */
      for (int i=0; i<ndim; i++){ 
        virt_rank[i] = old_rank[i]*old_virt_dim[i]; 
      }
      for (;;){
        memset(idx, 0, ndim*sizeof(long_int));
        memset(idx_offs, 0, ndim*sizeof(long_int));
        idx_offset = 0; 
        skip = 0;
        idx[ndim-1] = MAX(idx[ndim-1],start_ldim);
        for (dim=0; dim<ndim; dim++) {
          /* Warning: This next if block has a history of bugs */
          if (is_pad){
            //spad[dim] = padding[dim];
            if (sym[dim] != NS){
              LIBT_ASSERT(padding[dim] < old_phase[dim]);
              spad[dim] = 1;
              if (sym[dim] != SY && virt_rank[dim] < virt_rank[dim+1])
                spad[dim]--;
              if (sym[dim] == SY && virt_rank[dim] <= virt_rank[dim+1])
                spad[dim]--;
            }
          } else {
            spad[dim] = 0;
          }
          if (sym[dim] != NS && idx[dim] >= idx[dim+1]-spad[dim]){
            idx[dim+1] = idx[dim]+spad[dim];
  //        if (virt_rank[sym[dim]] + (sym[dim]==SY) <= virt_rank[dim])
  //          idx[sym[dim]]++;
          }
          if (dim > 0){
            imax = (old_edge_len[dim]-padding[dim])/old_phase[dim];
            if (virt_rank[dim] < (old_edge_len[dim]-padding[dim])%old_phase[dim])
              imax++;
            if (dim == ndim - 1)
              imax = MIN(imax, end_ldim);
            if (idx[dim] >= imax)
              skip = 1;
            else  {
              idx_offs[dim] = bucket_offset[dim][old_virt_idx[dim]*old_virt_edge_len[dim]+idx[dim]];
              idx_offset += idx_offs[dim];
            }
          }
        }
        /* determine how many elements belong to each virtual processor */
        /* Iterate over a block corresponding to a virtual rank */
        /* (INNER LOOP) */
        if (!skip){
          for (;;){
            imax = (old_edge_len[0]-padding[0])/old_phase[0];
            if (virt_rank[0] < (old_edge_len[0]-padding[0])%old_phase[0])
              imax++;
            if (sym[0] != NS) {
              imax = MIN(imax,idx[1]+1-spad[0]);
            }
            if (ndim == 1){
              imax = MIN(imax, end_ldim);
              i_st = start_ldim;
            } else
              i_st = 0;
            
            /* Increment virtual bucket */
            for (int i=i_st; i<imax; i++){
/*
              if (sym[0] != NS){
                if ((sym[0] != SY && virt_rank[0] >= virt_rank[1]) ||
                    (sym[0] == SY && virt_rank[0] > virt_rank[1])) {

                  LIBT_ASSERT(i<idx[1]);
                } else {
                  LIBT_ASSERT(i<=idx[1]);
                }
              }
              for (int dim = 1; dim < ndim-1; dim++){
                if (sym[dim] != NS){
                  if ((sym[dim] != SY && virt_rank[dim] >= virt_rank[dim+1]) ||
                      (sym[dim] == SY && virt_rank[dim] > virt_rank[dim+1])) {
                    LIBT_ASSERT(idx[dim]<idx[dim+1]);
                  } else
                    LIBT_ASSERT(idx[dim]<=idx[dim+1]);
                }
              }*/
              vc = bucket_offset[0][old_virt_idx[0]*old_virt_edge_len[0]+i];
              virt_counts[idx_offset+vc]++;
            }
            /* Increment indices and set up offsets */
            for (dim=1; dim < ndim; dim++){
              idx[dim]++;
              if (is_pad){
                act_max = (old_edge_len[dim]-padding[dim])/old_phase[dim];
                if (virt_rank[dim] <
                    (old_edge_len[dim]-padding[dim])%old_phase[dim])
                  act_max++;
              } else
                act_max = old_edge_len[dim]/old_phase[dim];
              if (dim == ndim - 1)
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
            if (dim == ndim) break;
          }
        }
        /* (OUTER LOOP) Iterate over virtual ranks on this pe */
        for (dim=0; dim<ndim; dim++){
          old_virt_idx[dim]++;
          if (old_virt_idx[dim] >= old_virt_dim[dim])
            old_virt_idx[dim] = 0;

          virt_rank[dim] = old_rank[dim]*old_virt_dim[dim]
                               +old_virt_idx[dim];
  
          if (old_virt_idx[dim] > 0)
            break;  
        }
        if (dim == ndim) break;
      }
      CTF_free(idx);
      CTF_free(idx_offs);
      CTF_free(old_virt_idx);
      CTF_free(virt_rank);
      CTF_free(spad);
      }
#ifdef USE_OMP
      for (int j=1; j<max_ntd; j++){
        for (int i=0; i<nbuf; i++){
          all_virt_counts[i] += all_virt_counts[i+nbuf*j];
        }
      }
#endif
    }
  } else
    memset(all_virt_counts, 0, sizeof(int)*nbuf);

  int tmp1, tmp2;
  int * virt_counts = all_virt_counts;

  memset(send_counts, 0, np*sizeof(int));

  /* Calculate All-to-all processor grid offsets from the virtual bucket counts */
  if (ndim == 0){
 //   send_counts[0] = virt_counts[0];
    memset(svirt_displs, 0, np*sizeof(int));
  } else {
    /* for (;;){
      for (i=0; i<new_phase[0]; i++){
        // Accumulate total pe-to-pe counts 
        send_counts[idx_offset + (i/new_virt_dim[0])*pe_lda[0]]
                     += virt_counts[buf_offset+i];
        // Calculate counts of virtual buckets going to each pe 
        svirt_displs[(idx_offset + (i/new_virt_dim[0])*pe_lda[0])*new_nvirt
                     +virt_offset+(i%new_virt_dim[0])] = virt_counts[buf_offset+i];
      }
      buf_offset += new_phase[0];//buf_lda[1];
      // Increment indices and set up offsets 
      for (act_lda=1; act_lda<ndim; act_lda++){
        idx[act_lda]++;
        if (idx[act_lda] >= new_phase[act_lda]){
          idx[act_lda] = 0;
        } 
        
        virt_offset -= (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
        idx_offset -= (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        idx_offs[act_lda] = idx[act_lda];
        virt_offset += (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
        idx_offset += (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        
        if (idx[act_lda] > 0)
          break;
      }
      if (act_lda == ndim) break;
    }*/
    for (int i=0; i<np; i++){
      for (int j=0; j<new_nvirt; j++){
        //printf("virt_count[%d][%d]=%d\n",i,j,virt_counts[i*new_nvirt+j]);
        send_counts[i] += virt_counts[i*new_nvirt+j];
        svirt_displs[i*new_nvirt+j] = virt_counts[i*new_nvirt+j];
      }
    }
  }

  /* Exchange counts */
  if (ord_glb_comm == NULL)
    recv_counts[0] = send_counts[0];
  else
    ALL_TO_ALL(send_counts, 1, COMM_INT_T, 
               recv_counts, 1, COMM_INT_T, ord_glb_comm);
  
  /* Calculate displacements out of the count arrays */
  send_displs[0] = 0;
  recv_displs[0] = 0;
  for (int i=1; i<np; i++){
    send_displs[i] = send_displs[i-1] + send_counts[i-1];
    recv_displs[i] = recv_displs[i-1] + recv_counts[i-1];
  }

  /* Calculate displacements for virt buckets in each message */
  for (int i=0; i<np; i++){
    tmp2 = svirt_displs[i*new_nvirt];
    svirt_displs[i*new_nvirt] = 0;
    for (int j=1; j<new_nvirt; j++){
      tmp1 = svirt_displs[i*new_nvirt+j];
      svirt_displs[i*new_nvirt+j] = svirt_displs[i*new_nvirt+j-1]+tmp2;
      tmp2 = tmp1;
    }
  }

  /* Exchange displacements for virt buckets */
  if (ord_glb_comm == NULL)
    memcpy(rvirt_displs, svirt_displs, new_nvirt*sizeof(int));
  else
    ALL_TO_ALL(svirt_displs, new_nvirt, COMM_INT_T, 
               rvirt_displs, new_nvirt, COMM_INT_T, ord_glb_comm);
  
  CTF_free(all_virt_counts);

}

/**
 * \brief move data between buckets and tensor layout
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] nbuf number of global virtual buckets
 * \param[in] new_nvirt new total virtualization factor
 * \param[in] edge_len current edge lengths of tensor
 * \param[in] sym symmetries of tensor
 * \param[in] old_phase current physical*virtual phase
 * \param[in] old_rank current physical rank
 * \param[in] new_phase new physical*virtual phase
 * \param[in] new_rank new physical rank
 * \param[in] old_virt_dim current virtualization dimensions on each process
 * \param[in] new_virt_dim new virtualization dimensions on each process
 * \param[in] pe_displs outgoing displs of pairs by pe
 * \param[in] virt_displs outgoing displs of pairs by virtual bucket
 * \param[in] old_virt_lda prefix sum of old_virt_dim
 * \param[in] new_virt_lda prefix sum of new_virt_dim
 * \param[in] pe_lda lda of processor grid
 * \param[in] vbs number of elements per virtual subtensor
 * \param[in] is_pad whether tensor is padded
 * \param[in] padding padding of tensor
 * \param[out] tsr_data data to retrieve from
 * \param[out] tsr_cyclic_data data to write to
 * \param[in] p_or_up whether to pack (1) or unpack (0)
 * \param[in] was_cyclic whether the old mapping was cyclic
 * \param[in] is_cyclic whether the new mapping is cyclic
 */
template<typename dtype>
void pup_virt_buff(int const            ndim, 
                   int const            nbuf,
                   int const            new_nvirt,
                   int const *          edge_len,
                   int const *          sym,
                   int const *          old_phase,
                   int const *          old_rank,
                   int const *          new_phase,
                   int const *          new_rank,
                   int const *          old_virt_dim,
                   int const *          new_virt_dim,
                   int const *          pe_displs,
                   int const *          virt_displs,
                   int const *          old_virt_lda,
                   int const *          new_virt_lda,
                   int const *          pe_lda,
                   long_int const       vbs,
                   int const            is_pad,
                   int const *          padding,
                   dtype *              tsr_data,
                   dtype *              tsr_cyclic_data,
                   int const            p_or_up,
                   int const            was_cyclic,
                   int const            is_cyclic){
  long_int i, imax, padimax, act_lda, act_max; 
  long_int poutside, pe_idx, virt_idx, overt_idx;
  long_int arr_idx, tmp1;
  long_int vr, vr_max, pact_max, ispm;
  int outside;
  long_int idx_offset, virt_offset;
  long_int *  idx, * old_virt_idx, * virt_rank;
  int * virt_counts;
  long_int * idx_offs;
  long_int * acc_idx;
  int * spad;

  CTF_mst_alloc_ptr(nbuf*sizeof(int), (void**)&virt_counts);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&acc_idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&old_virt_idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&virt_rank);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&spad);

  memset(old_virt_idx, 0, ndim*sizeof(long_int));
  memset(acc_idx, 0, ndim*sizeof(long_int));
  memset(virt_counts, 0, nbuf*sizeof(int));
  for (i=0; i<ndim; i++){ virt_rank[i] = old_rank[i]*old_virt_dim[i]; }

  /* Loop accross all virtual subtensors in order, and keep order within
     subtensors. We should move through one of the arrays contiguously */
  arr_idx = 0;
  overt_idx = 0;
  memset(idx, 0, ndim*sizeof(long_int));
  memset(idx_offs, 0, ndim*sizeof(long_int));

  idx_offset = 0;
  virt_offset = 0;
  for (i=1; i<ndim; i++) {
    idx_offs[i] = (virt_rank[i]%new_phase[i]);
    idx_offset += (idx_offs[i]/new_virt_dim[i])*pe_lda[i];
    virt_offset += (idx_offs[i]%new_virt_dim[i])*new_virt_lda[i];
  }

  outside = 0;
  poutside = -1;
  for (i=1; i<ndim; i++){
    if (sym[i] != NS){
      if (sym[i] == SY){
        if (old_rank[i+1]*old_virt_dim[i+1] <
            old_rank[i]*old_virt_dim[i]){
          outside=1;
          break;
        }
      }
      if (sym[i] != SY){
        if (old_rank[i+1]*old_virt_dim[i+1] <=
            old_rank[i]*old_virt_dim[i]){
          outside=1;
          break;
        }
      }
    }
  }
  imax = edge_len[0]/old_phase[0];
  for (;;){
    if (sym[0] != NS)
      imax = idx[1]+1;
    /* NOTE: Must iterate across all local data in correct global order
     *        to make the back-transformation correct */
    /* for each leading index owned by a virtual prc along an edge */
    if (outside == 0){
      if (is_pad) {
        ispm = (edge_len[0]-padding[0])/old_phase[0];
        if ((edge_len[0]-padding[0])%old_phase[0] > 
            old_rank[0]*old_virt_dim[0])
          ispm++;
        padimax = MIN(imax,ispm);
      } else
        padimax = imax;
      for (i=0; i<padimax; i++){
      /* for each virtual proc along leading dim */
        vr_max = old_virt_dim[0];
        if (is_pad){
          vr_max = MIN(vr_max, ((edge_len[0]-padding[0])-
                                (i*old_phase[0]+old_rank[0]*old_virt_dim[0])));
//        printf("vr_max = %d rather than %d\n",vr_max,old_virt_dim[0]);
        }
        if (sym[0] != NS && i == idx[1]){
          vr_max = MIN(vr_max, ((old_virt_idx[1]+(sym[0]==SY)+
                    old_rank[1]*old_virt_dim[1])-
                    (old_rank[0]*old_virt_dim[0])));
        }
        for (vr=0; vr<vr_max; vr++){
          virt_rank[0] = old_rank[0]*old_virt_dim[0]+vr;
          /* Find the offset of this virtual proc */
          arr_idx += (overt_idx + vr)*vbs;

          /* Find the processor of the new distribution */
          pe_idx = (old_phase[0]*i+virt_rank[0])%new_phase[0];
          virt_idx = (pe_idx%new_virt_dim[0]) + virt_offset;
          if (!p_or_up) virt_idx = overt_idx + vr;
          
          pe_idx = (pe_idx/new_virt_dim[0])*pe_lda[0] + idx_offset;
          
          /* Get the bucket indices and offsets */ 
          tmp1 = pe_displs[pe_idx];
          tmp1 += virt_displs[pe_idx*new_nvirt+virt_idx];
          tmp1 += virt_counts[pe_idx*new_nvirt+virt_idx];
          virt_counts[pe_idx*new_nvirt+virt_idx]++;
          /* Pack or unpack from bucket */
          if (p_or_up){
            tsr_cyclic_data[tmp1] = tsr_data[arr_idx+i];
          } else
            tsr_cyclic_data[arr_idx+i] = tsr_data[tmp1];
          arr_idx -= (overt_idx + vr)*vbs;
        }
      }
    }
    /* Adjust outer indices */
    for (act_lda=1; act_lda < ndim; act_lda++){
      overt_idx -= old_virt_idx[act_lda]*old_virt_lda[act_lda];
      /* get outer virtual index */
      old_virt_idx[act_lda]++;
      vr_max = old_virt_dim[act_lda];
      if (is_pad){
        vr_max = MIN(vr_max, ((edge_len[act_lda]-padding[act_lda])-
                      (idx[act_lda]*old_phase[act_lda]
                      +old_rank[act_lda]*old_virt_dim[act_lda])));
        //LIBT_ASSERT(vr_max!=0 || outside>0);
      }
      if (sym[act_lda] != NS && idx[act_lda] == idx[act_lda+1]){
        vr_max = MIN(vr_max, 
                ((old_virt_idx[act_lda+1]+(sym[act_lda]==SY)+
                  old_rank[act_lda+1]*old_virt_dim[act_lda+1])-
                  (old_rank[act_lda]*old_virt_dim[act_lda])));
        //LIBT_ASSERT(vr_max!=0 || outside>0);
      }
      if (old_virt_idx[act_lda] >= vr_max)
        old_virt_idx[act_lda] = 0;
      overt_idx += old_virt_idx[act_lda]*old_virt_lda[act_lda];

      /* adjust virtual rank */
      virt_rank[act_lda] = old_rank[act_lda]*old_virt_dim[act_lda]
                           +old_virt_idx[act_lda];
      /* adjust buffer offset */
      if (old_virt_idx[act_lda] == 0){
        /* Propogate buffer offset. When outer indices have multiple virtual ranks
           we must roll back the buffer to the correct position. So must keep history
           of the offsets. */
        if (idx[act_lda] == 0) acc_idx[act_lda] = 0;
        if (act_lda == 1) {
          acc_idx[act_lda] += imax;
          arr_idx += imax;
        } else {
          acc_idx[act_lda] += acc_idx[act_lda-1];
          arr_idx += acc_idx[act_lda-1];
        }
        idx[act_lda]++;
        act_max = edge_len[act_lda]/old_phase[act_lda];
        if (sym[act_lda] != NS) act_max = idx[act_lda+1]+1;
        if (idx[act_lda] >= act_max){
          idx[act_lda] = 0;
          arr_idx -= acc_idx[act_lda];
        }
        if (is_pad){
          pact_max = (edge_len[act_lda]-padding[act_lda])/old_phase[act_lda];
          if ((edge_len[act_lda]-padding[act_lda])%old_phase[act_lda] 
              > old_rank[act_lda]*old_virt_dim[act_lda])
            pact_max++;
          
          if (poutside == act_lda) poutside = -1;
          if (act_lda > poutside){
            if (idx[act_lda] >= pact_max) {
              poutside = act_lda;
            } /*else if (sym[act_lda] != NS] 
                        && idx[act_lda] == idx[sym[act_lda]]) {
              if (old_virt_idx[sym[act_lda]]+sym_type[act_lda]+
                  old_rank[sym[act_lda]]*old_virt_dim[sym[act_lda]]<=
                  (old_rank[act_lda]*old_virt_dim[act_lda])){
                outside = act_lda;
              }
            }*/
          }
        }
      }
      /* Adjust bucket locations by translating phases */
      idx_offset -= (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
      virt_offset -= (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
      idx_offs[act_lda] = ((idx[act_lda]*old_phase[act_lda]+virt_rank[act_lda])
                          %new_phase[act_lda]);
      idx_offset += (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
      virt_offset += (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];


      if (idx[act_lda] > 0 || old_virt_idx[act_lda] > 0)
        break;
    }
    if (poutside > -1) outside = 1;
    else {
      outside = 0;
      for (i=1; i<ndim; i++){
        if (sym[i] != NS && idx[i+1] == idx[i]){
          if (old_rank[i+1]*old_virt_dim[i+1]+
              old_virt_idx[i+1] + (sym[i]==SY) <=
              old_rank[i]*old_virt_dim[i] + old_virt_idx[i]){
            outside=1;
            break;
          }
        }
      }
    }
    if (act_lda == ndim) break;
  }
/*  for (i=1; i<nbuf; i++){
    printf("[%d] virt_counts = %d, virt_displs = %d\n",i-1, virt_counts[i-1],
            virt_displs[i]-virt_displs[i-1]+
            (pe_displs[i/new_nvirt]-pe_displs[(i-1)/new_nvirt]));
  }*/
  CTF_free(virt_counts);
  CTF_free(idx);
  CTF_free(acc_idx);
  CTF_free(idx_offs);
  CTF_free(old_virt_idx);
  CTF_free(virt_rank);
  CTF_free(spad);
}

int ** compute_bucket_offsets(int              ndim,
                              int const *      len,
                              int const *      old_phys_rank,
                              int const *      old_phys_dim,
                              int const *      old_virt_dim,
                              int const *      old_virt_lda,
                              int const *      old_offsets,
                              int * const *    old_permutation,
                              int const *      new_phys_dim,
                              int const *      new_phys_lda,
                              int const *      new_virt_dim,
                              int const *      new_virt_lda,
                              int              forward,
                              int              old_virt_np,
                              int              new_virt_np,
                              int const *      old_phys_edge_len,
                              int const *      old_virt_edge_len){

  TAU_FSTART(compute_bucket_offsets);
  
  int **bucket_offset; CTF_alloc_ptr(sizeof(int*)*ndim, (void**)&bucket_offset);
  
  for (int dim = 0;dim < ndim;dim++){
    CTF_alloc_ptr(sizeof(int)*old_phys_edge_len[dim], (void**)&bucket_offset[dim]);
    int pidx = 0;
    for (int vr = 0;vr < old_virt_dim[dim];vr++){
      for (int vidx = 0;vidx < old_virt_edge_len[dim];vidx++,pidx++){
        long_int _gidx = ((long_int)vidx*old_phys_dim[dim]+old_phys_rank[dim])*old_virt_dim[dim]+vr;
        long_int gidx;
        if (_gidx > len[dim] || (old_offsets != NULL && _gidx < old_offsets[dim])){
          gidx = -1;
        } else {
          if (old_permutation == NULL || old_permutation[dim] == NULL){
            gidx = _gidx;
          } else {
            gidx = old_permutation[dim][_gidx];
          }
        }
        if (gidx != -1){
          int total_rank = gidx%(new_phys_dim[dim]*new_virt_dim[dim]);
          int phys_rank = total_rank/new_virt_dim[dim];
          if (forward){
            int virt_rank = total_rank%new_virt_dim[dim];
            bucket_offset[dim][pidx] = phys_rank*MAX(1,new_phys_lda[dim])*new_virt_np+
                           virt_rank*new_virt_lda[dim];
            //printf("f %d - %d %d %d - %d - %d %d %d - %d\n", dim, vr, vidx, pidx, gidx, total_rank,
            //    phys_rank, virt_rank, bucket_offset[dim][pidx]);
          }
          else{
            bucket_offset[dim][pidx] = phys_rank*MAX(1,new_phys_lda[dim])*old_virt_np+
                           vr*old_virt_lda[dim];
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



/**
 * \brief version of pup_virt_buff which assumes padding and cyclic layout
 *
 * \param[in] ndim dimension of tensor
 * \param[in] len non-padded edge lengths of tensor
 * \param[in] sym symmetries of tensor
 * \param[in] old_phys_rank ranks of this processor on the old processor grid
 * \param[in] old_phys_dim edge lengths of the old processor grid
 * \param[in] old_virt_dim old virtualization factors along each dimension
 * \param[in] old_phys_edge_len the old tensor processor block lengths
 * \param[in] old_virt_edge_len the old tensor block lengths
 * \param[in] old_virt_nelem the old number of elements per block
 * \param[in] old_offsets old offsets of each tensor edge (corner 1 of slice)
 * \param[in] old_permutation permutation array for each edge length (no perm if NULL)
 * \param[in] new_phys_np the new number of processors
 * \param[in] new_phys_rank ranks of this processor on the new processor grid
 * \param[in] new_phys_dim edge lengths of the new processor grid
 * \param[in] new_virt_dim new virtualization factors along each dimension
 * \param[in] new_phys_edge_len the new tensor processor block lengths
 * \param[in] new_virt_edge_len the new tensor block lengths
 * \param[in] new_virt_nelem the new number of elements per block
 * \param[in,out] old_data the previous set of values stored locally
 * \param[in,out] new_data buffers to fill with data to send to each process and virtual bucket
 * \param[in] forward is 0 on the receiving side and reverses the role of all the previous parameters
 * \param[in] bucket_offset offsets for target index for each dimension
 
 */
template<typename dtype>
void pad_cyclic_pup_virt_buff(int const        ndim,
                              int const *      len,
                              int const *      sym,
                              int const *      old_phys_rank,
                              int const *      old_phys_dim,
                              int const *      old_virt_dim,
                              int const *      old_phys_edge_len,
                              int const *      old_virt_edge_len,
                              long_int const   old_virt_nelem,
                              int const *      old_offsets,
                              int * const *    old_permutation,
                              int const        new_phys_np,
                              int const *      new_phys_rank,
                              int const *      new_phys_dim,
                              int const *      new_virt_dim,
                              int const *      new_phys_edge_len,
                              int const *      new_virt_edge_len,
                              long_int const   new_virt_nelem,
                              dtype *          old_data,
                              dtype **         new_data,
                              int const        forward,
                              int * const *    bucket_offset){
  if (ndim == 0){
    if (forward)
      new_data[0][0] = old_data[0];
    else
      old_data[0] = new_data[0][0];
    return;
  }

  int old_virt_np = 1;
  for (int dim = 0;dim < ndim;dim++) old_virt_np *= old_virt_dim[dim];

  int new_virt_np = 1;
  for (int dim = 0;dim < ndim;dim++) new_virt_np *= new_virt_dim[dim];
  
  int nbucket = new_phys_np*(forward ? new_virt_np : old_virt_np);

#if DEBUG >= 1
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif

  TAU_FSTART(cyclic_pup_bucket);
#ifdef USE_OMP
  int max_ntd = omp_get_max_threads();
  max_ntd = MAX(1,MIN(max_ntd,new_virt_nelem/nbucket));

  long_int old_size, new_size;
  old_size = sy_packed_size(ndim, old_virt_edge_len, sym)*old_virt_np;
  new_size = sy_packed_size(ndim, new_virt_edge_len, sym)*new_virt_np;
  /*if (forward){
  } else {
    old_size = sy_packed_size(ndim, old_virt_edge_len, sym)*new_virt_np;
    new_size = sy_packed_size(ndim, new_virt_edge_len, sym)*old_virt_np;
  }*/
  /*printf("old_size=%d, new_size=%d,old_virt_np=%d,new_virt_np=%d\n",
          old_size,new_size,old_virt_np,new_virt_np);
*/
  int * bucket_store;  
  int * count_store;  
  int * thread_store;  
  CTF_mst_alloc_ptr(sizeof(int)*MAX(old_size,new_size), (void**)&bucket_store);
  CTF_mst_alloc_ptr(sizeof(int)*MAX(old_size,new_size), (void**)&count_store);
  CTF_mst_alloc_ptr(sizeof(int)*MAX(old_size,new_size), (void**)&thread_store);
  std::fill(bucket_store, bucket_store+MAX(old_size,new_size), -1);

  int ** par_virt_counts;
  CTF_alloc_ptr(sizeof(int*)*max_ntd, (void**)&par_virt_counts);
/*  for (int t=0; t<max_ntd; t++){
    CTF_mst_alloc_ptr(sizeof(int)*nbucket, (void**)&par_virt_counts[t]);
    std::fill(par_virt_counts[t], par_virt_counts[t]+nbucket, 0);
  }*/
  #pragma omp parallel num_threads(max_ntd)
  {
#endif

  int *offs; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&offs);
  if (old_offsets == NULL)
    for (int dim = 0;dim < ndim;dim++) offs[dim] = 0;
  else 
    for (int dim = 0;dim < ndim;dim++) offs[dim] = old_offsets[dim];

  int *ends; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&ends);
  for (int dim = 0;dim < ndim;dim++) ends[dim] = len[dim];

#ifdef USE_OMP
  int tid = omp_get_thread_num();
  int ntd = omp_get_num_threads();
  //partition the global tensor among threads, to preserve 
  //global ordering and load balance in partitioning
  int gidx_st[ndim];
  int gidx_end[ndim];
  if (ndim > 1){
    long_int all_size = packed_size(ndim-1, len+1, sym+1);
    long_int chnk = all_size/ntd;
    long_int glb_idx_st = chnk*tid + MIN(tid,all_size%ntd);
    long_int glb_idx_end = glb_idx_st+chnk+(tid<(all_size%ntd));
    //calculate global indices along each dimension corresponding to partition
//    printf("glb_idx_st = %ld, glb_idx_end = %ld\n",glb_idx_st,glb_idx_end);
    calc_idx_arr(ndim-1, len+1, sym+1, glb_idx_st, gidx_st+1);
    calc_idx_arr(ndim-1, len+1, sym+1, glb_idx_end, gidx_end+1);
    gidx_st[0] = 0;
    gidx_end[0] = 0;
#if DEBUG >= 1
    if (ntd == 1){
      if (gidx_end[ndim-1] != len[ndim-1]){
        for (int dim=0; dim<ndim; dim++){
          printf("glb_idx_end = %ld, gidx_end[%d]= %d, len[%d] = %d\n", 
                 glb_idx_end, dim, gidx_end[dim], dim, len[dim]);
        }
        ABORT;
      }
      LIBT_ASSERT(gidx_end[ndim-1] <= ends[ndim-1]);
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
  int *count; CTF_alloc_ptr(sizeof(int)*nbucket, (void**)&count);
  memset(count, 0, sizeof(int)*nbucket);

  int *gidx; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&gidx);
  memset(gidx, 0, sizeof(int)*ndim);
  for (int dim = 0;dim < ndim;dim++){
    gidx[dim] = old_phys_rank[dim]*old_virt_dim[dim];
  }

  int *virt_offset; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&virt_offset);
  memset(virt_offset, 0, sizeof(int)*ndim);

  int *idx; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&idx);
  memset(idx, 0, sizeof(int)*ndim);

  long_int *virt_acc; CTF_alloc_ptr(sizeof(long_int)*ndim, (void**)&virt_acc);
  memset(virt_acc, 0, sizeof(long_int)*ndim);

  long_int *idx_acc; CTF_alloc_ptr(sizeof(long_int)*ndim, (void**)&idx_acc);
  memset(idx_acc, 0, sizeof(long_int)*ndim);
  
  long_int *old_virt_lda; CTF_alloc_ptr(sizeof(long_int)*ndim, (void**)&old_virt_lda);
  old_virt_lda[0] = old_virt_nelem;
  for (int dim=1; dim<ndim; dim++){
    old_virt_lda[dim] = old_virt_lda[dim-1]*old_virt_dim[dim-1];
  }

  long_int offset = 0;

  long_int zero_len_toff = 0;

#ifdef USE_OMP
  for (int dim=ndim-1; dim>=0; dim--){
    int iist = MAX(0,(gidx_st[dim]-old_phys_rank[dim]*old_virt_dim[dim]));
    int ist = iist/(old_phys_dim[dim]*old_virt_dim[dim]);
    if (sym[dim] != NS) ist = MIN(ist,idx[dim+1]);
    int plen[ndim];
    memcpy(plen,old_virt_edge_len,ndim*sizeof(int));
    int idim = dim;
    do {
      plen[idim] = ist;
      idim--;
    } while (idim >= 0 && sym[idim] != NS);
    gidx[dim] += ist*old_phys_dim[dim]*old_virt_dim[dim];
    idx[dim] = ist;
    idx_acc[dim] = sy_packed_size(dim+1, plen, sym);
    offset += idx_acc[dim]; 

    LIBT_ASSERT(ist == 0 || gidx[dim] <= gidx_st[dim]);
    LIBT_ASSERT(ist < old_virt_edge_len[dim]);

    if (gidx[dim] > gidx_st[dim]) break;

    int vst = iist-ist*old_phys_dim[dim]*old_virt_dim[dim];
    if (vst > 0 ){
      vst = MIN(old_virt_dim[dim]-1,vst);
      gidx[dim] += vst;
      virt_offset[dim] = vst*old_virt_edge_len[dim];
      offset += vst*old_virt_lda[dim];
    } else vst = 0;
    if (gidx[dim] > gidx_st[dim]) break;
  }
#endif

  bool done = false;
  for (;!done;){
    int bucket0 = 0;
    bool outside0 = false;
    int len_zero_max = ends[0];
#ifdef USE_OMP
    bool is_at_end = true;
    bool is_at_start = true;
    for (int dim = ndim-1;dim >0;dim--){
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
    for (int dim = ndim-1;dim >0;dim--){
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
      for (int dim = 1;dim < ndim;dim++){
        if (bucket_offset[dim][virt_offset[dim]+idx[dim]] == -1) outside0 = true;
        bucket0 += bucket_offset[dim][virt_offset[dim]+idx[dim]];
      }
    }

    if (!outside0){
      for (int dim = 1;dim < ndim;dim++){
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
        int virt_min = MAX(0,MIN(old_virt_dim[0],gidx_min-gidx[0]));
        int virt_max = MAX(0,MIN(old_virt_dim[0],gidx_max-gidx[0]));

        offset += old_virt_nelem*virt_min;
        if (forward){
          for (virt_offset[0] = virt_min*old_virt_edge_len[0];
               virt_offset[0] < virt_max*old_virt_edge_len[0];
               virt_offset[0] += old_virt_edge_len[0])
          {
            int bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
#ifdef USE_OMP
            bucket_store[offset] = bucket;
            count_store[offset]  = count[bucket]++;
            thread_store[offset] = tid;
#else
            new_data[bucket][count[bucket]++] = old_data[offset];
#endif
            offset += old_virt_nelem;
          }
        }
        else{
          for (virt_offset[0] = virt_min*old_virt_edge_len[0];
               virt_offset[0] < virt_max*old_virt_edge_len[0];
               virt_offset[0] += old_virt_edge_len[0])
          {
            int bucket = bucket0+bucket_offset[0][virt_offset[0]+idx[0]];
#ifdef USE_OMP
            bucket_store[offset] = bucket;
            count_store[offset]  = count[bucket]++;
            thread_store[offset] = tid;
#else
            old_data[offset] = new_data[bucket][count[bucket]++];
#endif
            offset += old_virt_nelem;
          }
        }

        offset++;
        offset -= old_virt_nelem*virt_max;
        gidx[0] += old_phys_dim[0]*old_virt_dim[0];
      }

      offset -= idx_max;
      gidx[0] -= idx_max*old_phys_dim[0]*old_virt_dim[0];
    }
     
    idx_acc[0] = idx_max;

    idx[0] = 0;

    zero_len_toff = 0;

    /* Adjust outer indices */
    if (!done){
      for (int dim = 1;dim < ndim;dim++){
        offset += old_virt_lda[dim];
  
        virt_offset[dim] += old_virt_edge_len[dim];
        gidx[dim]++;

        if (virt_offset[dim] == old_virt_dim[dim]*old_virt_edge_len[dim]){
          offset -= old_virt_lda[dim]*old_virt_dim[dim];
          gidx[dim] -= old_virt_dim[dim];
          virt_offset[dim] = 0;

          offset += idx_acc[dim-1];
          idx_acc[dim] += idx_acc[dim-1];
          idx_acc[dim-1] = 0;

          gidx[dim] -= idx[dim]*old_phys_dim[dim]*old_virt_dim[dim];
          idx[dim]++;

          if (idx[dim] == (sym[dim] == NS ? old_virt_edge_len[dim] : idx[dim+1]+1)){
            offset -= idx_acc[dim];
            //index should always be zero here sicne everything is SY and not SH
            idx[dim] = 0;//(dim == 0 || sym[dim-1] == NS ? 0 : idx[dim-1]);
            //gidx[dim] += idx[dim]*old_phys_dim[dim]*old_virt_dim[dim];

            if (dim == ndim-1) done = true;
          }
          else{
            gidx[dim] += idx[dim]*old_phys_dim[dim]*old_virt_dim[dim];
            break;
          }
        }
        else{
          idx_acc[dim-1] = 0;
          break;
        }
      }
      if (ndim <= 1) done = true;
    }
  }
  CTF_free(gidx);
  CTF_free(idx_acc);
  CTF_free(virt_acc);
  CTF_free(idx);
  CTF_free(virt_offset);
  CTF_free(old_virt_lda);

#ifndef USE_OMP
#if DEBUG >= 1
  bool pass = true;
  for (int i = 0;i < nbucket-1;i++){
    if (count[i] != (long_int)(new_data[i+1]-new_data[i])){
      printf("rank = %d count %d should have been %d is %d\n", rank, i, (int)(new_data[i+1]-new_data[i]), count[i]);
      pass = false;
    }
  }
  if (!pass) ABORT;
#endif
#endif
  CTF_free(offs);
  CTF_free(ends);
 
#ifndef USE_OMP
  CTF_free(count);
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
    if (bckt < nbucket-1 && par_tmp != new_data[bckt+1]-new_data[bckt]){
      printf("rank = %d count for bucket %d is %d should have been %ld\n",rank,bckt,par_tmp,(long_int)(new_data[bckt+1]-new_data[bckt]));
      ABORT;
    }
#endif
  }
  TAU_FSTOP(cyclic_pup_bucket);
  TAU_FSTART(cyclic_pup_move);
  if (forward){
    #pragma omp parallel for
    for (int i=0; i<MAX(old_size,new_size); i++){
      if (bucket_store[i] != -1){
        int ct = count_store[i]+par_virt_counts[thread_store[i]][bucket_store[i]];
        new_data[bucket_store[i]][ct] = old_data[i];
      }
    }
  } else {
    #pragma omp parallel for
    for (int i=0; i<MAX(old_size,new_size); i++){
      if (bucket_store[i] != -1){
        int ct = count_store[i]+par_virt_counts[thread_store[i]][bucket_store[i]];
        old_data[i] = new_data[bucket_store[i]][ct];
      }
    }
  }
  TAU_FSTOP(cyclic_pup_move);
  for (int t=0; t<max_ntd; t++){
    CTF_free(par_virt_counts[t]);
  }
  CTF_free(par_virt_counts);
  CTF_free(count_store);
  CTF_free(bucket_store);
  CTF_free(thread_store);
#endif
}

/**
 * \brief optimized version of pup_virt_buff
 */
template<typename dtype>
void opt_pup_virt_buff(int const        ndim,
                       int const        nbuf,
                       int const        numPes,
                       long_int const   hvbs,
                       int const        old_nvirt,
                       int const        new_nvirt,
                       int const *      old_edge_len,
                       int const *      new_edge_len,
                       int const *      sym,
                       int const *      old_phase,
                       int const *      old_rank,
                       int const *      new_phase,
                       int const *      new_rank,
                       int const *      old_virt_dim,
                       int const *      new_virt_dim,
                       int const *      pe_displs,
                       int const *      virt_displs,
                       int const *      old_virt_lda,
                       int const *      new_virt_lda,
                       int const *      pe_lda,
                       long_int const   vbs,
                       int const        is_pad,
                       int const *      padding,
                       dtype *          tsr_data,
                       dtype *          tsr_cyclic_data,
                       int const        was_cyclic,
                       int const        is_cyclic,
                       int const        p_or_up){

/*  if (ndim==4){
    if (sym[ndim-2] != NS) printf("ncase with last sym\n");
    else printf("case with no last sym\n");
  }*/

  long_int old_size, new_size, ntd, tid, pe_idx, virt_idx;
  long_int i, tmp1;
  long_int * par_virt_counts;
  long_int * pe_idx_store, * target_store;
  int * sub_old_edge_len, * sub_new_edge_len;

#ifdef USE_OMP
  int max_ntd = omp_get_max_threads();
  max_ntd = MAX(1,MIN(max_ntd,vbs/nbuf));
  ntd = max_ntd;
#else
/*  int const max_ntd = 4;
  ntd = 4;*/
  int const max_ntd = 1;
  ntd = 1;
#endif
  //CTF_alloc_ptr(sizeof(long_int*)*max_ntd, (void**)&par_virt_counts);
  CTF_alloc_ptr(sizeof(long_int)*max_ntd*nbuf, (void**)&par_virt_counts);
  //for (i=0; i<max_ntd; i++)
  //  CTF_mst_alloc_ptr(nbuf*sizeof(long_int), (void**)&par_virt_counts[i]);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&sub_old_edge_len);
  for (i=0; i<ndim; i++){
    sub_old_edge_len[i] = old_edge_len[i]/old_phase[i];
  }
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&sub_new_edge_len);
  for (i=0; i<ndim; i++){
    sub_new_edge_len[i] = new_edge_len[i]/new_phase[i];
  }
  if (p_or_up){
    old_size = sy_packed_size(ndim, sub_old_edge_len, sym)*old_nvirt;
    new_size = sy_packed_size(ndim, sub_new_edge_len, sym)*new_nvirt;
  } else {
    old_size = sy_packed_size(ndim, sub_old_edge_len, sym)*new_nvirt;
    new_size = sy_packed_size(ndim, sub_new_edge_len, sym)*old_nvirt;
  }
  CTF_mst_alloc_ptr(sizeof(long_int)*MAX(old_size,new_size), 
                    (void**)&pe_idx_store);
  CTF_mst_alloc_ptr(sizeof(long_int)*MAX(old_size,new_size), 
                    (void**)&target_store);
  std::fill(target_store, target_store+MAX(old_size,new_size), -1);
  //for (i=0; i<max_ntd; i++)
  //  memset(par_virt_counts[i], 0, nbuf*sizeof(long_int));
  memset(par_virt_counts, 0, max_ntd*nbuf*sizeof(long_int));
  TAU_FSTART(opt_pup_virt_buf_calc_off);
#ifdef USE_OMP
#pragma omp parallel private(ntd, tid, pe_idx, virt_idx, i, tmp1) num_threads(max_ntd)
#else
//for (tid=0; tid<ntd; tid++)

#endif
{
/*  TAU_FSTART(thread_loop_time);
  if (tid != 1) TAU_FSTOP(thread_loop_time);*/
  long_int imax, padimax, act_lda, act_max, ledge_len_max, ledge_len_st, cptr;
  long_int poutside, overt_idx, part_size, idx_offset, csize, allsize;
  long_int virt_offset, arr_idx;
  long_int vr, outside, vr_max, pact_max, ispm;
  long_int * virt_counts, * idx, * old_virt_idx, * virt_rank;
  int * off_edge_len, * sub_edge_len;
  long_int * idx_offs;
  long_int * acc_idx, * spad;
#ifdef USE_OMP
  tid = omp_get_thread_num();
  ntd = omp_get_num_threads();
#else
  tid = 0;
  ntd = 1;
#endif


  if (ndim == 0){
    if (tid == 0){
      ledge_len_st = 0;
      ledge_len_max = 1;
    } else {
      ledge_len_st = 1;
      ledge_len_max = 1;
    }
  } else {
    if (ndim == 1){
      if (tid == 0){
        ledge_len_st = 0;
        ledge_len_max = old_edge_len[0]/old_phase[0];
      } else {
        ledge_len_st = old_edge_len[0]/old_phase[0];
        ledge_len_max = old_edge_len[0]/old_phase[0];
      }
    } else {
      if (sym[ndim-1] == 0){
        part_size = (old_edge_len[ndim-1]/old_phase[ndim-1])/ntd;
        if (tid < (old_edge_len[ndim-1]/old_phase[ndim-1])%ntd){
          part_size++;
          ledge_len_st = ((old_edge_len[ndim-1]/old_phase[ndim-1])/ntd + 1)*tid;
        } else
          ledge_len_st = ((old_edge_len[ndim-1]/old_phase[ndim-1])/ntd)*tid
            + (old_edge_len[ndim-1]/old_phase[ndim-1])%ntd;
        ledge_len_max = ledge_len_st+part_size;
      } else {
        CTF_alloc_ptr(ndim*sizeof(int), (void**)&sub_edge_len);
        part_size = (old_edge_len[ndim-1]/old_phase[ndim-1])/ntd;
        cptr = 0;
        for (i=0; i<ndim; i++){
          sub_edge_len[i] = old_edge_len[i]/old_phase[i];
        }
        allsize = packed_size(ndim, sub_edge_len, sym);
      
        ledge_len_st = -1;
        ledge_len_max = -1;
        for (i=0; i<old_edge_len[ndim-1]/old_phase[ndim-1]; i++){
          sub_edge_len[ndim-1] = i;
          cptr = ndim - 2;
          while (cptr >= 0 && sym[cptr] != 0){
            sub_edge_len[cptr] = i;
            cptr--;
          }
          csize = packed_size(ndim, sub_edge_len, sym);
          if (ledge_len_st == -1 && csize >= (allsize/ntd)*tid){
            ledge_len_st = i;
          }
          if (ledge_len_max == -1 && csize >= (allsize/ntd)*(tid+1)){
            ledge_len_max = i;
          }
        }
        LIBT_ASSERT(ledge_len_max != -1);
      }

    }
  }

  if (ledge_len_max!=ledge_len_st) {

  //CTF_alloc_ptr(nbuf*sizeof(long_int), (void**)&virt_counts);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&acc_idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&old_virt_idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&virt_rank);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&spad);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&off_edge_len);

  //virt_counts = par_virt_counts[tid];
  virt_counts = par_virt_counts+tid*nbuf;

  memset(old_virt_idx, 0, ndim*sizeof(long_int));
  memset(acc_idx, 0, ndim*sizeof(long_int));
//  memset(virt_counts, 0, nbuf*sizeof(long_int));

  for (i=0; i<ndim; i++){ virt_rank[i] = old_rank[i]*old_virt_dim[i]; }

  /* Loop accross all virtual subtensors in order, and keep order within
     subtensors. We should move through one of the arrays contiguously */
  arr_idx = 0;
  overt_idx = 0;
  memset(idx, 0, ndim*sizeof(long_int));
  memset(idx_offs, 0, ndim*sizeof(long_int));

  if (ndim != 0){
    idx[ndim-1] = ledge_len_st;

    for (i=0; i<ndim; i++){
      off_edge_len[i] = old_edge_len[i]/old_phase[i];
    }
    i=ndim-1;
    do {
      off_edge_len[i] = ledge_len_st;
      i--;
    } while (i>=0 && sym[i] != NS);

    arr_idx += sy_packed_size(ndim, off_edge_len, sym);
  }

  idx_offset = 0;
  virt_offset = 0;
  for (i=1; i<ndim; i++) {
    if (was_cyclic)
      idx_offs[i] = idx[i]*old_phase[i]+virt_rank[i];
    else
      idx_offs[i] = idx[i]+virt_rank[i]*old_edge_len[i]/old_phase[i];
    if (is_cyclic)
      idx_offs[i] = idx_offs[i]%new_phase[i];
    else
      idx_offs[i] = idx_offs[i]/(new_edge_len[i]/new_phase[i]);
    idx_offset += (idx_offs[i]/new_virt_dim[i])*pe_lda[i];
    virt_offset += (idx_offs[i]%new_virt_dim[i])*new_virt_lda[i];
  }
  poutside = -1;
  if (is_pad){
    for (i=1; i<ndim; i++){
      pact_max = (old_edge_len[i]-padding[i])/old_phase[i];
      if ((old_edge_len[i]-padding[i])%old_phase[i] 
          > old_rank[i]*old_virt_dim[i])
        pact_max++;
      
      if (poutside == i) poutside = -1;
      if (i > poutside){
        if (idx[i] >= pact_max) {
          poutside = i;
        } 
      }
    }
  }
  outside = 0;
  if (poutside > -1) outside = 1;
  for (i=1; i<ndim; i++){
    if (sym[i] != NS && idx[i+1] == idx[i]){
      if (old_rank[i+1]*old_virt_dim[i+1]+(sym[i]==SY)<=
          old_rank[i]*old_virt_dim[i]){
        outside=1;
        break;
      }
    }
  }
  if (ndim > 0){
    imax = old_edge_len[0]/old_phase[0];
    for (;;){
      if (sym[0] != NS)
        imax = idx[1]+1;
      /* NOTE: Must iterate across all local data in correct global order
       *        to make the back-transformation correct */
      /* for each leading index owned by a virtual prc along an edge */
      if (outside == 0){
        if (is_pad) {
          ispm = (old_edge_len[0]-padding[0])/old_phase[0];
          if ((old_edge_len[0]-padding[0])%old_phase[0] > 
              old_rank[0]*old_virt_dim[0])
            ispm++;
          padimax = MIN(imax,ispm);
        } else
          padimax = imax;
        if (ndim == 1) {
          i=ledge_len_st;
          padimax = MIN(padimax,ledge_len_max);
        } else i=0;
        for (; i<padimax; i++){
        /* for each virtual proc along leading dim */
          vr_max = old_virt_dim[0];
          if (is_pad){
            vr_max = MIN(vr_max, ((old_edge_len[0]-padding[0])-
                                  (i*old_phase[0]+old_rank[0]*old_virt_dim[0])));
  //      printf("vr_max = %d rather than %d\n",vr_max,old_virt_dim[0]);
          }
          if (sym[0] != NS && i == idx[1]){
            vr_max = MIN(vr_max, ((old_virt_idx[1]+(sym[0]==SY)+
                      old_rank[1]*old_virt_dim[1])-
                      (old_rank[0]*old_virt_dim[0])));
          }
          if (was_cyclic&&is_cyclic)
          {
            long_int pe_idx_0 = (old_phase[0]*i+old_rank[0]*old_virt_dim[0])%new_phase[0];
            long_int pe_idx_1 = (pe_idx_0/new_virt_dim[0])*pe_lda[0];
            long_int pe_idx_2 = pe_idx_0;
            long_int virt_idx_0 = (pe_idx_0%new_virt_dim[0]);
            for (vr=0; vr<vr_max; vr++){
              virt_rank[0] = old_rank[0]*old_virt_dim[0]+vr;
              /* Find the offset of this virtual proc */
              arr_idx += (overt_idx + vr)*vbs;
            
              /* Find the processor of the new distribution */
              if (!p_or_up) virt_idx = overt_idx + vr;
              else virt_idx = virt_idx_0 + virt_offset;

              pe_idx = pe_idx_1 + idx_offset;
              pe_idx_store[arr_idx+i] = pe_idx*new_nvirt+virt_idx+tid*nbuf;

              /* Get the bucket indices and offsets */
              tmp1 = pe_displs[pe_idx];
              tmp1 += virt_displs[pe_idx*new_nvirt+virt_idx];
              tmp1 += virt_counts[pe_idx*new_nvirt+virt_idx];
              virt_counts[pe_idx*new_nvirt+virt_idx]++;
  //            /* Pack or unpack from bucket */
              target_store[arr_idx+i]=tmp1;
    /*      if (p_or_up){
                tsr_cyclic_data[tmp1] = tsr_data[arr_idx+i];
              } else
                tsr_cyclic_data[arr_idx+i] = tsr_data[tmp1];*/
              arr_idx -= (overt_idx + vr)*vbs;

              if (pe_idx_0 == new_phase[0]-1)
              {
                  virt_idx_0 = 0;
                  pe_idx_0 = 0;
                  pe_idx_1 = 0;
                  pe_idx_2 = 0;
              }
              else
              {
                  virt_idx_0 = (virt_idx_0 == new_virt_dim[0]-1 ? 0 : virt_idx_0+1);
                  pe_idx_0++;
                  pe_idx_2++;
                  if (pe_idx_2 == new_virt_dim[0])
                  {
                      pe_idx_2 = 0;
                      pe_idx_1 += pe_lda[0];
                  }
              }
            }
          }
          else
          {
            for (vr=0; vr<vr_max; vr++){
              virt_rank[0] = old_rank[0]*old_virt_dim[0]+vr;
              /* Find the offset of this virtual proc */
              arr_idx += (overt_idx + vr)*vbs;
            
              /* Find the processor of the new distribution */
              //pe_idx = (old_phase[0]*i+virt_rank[0])%new_phase[0];
              if (was_cyclic)
                pe_idx = i*old_phase[0]+virt_rank[0];
              else
                pe_idx = i+virt_rank[0]*old_edge_len[0]/old_phase[0];
              if (is_cyclic)
                pe_idx = pe_idx%new_phase[0];
              else
                pe_idx = pe_idx/(new_edge_len[0]/new_phase[0]);
              virt_idx = (pe_idx%new_virt_dim[0]) + virt_offset;
              if (!p_or_up) virt_idx = overt_idx + vr;

              pe_idx = (pe_idx/new_virt_dim[0])*pe_lda[0] + idx_offset;
              //pe_idx_store[arr_idx+i] = pe_idx*new_nvirt+virt_idx;
              //pe_idx_store[arr_idx+i] = pe_idx_store[arr_idx+i]*ntd+tid;
              pe_idx_store[arr_idx+i] = pe_idx*new_nvirt+virt_idx+tid*nbuf;

              /* Get the bucket indices and offsets */
              tmp1 = pe_displs[pe_idx];
              tmp1 += virt_displs[pe_idx*new_nvirt+virt_idx];
              tmp1 += virt_counts[pe_idx*new_nvirt+virt_idx];
              virt_counts[pe_idx*new_nvirt+virt_idx]++;
//              /* Pack or unpack from bucket */
              target_store[arr_idx+i]=tmp1;
  /*        if (p_or_up){
                tsr_cyclic_data[tmp1] = tsr_data[arr_idx+i];
              } else
                tsr_cyclic_data[arr_idx+i] = tsr_data[tmp1];*/
              arr_idx -= (overt_idx + vr)*vbs;
            }
          }
        } 
      }
      /* Adjust outer indices */
      for (act_lda=1; act_lda < ndim; act_lda++){
        overt_idx -= old_virt_idx[act_lda]*old_virt_lda[act_lda];
        /* get outer virtual index */
        old_virt_idx[act_lda]++;
        vr_max = old_virt_dim[act_lda];
        if (is_pad){
          vr_max = MIN(vr_max, ((old_edge_len[act_lda]-padding[act_lda])-
                        (idx[act_lda]*old_phase[act_lda]
                        +old_rank[act_lda]*old_virt_dim[act_lda])));
          //LIBT_ASSERT(vr_max!=0 || outside>0);
        }
        if (sym[act_lda] != NS && idx[act_lda] == idx[act_lda+1]){
          vr_max = MIN(vr_max, 
                  ((old_virt_idx[act_lda+1]+(sym[act_lda]==SY)+
                    old_rank[act_lda+1]*old_virt_dim[act_lda+1])-
                    (old_rank[act_lda]*old_virt_dim[act_lda])));
          //LIBT_ASSERT(vr_max!=0 || outside>0);
        }
        if (old_virt_idx[act_lda] >= vr_max)
          old_virt_idx[act_lda] = 0;
        overt_idx += old_virt_idx[act_lda]*old_virt_lda[act_lda];

        /* adjust virtual rank */
        virt_rank[act_lda] = old_rank[act_lda]*old_virt_dim[act_lda]
                             +old_virt_idx[act_lda];
        /* adjust buffer offset */
        if (old_virt_idx[act_lda] == 0){
          /* Propogate buffer offset. When outer indices have multiple virtual ranks
             we must roll back the buffer to the correct position. So must keep history
             of the offsets. */
          if (idx[act_lda] == 0) acc_idx[act_lda] = 0;
          if (act_lda == 1) {
            acc_idx[act_lda] += imax;
            arr_idx += imax;
          } else {
            acc_idx[act_lda] += acc_idx[act_lda-1];
            arr_idx += acc_idx[act_lda-1];
          }
          idx[act_lda]++;
          if (act_lda == ndim - 1)
            act_max = ledge_len_max;
          else
            act_max = old_edge_len[act_lda]/old_phase[act_lda];
          if (sym[act_lda] != NS) act_max = idx[act_lda+1]+1;
          if (idx[act_lda] >= act_max){
            idx[act_lda] = 0;
            arr_idx -= acc_idx[act_lda];
          }
          if (is_pad){
            pact_max = (old_edge_len[act_lda]-padding[act_lda])/old_phase[act_lda];
            if ((old_edge_len[act_lda]-padding[act_lda])%old_phase[act_lda] 
                > old_rank[act_lda]*old_virt_dim[act_lda])
              pact_max++;
            
            if (poutside == act_lda) poutside = -1;
            if (act_lda > poutside){
              if (idx[act_lda] >= pact_max) {
                poutside = act_lda;
              } /*else if (sym[act_lda] != NS] 
                          && idx[act_lda] == idx[sym[act_lda]]) {
                if (old_virt_idx[sym[act_lda]]+sym_type[act_lda]+
                    +old_rank[sym[act_lda]]*old_virt_dim[sym[act_lda]]<=
                    (old_rank[act_lda]*old_virt_dim[act_lda])){
                  outside = act_lda;
                }
              }*/
            }
          }
        }
        /* Adjust bucket locations by translating phases */     
        idx_offset -= (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        virt_offset -= (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
        if (was_cyclic)
          idx_offs[act_lda] = idx[act_lda]*old_phase[act_lda]+virt_rank[act_lda];
        else 
          idx_offs[act_lda] = idx[act_lda]+virt_rank[act_lda]*old_edge_len[act_lda]/old_phase[act_lda];
        if (is_cyclic)
          idx_offs[act_lda] = idx_offs[act_lda]%new_phase[act_lda];
        else
          idx_offs[act_lda] = idx_offs[act_lda]/(new_edge_len[act_lda]/new_phase[act_lda]);
        idx_offset += (idx_offs[act_lda]/new_virt_dim[act_lda])*pe_lda[act_lda];
        virt_offset += (idx_offs[act_lda]%new_virt_dim[act_lda])*new_virt_lda[act_lda];
    

        if (idx[act_lda] > 0 || old_virt_idx[act_lda] > 0)
          break;
      }
      if (poutside > -1) outside = 1;
      else {
        outside = 0;
        for (i=1; i<ndim; i++){
          if (sym[i] != NS && idx[i+1] == idx[i]){
            if (old_rank[i+1]*old_virt_dim[i+1]+
                old_virt_idx[i+1] + (sym[i]==SY)<=
                old_rank[i]*old_virt_dim[i] + old_virt_idx[i]){
              outside=1;
              break;
            }
          }
        }
      }
      if (act_lda == ndim) {
        break;
      }
    }
  }

  CTF_free(idx);
  CTF_free(acc_idx);
  CTF_free(idx_offs);
  CTF_free(old_virt_idx);
  CTF_free(virt_rank);
  CTF_free(spad);
  CTF_free(off_edge_len);
  }
  //if (tid == 1) TAU_FSTOP(thread_loop_time);
}
  TAU_FSTOP(opt_pup_virt_buf_calc_off);
  long_int par_i, par_j, par_tmp;
  for (par_i=0; par_i<nbuf; par_i++){
    par_tmp = 0;
    for (par_j=0; par_j<max_ntd; par_j++){
      //par_tmp += par_virt_counts[par_j][par_i];
      //par_virt_counts[par_j][par_i] = par_tmp - par_virt_counts[par_j][par_i];
      par_tmp += par_virt_counts[par_i+par_j*nbuf];
      par_virt_counts[par_i+par_j*nbuf] = par_tmp - par_virt_counts[par_i+par_j*nbuf];
    }
  }
  TAU_FSTART(opt_pup_virt_buf_move);
  if (ndim == 0){
    tsr_cyclic_data[0] = tsr_data[0];
  } else {
    if (p_or_up){
#ifdef USE_OMP
      #pragma omp parallel for private(i, tmp1, virt_idx)
#endif
      for (i=0; i<MAX(new_size,old_size); i++){
        tmp1 = target_store[i];
        if (tmp1 != -1){
          //virt_idx = pe_idx_store[i]/ntd;
          /*tmp1 += par_virt_counts[(pe_idx_store[i]%ntd)*nbuf +
                                  virt_idx];*/
          //tmp1 += par_virt_counts[(pe_idx_store[i]%ntd)][virt_idx];
          tmp1 += par_virt_counts[pe_idx_store[i]];
          tsr_cyclic_data[tmp1] = tsr_data[i];
        }
      }
    } else {
#ifdef USE_OMP
      #pragma omp parallel for private(i, tmp1, virt_idx)
#endif
      for (i=0; i<MAX(new_size,old_size); i++){
        tmp1 = target_store[i];
        if (tmp1 != -1){
          //virt_idx = pe_idx_store[i]/ntd;
          /*tmp1 += par_virt_counts[(pe_idx_store[i]%ntd)*nbuf +
                                  virt_idx];*/
          //tmp1 += par_virt_counts[(pe_idx_store[i]%ntd)][virt_idx];
          tmp1 += par_virt_counts[pe_idx_store[i]];
          tsr_cyclic_data[i] = tsr_data[tmp1];
        }
      }
    }
  }

  TAU_FSTOP(opt_pup_virt_buf_move);
  CTF_free(sub_old_edge_len);
  CTF_free(sub_new_edge_len);
  CTF_free(pe_idx_store);
  CTF_free(target_store);
  //for (i=0; i<ntd; i++)
  //  CTF_free(par_virt_counts[i]);
  CTF_free(par_virt_counts);
//#endif
}

/**
 * \brief Repad and reshuffle elements
 *
 * \param ndim number of tensor dimensions
 * \param nval number of elements in each subtensor
 * \param old_edge_len old edge lengths of tensor
 * \param sym symmetries of tensor
 * \param old_phase current physical*virtual phase
 * \param old_rank current physical rank
 * \param old_pe_lda old lda of processor grid
 * \param is_old_pad whether this tensor is currently padded
 * \param old_padding the padding in each dimension
 * \param new_edge_len new edge lengths of tensor
 * \param new_phase new physival*virtual phase
 * \param new_rank new physical rank
 * \param new_pe_lda lda of processor grid
 * \param is_new_pad tells function whether new layout is padded
 * \param new_padding the new padding to apply to edge_len
 * \param old_virt_dim current virtualization dimensions on each process
 * \param new_virt_dim new virtualization dimensions on each process
 * \param tsr_data current tensor data
 * \param tsr_cyclic_data pointer to a a pointer that will point to 
 *              new tensor of data that will be alloced
 * \param ord_glb_comm the global communicator
 */
template<typename dtype>
int padded_reshuffle(int const          tid,
                     int const          ndim, 
                     long_int const     nval,
                     int const *        old_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const *        old_pe_lda,
                     int const          is_old_pad,
                     int const *        old_padding,
                     int const *        new_edge_len,
                     int const *        new_phase,
                     int const *        new_rank,
                     int const *        new_pe_lda,
                     int const          is_new_pad,
                     int const *        new_padding,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     dtype *            tsr_data,
                     dtype **           tsr_cyclic_data,
                     CommData_t *       ord_glb_comm){
  int i, old_num_virt, new_num_virt, numPes;
  long_int new_nval, swp_nval;
  long_int old_size;
  int idx_lyr;
  int * virt_phase_rank, * old_virt_phase_rank, * sub_edge_len;
  tkv_pair<dtype> * pairs;
  dtype * tsr_new_data;
  DEBUG_PRINTF("Performing padded reshuffle\n");

  TAU_FSTART(padded_reshuffle);

  numPes = ord_glb_comm->np;

  CTF_alloc_ptr(ndim*sizeof(int), (void**)&virt_phase_rank);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&old_virt_phase_rank);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&sub_edge_len);

  new_num_virt = 1;
  old_num_virt = 1;
  idx_lyr = ord_glb_comm->rank;
  for (i=0; i<ndim; i++){
/*  if (ord_glb_comm == NULL || ord_glb_comm->rank == 0){
      printf("old_pe_lda[%d] = %d, old_phase = %d, old_virt_dim = %d\n",
              i,old_pe_lda[i], old_phase[i], old_virt_dim[i]);
      printf("new_pe_lda[%d] = %d, new_phase = %d, new_virt_dim = %d\n",
            i,new_pe_lda[i], new_phase[i], new_virt_dim[i]);
    printf("is_new_pad = %d\n", is_new_pad);
    if (is_new_pad)
      printf("new_padding[%d] = %d\n", i, new_padding[i]);
    printf("is_old_pad = %d\n", is_old_pad);
    if (is_old_pad)
      printf("old_padding[%d] = %d\n", i, old_padding[i]);
    }*/
    old_num_virt = old_num_virt*old_virt_dim[i];
    new_num_virt = new_num_virt*new_virt_dim[i];
    virt_phase_rank[i] = new_rank[i]*new_virt_dim[i];
    old_virt_phase_rank[i] = old_rank[i]*old_virt_dim[i];
    //idx_lyr -= (old_phase[i]/old_virt_dim[i])*old_rank[i];
    idx_lyr -= old_rank[i]*old_pe_lda[i];
  }
  if (idx_lyr == 0 ){
    read_loc_pairs(ndim, nval, is_old_pad, old_num_virt, sym,
                   old_edge_len, old_padding, old_virt_dim,
                   old_phase, old_virt_phase_rank, &new_nval, tsr_data,
                   &pairs);
  } else {
    new_nval = 0;
    pairs = NULL;
  }

  old_size = sy_packed_size(ndim, new_edge_len, sym);

  for (i=0; i<ndim; i++){
    sub_edge_len[i] = new_edge_len[i] / new_phase[i];
  }
  if (ord_glb_comm->rank == 0){
    DPRINTF(1,"Tensor %d now has virtualization factor of %d\n",tid,new_num_virt);
  }
  swp_nval = new_num_virt*sy_packed_size(ndim, sub_edge_len, sym);
  if (ord_glb_comm->rank == 0){
    DPRINTF(1,"Tensor %d is of size "PRId64", has factor of %lf growth due to padding\n", 
    
          tid, swp_nval,
          ord_glb_comm->np*(swp_nval/(double)old_size));
  }

  CTF_alloc_ptr(swp_nval*sizeof(dtype), (void**)&tsr_new_data);

  std::fill(tsr_new_data, tsr_new_data+swp_nval, get_zero<dtype>());


  wr_pairs_layout(ndim,
                  numPes,
                  new_nval,
                  (dtype)1.0,
                  (dtype)0.0,
                  (is_old_pad | is_new_pad),
                  'w',
                  new_num_virt,
                  sym,
                  new_edge_len,
                  new_padding,
                  new_phase,
                  new_virt_dim,
                  virt_phase_rank,
                  new_pe_lda,
                  pairs,
                  tsr_new_data,
                  ord_glb_comm);
                
  *tsr_cyclic_data = tsr_new_data;


  CTF_free(old_virt_phase_rank);
  if (pairs != NULL)
    CTF_free(pairs);
  CTF_free(virt_phase_rank);
  CTF_free(sub_edge_len);
  TAU_FSTOP(padded_reshuffle);

  return DIST_TENSOR_SUCCESS;
}


/* Goes from any set of phases to any new set of phases */
/**
 * \brief Reshuffle elements
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] nval number of elements in the subtensors each proc owns
 * \param[in] old_edge_len old edge lengths of tensor
 * \param[in] sym symmetries of tensor
 * \param[in] old_phase current physical*virtual phase
 * \param[in] old_rank current physical rank
 * \param[in] old_pe_lda old lda of processor grid
 * \param[in] is_old_pad whether tensor was padded
 * \param[in] old_padding padding of current tensor
 * \param[in] old_offsets old offsets of each tensor edge (corner 1 of slice)
 * \param[in] old_permutation permutation array for each edge length (no perm if NULL)
 * \param[in] new_edge_len new edge lengths of tensor
 * \param[in] new_phase new physical*virtual phase
 * \param[in] new_rank new physical rank
 * \param[in] new_pe_lda new lda of processor grid
 * \param[in] is_new_pad whether tensor will be padded
 * \param[in] new_padding padding we want
 * \param[in] new_offsets old offsets of each tensor edge (corner 1 of slice)
 * \param[in] new_permutation permutation array for each edge length (no perm if NULL)
 * \param[in] old_virt_dim current virtualization dimensions on each process
 * \param[in] new_virt_dim new virtualization dimensions on each process
 * \param[in,out] ptr_tsr_data current tensor data
 * \param[out] ptr_tsr_cyclic_data pointer to a new tensor of 
              data that will be filled
 * \param[in] ord_glb_comm the global communicator
 * \param[in] was_cyclic whether the mapping was cyclic
 * \param[in] is_cyclic whether the mapping is cyclic
 */
template<typename dtype>
int cyclic_reshuffle(int const          ndim, 
                     long_int const     nval,
                     int const *        old_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const *        old_pe_lda,
                     int const          is_old_pad,
                     int const *        old_padding,
                     int const *        old_offsets,
                     int * const *      old_permutation,
                     int const *        new_edge_len,
                     int const *        new_phase,
                     int const *        new_rank,
                     int const *        new_pe_lda,
                     int const          is_new_pad,
                     int const *        new_padding,
                     int const *        new_offsets,
                     int * const *      new_permutation,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     dtype **           ptr_tsr_data,
                     dtype **           ptr_tsr_cyclic_data,
                     CommData_t *       ord_glb_comm,
                     int const          was_cyclic,
                     int const          is_cyclic){
  /* If we want to use key value pairs to reshuffle */
  int i, nbuf, np, old_nvirt, new_nvirt, old_np, new_np, idx_lyr;
  long_int vbs_old, vbs_new;
  long_int hvbs_old, hvbs_new;
  long_int swp_nval;
  int * hsym;
  int * send_counts, * recv_counts, * idx, * buf_lda;
  long_int * idx_offs;
  int * svirt_displs, * rvirt_displs, * send_displs;
  int * recv_displs, * new_virt_lda, * old_virt_lda;
  int * old_sub_edge_len, * new_sub_edge_len;

  dtype * tsr_cyclic_data, * swp_ptr;
  dtype * tsr_data = *ptr_tsr_data;
  if (ndim == 0){
    CTF_alloc_ptr(sizeof(dtype), (void**)&tsr_cyclic_data);
    if (ord_glb_comm->rank == 0){
      tsr_cyclic_data[0] = tsr_data[0];
    } else {
      tsr_cyclic_data[0] = get_zero<dtype>();
    }
    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    return DIST_TENSOR_SUCCESS;
  }

  if (ndim == 0){
    CTF_alloc_ptr(sizeof(dtype), (void**)&tsr_cyclic_data);
    if (ord_glb_comm->rank == 0){
      tsr_cyclic_data[0] = tsr_data[0];
    } else {
      tsr_cyclic_data[0] = get_zero<dtype>();
    }
    *ptr_tsr_cyclic_data = tsr_cyclic_data;
    return DIST_TENSOR_SUCCESS;
  }

  TAU_FSTART(cyclic_reshuffle);
  if (ord_glb_comm == NULL)
    np = 1;
  else
    np = ord_glb_comm->np;

  CTF_alloc_ptr(ndim*sizeof(int), (void**)&hsym);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&idx);
  CTF_alloc_ptr(ndim*sizeof(long_int), (void**)&idx_offs);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&old_virt_lda);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&new_virt_lda);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&buf_lda);

  nbuf = 1;
  new_nvirt = 1;
  old_nvirt = 1;
  old_np = 1;
  new_np = 1;
  if (ord_glb_comm == NULL || ord_glb_comm->rank == 0){
    DPRINTF(3,"is_cyclic = %d, was_cyclic = %d\n",is_cyclic,was_cyclic);
  }
  if (ord_glb_comm == NULL)
    idx_lyr = 0;
  else
    idx_lyr = ord_glb_comm->rank;
  for (i=0; i<ndim; i++) {
    if (ord_glb_comm == NULL || ord_glb_comm->rank == 0){
      /*printf("old_pe_lda[%d] = %d, old_phase = %d, old_virt_dim = %d\n",
              i,old_pe_lda[i], old_phase[i], old_virt_dim[i]);
      printf("new_pe_lda[%d] = %d, new_phase = %d, new_virt_dim = %d\n",
            i,new_pe_lda[i], new_phase[i], new_virt_dim[i]);*/
    }
    buf_lda[i] = nbuf;
    new_virt_lda[i] = new_nvirt;
    old_virt_lda[i] = old_nvirt;
    nbuf = nbuf*new_phase[i];
    /*printf("is_new_pad = %d\n", is_new_pad);
    if (is_new_pad)
      printf("new_padding[%d] = %d\n", i, new_padding[i]);
    printf("is_old_pad = %d\n", is_old_pad);
    if (is_old_pad)
      printf("old_padding[%d] = %d\n", i, old_padding[i]);*/
    old_nvirt = old_nvirt*old_virt_dim[i];
    new_nvirt = new_nvirt*new_virt_dim[i];
    new_np = new_np*new_phase[i]/new_virt_dim[i];
    old_np = old_np*old_phase[i]/old_virt_dim[i];
    idx_lyr -= old_rank[i]*old_pe_lda[i];
  }
  vbs_old = nval/old_nvirt;
  nbuf = np*new_nvirt;

  CTF_mst_alloc_ptr(np*sizeof(int), (void**)&recv_counts);
  CTF_mst_alloc_ptr(np*sizeof(int), (void**)&send_counts);
  CTF_mst_alloc_ptr(nbuf*sizeof(int), (void**)&rvirt_displs);
  CTF_mst_alloc_ptr(nbuf*sizeof(int), (void**)&svirt_displs);
  CTF_mst_alloc_ptr(np*sizeof(int), (void**)&send_displs);
  CTF_mst_alloc_ptr(np*sizeof(int), (void**)&recv_displs);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&old_sub_edge_len);
  CTF_alloc_ptr(ndim*sizeof(int), (void**)&new_sub_edge_len);
  int ** bucket_offset;
  
  int *real_edge_len; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&real_edge_len);
  for (i=0; i<ndim; i++) real_edge_len[i] = old_edge_len[i]-old_padding[i];
  
  int *old_phys_dim; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&old_phys_dim);
  for (i=0; i<ndim; i++) old_phys_dim[i] = old_phase[i]/old_virt_dim[i];

  int *new_phys_dim; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&new_phys_dim);
  for (i=0; i<ndim; i++) new_phys_dim[i] = new_phase[i]/new_virt_dim[i];
  
  int *old_phys_edge_len; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&old_phys_edge_len);
  for (int dim = 0;dim < ndim;dim++) old_phys_edge_len[dim] = (real_edge_len[dim]+old_padding[dim])/old_phys_dim[dim];

  int *new_phys_edge_len; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&new_phys_edge_len);
  for (int dim = 0;dim < ndim;dim++) new_phys_edge_len[dim] = (real_edge_len[dim]+new_padding[dim])/new_phys_dim[dim];

  int *old_virt_edge_len; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&old_virt_edge_len);
  for (int dim = 0;dim < ndim;dim++) old_virt_edge_len[dim] = old_phys_edge_len[dim]/old_virt_dim[dim];

  int *new_virt_edge_len; CTF_alloc_ptr(sizeof(int)*ndim, (void**)&new_virt_edge_len);
  for (int dim = 0;dim < ndim;dim++) new_virt_edge_len[dim] = new_phys_edge_len[dim]/new_virt_dim[dim];
  
  LIBT_ASSERT(is_old_pad);
  LIBT_ASSERT(is_new_pad);


//  if (idx_lyr == 0){
    bucket_offset = compute_bucket_offsets(
                              ndim, real_edge_len, old_rank, old_phys_dim,
                              old_virt_dim, old_virt_lda, old_offsets,
                              old_permutation, new_phys_dim, new_pe_lda,
                              new_virt_dim, new_virt_lda, 1,
                              old_nvirt, new_nvirt, old_phys_edge_len, old_virt_edge_len);



    TAU_FSTART(calc_cnt_displs);
    /* Calculate bucket counts to begin exchange */
    calc_cnt_displs<dtype>( ndim,         nbuf,           new_nvirt,
                            np,           old_edge_len,   new_edge_len,
                            old_virt_edge_len,
                            sym,          old_phase,      old_rank,       
                            new_phase,    old_virt_dim,   new_virt_dim,   
                            new_virt_lda, buf_lda,        new_pe_lda,     
                            is_old_pad,   old_padding,    send_counts,    
                            recv_counts,  send_displs,    recv_displs,    
                            svirt_displs, rvirt_displs,   ord_glb_comm, 
                            idx_lyr,      was_cyclic,     is_cyclic,
                            bucket_offset);
    
    TAU_FSTOP(calc_cnt_displs);
  /*for (i=0; i<np; i++){
    printf("[%d] send_counts[%d] = %d recv_counts[%d] = %d\n", ord_glb_comm->rank, i, send_counts[i], i, recv_counts[i]);
  }
  for (i=0; i<nbuf; i++){
    printf("[%d] svirt_displs[%d] = %d rvirt_displs[%d] = %d\n", ord_glb_comm->rank, i, svirt_displs[i], i, rvirt_displs[i]);
  }*/

//  }
  for (i=0; i<ndim; i++){
    new_sub_edge_len[i] = new_edge_len[i];
    old_sub_edge_len[i] = old_edge_len[i];
  }
  for (i=0; i<ndim; i++){
    new_sub_edge_len[i] = new_sub_edge_len[i] / new_phase[i];
    old_sub_edge_len[i] = old_sub_edge_len[i] / old_phase[i];
  }
  for (i=1; i<ndim; i++){
    hsym[i-1] = sym[i];
  }
  hvbs_old = sy_packed_size(ndim-1, old_sub_edge_len+1, hsym);
  hvbs_new = sy_packed_size(ndim-1, new_sub_edge_len+1, hsym);
  swp_nval = new_nvirt*sy_packed_size(ndim, new_sub_edge_len, sym);
  CTF_mst_alloc_ptr(MAX(nval,swp_nval)*sizeof(dtype), (void**)&tsr_cyclic_data);

  vbs_new = swp_nval/new_nvirt;

  if (ord_glb_comm != NULL){
    DPRINTF(4,"[%d] send = %d, had= "PRId64" recv = %d, should get = "PRId64"\n", 
            ord_glb_comm->rank, send_displs[ord_glb_comm->np-1] + send_counts[ord_glb_comm->np-1], nval,
            recv_displs[ord_glb_comm->np-1] + recv_counts[ord_glb_comm->np-1], swp_nval);
  }
  TAU_FSTART(pack_virt_buf);
  if (idx_lyr == 0){
    if (was_cyclic&&is_cyclic) {
      dtype **new_data; CTF_alloc_ptr(sizeof(dtype*)*np*new_nvirt, (void**)&new_data);
      for (int p = 0,b = 0;p < np;p++)
      {
        for (int v = 0;v < new_nvirt;v++,b++)
            new_data[b] = tsr_cyclic_data+send_displs[p]+svirt_displs[b];
      }

      //long_int ndata = send_displs[np-1] + send_counts[np-1];

      //dtype *test_data; CTF_alloc_ptr(sizeof(dtype)*ndata, (void**)&test_data);
      //memset(test_data, 0, sizeof(dtype)*ndata);

      pad_cyclic_pup_virt_buff(ndim, real_edge_len, sym, 
                               old_rank, old_phys_dim, old_virt_dim,
                               old_phys_edge_len, old_virt_edge_len,
                               vbs_old, old_offsets, old_permutation,
                               new_np, new_rank, new_phys_dim, new_virt_dim,
                               new_phys_edge_len, new_virt_edge_len,
                               vbs_new,  
                               tsr_data, new_data, 1, bucket_offset);

      /*
      opt_pup_virt_buff(ndim,             nbuf,           np,
                        hvbs_old,         old_nvirt,      new_nvirt,
                        old_edge_len,     new_edge_len,
                        sym,              old_phase,
                        old_rank,         new_phase,      new_rank,
                        old_virt_dim,     new_virt_dim,   send_displs,
                        svirt_displs,     old_virt_lda,   new_virt_lda,
                        new_pe_lda,       vbs_old,        is_old_pad,
                        old_padding,      tsr_data,       test_data,
                        was_cyclic,       is_cyclic,      1);

      for (long_int k = 0;k < ndata;k++)
      {
          if (test_data[k] != tsr_cyclic_data[k])
          {
              for (long_int l = 0;l < ndata;l++)
              {
                  printf("%lld: %f %f\n", l, test_data[l], tsr_cyclic_data[l]);
              }
              assert(0);
          }
      }

      CTF_free(test_data);
      */
      CTF_free(new_data);
    } else {
      opt_pup_virt_buff(ndim,             nbuf,           np,
                        hvbs_old,         old_nvirt,      new_nvirt,
                        old_edge_len,     new_edge_len,
                        sym,              old_phase,
                        old_rank,         new_phase,      new_rank,
                        old_virt_dim,     new_virt_dim,   send_displs,
                        svirt_displs,     old_virt_lda,   new_virt_lda,
                        new_pe_lda,       vbs_old,        is_old_pad,
                        old_padding,      tsr_data,       tsr_cyclic_data,
                        was_cyclic,       is_cyclic,      1);
    }
  }
  for (int dim = 0;dim < ndim;dim++){
    CTF_free(bucket_offset[dim]);
  }
  CTF_free(bucket_offset);

  TAU_FSTOP(pack_virt_buf);

  if (swp_nval > nval){
    CTF_free(tsr_data);
    CTF_mst_alloc_ptr(swp_nval*sizeof(dtype), (void**)&tsr_data);
  }

  /* Communicate data */
  if (ord_glb_comm != NULL){
    TAU_FSTART(ALL_TO_ALL_V);
    /* FIXME: not general to all types */
    if (sizeof(dtype) == 4)
      ALL_TO_ALLV(tsr_cyclic_data, send_counts, send_displs, MPI_FLOAT,
                  tsr_data, recv_counts, recv_displs, MPI_FLOAT, ord_glb_comm);
    if (sizeof(dtype) == 8)
      ALL_TO_ALLV(tsr_cyclic_data, send_counts, send_displs, COMM_DOUBLE_T,
                  tsr_data, recv_counts, recv_displs, COMM_DOUBLE_T, ord_glb_comm);
    if (sizeof(dtype) == 16)
      ALL_TO_ALLV(tsr_cyclic_data, send_counts, send_displs, MPI::DOUBLE_COMPLEX,
                  tsr_data, recv_counts, recv_displs, MPI::DOUBLE_COMPLEX, ord_glb_comm);
    TAU_FSTOP(ALL_TO_ALL_V);
  } else {
//    memcpy(tsr_data, tsr_cyclic_data, swp_nval*sizeof(dtype));
    swp_ptr = tsr_cyclic_data;
    tsr_cyclic_data = tsr_data;
    tsr_data = swp_ptr;
  }

  std::fill(tsr_cyclic_data, tsr_cyclic_data+swp_nval, get_zero<dtype>());
  TAU_FSTART(unpack_virt_buf);
  /* Deserialize data into correctly ordered virtual sub blocks */
  if (ord_glb_comm == NULL ||
      recv_displs[ord_glb_comm->np-1] + recv_counts[ord_glb_comm->np-1] > 0){

    if (was_cyclic&&is_cyclic)
    {
      dtype **new_data; CTF_alloc_ptr(sizeof(dtype*)*np*new_nvirt, (void**)&new_data);
      for (int p = 0,b = 0;p < np;p++)
      {
        for (int v = 0;v < new_nvirt;v++,b++)
            new_data[b] = tsr_data+recv_displs[p]+rvirt_displs[b];
      }
      bucket_offset = compute_bucket_offsets(
                              ndim, real_edge_len, new_rank, new_phys_dim,
                              new_virt_dim, new_virt_lda, new_offsets,
                              new_permutation, old_phys_dim, old_pe_lda,
                              old_virt_dim, old_virt_lda, 0,
                              new_nvirt, old_nvirt, new_phys_edge_len, new_virt_edge_len);


      /*
      dtype *test_data; CTF_alloc_ptr(sizeof(dtype)*swp_nval, (void**)&test_data);
      memset(test_data, 0, sizeof(dtype)*swp_nval);

      opt_pup_virt_buff(ndim,             nbuf,           np,
                        hvbs_new,         old_nvirt,      new_nvirt,
                        new_edge_len,     old_edge_len,
                        sym,              new_phase,
                        new_rank,         old_phase,      old_rank,
                        new_virt_dim,     old_virt_dim,   recv_displs,
                        rvirt_displs,     new_virt_lda,   old_virt_lda,
                        old_pe_lda,       vbs_new,        is_new_pad,
                        new_padding,      tsr_data,       test_data,
                        is_cyclic,        was_cyclic,     0);
                        */

      pad_cyclic_pup_virt_buff(ndim, real_edge_len, sym, 
                               new_rank, new_phys_dim, new_virt_dim,
                               new_phys_edge_len, new_virt_edge_len,
                               vbs_new, new_offsets, new_permutation,
                               old_np, old_rank, old_phys_dim,  old_virt_dim,
                               old_phys_edge_len, old_virt_edge_len,
                               vbs_old,  
                               tsr_cyclic_data, new_data, 0, bucket_offset);
      for (int dim = 0;dim < ndim;dim++){
        CTF_free(bucket_offset[dim]);
      }
      CTF_free(bucket_offset);


      /*
      for (long_int k = 0;k < swp_nval;k++)
      {
          if (test_data[k] != tsr_cyclic_data[k])
          {
              for (long_int l = 0;l < swp_nval;l++)
              {
                  printf("%lld: %f %f\n", l, test_data[l], tsr_cyclic_data[l]);
              }
              assert(0);
          }
      }

      CTF_free(test_data);
      */
      CTF_free(new_data);
    }
    else
    {
      opt_pup_virt_buff(ndim,             nbuf,           np,
                        hvbs_new,         old_nvirt,      new_nvirt,
                        new_edge_len,     old_edge_len,
                        sym,              new_phase,
                        new_rank,         old_phase,      old_rank,
                        new_virt_dim,     old_virt_dim,   recv_displs,
                        rvirt_displs,     new_virt_lda,   old_virt_lda,
                        old_pe_lda,       vbs_new,        is_new_pad,
                        new_padding,      tsr_data,       tsr_cyclic_data,
                        is_cyclic,        was_cyclic,     0);
    }
  }
  TAU_FSTOP(unpack_virt_buf);

  *ptr_tsr_cyclic_data = tsr_cyclic_data;
  *ptr_tsr_data = tsr_data;

  CTF_free(real_edge_len);
  CTF_free(old_phys_dim);
  CTF_free(new_phys_dim);
  CTF_free(hsym);
  CTF_free(idx);
  CTF_free(idx_offs);
  CTF_free(old_virt_lda);
  CTF_free(new_virt_lda);
  CTF_free(buf_lda);
  CTF_free(recv_counts);
  CTF_free(send_counts);
  CTF_free(rvirt_displs);
  CTF_free(svirt_displs);
  CTF_free(send_displs);
  CTF_free(recv_displs);
  CTF_free(old_sub_edge_len);
  CTF_free(new_sub_edge_len);
  CTF_free(new_virt_edge_len);
  CTF_free(old_virt_edge_len);
  CTF_free(new_phys_edge_len);
  CTF_free(old_phys_edge_len);

  TAU_FSTOP(cyclic_reshuffle);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief Reshuffle elements given the global phases stay the same
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] phase physical*virtual phase
 * \param[in] old_size number of elements in the subtensor each proc owned
 * \param[in] old_virt_dim current virtualization dimensions on each process
 * \param[in] old_rank current physical rank
 * \param[in] old_pe_lda old lda of processor grid
 * \param[in] new_size number of elements in the subtensor each proc owns
 * \param[in] new_virt_dim new virtualization dimensions on each process
 * \param[in] new_rank new physical rank
 * \param[in] new_pe_lda new lda of processor grid
 * \param[in] tsr_data current tensor data
 * \param[out] ptr_tsr_cyclic_data pointer to a new tensor of 
              data that will be filled
 * \param[in] glb_comm the global communicator
 */
template<typename dtype>
void block_reshuffle(int const        ndim,
                     int const *      phase,
                     long_int const   old_size,
                     int const *      old_virt_dim,
                     int const *      old_rank,
                     int const *      old_pe_lda,
                     long_int const   new_size,
                     int const *      new_virt_dim,
                     int const *      new_rank,
                     int const *      new_pe_lda,
                     dtype *          tsr_data,
                     dtype *&         tsr_cyclic_data,
                     CommData_t *     glb_comm){
  int i, idx_lyr_new, idx_lyr_old, blk_idx, prc_idx, loc_idx;
  int num_old_virt, num_new_virt;
  int * idx, * old_loc_lda, * new_loc_lda, * phase_lda;
  long_int blk_sz;
  MPI_Request * reqs;

  if (ndim == 0){
    CTF_alloc_ptr(sizeof(dtype)*new_size, (void**)&tsr_cyclic_data);
    if (glb_comm->rank == 0)
      tsr_cyclic_data[0] = tsr_data[0];
    else
      tsr_cyclic_data[0] = 0.0;
    return;
  }

  TAU_FSTART(block_reshuffle);

  CTF_mst_alloc_ptr(sizeof(dtype)*new_size, (void**)&tsr_cyclic_data);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&idx);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&old_loc_lda);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&new_loc_lda);
  CTF_alloc_ptr(sizeof(int)*ndim, (void**)&phase_lda);

  blk_sz = old_size;
  old_loc_lda[0] = 1;
  new_loc_lda[0] = 1;
  phase_lda[0] = 1;
  num_old_virt = 1;
  num_new_virt = 1;
  idx_lyr_old = glb_comm->rank;
  idx_lyr_new = glb_comm->rank;

  for (i=0; i<ndim; i++){
    num_old_virt *= old_virt_dim[i];
    num_new_virt *= new_virt_dim[i];
    blk_sz = blk_sz/old_virt_dim[i];
    idx_lyr_old -= old_rank[i]*old_pe_lda[i];
    idx_lyr_new -= new_rank[i]*new_pe_lda[i];
    if (i>0){
      old_loc_lda[i] = old_loc_lda[i-1]*old_virt_dim[i-1];
      new_loc_lda[i] = new_loc_lda[i-1]*new_virt_dim[i-1];
      phase_lda[i] = phase_lda[i-1]*phase[i-1];
    }
  }
  
  CTF_alloc_ptr(sizeof(MPI_Request)*(num_old_virt+num_new_virt), (void**)&reqs);

  if (idx_lyr_new == 0){
    memset(idx, 0, sizeof(int)*ndim);

    for (;;){
      loc_idx = 0;
      blk_idx = 0;
      prc_idx = 0;
      for (i=0; i<ndim; i++){
        loc_idx += idx[i]*new_loc_lda[i];
        blk_idx += ( idx[i] + new_rank[i]*new_virt_dim[i])                 *phase_lda[i];
        prc_idx += ((idx[i] + new_rank[i]*new_virt_dim[i])/old_virt_dim[i])*old_pe_lda[i];
      }
      DPRINTF(3,"proc %d receiving blk %d (loc %d, size "PRId64") from proc %d\n", 
              glb_comm->rank, blk_idx, loc_idx, blk_sz, prc_idx);
      MPI_Irecv(tsr_cyclic_data+loc_idx*blk_sz, blk_sz*sizeof(dtype), 
                MPI_CHAR, prc_idx, blk_idx, glb_comm->cm, reqs+loc_idx);
      for (i=0; i<ndim; i++){
        idx[i]++;
        if (idx[i] >= new_virt_dim[i])
          idx[i] = 0;
        else 
          break;
      }
      if (i==ndim) break;
    }
  }

  if (idx_lyr_old == 0){
    memset(idx, 0, sizeof(int)*ndim);

    for (;;){
      loc_idx = 0;
      blk_idx = 0;
      prc_idx = 0;
      for (i=0; i<ndim; i++){
        loc_idx += idx[i]*old_loc_lda[i];
        blk_idx += ( idx[i] + old_rank[i]*old_virt_dim[i])                 *phase_lda[i];
        prc_idx += ((idx[i] + old_rank[i]*old_virt_dim[i])/new_virt_dim[i])*new_pe_lda[i];
      }
      DPRINTF(3,"proc %d sending blk %d (loc %d) to proc %d\n", 
              glb_comm->rank, blk_idx, loc_idx, prc_idx);
      MPI_Isend(tsr_data+loc_idx*blk_sz, blk_sz*sizeof(dtype), 
                MPI_CHAR, prc_idx, blk_idx, glb_comm->cm, reqs+num_new_virt+loc_idx);
      for (i=0; i<ndim; i++){
        idx[i]++;
        if (idx[i] >= old_virt_dim[i])
          idx[i] = 0;
        else 
          break;
      }
      if (i==ndim) break;
    }
  }

  if (idx_lyr_new == 0 && idx_lyr_old == 0){
    MPI_Waitall(num_new_virt+num_old_virt, reqs, MPI_STATUSES_IGNORE);
  } else if (idx_lyr_new == 0){
    MPI_Waitall(num_new_virt, reqs, MPI_STATUSES_IGNORE);
  } else if (idx_lyr_old == 0){
    MPI_Waitall(num_old_virt, reqs+num_new_virt, MPI_STATUSES_IGNORE);
    std::fill(tsr_cyclic_data, tsr_cyclic_data+new_size, get_zero<dtype>());
  } else {
    std::fill(tsr_cyclic_data, tsr_cyclic_data+new_size, get_zero<dtype>());
  }

  CTF_free(idx);
  CTF_free(old_loc_lda);
  CTF_free(new_loc_lda);
  CTF_free(phase_lda);
  CTF_free(reqs);

  TAU_FSTOP(block_reshuffle);
}
#endif
