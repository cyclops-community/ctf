#include "summation.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_sum.h"
#include "sum_tsr.h"

namespace CTF_int {

  using namespace CTF;

  summation::summation(tensor * A_, 
                int const * idx_A_,
                char const * alpha_, 
                tensor * B_, 
                int const * idx_B_,
                char const * beta_){
    A = A_;
    idx_A = idx_A_;
    alpha = alpha_;
    B = B_;
    idx_B = idx_B_;
    beta = beta_;
    is_custom = 0;
  }
 
  summation::summation(tensor * A_, 
                int const * idx_A_,
                tensor * B_, 
                int const * idx_B_,
                univar_function func_){
    A = A_;
    idx_A = idx_A_;
    B = B_;
    idx_B = idx_B_;
    func = func_;
    is_custom = 1;
  }

  void summation::get_fold_indices(int *        num_fold,
                            int **         fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, inA, inB, iiA, iiB;
    int * idx_arr, * idx;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);
    CTF_alloc_ptr(num_tot*sizeof(int), (void**)&idx);

    for (i=0; i<num_tot; i++){
      idx[i] = 1;
    }
    
    for (iA=0; iA<A->order; iA++){
      i = idx_A[iA];
      iB = idx_arr[2*i+1];
      broken = 0;
      inA = iA;
      do {
        in = idx_A[inA];
        inB = idx_arr[2*in+1];
        if (((inA>=0) + (inB>=0) != 2) ||
            (iB != -1 && inB - iB != in-i) ||
            (iB != -1 && A->sym[inA] != B->sym[inB])){
          broken = 1;
        }
        inA++;
      } while (A->sym[inA-1] != NS);
      if (broken){
        for (iiA=iA;iiA<inA;iiA++){
          idx[idx_A[iiA]] = 0;
        }
      }
    }

    for (iB=0; iB<B->order; iB++){
      i = idx_B[iB];
      iA = idx_arr[2*i+0];
      broken = 0;
      inB = iB;
      do {
        in = idx_B[inB];
        inA = idx_arr[2*in+0];
        if (((inB>=0) + (inA>=0) != 2) ||
            (iA != -1 && inA - iA != in-i) ||
            (iA != -1 && B->sym[inB] != A->sym[inA])){
          broken = 1;
        }
        inB++;
      } while (B->sym[inB-1] != NS);
      if (broken){
        for (iiB=iB;iiB<inB;iiB++){
          idx[idx_B[iiB]] = 0;
        }
      }
    }
    

    nfold = 0;
    for (i=0; i<num_tot; i++){
      if (idx[i] == 1){
        idx[nfold] = i;
        nfold++;
      }
    }
    *num_fold = nfold;
    *fold_idx = idx;
    CTF_free(idx_arr);
  }

  int summation::can_fold(){
    int i, j, nfold, * fold_idx;
    for (i=0; i<A->order; i++){
      for (j=i+1; j<A->order; j++){
        if (idx_A[i] == idx_A[j]) return 0;
      }
    }
    for (i=0; i<B->order; i++){
      for (j=i+1; j<B->order; j++){
        if (idx_B[i] == idx_B[j]) return 0;
      }
    }
    get_fold_indices(&nfold, &fold_idx);
    CTF_free(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }

  void summation::get_len_ordering(
                                        int **      new_ordering_A,
                                        int **      new_ordering_B){
    int i, num_tot;
    int * ordering_A, * ordering_B, * idx_arr;
    
    CTF_alloc_ptr(sizeof(int)*A->order, (void**)&ordering_A);
    CTF_alloc_ptr(sizeof(int)*B->order, (void**)&ordering_B);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);
    for (i=0; i<num_tot; i++){
      ordering_A[i] = idx_arr[2*i];
      ordering_B[i] = idx_arr[2*i+1];
    }
    CTF_free(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
  }

  tsum * summation::construct_sum(int inner_stride){
    int nvirt, i, iA, iB, order_tot, is_top, sA, sB, need_rep, i_A, i_B, j, k;
    int64_t blk_sz_A, blk_sz_B, vrt_sz_A, vrt_sz_B;
    int nphys_dim;
    int * idx_arr, * virt_dim, * phys_mapped;
    int * virt_blk_len_A, * virt_blk_len_B;
    int * blk_len_A, * blk_len_B;
    tensor<dtype> * tsr_A, * tsr_B;
    tsum<dtype> * htsum = NULL , ** rec_tsum = NULL;
    mapping * map;
    strp_tsr<dtype> * str_A, * str_B;

    is_top = 1;

    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];

    inv_idx(tsr_A->order, idx_A, tsr_A->edge_map,
            tsr_B->order, idx_B, tsr_B->edge_map,
            &order_tot, &idx_arr);

    nphys_dim = topovec[tsr_A->itopo].order;

    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&blk_len_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&blk_len_B);
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&virt_blk_len_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&virt_blk_len_B);
    CTF_alloc_ptr(sizeof(int)*order_tot, (void**)&virt_dim);
    CTF_alloc_ptr(sizeof(int)*nphys_dim*2, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*2);


    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = tsr_A->size;
    blk_sz_B = tsr_B->size;
    calc_dim(tsr_A->order, blk_sz_A, tsr_A->edge_len, tsr_A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(tsr_B->order, blk_sz_B, tsr_B->edge_len, tsr_B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);

    /* Strip out the relevant part of the tensor if we are contracting over diagonal */
    sA = strip_diag<dtype>(tsr_A->order, order_tot, idx_A, vrt_sz_A,
                           tsr_A->edge_map, &topovec[tsr_A->itopo],
                           blk_len_A, &blk_sz_A, &str_A);
    sB = strip_diag<dtype>(tsr_B->order, order_tot, idx_B, vrt_sz_B,
                           tsr_B->edge_map, &topovec[tsr_B->itopo],
                           blk_len_B, &blk_sz_B, &str_B);
    if (sA || sB){
      if (global_comm.rank == 0)
        DPRINTF(1,"Stripping tensor\n");
      strp_sum<dtype> * ssum = new strp_sum<dtype>;
      htsum = ssum;
      is_top = 0;
      rec_tsum = &ssum->rec_tsum;

      ssum->rec_strp_A = str_A;
      ssum->rec_strp_B = str_B;
      ssum->strip_A = sA;
      ssum->strip_B = sB;
    }

    nvirt = 1;
    for (i=0; i<order_tot; i++){
      iA = idx_arr[2*i];
      iB = idx_arr[2*i+1];
      if (iA != -1){
        map = &tsr_A->edge_map[iA];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
          if (sA) virt_dim[i] = virt_dim[i]/str_A->strip_dim[iA];
        }
        else virt_dim[i] = 1;
      } else {
        ASSERT(iB!=-1);
        map = &tsr_B->edge_map[iB];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
          if (sB) virt_dim[i] = virt_dim[i]/str_B->strip_dim[iA];
        }
        else virt_dim[i] = 1;
      }
      nvirt *= virt_dim[i];
    }

    for (i=0; i<tsr_A->order; i++){
      map = &tsr_A->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[2*map->cdt+0] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[2*map->cdt+0] = 1;
        }
      }
    }
    for (i=0; i<tsr_B->order; i++){
      map = &tsr_B->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[2*map->cdt+1] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[2*map->cdt+1] = 1;
        }
      }
    }
    need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 ||
          phys_mapped[2*i+1] == 0){
        need_rep = 1;
        break;
      }
    }
    if (need_rep){
      if (global_comm.rank == 0)
        DPRINTF(1,"Replicating tensor\n");

      tsum_replicate<dtype> * rtsum = new tsum_replicate<dtype>;
      if (is_top){
        htsum = rtsum;
        is_top = 0;
      } else {
        *rec_tsum = rtsum;
      }
      rec_tsum = &rtsum->rec_tsum;
      rtsum->ncdt_A = 0;
      rtsum->ncdt_B = 0;
      rtsum->size_A = blk_sz_A;
      rtsum->size_B = blk_sz_B;
      rtsum->cdt_A = NULL;
      rtsum->cdt_B = NULL;
      for (i=0; i<nphys_dim; i++){
        if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
          rtsum->ncdt_A++;
        }
        if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
          rtsum->ncdt_B++;
        }
      }
      if (rtsum->ncdt_A > 0)
        CTF_alloc_ptr(sizeof(CommData)*rtsum->ncdt_A, (void**)&rtsum->cdt_A);
      if (rtsum->ncdt_B > 0)
        CTF_alloc_ptr(sizeof(CommData)*rtsum->ncdt_B, (void**)&rtsum->cdt_B);
      rtsum->ncdt_A = 0;
      rtsum->ncdt_B = 0;
      for (i=0; i<nphys_dim; i++){
        if (phys_mapped[2*i+0] == 0 && phys_mapped[2*i+1] == 1){
          rtsum->cdt_A[rtsum->ncdt_A] = topovec[tsr_A->itopo].dim_comm[i];
          if (rtsum->cdt_A[rtsum->ncdt_A].alive == 0)
            SHELL_SPLIT(global_comm, rtsum->cdt_A[rtsum->ncdt_A]);
          rtsum->ncdt_A++;
        }
        if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
          rtsum->cdt_B[rtsum->ncdt_B] = topovec[tsr_B->itopo].dim_comm[i];
          if (rtsum->cdt_B[rtsum->ncdt_B].alive == 0)
            SHELL_SPLIT(global_comm, rtsum->cdt_B[rtsum->ncdt_B]);
          rtsum->ncdt_B++;
        }
      }
      ASSERT(rtsum->ncdt_A == 0 || rtsum->cdt_B == 0);
    }

    int * new_sym_A, * new_sym_B;
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, tsr_A->sym, sizeof(int)*tsr_A->order);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, tsr_B->sym, sizeof(int)*tsr_B->order);

    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
      tsum_virt<dtype> * tsumv = new tsum_virt<dtype>;
      if (is_top) {
        htsum = tsumv;
        is_top = 0;
      } else {
        *rec_tsum = tsumv;
      }
      rec_tsum = &tsumv->rec_tsum;

      tsumv->num_dim  = order_tot;
      tsumv->virt_dim   = virt_dim;
      tsumv->order_A = tsr_A->order;
      tsumv->blk_sz_A = vrt_sz_A;
      tsumv->idx_map_A  = idx_A;
      tsumv->order_B = tsr_B->order;
      tsumv->blk_sz_B = vrt_sz_B;
      tsumv->idx_map_B  = idx_B;
      tsumv->buffer = NULL;
    } else CTF_free(virt_dim);

    seq_tsr_sum<dtype> * tsumseq = new seq_tsr_sum<dtype>;
    if (inner_stride == -1){
      tsumseq->is_inner = 0;
    } else {
      tsumseq->is_inner = 1;
      tsumseq->inr_stride = inner_stride;
      tensor<dtype> * itsr;
      itsr = tensors[tsr_A->rec_tid];
      i_A = 0;
      for (i=0; i<tsr_A->order; i++){
        if (tsr_A->sym[i] == NS){
          for (j=0; j<itsr->order; j++){
            if (tsr_A->inner_ordering[j] == i_A){
              j=i;
              do {
                j--;
              } while (j>=0 && tsr_A->sym[j] != NS);
              for (k=j+1; k<=i; k++){
                virt_blk_len_A[k] = 1;
                new_sym_A[k] = NS;
              }
              break;
            }
          }
          i_A++;
        }
      }
      itsr = tensors[tsr_B->rec_tid];
      i_B = 0;
      for (i=0; i<tsr_B->order; i++){
        if (tsr_B->sym[i] == NS){
          for (j=0; j<itsr->order; j++){
            if (tsr_B->inner_ordering[j] == i_B){
              j=i;
              do {
                j--;
              } while (j>=0 && tsr_B->sym[j] != NS);
              for (k=j+1; k<=i; k++){
                virt_blk_len_B[k] = 1;
                new_sym_B[k] = NS;
              }
              break;
            }
          }
          i_B++;
        }
      }
    }
    if (is_top) {
      htsum = tsumseq;
      is_top = 0;
    } else {
      *rec_tsum = tsumseq;
    }
    tsumseq->order_A         = tsr_A->order;
    tsumseq->idx_map_A      = idx_A;
    tsumseq->edge_len_A     = virt_blk_len_A;
    tsumseq->sym_A          = new_sym_A;
    tsumseq->order_B         = tsr_B->order;
    tsumseq->idx_map_B      = idx_B;
    tsumseq->edge_len_B     = virt_blk_len_B;
    tsumseq->sym_B          = new_sym_B;
    tsumseq->func_ptr       = ftsr;
    tsumseq->custom_params  = felm;
    tsumseq->is_custom      = (felm.func_ptr != NULL);

    htsum->A      = tsr_A->data;
    htsum->B      = tsr_B->data;
    htsum->alpha  = alpha;
    htsum->beta   = beta;

    CTF_free(idx_arr);
    CTF_free(blk_len_A);
    CTF_free(blk_len_B);
    CTF_free(phys_mapped);

    return htsum;
  }

  int summation::home_sum_tsr(bool run_diag){
    int ret, was_home_A, was_home_B;
    tensor<dtype> * tsr_A, * tsr_B, * ntsr_A, * ntsr_B;
    int was_cyclic_B;
    int64_t old_size_B;
    int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B;
    int * old_padding_B, * old_edge_len_B;
    CTF_sum_type_t type;
    type.tid_A = tid_A;
    type.tid_B = tid_B;
    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    
    contract_mst();

    CTF_alloc_ptr(sizeof(int)*tsr_A->order, (void**)&type.idx_map_A);
    CTF_alloc_ptr(sizeof(int)*tsr_B->order, (void**)&type.idx_map_B);

    memcpy(type.idx_map_A, idx_map_A, sizeof(int)*tsr_A->order);
    memcpy(type.idx_map_B, idx_map_B, sizeof(int)*tsr_B->order);
  #ifndef HOME_CONTRACT
    #ifdef USE_SYM_SUM
      ret = sym_sum_tsr(alpha_, beta, &type, ftsr, felm, run_diag);
      free_type(&type);
      return ret;
    #else
      ret = sum_tensors(alpha_, beta, tid_A, tid_B, idx_map_A, idx_map_B, ftsr, felm, run_diag);
      free_type(&type);
      return ret;
    #endif
  #else
    int new_tid;
    if (tsr_A->has_zero_edge_len || 
        tsr_B->has_zero_edge_len){
      if (beta != 1.0 && !tsr_B->has_zero_edge_len){ 
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        int sub_idx_map_B[tsr_B->order];
        int sm_idx=0;
        for (int i=0; i<tsr_B->order; i++){
          sub_idx_map_B[i]=sm_idx;
          sm_idx++;
          for (int j=0; j<i; j++){
            if (idx_map_B[i]==idx_map_B[j]){
              sub_idx_map_B[i]=sub_idx_map_B[j];
              sm_idx--;
              break;
            }
          }
        }
        scale_tsr(beta, tid_B, sub_idx_map_B, fs, felm); 
      }
      free_type(&type);
      return CTF_SUCCESS;
    }
    if (tid_A == tid_B){
      clone_tensor(tid_A, 1, &new_tid);
      ret = home_sum_tsr(alpha_, beta, new_tid, tid_B, 
                          idx_map_A, idx_map_B, ftsr, felm, run_diag);
      del_tsr(new_tid);
      free_type(&type);
      return ret;
    }
    was_home_A = tsr_A->is_home;
    was_home_B = tsr_B->is_home;
    if (was_home_A){
      clone_tensor(tid_A, 0, &type.tid_A, 0);
      ntsr_A = tensors[type.tid_A];
      ntsr_A->data = tsr_A->data;
      ntsr_A->home_buffer = tsr_A->home_buffer;
      ntsr_A->is_home = 1;
      ntsr_A->is_mapped = 1;
      ntsr_A->itopo = tsr_A->itopo;
      copy_mapping(tsr_A->order, tsr_A->edge_map, ntsr_A->edge_map);
      set_padding(ntsr_A);
    }     
    if (was_home_B){
      clone_tensor(tid_B, 0, &type.tid_B, 0);
      ntsr_B = tensors[type.tid_B];
      ntsr_B->data = tsr_B->data;
      ntsr_B->home_buffer = tsr_B->home_buffer;
      ntsr_B->is_home = 1;
      ntsr_B->is_mapped = 1;
      ntsr_B->itopo = tsr_B->itopo;
      copy_mapping(tsr_B->order, tsr_B->edge_map, ntsr_B->edge_map);
      set_padding(ntsr_B);
    }
  #if DEBUG >= 1
    if (get_global_comm().rank == 0)
      printf("Start head sum:\n");
  #endif
    
    #ifdef USE_SYM_SUM
    ret = sym_sum_tsr(alpha_, beta, &type, ftsr, felm, run_diag);
    #else
    ret = sum_tensors(alpha_, beta, type.tid_A, type.tid_B, idx_map_A, idx_map_B, ftsr, felm, run_diag);
    #endif
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("End head sum:\n");
  #endif

    if (ret!= CTF_SUCCESS) return ret;
    if (was_home_A) unmap_inner(ntsr_A);
    if (was_home_B) unmap_inner(ntsr_B);

    if (was_home_B && !ntsr_B->is_home){
      if (global_comm.rank == 0)
        DPRINTF(2,"Migrating tensor %d back to home\n", tid_B);
      save_mapping(ntsr_B,
                   &old_phase_B, &old_rank_B, 
                   &old_virt_dim_B, &old_pe_lda_B, 
                   &old_size_B, 
                   &was_cyclic_B, &old_padding_B, 
                   &old_edge_len_B, &topovec[ntsr_B->itopo]);
      tsr_B->data = ntsr_B->data;
      tsr_B->is_home = 0;
      TAU_FSTART(redistribute_for_sum_home);
      remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size_B, 
                   old_phase_B, old_rank_B, old_virt_dim_B, 
                   old_pe_lda_B, was_cyclic_B, 
                   old_padding_B, old_edge_len_B, global_comm);
      TAU_FSTOP(redistribute_for_sum_home);
      memcpy(tsr_B->home_buffer, tsr_B->data, tsr_B->size*sizeof(dtype));
      CTF_free(tsr_B->data);
      tsr_B->data = tsr_B->home_buffer;
      tsr_B->is_home = 1;
      ntsr_B->is_data_aliased = 1;
      del_tsr(type.tid_B);
      CTF_free(old_phase_B);
      CTF_free(old_rank_B);
      CTF_free(old_virt_dim_B);
      CTF_free(old_pe_lda_B);
      CTF_free(old_padding_B);
      CTF_free(old_edge_len_B);
    } else if (was_home_B){
      if (ntsr_B->data != tsr_B->data){
        printf("Tensor %d is a copy of %d and did not leave home but buffer is %p was %p\n", type.tid_B, tid_B, ntsr_B->data, tsr_B->data);
        ABORT;

      }
      ntsr_B->has_home = 0;
      ntsr_B->is_data_aliased = 1;
      del_tsr(type.tid_B);
    }
    if (was_home_A && !ntsr_A->is_home){
      ntsr_A->has_home = 0;
      del_tsr(type.tid_A);
    } else if (was_home_A) {
      ntsr_A->has_home = 0;
      ntsr_A->is_data_aliased = 1;
      del_tsr(type.tid_A);
    }
    free_type(&type);
    return ret;
  #endif
  }

  int summation::sym_sum_tsr(bool run_diag){
    int stat, sidx, i, new_tid, * new_idx_map;
    int * map_A, * map_B, * dstack_tid_B;
    int ** dstack_map_B;
    int ntid_A, ntid_B, nst_B;
    std::vector<CTF_sum_type_t> perm_types;
    std::vector<dtype> signs;
    dtype dbeta;
    CTF_sum_type_t unfold_type, new_type;
  //#if (DEBUG >= 1 || VERBOSE >= 1)
  //  print_sum(type,alpha_,beta);
  //#endif
    check_sum(type);
    if (tensors[type->tid_A]->has_zero_edge_len || 
        tensors[type->tid_B]->has_zero_edge_len){
      tensor<dtype> * tsr_B = tensors[type->tid_B];
      if (beta != 1.0 && !tsr_B->has_zero_edge_len){ 
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        int sub_idx_map_B[tsr_B->order];
        int sm_idx=0;
        for (int i=0; i<tsr_B->order; i++){
          sub_idx_map_B[i]=sm_idx;
          sm_idx++;
          for (int j=0; j<i; j++){
            if (type->idx_map_B[i]==type->idx_map_B[j]){
              sub_idx_map_B[i]=sub_idx_map_B[j];
              sm_idx--;
              break;
            }
          }
        }
        scale_tsr(beta, type->tid_B, sub_idx_map_B, fs, felm); 
      }
      return CTF_SUCCESS;
    }
    ntid_A = type->tid_A;
    ntid_B = type->tid_B;
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_A]->order,   (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->order,   (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*tensors[ntid_B]->order,   (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(int)*tensors[ntid_B]->order,   (void**)&dstack_tid_B);
    memcpy(map_A, type->idx_map_A, tensors[ntid_A]->order*sizeof(int));
    memcpy(map_B, type->idx_map_B, tensors[ntid_B]->order*sizeof(int));
    while (!run_diag && extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      if (ntid_A != type->tid_A) del_tsr(ntid_A);
      CTF_free(map_A);
      ntid_A = new_tid;
      new_type.tid_A = new_tid;
      map_A = new_idx_map;
    }
    nst_B = 0;
    while (!run_diag && extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      dstack_map_B[nst_B] = map_B;
      dstack_tid_B[nst_B] = ntid_B;
      nst_B++;
      ntid_B = new_tid;
      new_type.tid_B = new_tid;
      map_B = new_idx_map;
    }

    if (ntid_A == ntid_B){
      clone_tensor(ntid_A, 1, &new_tid);
      new_type = *type;
      new_type.tid_A = new_tid;
      stat = sym_sum_tsr(alpha_, beta, &new_type, ftsr, felm, run_diag);
      del_tsr(new_tid);
      return stat;
    }
    new_type.tid_A = ntid_A;
    new_type.tid_B = ntid_B;
    new_type.idx_map_A = map_A;
    new_type.idx_map_B = map_B;

    dtype alpha = alpha_*align_symmetric_indices(tensors[ntid_A]->order,
                                                 map_A,
                                                 tensors[ntid_A]->sym,
                                                 tensors[ntid_B]->order,
                                                 map_B,
                                                 tensors[ntid_B]->sym);
    double ocfact = overcounting_factor(tensors[ntid_A]->order,
                                         map_A,
                                         tensors[ntid_A]->sym,
                                         tensors[ntid_B]->order,
                                         map_B,
                                         tensors[ntid_B]->sym);

    alpha *= ocfact;

    if (unfold_broken_sym(&new_type, NULL) != -1){
      if (global_comm.rank == 0)
        DPRINTF(1,"Contraction index is broken\n");

      sidx = unfold_broken_sym(&new_type, &unfold_type);
      int * sym, dim, sy;
      sy = 0;
      sym = get_sym(ntid_A);
      dim = get_dim(ntid_A);
      for (i=0; i<dim; i++){
        if (sym[i] == SY) sy = 1;
      }
      CTF_free(sym);
      sym = get_sym(ntid_B);
      dim = get_dim(ntid_B);
      for (i=0; i<dim; i++){
        if (sym[i] == SY) sy = 1;
      }
      CTF_free(sym);
      if (sy && sidx%2 == 0){/* && map_tensors(&unfold_type,
                            ftsr, felm, alpha, beta, &ctrf, 0) == CTF_SUCCESS){*/
        if (global_comm.rank == 0)
          DPRINTF(1,"Performing index desymmetrization\n");
        desymmetrize(ntid_A, unfold_type.tid_A, 0);
        unfold_type.tid_B = ntid_B;
        sym_sum_tsr(alpha, beta, &unfold_type, ftsr, felm, run_diag);
        if (ntid_A != unfold_type.tid_A){
          unmap_inner(tensors[unfold_type.tid_A]);
          dealias(ntid_A, unfold_type.tid_A);
          del_tsr(unfold_type.tid_A);
        }
      } else {
        get_sym_perms(&new_type, alpha, perm_types, signs);
        if (global_comm.rank == 0)
          DPRINTF(1,"Performing %d summation permutations\n", 
                  (int)perm_types.size());
        dbeta = beta;
        for (i=0; i<(int)perm_types.size(); i++){
          sum_tensors(signs[i], dbeta, perm_types[i].tid_A, perm_types[i].tid_B,
                      perm_types[i].idx_map_A, perm_types[i].idx_map_B, ftsr, felm, run_diag);
          dbeta = 1.0;
        }
        for (i=0; i<(int)perm_types.size(); i++){
          free_type(&perm_types[i]);
        }
        perm_types.clear();
        signs.clear();
      }
      CTF_free(unfold_type.idx_map_A);
      CTF_free(unfold_type.idx_map_B);
    } else {
      sum_tensors(alpha, beta, new_type.tid_A, new_type.tid_B, new_type.idx_map_A, 
                  new_type.idx_map_B, ftsr, felm, run_diag);
    }
    if (ntid_A != type->tid_A) del_tsr(ntid_A);
    for (i=nst_B-1; i>=0; i--){
      extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
      del_tsr(ntid_B);
      ntid_B = dstack_tid_B[i];
    }
    ASSERT(ntid_B == type->tid_B);
    CTF_free(map_A);
    CTF_free(map_B);
    CTF_free(dstack_map_B);
    CTF_free(dstack_tid_B);

    return CTF_SUCCESS;
  }


  /**
   * \brief PDAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
   * \param[in] run_diag if 1 run diagonal sum
   */
  int summation::sum_tensors(bool run_diag){
    int stat, new_tid, * new_idx_map;
    int * map_A, * map_B, * dstack_tid_B;
    int ** dstack_map_B;
    int ntid_A, ntid_B, nst_B;
    tsum<dtype> * sumf;
    check_sum(tid_A, tid_B, idx_map_A, idx_map_B);
    if (tensors[tid_A]->has_zero_edge_len || tensors[tid_B]->has_zero_edge_len){
      tensor<dtype> * tsr_B = tensors[tid_B];
      if (beta != 1.0 && !tsr_B->has_zero_edge_len){ 
        fseq_tsr_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        int sub_idx_map_B[tsr_B->order];
        int sm_idx=0;
        for (int i=0; i<tsr_B->order; i++){
          sub_idx_map_B[i]=sm_idx;
          sm_idx++;
          for (int j=0; j<i; j++){
            if (idx_map_B[i]==idx_map_B[j]){
              sub_idx_map_B[i]=sub_idx_map_B[j];
              sm_idx--;
              break;
            }
          }
        }
        scale_tsr(beta, tid_B, sub_idx_map_B, fs, felm); 
      }
      return CTF_SUCCESS;
    }


    CTF_alloc_ptr(sizeof(int)*tensors[tid_A]->order,   (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->order,   (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*tensors[tid_B]->order,   (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->order,   (void**)&dstack_tid_B);
    memcpy(map_A, idx_map_A, tensors[tid_A]->order*sizeof(int));
    memcpy(map_B, idx_map_B, tensors[tid_B]->order*sizeof(int));
    ntid_A = tid_A;
    ntid_B = tid_B;
    while (!run_diag && extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      if (ntid_A != tid_A) del_tsr(ntid_A);
      CTF_free(map_A);
      ntid_A = new_tid;
      map_A = new_idx_map;
    }
    nst_B = 0;
    while (!run_diag && extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == CTF_SUCCESS){
      dstack_map_B[nst_B] = map_B;
      dstack_tid_B[nst_B] = ntid_B;
      nst_B++;
      ntid_B = new_tid;
      map_B = new_idx_map;
    }
    if (ntid_A == ntid_B){
      clone_tensor(ntid_A, 1, &new_tid);
      stat = sum_tensors(alpha_, beta, new_tid, ntid_B, map_A, map_B, ftsr, felm);
      del_tsr(new_tid);
      ASSERT(stat == CTF_SUCCESS);
    } else{ 

      dtype alpha = alpha_*align_symmetric_indices(tensors[ntid_A]->order,
                                                   map_A,
                                                   tensors[ntid_A]->sym,
                                                   tensors[ntid_B]->order,
                                                   map_B,
                                                   tensors[ntid_B]->sym);

      CTF_sum_type_t type = {(int)ntid_A, (int)ntid_B,
                             (int*)map_A, (int*)map_B};
  #if DEBUG >= 1 //|| VERBOSE >= 1)
      print_sum(&type,alpha,beta);
  #endif

  #if VERIFY
      int64_t nsA, nsB;
      int64_t nA, nB;
      dtype * sA, * sB;
      dtype * uA, * uB;
      int order_A, order_B,  i;
      int * edge_len_A, * edge_len_B;
      int * sym_A, * sym_B;
      stat = allread_tsr(ntid_A, &nsA, &sA);
      assert(stat == CTF_SUCCESS);

      stat = allread_tsr(ntid_B, &nsB, &sB);
      assert(stat == CTF_SUCCESS);
  #endif

      TAU_FSTART(sum_tensors);

      /* Check if the current tensor mappings can be summed on */
  #if REDIST
      if (1) {
  #else
      if (check_sum_mapping(ntid_A, map_A, ntid_B, map_B) == 0) {
  #endif
        /* remap if necessary */
        stat = map_tensor_pair(ntid_A, map_A, ntid_B, map_B);
        if (stat == CTF_ERROR) {
          printf("Failed to map tensors to physical grid\n");
          return CTF_ERROR;
        }
      } else {
  #if DEBUG >= 2
        if (get_global_comm().rank == 0){
          printf("Keeping mappings:\n");
        }
        print_map(stdout, ntid_A);
        print_map(stdout, ntid_B);
  #endif
      }
      /* Construct the tensor algorithm we would like to use */
      ASSERT(check_sum_mapping(ntid_A, map_A, ntid_B, map_B));
  #if FOLD_TSR
      if (felm.func_ptr == NULL && can_fold(&type)){
        int inner_stride;
        TAU_FSTART(map_fold);
        stat = map_fold(&type, &inner_stride);
        TAU_FSTOP(map_fold);
        if (stat == CTF_SUCCESS){
          sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                                ftsr, felm, inner_stride);
        } else
          return CTF_ERROR;
      } else
        sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                             ftsr, felm);
  #else
      sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                           ftsr, felm);
  #endif
      /*TAU_FSTART(zero_sum_padding);
      stat = zero_out_padding(ntid_A);
      TAU_FSTOP(zero_sum_padding);
      TAU_FSTART(zero_sum_padding);
      stat = zero_out_padding(ntid_B);
      TAU_FSTOP(zero_sum_padding);*/
      DEBUG_PRINTF("[%d] performing tensor sum\n", get_global_comm().rank);
  #if DEBUG >=3
      if (get_global_comm().rank == 0){
        for (int i=0; i<tensors[ntid_A]->order; i++){
          printf("padding[%d] = %d\n",i, tensors[ntid_A]->padding[i]);
        }
        for (int i=0; i<tensors[ntid_B]->order; i++){
          printf("padding[%d] = %d\n",i, tensors[ntid_B]->padding[i]);
        }
      }
  #endif

      TAU_FSTART(sum_func);
      /* Invoke the contraction algorithm */

      sumf->run();
      TAU_FSTOP(sum_func);
  #ifndef SEQ
      stat = zero_out_padding(ntid_B);
  #endif

  #if VERIFY
      stat = allread_tsr(ntid_A, &nA, &uA);
      assert(stat == CTF_SUCCESS);
      stat = get_tsr_info(ntid_A, &order_A, &edge_len_A, &sym_A);
      assert(stat == CTF_SUCCESS);

      stat = allread_tsr(ntid_B, &nB, &uB);
      assert(stat == CTF_SUCCESS);
      stat = get_tsr_info(ntid_B, &order_B, &edge_len_B, &sym_B);
      assert(stat == CTF_SUCCESS);

      if (nsA != nA) { printf("nsA = " PRId64 ", nA = " PRId64 "\n",nsA,nA); ABORT; }
      if (nsB != nB) { printf("nsB = " PRId64 ", nB = " PRId64 "\n",nsB,nB); ABORT; }
      for (i=0; (uint64_t)i<nA; i++){
        if (fabs(uA[i] - sA[i]) > 1.E-6){
          printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
        }
      }

      cpy_sym_sum(alpha, uA, order_A, edge_len_A, edge_len_A, sym_A, map_A,
                  beta, sB, order_B, edge_len_B, edge_len_B, sym_B, map_B);
      assert(stat == CTF_SUCCESS);

      for (i=0; (uint64_t)i<nB; i++){
        if (fabs(uB[i] - sB[i]) > 1.E-6){
          printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
        }
        assert(fabs(sB[i] - uB[i]) < 1.E-6);
      }
      CTF_free(uA);
      CTF_free(uB);
      CTF_free(sA);
      CTF_free(sB);
  #endif

      delete sumf;
      if (ntid_A != tid_A) del_tsr(ntid_A);
      for (int i=nst_B-1; i>=0; i--){
        int ret = extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
        ASSERT(ret == CTF_SUCCESS);
        del_tsr(ntid_B);
        ntid_B = dstack_tid_B[i];
      }
      ASSERT(ntid_B == tid_B);
    }
    CTF_free(map_A);
    CTF_free(map_B);
    CTF_free(dstack_map_B);
    CTF_free(dstack_tid_B);

    TAU_FSTOP(sum_tensors);
    return CTF_SUCCESS;
  }


}
