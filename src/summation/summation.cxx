#include "summation.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_sum.h"
#include "sum_tsr.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"

namespace CTF_int {

  using namespace CTF;

  summation::~summation(){
    if (idx_A != NULL) free(idx_A);
    if (idx_B != NULL) free(idx_B);
  }

  summation::summation(summation const & other){
    A= other.A;
    idx_A = (int*)malloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B= other.B;
    idx_B = (int*)malloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    if (other.is_custom){
      func = other.func;
      is_custom = 1;
    } else {
      alpha = other.alpha;
      beta = other.beta;
    }
  }

  summation::summation(tensor * A_, 
                int const * idx_A_,
                char const * alpha_, 
                tensor * B_, 
                int const * idx_B_,
                char const * beta_){
    A = A_;
    idx_A = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    alpha = alpha_;
    B = B_;
    idx_B = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    beta = beta_;
    is_custom = 0;
  }
 
  summation::summation(tensor * A_, 
                int const * idx_A_,
                tensor * B_, 
                int const * idx_B_,
                univar_function func_){
    A = A_;
    idx_A = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    B = B_;
    idx_B = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
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
    tsum * htsum = NULL , ** rec_tsum = NULL;
    mapping * map;
    strp_tsr * str_A, * str_B;

    is_top = 1;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &order_tot, &idx_arr);

    nphys_dim = A->topo->order;

    CTF_alloc_ptr(sizeof(int)*A->order, (void**)&blk_len_A);
    CTF_alloc_ptr(sizeof(int)*B->order, (void**)&blk_len_B);
    CTF_alloc_ptr(sizeof(int)*A->order, (void**)&virt_blk_len_A);
    CTF_alloc_ptr(sizeof(int)*B->order, (void**)&virt_blk_len_B);
    CTF_alloc_ptr(sizeof(int)*order_tot, (void**)&virt_dim);
    CTF_alloc_ptr(sizeof(int)*nphys_dim*2, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*2);


    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = A->size;
    blk_sz_B = B->size;
    calc_dim(A->order, blk_sz_A, A->pad_edge_len, A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(B->order, blk_sz_B, B->pad_edge_len, B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);

    /* Strip out the relevant part of the tensor if we are contracting over diagonal */
    sA = strip_diag(A->order, order_tot, idx_A, vrt_sz_A,
                           A->edge_map, A->topo, A->sr,
                           blk_len_A, &blk_sz_A, &str_A);
    sB = strip_diag(B->order, order_tot, idx_B, vrt_sz_B,
                           B->edge_map, B->topo, B->sr,
                           blk_len_B, &blk_sz_B, &str_B);
    if (sA || sB){
      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Stripping tensor\n");
      strp_sum * ssum = new strp_sum;
      ssum->sr_A = A->sr;
      ssum->sr_B = B->sr;
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
        map = &A->edge_map[iA];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
          if (sA) virt_dim[i] = virt_dim[i]/str_A->strip_dim[iA];
        }
        else virt_dim[i] = 1;
      } else {
        ASSERT(iB!=-1);
        map = &B->edge_map[iB];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
          if (sB) virt_dim[i] = virt_dim[i]/str_B->strip_dim[iA];
        }
        else virt_dim[i] = 1;
      }
      nvirt *= virt_dim[i];
    }

    for (i=0; i<A->order; i++){
      map = &A->edge_map[i];
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
    for (i=0; i<B->order; i++){
      map = &B->edge_map[i];
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
      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Replicating tensor\n");

      tsum_replicate * rtsum = new tsum_replicate;
      rtsum->sr_A = A->sr;
      rtsum->sr_B = B->sr;
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
          rtsum->cdt_A[rtsum->ncdt_A] = A->topo->dim_comm[i];
          if (rtsum->cdt_A[rtsum->ncdt_A].alive == 0)
            rtsum->cdt_A[rtsum->ncdt_A].activate(A->wrld->comm);
          rtsum->ncdt_A++;
        }
        if (phys_mapped[2*i+1] == 0 && phys_mapped[2*i+0] == 1){
          rtsum->cdt_B[rtsum->ncdt_B] = B->topo->dim_comm[i];
          if (rtsum->cdt_B[rtsum->ncdt_B].alive == 0)
            rtsum->cdt_B[rtsum->ncdt_B].activate(B->wrld->comm);
          rtsum->ncdt_B++;
        }
      }
      ASSERT(rtsum->ncdt_A == 0 || rtsum->cdt_B == 0);
    }

    int * new_sym_A, * new_sym_B;
    CTF_alloc_ptr(sizeof(int)*A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, A->sym, sizeof(int)*A->order);
    CTF_alloc_ptr(sizeof(int)*B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, B->sym, sizeof(int)*B->order);

    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
      tsum_virt * tsumv = new tsum_virt;
      tsumv->sr_A = A->sr;
      tsumv->sr_B = B->sr;
      if (is_top) {
        htsum = tsumv;
        is_top = 0;
      } else {
        *rec_tsum = tsumv;
      }
      rec_tsum = &tsumv->rec_tsum;

      tsumv->num_dim  = order_tot;
      tsumv->virt_dim   = virt_dim;
      tsumv->order_A = A->order;
      tsumv->blk_sz_A = vrt_sz_A;
      tsumv->idx_map_A  = idx_A;
      tsumv->order_B = B->order;
      tsumv->blk_sz_B = vrt_sz_B;
      tsumv->idx_map_B  = idx_B;
      tsumv->buffer = NULL;
    } else CTF_free(virt_dim);

    seq_tsr_sum * tsumseq = new seq_tsr_sum;
    tsumseq->sr_A = A->sr;
    tsumseq->sr_B = B->sr;
    if (inner_stride == -1){
      tsumseq->is_inner = 0;
    } else {
      tsumseq->is_inner = 1;
      tsumseq->inr_stride = inner_stride;
      tensor * itsr;
      itsr = A->rec_tsr;
      i_A = 0;
      for (i=0; i<A->order; i++){
        if (A->sym[i] == NS){
          for (j=0; j<itsr->order; j++){
            if (A->inner_ordering[j] == i_A){
              j=i;
              do {
                j--;
              } while (j>=0 && A->sym[j] != NS);
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
      itsr = B->rec_tsr;
      i_B = 0;
      for (i=0; i<B->order; i++){
        if (B->sym[i] == NS){
          for (j=0; j<itsr->order; j++){
            if (B->inner_ordering[j] == i_B){
              j=i;
              do {
                j--;
              } while (j>=0 && B->sym[j] != NS);
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
    tsumseq->order_A        = A->order;
    tsumseq->idx_map_A      = idx_A;
    tsumseq->edge_len_A     = virt_blk_len_A;
    tsumseq->sym_A          = new_sym_A;
    tsumseq->order_B        = B->order;
    tsumseq->idx_map_B      = idx_B;
    tsumseq->edge_len_B     = virt_blk_len_B;
    tsumseq->sym_B          = new_sym_B;
    tsumseq->is_custom      = is_custom;
    if (is_custom){
      tsumseq->func     = func;
    } else {
      tsumseq->is_inner = 1;
      tsumseq->inr_stride = inner_stride;
      htsum->alpha  = alpha;
      htsum->beta   = beta;
    }

    htsum->A      = A->data;
    htsum->B      = B->data;

    CTF_free(idx_arr);
    CTF_free(blk_len_A);
    CTF_free(blk_len_B);
    CTF_free(phys_mapped);

    return htsum;
  }

  int summation::home_sum_tsr(bool run_diag){
    int ret, was_home_A, was_home_B;
    tensor * tnsr_A, * tnsr_B;
    summation osum = summation(*this);
   
    CTF_contract_mst();

  #ifndef HOME_CONTRACT
    #ifdef USE_SYM_SUM
      ret = sym_sum_tsr(run_diag);
      return ret;
    #else
      ret = sum_tensors(run_diag);
      return ret;
    #endif
  #else
    if (A->has_zero_edge_len || 
        B->has_zero_edge_len){
      if (!is_custom && !B->sr.isequal(beta,B->sr.mulid) && !B->has_zero_edge_len){ 
        int sub_idx_map_B[B->order];
        int sm_idx=0;
        for (int i=0; i<B->order; i++){
          sub_idx_map_B[i]=sm_idx;
          sm_idx++;
          for (int j=0; j<i; j++){
            if (idx_B[i]==idx_B[j]){
              sub_idx_map_B[i]=sub_idx_map_B[j];
              sm_idx--;
              break;
            }
          }
        }
        scaling scl = scaling(B, sub_idx_map_B, beta);
        scl.execute();
      }
      return SUCCESS;
    }
    if (A == B){
      tensor * cpy_tsr_A = new tensor(A);
      osum.A = cpy_tsr_A;
      osum.execute();
      return SUCCESS;
    }
    was_home_A = A->is_home;
    was_home_B = B->is_home;
    if (was_home_A){
      tnsr_A = new tensor(A,0,0);
      tnsr_A->data = A->data;
      tnsr_A->home_buffer = A->home_buffer;
      tnsr_A->is_home = 1;
      tnsr_A->is_mapped = 1;
      tnsr_A->topo = A->topo;
      copy_mapping(A->order, A->edge_map, tnsr_A->edge_map);
      tnsr_A->set_padding();
      osum.A = tnsr_A;
    }     
    if (was_home_B){
      tnsr_B = new tensor(B,0,0);
      tnsr_B->data = B->data;
      tnsr_B->home_buffer = B->home_buffer;
      tnsr_B->is_home = 1;
      tnsr_B->is_mapped = 1;
      tnsr_B->topo = B->topo;
      copy_mapping(B->order, B->edge_map, tnsr_B->edge_map);
      tnsr_B->set_padding();
      osum.B = tnsr_A;
    }
  #if DEBUG >= 1
    if (A->wrld->cdt.rank == 0)
      printf("Start head sum:\n");
  #endif
    
    #ifdef USE_SYM_SUM
    ret = osum.sym_sum_tsr(run_diag);
    #else
    ret = osum.sum_tensors(run_diag);
    #endif
  #if DEBUG >= 1
    if (A->wrld->cdt.rank == 0)
      printf("End head sum:\n");
  #endif

    if (ret!= SUCCESS) return ret;
    if (was_home_A) tnsr_A->unfold(); //FIXME: set_padding?
    if (was_home_B) tnsr_B->unfold();

    if (was_home_B && !tnsr_B->is_home){
      if (A->wrld->cdt.rank == 0)
        DPRINTF(2,"Migrating tensor %s back to home\n", B->name);
      distribution odst = distribution(tnsr_B);
      B->data = tnsr_B->data;
      B->is_home = 0;
      TAU_FSTART(redistribute_for_sum_home);
      B->redistribute(odst);
      TAU_FSTOP(redistribute_for_sum_home);
      memcpy(B->home_buffer, B->data, B->size*B->sr.el_size);
      CTF_free(B->data);
      B->data = B->home_buffer;
      B->is_home = 1;
      tnsr_B->is_data_aliased = 1;
      delete tnsr_B;
    } else if (was_home_B){
      if (tnsr_B->data != B->data){
        printf("Tensor %s is a copy of %s and did not leave home but buffer is %p was %p\n", tnsr_B->name, B->name, tnsr_B->data, B->data);
        ABORT;

      }
      tnsr_B->has_home = 0;
      tnsr_B->is_data_aliased = 1;
      delete tnsr_B;
    }
    if (was_home_A && !tnsr_A->is_home){
      tnsr_A->has_home = 0;
      delete tnsr_A;
    } else if (was_home_A) {
      tnsr_A->has_home = 0;
      tnsr_A->is_data_aliased = 1;
      delete tnsr_A;
    }
    return ret;
  #endif
  }

  int summation::sym_sum_tsr(bool run_diag){
    int sidx, i, nst_B, * new_idx_map;
    int * map_A, * map_B;
    int ** dstack_map_B;
    tensor * tnsr_A, * tnsr_B, * new_tsr, ** dstack_tsr_B;
    std::vector<summation> perm_types;
    std::vector<int> signs;
    char const * dbeta;
  //#if (DEBUG >= 1 || VERBOSE >= 1)
  //  print_sum(type,alpha_,beta);
  //#endif
    check_consistency();
    if (A->has_zero_edge_len || B->has_zero_edge_len){
      if (!is_custom && !B->sr.isequal(beta, B->sr.mulid) && !B->has_zero_edge_len){ 
        int sub_idx_map_B[B->order];
        int sm_idx=0;
        for (int i=0; i<B->order; i++){
          sub_idx_map_B[i]=sm_idx;
          sm_idx++;
          for (int j=0; j<i; j++){
            if (idx_B[i]==idx_B[j]){
              sub_idx_map_B[i]=sub_idx_map_B[j];
              sm_idx--;
              break;
            }
          }
        }
        scaling scl = scaling(B, sub_idx_map_B, beta);
        scl.execute();
      }
      return SUCCESS;
    }
    tnsr_A = A;
    tnsr_B = B;
    CTF_alloc_ptr(sizeof(int)*tnsr_A->order,   (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tnsr_B->order,   (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*tnsr_B->order,   (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(tensor*)*tnsr_B->order,   (void**)&dstack_tsr_B);
    memcpy(map_A, idx_A, tnsr_A->order*sizeof(int));
    memcpy(map_B, idx_B, tnsr_B->order*sizeof(int));
    while (!run_diag && tnsr_A->extract_diag(map_A, 1, new_tsr, &new_idx_map) == SUCCESS){
      if (tnsr_A != A) delete tnsr_A;
      CTF_free(map_A);
      tnsr_A = new_tsr;
      map_A = new_idx_map;
    }
    nst_B = 0;
    while (!run_diag && tnsr_B->extract_diag(map_B, 1, new_tsr, &new_idx_map) == SUCCESS){
      dstack_map_B[nst_B] = map_B;
      dstack_tsr_B[nst_B] = tnsr_B;
      nst_B++;
      tnsr_B = new_tsr;
      map_B = new_idx_map;
    }

    summation newsum = summation(*this);
    newsum.A = tnsr_A;
    newsum.B = tnsr_B;
    if (tnsr_A == tnsr_B){
      new_tsr = new tensor(tnsr_A);
      newsum.A = new_tsr;
      newsum.B = tnsr_B;
      return newsum.sym_sum_tsr(run_diag);
      
      /*clone_tensor(ntid_A, 1, &new_tid);
      new_type = *type;
      new_type.tid_A = new_tid;
      stat = sym_sum_tsr(alpha_, beta, &new_type, ftsr, felm, run_diag);
      del_tsr(new_tid);
      return stat;*/
    }
    
/*    new_type.tid_A = ntid_A;
    new_type.tid_B = ntid_B;
    new_type.idx_map_A = map_A;
    new_type.idx_map_B = map_B;*/

    //FIXME: make these typefree...
    int sign = align_symmetric_indices(tnsr_A->order,
                                                 map_A,
                                                 tnsr_A->sym,
                                                 tnsr_B->order,
                                                 map_B,
                                                 tnsr_B->sym);
    int ocfact = overcounting_factor(tnsr_A->order,
                                        map_A,
                                        tnsr_A->sym,
                                        tnsr_B->order,
                                        map_B,
                                        tnsr_B->sym);

    if (ocfact != 1 || sign != 1){
      if (is_custom) //PANIC
        ABORT; //FIXME!
      else {
        if (ocfact != 1){
          char * new_alpha = (char*)malloc(tnsr_B->sr.el_size);
          tnsr_B->sr.copy(new_alpha, tnsr_B->sr.addid);
          
          for (int i=0; i<ocfact; i++){
            tnsr_B->sr.add(new_alpha, alpha, new_alpha);
          }
          alpha = new_alpha;
        }
        if (sign == -1){
          char * new_alpha = (char*)malloc(tnsr_B->sr.el_size);
          tnsr_B->sr.addinv(alpha, new_alpha);
          alpha = new_alpha;
        }
      }
    }


    if (unfold_broken_sym(NULL) != -1){
      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Contraction index is broken\n");

      summation * unfold_sum;
      sidx = unfold_broken_sym(&unfold_sum);
      int sy;
      sy = 0;
      for (i=0; i<A->order; i++){
        if (A->sym[i] == SY) sy = 1;
      }
      for (i=0; i<B->order; i++){
        if (B->sym[i] == SY) sy = 1;
      }
      if (sy && sidx%2 == 0){/* && map_tensors(&unfold_type,
                            ftsr, felm, alpha, beta, &ctrf, 0) == SUCCESS){*/
        if (A->wrld->cdt.rank == 0)
          DPRINTF(1,"Performing index desymmetrization\n");
        desymmetrize(tnsr_A, unfold_sum->A, 0);
        unfold_sum->B = tnsr_B;
        unfold_sum->sym_sum_tsr(run_diag);
//        sym_sum_tsr(alpha, beta, &unfold_type, ftsr, felm, run_diag);
        if (tnsr_A != unfold_sum->A){
          unfold_sum->A->unfold();
          tnsr_A->pull_alias(unfold_sum->A);
          delete unfold_sum->A;
        }
      } else {
        //get_sym_perms(&new_type, alpha, perm_types, signs);
        get_sym_perms(newsum, perm_types, signs);
        if (A->wrld->cdt.rank == 0)
          DPRINTF(1,"Performing %d summation permutations\n", 
                  (int)perm_types.size());
        dbeta = beta;
        char * new_alpha = (char*)malloc(tnsr_B->sr.el_size);
        for (i=0; i<(int)perm_types.size(); i++){
          if (signs[i] == 1)
            B->sr.copy(new_alpha, alpha);
          else
            tnsr_B->sr.addinv(alpha, new_alpha);
          perm_types[i].alpha = new_alpha;
          perm_types[i].beta = dbeta;
          perm_types[i].execute();
          /*sum_tensors(new_alpha, dbeta, perm_types[i].tid_A, perm_types[i].tid_B,
                      perm_types[i].idx_map_A, perm_types[i].idx_map_B, ftsr, felm, run_diag);*/
          dbeta = newsum.B->sr.addid;
        }
/*        for (i=0; i<(int)perm_types.size(); i++){
          free_type(&perm_types[i]);
        }*/
        perm_types.clear();
        signs.clear();
      }
    } else {
      newsum.sum_tensors(run_diag);
/*      sum_tensors(alpha, beta, new_type.tid_A, new_type.tid_B, new_type.idx_map_A, 
                  new_type.idx_map_B, ftsr, felm, run_diag);*/
    }
    if (tnsr_A != A) delete tnsr_A;
    for (i=nst_B-1; i>=0; i--){
//      extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
      dstack_tsr_B[i]->extract_diag(dstack_map_B[i], 0, tnsr_B, &new_idx_map);
      //del_tsr(ntid_B);
      delete tnsr_B;
      tnsr_B = dstack_tsr_B[i];
    }
    ASSERT(tnsr_B == B);
    CTF_free(map_A);
    CTF_free(map_B);
    CTF_free(dstack_map_B);
    CTF_free(dstack_tsr_B);

    return SUCCESS;
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
      tensor<dtype> * B = tensors[tid_B];
      if (beta != 1.0 && !B->has_zero_edge_len){ 
        fseq_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;
        int sub_idx_map_B[B->order];
        int sm_idx=0;
        for (int i=0; i<B->order; i++){
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
      return SUCCESS;
    }


    CTF_alloc_ptr(sizeof(int)*tensors[tid_A]->order,   (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->order,   (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*tensors[tid_B]->order,   (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(int)*tensors[tid_B]->order,   (void**)&dstack_tid_B);
    memcpy(map_A, idx_map_A, tensors[tid_A]->order*sizeof(int));
    memcpy(map_B, idx_map_B, tensors[tid_B]->order*sizeof(int));
    ntid_A = tid_A;
    ntid_B = tid_B;
    while (!run_diag && extract_diag(ntid_A, map_A, 1, &new_tid, &new_idx_map) == SUCCESS){
      if (ntid_A != tid_A) del_tsr(ntid_A);
      CTF_free(map_A);
      ntid_A = new_tid;
      map_A = new_idx_map;
    }
    nst_B = 0;
    while (!run_diag && extract_diag(ntid_B, map_B, 1, &new_tid, &new_idx_map) == SUCCESS){
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
      ASSERT(stat == SUCCESS);
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
      assert(stat == SUCCESS);

      stat = allread_tsr(ntid_B, &nsB, &sB);
      assert(stat == SUCCESS);
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
        if (A->wrld->cdt.rank == 0){
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
        if (stat == SUCCESS){
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
      DEBUG_PRINTF("[%d] performing tensor sum\n", A->wrld->cdt.rank);
  #if DEBUG >=3
      if (A->wrld->cdt.rank == 0){
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
      assert(stat == SUCCESS);
      stat = get_info(ntid_A, &order_A, &edge_len_A, &sym_A);
      assert(stat == SUCCESS);

      stat = allread_tsr(ntid_B, &nB, &uB);
      assert(stat == SUCCESS);
      stat = get_info(ntid_B, &order_B, &edge_len_B, &sym_B);
      assert(stat == SUCCESS);

      if (nsA != nA) { printf("nsA = " PRId64 ", nA = " PRId64 "\n",nsA,nA); ABORT; }
      if (nsB != nB) { printf("nsB = " PRId64 ", nB = " PRId64 "\n",nsB,nB); ABORT; }
      for (i=0; (uint64_t)i<nA; i++){
        if (fabs(uA[i] - sA[i]) > 1.E-6){
          printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
        }
      }

      cpy_sym_sum(alpha, uA, order_A, edge_len_A, edge_len_A, sym_A, map_A,
                  beta, sB, order_B, edge_len_B, edge_len_B, sym_B, map_B);
      assert(stat == SUCCESS);

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
        ASSERT(ret == SUCCESS);
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
    return SUCCESS;
  }

  int summation::unfold_broken_sym(summation ** nnew_sum){
    int sidx, i, num_tot, iA, iA2, iB;
    int * idx_arr;

    sumation * new_sum;
   
    if (nnew_sum != NULL){
      new_sum = new summation(*this);
      *nnew_sum = new_sum;
    }

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);

    sidx = -1;
    for (i=0; i<A->order; i++){
      if (A->sym[i] != NS){
        iA = idx_A[i];
        if (idx_arr[2*iA+1] != -1){
          if (B->sym[idx_arr[2*iA+1]] == NS ||
              idx_arr[2*idx_A[i+1]+1] == -1 ||
              idx_A[i+1] != idx_B[idx_arr[2*iA+1]+1]){
            sidx = 2*i;
            break;
          }
        } else if (idx_arr[2*idx_A[i+1]+1] != -1){
          sidx = 2*i;
          break;
        }
      }
    } 
    if (sidx == -1){
      for (i=0; i<B->order; i++){
        if (B->sym[i] != NS){
          iB = idx_B[i];
          if (idx_arr[2*iB+0] != -1){
            if (A->sym[idx_arr[2*iB+0]] == NS ||
                idx_arr[2*idx_B[i+1]+0] == -1 ||
                idx_B[i+1] != idx_A[idx_arr[2*iB+0]+1]){
              sidx = 2*i+1;
              break;
            }
          } else if (idx_arr[2*idx_B[i+1]+0] != -1){
            sidx = 2*i+1;
            break;
          }
        }
      }
    }
    if (sidx == -1){
      for (i=0; i<A->order; i++){
        if (A->sym[i] == SY){
          iA = idx_A[i];
          iA2 = idx_A[i+1];
          if (idx_arr[2*iA+1] == -1 &&
              idx_arr[2*iA2+1] == -1){
            sidx = 2*i;
            break;
          }
        }
      }
    } 
    if (nnew_sum != NULL && sidx != -1){
      if(sidx%2 == 0){
        new_sum->A = new tensor(A, 0, 0);
        new_sum->A->sym[sidx/2] = NS;
      } else {
        new_sum->B = new tensor(B, 0, 0);
        new_sum->B->sym[sidx/2] = NS;
      }
    }
    CTF_free(idx_arr);
    return sidx;
  }

  void summation::check_consistency(){
    int i, num_tot, len;
    int iA, iB;
    int order_A, order_B;
    int * idx_arr;
       
    inv_idx(A->order, idx_map_A,
            B->order, idx_map_B,
            &num_tot, &idx_arr);

    for (i=0; i<num_tot; i++){
      len = -1;
      iA = idx_arr[2*i+0];
      iB = idx_arr[2*i+1];
      if (iA != -1){
        len = A->lens[iA];
      }
      if (len != -1 && iB != -1 && len != B->lens[iB]){
        if (global_comm.rank == 0){
          printf("i = %d Error in sum call: The %dth edge length (%d) of tensor %s does not",
                  i, iA, len, tsr_A->name);
          printf("match the %dth edge length (%d) of tensor %s.\n",
                  iB, B->lens[iB], tsr_B->name);
        }
        ABORT;
      }
    }
    CTF_free(idx_arr);
    return CTF_SUCCESS;

  }


  int summation::is_equal(summation const & os){
    int i;

    if (A != os.A) return 0;
    if (B != os.B) return 0;

    for (i=0; i<A->ndim; i++){
      if (idx_A[i] != os.idx_A[i]) return 0;
    }
    for (i=0; i<B->ndim; i++){
      if (idx_B[i] != os.idx_B[i]) return 0;
    }
    return 1;
  }

}
