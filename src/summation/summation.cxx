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
#include "../redistribution/folding.h"

namespace CTF_int {

  using namespace CTF;

  summation::~summation(){
    if (idx_A != NULL) free(idx_A);
    if (idx_B != NULL) free(idx_B);
  }

  summation::summation(summation const & other){
    A     = other.A;
    idx_A = (int*)malloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B     = other.B;
    idx_B = (int*)malloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    if (other.is_custom){
      func      = other.func;
      is_custom = 1;
    } else {
      alpha = other.alpha;
      beta  = other.beta;
    }
  }

  summation::summation(tensor *     A_,
                       int const *  idx_A_,
                       char const * alpha_,
                       tensor *     B_,
                       int const *  idx_B_,
                       char const * beta_){
    A         = A_;
    idx_A     = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    alpha     = alpha_;
    B         = B_;
    idx_B     = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    beta      = beta_;
    is_custom = 0;
  }
 
  summation::summation(tensor *        A_,
                       int const *     idx_A_,
                       tensor *        B_,
                       int const *     idx_B_,
                       univar_function func_){
    A         = A_;
    idx_A     = (int*)malloc(sizeof(int)*A->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    B         = B_;
    idx_B     = (int*)malloc(sizeof(int)*B->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    func      = func_;
    is_custom = 1;
  }

  void summation::get_fold_indices(int *  num_fold,
                                   int ** fold_idx){
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
      i      = idx_A[iA];
      iB     = idx_arr[2*i+1];
      broken = 0;
      inA    = iA;
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
      i      = idx_B[iB];
      iA     = idx_arr[2*i+0];
      broken = 0;
      inB    = iB;
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

  int summation::map_fold(){
    int i, j, nfold, nf, all_fdim_A, all_fdim_B;
    int nvirt_A, nvirt_B;
    int * fold_idx, * fidx_map_A, * fidx_map_B;
    int * fnew_ord_A, * fnew_ord_B;
    int * all_flen_A, * all_flen_B;
    tensor * ftsr_A, * ftsr_B;
    summation fold_sum;
    int inr_stride;

    get_fold_indices(&nfold, &fold_idx);
    if (nfold == 0){
      CTF_free(fold_idx);
      return ERROR;
    }

    /* overestimate this space to not bother with it later */
    CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_A);
    CTF_alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_B);


    A->fold(nfold, fold_idx, idx_A, 
            &all_fdim_A, &all_flen_A);
    B->fold(nfold, fold_idx, idx_B, 
            &all_fdim_B, &all_flen_B);

    nf = 0;
    for (i=0; i<A->order; i++){
      for (j=0; j<nfold; j++){
        if (A->sym[i] == NS && idx_A[i] == fold_idx[j]){
          fidx_map_A[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<B->order; i++){
      for (j=0; j<nfold; j++){
        if (B->sym[i] == NS && idx_B[i] == fold_idx[j]){
          fidx_map_B[nf] = j;
          nf++;
        }
      }
    }

    ftsr_A = A->rec_tsr;
    ftsr_B = B->rec_tsr;

    fold_sum.A = A->rec_tsr;
    fold_sum.B = B->rec_tsr;

    conv_idx<int>(ftsr_A->order, fidx_map_A, &fold_sum.idx_A,
                  ftsr_B->order, fidx_map_B, &fold_sum.idx_B);

  #if DEBUG>=2
    if (global_comm.rank == 0){
      printf("Folded summation type:\n");
    }
    //fold_sum.print();//print_sum(&fold_type,0.0,0.0);
  #endif
   
    //for type order 1 to 3 
    get_len_ordering(&fnew_ord_A, &fnew_ord_B); 



    permute_target(ftsr_A->order, fnew_ord_A, A->inner_ordering);
    permute_target(ftsr_B->order, fnew_ord_B, B->inner_ordering);
    

    nvirt_A = A->calc_nvirt();
    for (i=0; i<nvirt_A; i++){
      nosym_transpose(all_fdim_A, A->inner_ordering, all_flen_A, 
                      A->data + i*(A->size/nvirt_A), 1, A->sr);
    }
    nvirt_B = B->calc_nvirt();
    for (i=0; i<nvirt_B; i++){
      nosym_transpose(all_fdim_B, B->inner_ordering, all_flen_B, 
                      B->data + i*(B->size/nvirt_B), 1, B->sr);
    }

    inr_stride = 1;
    for (i=0; i<ftsr_A->order; i++){
      inr_stride *= ftsr_A->pad_edge_len[i];
    }

    CTF_free(fidx_map_A);
    CTF_free(fidx_map_B);
    CTF_free(fnew_ord_A);
    CTF_free(fnew_ord_B);
    CTF_free(all_flen_A);
    CTF_free(all_flen_B);
    CTF_free(fold_idx);

    return inr_stride; 
  }

  void summation::get_len_ordering(int ** new_ordering_A,
                                   int ** new_ordering_B){
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

    CTF_alloc_ptr(sizeof(int)*A->order,    (void**)&blk_len_A);
    CTF_alloc_ptr(sizeof(int)*B->order,    (void**)&blk_len_B);
    CTF_alloc_ptr(sizeof(int)*A->order,    (void**)&virt_blk_len_A);
    CTF_alloc_ptr(sizeof(int)*B->order,    (void**)&virt_blk_len_B);
    CTF_alloc_ptr(sizeof(int)*order_tot,   (void**)&virt_dim);
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
      rec_tsum      = &rtsum->rec_tsum;
      rtsum->ncdt_A = 0;
      rtsum->ncdt_B = 0;
      rtsum->size_A = blk_sz_A;
      rtsum->size_B = blk_sz_B;
      rtsum->cdt_A  = NULL;
      rtsum->cdt_B  = NULL;
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
      rec_tsum         = &tsumv->rec_tsum;

      tsumv->num_dim   = order_tot;
      tsumv->virt_dim  = virt_dim;
      tsumv->order_A   = A->order;
      tsumv->blk_sz_A  = vrt_sz_A;
      tsumv->idx_map_A = idx_A;
      tsumv->order_B   = B->order;
      tsumv->blk_sz_B  = vrt_sz_B;
      tsumv->idx_map_B = idx_B;
      tsumv->buffer    = NULL;
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
    tsumseq->order_A    = A->order;
    tsumseq->idx_map_A  = idx_A;
    tsumseq->edge_len_A = virt_blk_len_A;
    tsumseq->sym_A      = new_sym_A;
    tsumseq->order_B    = B->order;
    tsumseq->idx_map_B  = idx_B;
    tsumseq->edge_len_B = virt_blk_len_B;
    tsumseq->sym_B      = new_sym_B;
    tsumseq->is_custom  = is_custom;
    if (is_custom){
      tsumseq->func     = func;
    } else {
      tsumseq->is_inner   = 1;
      tsumseq->inr_stride = inner_stride;
      htsum->alpha        = alpha;
      htsum->beta         = beta;
    }

    htsum->A = A->data;
    htsum->B = B->data;

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
      tnsr_A              = new tensor(A,0,0);
      tnsr_A->data        = A->data;
      tnsr_A->home_buffer = A->home_buffer;
      tnsr_A->is_home     = 1;
      tnsr_A->is_mapped   = 1;
      tnsr_A->topo        = A->topo;
      copy_mapping(A->order, A->edge_map, tnsr_A->edge_map);
      tnsr_A->set_padding();
      osum.A              = tnsr_A;
    }     
    if (was_home_B){
      tnsr_B              = new tensor(B,0,0);
      tnsr_B->data        = B->data;
      tnsr_B->home_buffer = B->home_buffer;
      tnsr_B->is_home     = 1;
      tnsr_B->is_mapped   = 1;
      tnsr_B->topo        = B->topo;
      copy_mapping(B->order, B->edge_map, tnsr_B->edge_map);
      tnsr_B->set_padding();
      osum.B              = tnsr_A;
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
    CTF_alloc_ptr(sizeof(int)*tnsr_A->order,     (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*tnsr_B->order,     (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*tnsr_B->order,    (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(tensor*)*tnsr_B->order, (void**)&dstack_tsr_B);
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
      tensor nnew_tsr = tensor(tnsr_A);
      newsum.A = &nnew_tsr;
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
    int stat, * new_idx_map;
    int * map_A, * map_B;
    int nst_B;
    int ** dstack_map_B;
    tensor * tnsr_A, * tnsr_B, * new_tsr, ** dstack_tsr_B;
//    tsum<dtype> * sumf;
    tsum * sumf;
    //check_sum(tid_A, tid_B, idx_map_A, idx_map_B);
    //FIXME: hmm all of the below already takes place in sym_sum
    check_consistency();
    if (A->has_zero_edge_len || B->has_zero_edge_len){
      if (!is_custom && !B->sr.isequal(beta,B->sr.mulid) && !B->has_zero_edge_len){ 
    /*    fseq_scl<dtype> fs;
        fs.func_ptr=sym_seq_scl_ref<dtype>;
        fseq_elm_scl<dtype> felm;
        felm.func_ptr = NULL;*/
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


    //FIXME: remove all of the below, sum_tensors should never be called without sym_sum
    CTF_alloc_ptr(sizeof(int)*A->order,  (void**)&map_A);
    CTF_alloc_ptr(sizeof(int)*B->order,  (void**)&map_B);
    CTF_alloc_ptr(sizeof(int*)*B->order, (void**)&dstack_map_B);
    CTF_alloc_ptr(sizeof(tensor*)*B->order, (void**)&dstack_tsr_B);
    tnsr_A = A;
    tnsr_B = B;
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
      tensor * nnew_tsr = new tensor(tnsr_A);
      newsum.A = nnew_tsr;
      newsum.B = tnsr_B;
    } else{ 
     //FIXME: remove the below, sum_tensors should never be called without sym_sum
     int sign = align_symmetric_indices(tnsr_A->order,
                                        map_A,
                                        tnsr_A->sym,
                                        tnsr_B->order,
                                        map_B,
                                        tnsr_B->sym);
      ASSERT(sign == 1);
/*        if (sign == -1){
          char * new_alpha = (char*)malloc(tnsr_B->sr.el_size);
          tnsr_B->sr.addinv(alpha, new_alpha);
          alpha = new_alpha;
        }*/

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
      if (check_mapping() == 0) {
  #endif
        /* remap if necessary */
        stat = map();
        if (stat == ERROR) {
          printf("Failed to map tensors to physical grid\n");
          return ERROR;
        }
      } else {
  #if DEBUG >= 2
        if (A->wrld->cdt.rank == 0){
          printf("Keeping mappings:\n");
        }
       // print_map(stdout, ntid_A);
       // print_map(stdout, ntid_B);
  #endif
      }
      /* Construct the tensor algorithm we would like to use */
      ASSERT(check_mapping());
  #if FOLD_TSR
      if (is_custom == false && can_fold()){
        int inner_stride;
        TAU_FSTART(map_fold);
        inner_stride = map_fold();
        TAU_FSTOP(map_fold);
        sumf = newsum.construct_sum(inner_stride);
        /*alpha, beta, ntid_A, map_A, ntid_B, map_B,
                              ftsr, felm, inner_stride);*/
      } else
        sumf = newsum.construct_sum();
        /*sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                             ftsr, felm);*/
  #else
      sumf = newsum.construct_sum();
      /*sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                           ftsr, felm);*/
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
      stat = tnsr_B->zero_out_padding();
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
      if (tnsr_A != A) delete tnsr_A;
      for (int i=nst_B-1; i>=0; i--){
        int ret = dstack_tsr_B[i]->extract_diag(dstack_map_B[i], 0, tnsr_B, &new_idx_map);
        ASSERT(ret == SUCCESS);
        delete tnsr_B;
        tnsr_B = dstack_tsr_B[i];
      }
      ASSERT(tnsr_B == B);
    }
    CTF_free(map_A);
    CTF_free(map_B);
    CTF_free(dstack_map_B);
    CTF_free(dstack_tsr_B);

    TAU_FSTOP(sum_tensors);
    return SUCCESS;
  }

  int summation::unfold_broken_sym(summation ** nnew_sum){
    int sidx, i, num_tot, iA, iA2, iB;
    int * idx_arr;

    summation * new_sum;
   
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
    int * idx_arr;
       
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);

    for (i=0; i<num_tot; i++){
      len = -1;
      iA = idx_arr[2*i+0];
      iB = idx_arr[2*i+1];
      if (iA != -1){
        len = A->lens[iA];
      }
      if (len != -1 && iB != -1 && len != B->lens[iB]){
        if (A->wrld->cdt.rank == 0){
          printf("i = %d Error in sum call: The %dth edge length (%d) of tensor %s does not",
                  i, iA, len, A->name);
          printf("match the %dth edge length (%d) of tensor %s.\n",
                  iB, B->lens[iB], B->name);
        }
        ABORT;
      }
    }
    CTF_free(idx_arr);

  }


  int summation::is_equal(summation const & os){
    int i;

    if (A != os.A) return 0;
    if (B != os.B) return 0;

    for (i=0; i<A->order; i++){
      if (idx_A[i] != os.idx_A[i]) return 0;
    }
    for (i=0; i<B->order; i++){
      if (idx_B[i] != os.idx_B[i]) return 0;
    }
    return 1;
  }

  int summation::check_mapping(){
    int i, pass, order_tot, iA, iB;
    int * idx_arr, * phys_map;
    mapping * map;

    TAU_FSTART(check_sum_mapping);
    pass = 1;
    
    if (A->is_mapped == 0) pass = 0;
    if (B->is_mapped == 0) pass = 0;
    
    
    if (A->is_folded == 1) pass = 0;
    if (B->is_folded == 1) pass = 0;
    
    if (A->topo != B->topo) pass = 0;

    if (pass==0){
      TAU_FSTOP(check_sum_mapping);
      return 0;
    }
    
    CTF_alloc_ptr(sizeof(int)*A->topo->order, (void**)&phys_map);
    memset(phys_map, 0, sizeof(int)*A->topo->order);

    inv_idx(A->order, idx_A, A->edge_map,
            B->order, idx_B, B->edge_map,
            &order_tot, &idx_arr);

    if (!check_self_mapping(A, idx_A))
      pass = 0;
    if (!check_self_mapping(B, idx_B))
      pass = 0;
    if (pass == 0)
      DPRINTF(4,"failed confirmation here\n");

    for (i=0; i<order_tot; i++){
      iA = idx_arr[2*i];
      iB = idx_arr[2*i+1];
      if (iA != -1 && iB != -1) {
        if (!comp_dim_map(&A->edge_map[iA], &B->edge_map[iB])){
          pass = 0;
          DPRINTF(4,"failed confirmation here i=%d\n",i);
        }
      }
      if (iA != -1) {
        map = &A->edge_map[iA];
        if (map->type == PHYSICAL_MAP)
          phys_map[map->cdt] = 1;
        while (map->has_child) {
          map = map->child;
          if (map->type == PHYSICAL_MAP)
            phys_map[map->cdt] = 1;
        }
      }
      if (iB != -1){
        map = &B->edge_map[iB];
        if (map->type == PHYSICAL_MAP)
          phys_map[map->cdt] = 1;
        while (map->has_child) {
          map = map->child;
          if (map->type == PHYSICAL_MAP)
            phys_map[map->cdt] = 1;
        }
      }
    }
    /* Ensure that something is mapped to each dimension, since replciation
       does not make sense in sum for all tensors */
  /*  for (i=0; i<topovec[A->itopo].order; i++){
      if (phys_map[i] == 0) {
        pass = 0;
        DPRINTF(3,"failed confirmation here i=%d\n",i);
      }
    }*/

    CTF_free(phys_map);
    CTF_free(idx_arr);

    TAU_FSTOP(check_sum_mapping);

    return pass;
  }

  int summation::map(){
    int i, ret, num_sum, num_tot, need_remap;
    int was_cyclic_A, was_cyclic_B, need_remap_A, need_remap_B;

    int d, old_topo_A, old_topo_B;
    int64_t old_size_A, old_size_B;
    int * idx_arr, * idx_sum;
    mapping * old_map_A, * old_map_B;
    int * old_phase_A, * old_rank_A, * old_virt_dim_A, * old_pe_lda_A, 
        * old_padding_A, * old_edge_len_A;
    int * old_phase_B, * old_rank_B, * old_virt_dim_B, * old_pe_lda_B, 
        * old_padding_B, * old_edge_len_B;
  //  uint64_t nvirt, tnvirt, bnvirt;
    int btopo;
    int gtopo;
    tensor<dtype> * tsr_A, * tsr_B;
    tsr_A = tensors[tid_A];
    tsr_B = tensors[tid_B];
    
    TAU_FSTART(map_tensor_pair);

    inv_idx(tsr_A->order, idx_map_A, tsr_A->edge_map,
            tsr_B->order, idx_map_B, tsr_B->edge_map,
            &num_tot, &idx_arr);

    CTF_alloc_ptr(sizeof(int)*num_tot, (void**)&idx_sum);
    
    num_sum = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[2*i] != -1 && idx_arr[2*i+1] != -1){
        idx_sum[num_sum] = i;
        num_sum++;
      }
    }
  #if DEBUG >= 2
    if (global_comm.rank == 0)
      printf("Initial mappings:\n");
    print_map(stdout, tid_A);
    print_map(stdout, tid_B);
  #endif

    unmap_inner(tsr_A);
    unmap_inner(tsr_B);
    set_padding(tsr_A);
    set_padding(tsr_B);
    save_mapping(tsr_A, &old_phase_A, &old_rank_A, &old_virt_dim_A, &old_pe_lda_A, 
                 &old_size_A, &was_cyclic_A, &old_padding_A, &old_edge_len_A, &topovec[tsr_A->itopo]);  
    save_mapping(tsr_B, &old_phase_B, &old_rank_B, &old_virt_dim_B, &old_pe_lda_B, 
                 &old_size_B, &was_cyclic_B, &old_padding_B, &old_edge_len_B, &topovec[tsr_B->itopo]);  
    old_topo_A = tsr_A->itopo;
    old_topo_B = tsr_B->itopo;
    CTF_alloc_ptr(sizeof(mapping)*tsr_A->order,         (void**)&old_map_A);
    CTF_alloc_ptr(sizeof(mapping)*tsr_B->order,         (void**)&old_map_B);
    for (i=0; i<tsr_A->order; i++){
      old_map_A[i].type         = NOT_MAPPED;
      old_map_A[i].has_child    = 0;
      old_map_A[i].np           = 1;
    }
    for (i=0; i<tsr_B->order; i++){
      old_map_B[i].type         = NOT_MAPPED;
      old_map_B[i].has_child    = 0;
      old_map_B[i].np           = 1;
    }
    copy_mapping(tsr_A->order, tsr_A->edge_map, old_map_A);
    copy_mapping(tsr_B->order, tsr_B->edge_map, old_map_B);
  //  bnvirt = 0;  
    btopo = -1;
    uint64_t size;
    uint64_t min_size = UINT64_MAX;
    /* Attempt to map to all possible permutations of processor topology */
    for (i=global_comm.rank; i<2*(int)topovec.size(); i+=global_comm.np){
  //  for (i=global_comm.rank*topovec.size(); i<2*(int)topovec.size(); i++){
      clear_mapping(tsr_A);
      clear_mapping(tsr_B);
      set_padding(tsr_A);
      set_padding(tsr_B);

      tsr_A->itopo = i/2;
      tsr_B->itopo = i/2;
      tsr_A->is_mapped = 1;
      tsr_B->is_mapped = 1;

      if (i%2 == 0){
        ret = map_self_indices(tid_A, idx_map_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(tid_B, idx_map_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }
      ret = map_sum_indices(idx_arr, idx_sum, num_tot, num_sum, 
                            tid_A, tid_B, &topovec[tsr_A->itopo], 2);
      if (ret == CTF_NEGATIVE) continue;
      else if (ret != SUCCESS){
        return ret;
      }
      if (i%2 == 0){
        ret = map_self_indices(tid_A, idx_map_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(tid_B, idx_map_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }

      if (i%2 == 0){
        ret = map_self_indices(tid_A, idx_map_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        ret = map_tensor_rem(topovec[tsr_A->itopo].order, 
                             topovec[tsr_A->itopo].dim_comm, tsr_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        copy_mapping(tsr_A->order, tsr_B->order,
                     idx_map_A, tsr_A->edge_map, 
                     idx_map_B, tsr_B->edge_map,0);
        ret = map_tensor_rem(topovec[tsr_B->itopo].order, 
                             topovec[tsr_B->itopo].dim_comm, tsr_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(tid_B, idx_map_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        ret = map_tensor_rem(topovec[tsr_B->itopo].order, 
                             topovec[tsr_B->itopo].dim_comm, tsr_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        copy_mapping(tsr_B->order, tsr_A->order,
                     idx_map_B, tsr_B->edge_map, 
                     idx_map_A, tsr_A->edge_map,0);
        ret = map_tensor_rem(topovec[tsr_A->itopo].order, 
                             topovec[tsr_A->itopo].dim_comm, tsr_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }
      if (i%2 == 0){
        ret = map_self_indices(tid_B, idx_map_B);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(tid_A, idx_map_A);
        if (ret == CTF_NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }

  /*    ret = map_symtsr(tsr_A->order, tsr_A->sym_table, tsr_A->edge_map);
      ret = map_symtsr(tsr_B->order, tsr_B->sym_table, tsr_B->edge_map);
      if (ret!=SUCCESS) return ret;
      return SUCCESS;*/

  #if DEBUG >= 3  
      print_map(stdout, tid_A,0);
      print_map(stdout, tid_B,0);
  #endif
      if (!check_sum_mapping(tid_A, idx_map_A, tid_B, idx_map_B)) continue;
      set_padding(tsr_A);
      set_padding(tsr_B);
      size = tsr_A->size + tsr_B->size;

      need_remap_A = 0;
      need_remap_B = 0;

      if (tsr_A->itopo == old_topo_A){
        for (d=0; d<tsr_A->order; d++){
          if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
            need_remap_A = 1;
        }
      } else
        need_remap_A = 1;
      if (need_remap_A){
        if (can_block_reshuffle(tsr_A->order, old_phase_A, tsr_A->edge_map)){
          size += tsr_A->size*log2(global_comm.np);
        } else {
          size += 5.*tsr_A->size*log2(global_comm.np);
        }
      }
      if (tsr_B->itopo == old_topo_B){
        for (d=0; d<tsr_B->order; d++){
          if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
            need_remap_B = 1;
        }
      } else
        need_remap_B = 1;
      if (need_remap_B){
        if (can_block_reshuffle(tsr_B->order, old_phase_B, tsr_B->edge_map)){
          size += tsr_B->size*log2(global_comm.np);
        } else {
          size += 5.*tsr_B->size*log2(global_comm.np);
        }
      }

      /*nvirt = (uint64_t)calc_nvirt(tsr_A);
      tnvirt = nvirt*(uint64_t)calc_nvirt(tsr_B);
      if (tnvirt < nvirt) nvirt = UINT64_MAX;
      else nvirt = tnvirt;
      if (btopo == -1 || nvirt < bnvirt ) {
        bnvirt = nvirt;
        btopo = i;      
      }*/
      if (btopo == -1 || size < min_size){
        min_size = size;
        btopo = i;      
      }
    }
    if (btopo == -1)
      min_size = UINT64_MAX;
    /* pick lower dimensional mappings, if equivalent */
    gtopo = get_best_topo(min_size, btopo, global_comm);
    TAU_FSTOP(map_tensor_pair);
    if (gtopo == -1){
      printf("ERROR: Failed to map pair!\n");
      ABORT;
      return ERROR;
    }
    
    clear_mapping(tsr_A);
    clear_mapping(tsr_B);
    set_padding(tsr_A);
    set_padding(tsr_B);

    tsr_A->itopo = gtopo/2;
    tsr_B->itopo = gtopo/2;
      
    if (gtopo%2 == 0){
      ret = map_self_indices(tid_A, idx_map_A);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(tid_B, idx_map_B);
      ASSERT(ret == SUCCESS);
    }
    ret = map_sum_indices(idx_arr, idx_sum, num_tot, num_sum, 
                          tid_A, tid_B, &topovec[tsr_A->itopo], 2);
    ASSERT(ret == SUCCESS);

    if (gtopo%2 == 0){
      ret = map_self_indices(tid_A, idx_map_A);
      ASSERT(ret == SUCCESS);
      ret = map_tensor_rem(topovec[tsr_A->itopo].order, 
                           topovec[tsr_A->itopo].dim_comm, tsr_A);
      ASSERT(ret == SUCCESS);
      copy_mapping(tsr_A->order, tsr_B->order,
                   idx_map_A, tsr_A->edge_map, 
                   idx_map_B, tsr_B->edge_map,0);
      ret = map_tensor_rem(topovec[tsr_B->itopo].order, 
                           topovec[tsr_B->itopo].dim_comm, tsr_B);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(tid_B, idx_map_B);
      ASSERT(ret == SUCCESS);
      ret = map_tensor_rem(topovec[tsr_B->itopo].order, 
                           topovec[tsr_B->itopo].dim_comm, tsr_B);
      ASSERT(ret == SUCCESS);
      copy_mapping(tsr_B->order, tsr_A->order,
                   idx_map_B, tsr_B->edge_map, 
                   idx_map_A, tsr_A->edge_map,0);
      ret = map_tensor_rem(topovec[tsr_A->itopo].order, 
                           topovec[tsr_A->itopo].dim_comm, tsr_A);
      ASSERT(ret == SUCCESS);
    }

    tsr_A->is_mapped = 1;
    tsr_B->is_mapped = 1;


    set_padding(tsr_A);
    set_padding(tsr_B);
  #if DEBUG >= 2
    if (global_comm.rank == 0)
      printf("New mappings:\n");
    print_map(stdout, tid_A);
    print_map(stdout, tid_B);
  #endif

    TAU_FSTART(redistribute_for_sum);
   
    tsr_A->is_cyclic = 1;
    tsr_B->is_cyclic = 1;
    need_remap = 0;
    if (tsr_A->itopo == old_topo_A){
      for (d=0; d<tsr_A->order; d++){
        if (!comp_dim_map(&tsr_A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      remap_tensor(tid_A, tsr_A, &topovec[tsr_A->itopo], old_size_A, old_phase_A, old_rank_A, old_virt_dim_A, 
                   old_pe_lda_A, was_cyclic_A, old_padding_A, old_edge_len_A, global_comm);   
    need_remap = 0;
    if (tsr_B->itopo == old_topo_B){
      for (d=0; d<tsr_B->order; d++){
        if (!comp_dim_map(&tsr_B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      remap_tensor(tid_B, tsr_B, &topovec[tsr_B->itopo], old_size_B, old_phase_B, old_rank_B, old_virt_dim_B, 
                   old_pe_lda_B, was_cyclic_B, old_padding_B, old_edge_len_B, global_comm);   

    TAU_FSTOP(redistribute_for_sum);
    CTF_free(idx_sum);
    CTF_free(old_phase_A);
    CTF_free(old_rank_A);
    CTF_free(old_virt_dim_A);
    CTF_free(old_pe_lda_A);
    CTF_free(old_padding_A);
    CTF_free(old_edge_len_A);
    CTF_free(old_phase_B);
    CTF_free(old_rank_B);
    CTF_free(old_virt_dim_B);
    CTF_free(old_pe_lda_B);
    CTF_free(old_padding_B);
    CTF_free(old_edge_len_B);
    CTF_free(idx_arr);
    for (i=0; i<tsr_A->order; i++){
      clear_mapping(old_map_A+i);
    }
    for (i=0; i<tsr_B->order; i++){
      clear_mapping(old_map_B+i);
    }
    CTF_free(old_map_A);
    CTF_free(old_map_B);

    return SUCCESS;
  }
}
