#include "summation.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_sum.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include "../scaling/scaling.h"

namespace CTF_int {

  using namespace CTF;

  summation::~summation(){
    if (idx_A != NULL) cdealloc(idx_A);
    if (idx_B != NULL) cdealloc(idx_B);
  }

  summation::summation(summation const & other){
    A     = other.A;
    idx_A = (int*)alloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B     = other.B;
    idx_B = (int*)alloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    func      = other.func;
    is_custom = other.is_custom;
    alpha = other.alpha;
    beta  = other.beta;
  }

  summation::summation(tensor *     A_,
                       int const *  idx_A_,
                       char const * alpha_,
                       tensor *     B_,
                       int const *  idx_B_,
                       char const * beta_){
    A         = A_;
    alpha     = alpha_;
    B         = B_;
    beta      = beta_;
    is_custom = 0;
    func      = NULL;

    idx_A     = (int*)alloc(sizeof(int)*A->order);
    idx_B     = (int*)alloc(sizeof(int)*B->order);

    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
  }

  summation::summation(tensor *     A_,
                       char const * cidx_A,
                       char const * alpha_,
                       tensor *     B_,
                       char const * cidx_B,
                       char const * beta_){
    A         = A_;
    alpha     = alpha_;
    B         = B_;
    beta      = beta_;
    is_custom = 0;
    func      = NULL;
    
    conv_idx(A->order, cidx_A, &idx_A, B->order, cidx_B, &idx_B);
  }

 
  summation::summation(tensor *                A_,
                       int const *             idx_A_,
                       char const *            alpha_,
                       tensor *                B_,
                       int const *             idx_B_,
                       char const *            beta_,
                       univar_function const * func_){
    A         = A_;
    alpha     = alpha_;
    B         = B_;
    beta      = beta_;
    func      = func_;
    if (func == NULL)
      is_custom = 0;
    else
      is_custom = 1;

    idx_A     = (int*)alloc(sizeof(int)*A->order);
    idx_B     = (int*)alloc(sizeof(int)*B->order);

    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
  }

 
  summation::summation(tensor *                A_,
                       char const *            cidx_A,
                       char const *            alpha_,
                       tensor *                B_,
                       char const *            cidx_B,
                       char const *            beta_,
                       univar_function const * func_){
    A         = A_;
    alpha     = alpha_;
    B         = B_;
    beta      = beta_;
    func      = func_;
    if (func == NULL)
      is_custom = 0;
    else
      is_custom = 1;

    conv_idx(A->order, cidx_A, &idx_A, B->order, cidx_B, &idx_B);
  }

  void summation::execute(bool run_diag){
#if (DEBUG >= 2 || VERBOSE >= 1)
  #if DEBUG >= 2
    if (A->wrld->cdt.rank == 0) printf("Summation::execute (head):\n");
  #endif
    print();
#endif
    //update_all_models(A->wrld->cdt.cm);
    int stat = home_sum_tsr(run_diag);
    if (stat != SUCCESS){
      printf("CTF ERROR: Failed to perform summation\n");
      ASSERT(0);
    }
  }
  
  double summation::estimate_time(){
    int np = std::max(A->wrld->np,B->wrld->np);
    double flop_rate = 1.E9*np;
    double flops = 0.;
    if (A->is_sparse)
      flops += A->nnz_tot;
    else
      flops += A->size*np;
    if (B->is_sparse)
      flops += B->nnz_tot;
    else
      flops += B->size*np;
    return flops/flop_rate;
  }

  void summation::get_fold_indices(int *  num_fold,
                                   int ** fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, inA, inB, iiA, iiB;
    int * idx_arr, * idx;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);
    CTF_int::alloc_ptr(num_tot*sizeof(int), (void**)&idx);

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
      //    printf("index in = %d inA = %d inB = %d is broken symA = %d symB = %d\n",in, inA, inB, A->sym[inA], B->sym[inB]);
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
    CTF_int::cdealloc(idx_arr);
  }

  int summation::can_fold(){
    int i, j, nfold, * fold_idx;
    //FIXME: fold sparse tensors into CSR form
    if (A->is_sparse || B->is_sparse) return 0;

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
    CTF_int::cdealloc(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }
 
  void summation::get_fold_sum(summation *& fold_sum,
                               int &        all_fdim_A,
                               int &        all_fdim_B,
                               int64_t *&   all_flen_A,
                               int64_t *&   all_flen_B){
    int i, j, nfold, nf;
    int * fold_idx, * fidx_map_A, * fidx_map_B;
    tensor * ftsr_A, * ftsr_B;

    get_fold_indices(&nfold, &fold_idx);
    if (nfold == 0){
      CTF_int::cdealloc(fold_idx);
      assert(0);
    }

    /* overestimate this space to not bother with it later */
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_A);
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_map_B);


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

    int * sidx_A, * sidx_B;
    CTF_int::conv_idx<int>(ftsr_A->order, fidx_map_A, &sidx_A,
                           ftsr_B->order, fidx_map_B, &sidx_B);
    
    cdealloc(fidx_map_A);
    cdealloc(fidx_map_B);
    cdealloc(fold_idx);

    fold_sum = new summation(A->rec_tsr, sidx_A, alpha, B->rec_tsr, sidx_B, beta);
    cdealloc(sidx_A);
    cdealloc(sidx_B);
  }

  int summation::map_fold(){
    int i, all_fdim_A, all_fdim_B;
    int * fnew_ord_A, * fnew_ord_B;
    int64_t * all_flen_A, * all_flen_B;
    int inr_stride;

    summation * fold_sum;
    get_fold_sum(fold_sum, all_fdim_A, all_fdim_B, all_flen_A, all_flen_B);
  #if DEBUG>=2
    if (A->wrld->rank == 0){
      printf("Folded summation type:\n");
    }
    fold_sum->print();//print_sum(&fold_type,0.0,0.0);
  #endif
   
    //for type order 1 to 3 
    fold_sum->get_len_ordering(&fnew_ord_A, &fnew_ord_B); 
    permute_target(fold_sum->A->order, fnew_ord_A, A->inner_ordering);
    permute_target(fold_sum->B->order, fnew_ord_B, B->inner_ordering);
    

    nosym_transpose(A, all_fdim_A, all_flen_A, A->inner_ordering, 1);
    /*for (i=0; i<nvirt_A; i++){
      nosym_transpose(all_fdim_A, A->inner_ordering, all_flen_A, 
                      A->data + A->sr->el_size*i*(A->size/nvirt_A), 1, A->sr);
    }*/
    nosym_transpose(B, all_fdim_B, all_flen_B, B->inner_ordering, 1);
    /*for (i=0; i<nvirt_B; i++){
      nosym_transpose(all_fdim_B, B->inner_ordering, all_flen_B, 
                      B->data + B->sr->el_size*i*(B->size/nvirt_B), 1, B->sr);
    }*/

    inr_stride = 1;
    for (i=0; i<fold_sum->A->order; i++){
      inr_stride *= fold_sum->A->pad_edge_len[i];
    }

    CTF_int::cdealloc(fnew_ord_A);
    CTF_int::cdealloc(fnew_ord_B);
    CTF_int::cdealloc(all_flen_A);
    CTF_int::cdealloc(all_flen_B);

    delete fold_sum;

    return inr_stride; 
  }

  double summation::est_time_fold(){
    int all_fdim_A, all_fdim_B;
    int * fnew_ord_A, * fnew_ord_B;
    int64_t * all_flen_A, * all_flen_B;
    int * tAiord, * tBiord;

    summation * fold_sum;
    get_fold_sum(fold_sum, all_fdim_A, all_fdim_B, all_flen_A, all_flen_B);
    fold_sum->get_len_ordering(&fnew_ord_A, &fnew_ord_B); 
    CTF_int::alloc_ptr(all_fdim_A*sizeof(int), (void**)&tAiord);
    CTF_int::alloc_ptr(all_fdim_B*sizeof(int), (void**)&tBiord);
    memcpy(tAiord, A->inner_ordering, all_fdim_A*sizeof(int));
    memcpy(tBiord, B->inner_ordering, all_fdim_B*sizeof(int));

    permute_target(fold_sum->A->order, fnew_ord_A, tAiord);
    permute_target(fold_sum->B->order, fnew_ord_B, tBiord);
  
    A->is_folded = 0; 
    delete A->rec_tsr; 
    cdealloc(A->inner_ordering); 
    B->is_folded = 0; 
    delete B->rec_tsr; 
    cdealloc(B->inner_ordering); 

    double esttime = 0.0;

    esttime += A->calc_nvirt()*est_time_transp(all_fdim_A, tAiord, all_flen_A, 1, A->sr);
    esttime += 2.*B->calc_nvirt()*est_time_transp(all_fdim_B, tBiord, all_flen_B, 1, B->sr);

    delete fold_sum;

    cdealloc(all_flen_A);
    cdealloc(all_flen_B);
    cdealloc(tAiord);
    cdealloc(tBiord);
    cdealloc(fnew_ord_A);
    cdealloc(fnew_ord_B);
    return esttime;
  }


 
  void summation::get_len_ordering(int ** new_ordering_A,
                                   int ** new_ordering_B){
    int num_tot;
    int * ordering_A, * ordering_B, * idx_arr;
    
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&ordering_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&ordering_B);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);
    ASSERT(num_tot == A->order);
    ASSERT(num_tot == B->order);
    /*for (i=0; i<num_tot; i++){
      ordering_A[i] = idx_arr[2*i];
      ordering_B[i] = idx_arr[2*i+1];
    }*/
    for (int i=0; i<num_tot; i++){
      ordering_B[i] = i;
      for (int j=0; j<num_tot; j++){
        if (idx_A[j] == idx_B[i])
          ordering_A[i] = j;
      }
    }
    CTF_int::cdealloc(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
  }


  tspsum * summation::construct_sparse_sum(int const * phys_mapped){
    int nvirt, i, iA, iB, order_tot, is_top, need_rep;
    int64_t blk_sz_A, blk_sz_B, vrt_sz_A, vrt_sz_B;
    int nphys_dim;
    int * virt_dim;
    int * idx_arr;
    int64_t * virt_blk_len_A, * virt_blk_len_B;
    int64_t * blk_len_A, * blk_len_B;
    mapping * map;
    tspsum * htsum = NULL , ** rec_tsum = NULL;

    is_top = 1;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &order_tot, &idx_arr);

    nphys_dim = A->topo->order;

    CTF_int::alloc_ptr(sizeof(int)*order_tot,   (void**)&virt_dim);
    CTF_int::alloc_ptr(sizeof(int64_t)*A->order,    (void**)&blk_len_A);
    CTF_int::alloc_ptr(sizeof(int64_t)*B->order,    (void**)&blk_len_B);
    CTF_int::alloc_ptr(sizeof(int64_t)*A->order,    (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int64_t)*B->order,    (void**)&virt_blk_len_B);

    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = A->size;
    blk_sz_B = B->size;
    calc_dim(A->order, blk_sz_A, A->pad_edge_len, A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(B->order, blk_sz_B, B->pad_edge_len, B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);

    nvirt = 1;
    for (i=0; i<order_tot; i++){
      iA = idx_arr[2*i];
      iB = idx_arr[2*i+1];
      if (iA != -1){
        map = &A->edge_map[iA];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
        }
        else virt_dim[i] = 1;
      } else {
        ASSERT(iB!=-1);
        map = &B->edge_map[iB];
        while (map->has_child) map = map->child;
        if (map->type == VIRTUAL_MAP){
          virt_dim[i] = map->np;
        }
        else virt_dim[i] = 1;
      }
      nvirt *= virt_dim[i];
    }



    if (A->is_sparse){
      if (A->wrld->np > 1){
        tspsum_pin_keys * sksum = new tspsum_pin_keys(this, 1);
        if (is_top){
          htsum = sksum;
          is_top = 0;
        } else {
          *rec_tsum = sksum;
        }
        rec_tsum = &sksum->rec_tsum;
      }

      tspsum_permute * pmsum = new tspsum_permute(this, 1, virt_blk_len_A);
      if (is_top){
        htsum = pmsum;
        is_top = 0;
      } else {
        *rec_tsum = pmsum;
      }
      rec_tsum = &pmsum->rec_tsum;
    }
    if (B->is_sparse){
      if (B->wrld->np > 1){
        tspsum_pin_keys * sksum = new tspsum_pin_keys(this, 0);
        if (is_top){
          htsum = sksum;
          is_top = 0;
        } else {
          *rec_tsum = sksum;
        }
        rec_tsum = &sksum->rec_tsum;
      }

      tspsum_permute * pmsum = new tspsum_permute(this, 0, virt_blk_len_B);
      if (is_top){
        htsum = pmsum;
        is_top = 0;
      } else {
        *rec_tsum = pmsum;
      }
      rec_tsum = &pmsum->rec_tsum;
    }

/*    bool need_sp_map = false;
    if (A->is_sparse || B->is_sparse){
      for (int i=0; i<B->order; i++){
        bool found_match = false;
        for (int j=0; j<A->order; j++){
          if (idx_B[i] == idx_A[j]) found_match = true;
        }
        if (!found_match) need_sp_map = true;
      }
    }

    if (need_sp_map){
      tspsum_map * smsum = new tspsum_map(this);
      if (is_top){
        htsum = smsum;
        is_top = 0;
      } else {
        *rec_tsum = smsum;
      }
      rec_tsum = &smsum->rec_tsum;
    }*/


    need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 ||
          phys_mapped[2*i+1] == 0){
        need_rep = 1;
        break;
      }
    }

    if (need_rep){
/*      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Replicating tensor\n");*/

      tspsum_replicate * rtsum = new tspsum_replicate(this, phys_mapped, blk_sz_A, blk_sz_B);

      if (is_top){
        htsum = rtsum;
        is_top = 0;
      } else {
        *rec_tsum = rtsum;
      }
      rec_tsum      = &rtsum->rec_tsum;
    }

    /* Multiply over virtual sub-blocks */
    tspsum_virt * tsumv = new tspsum_virt(this);
    if (is_top) {
      htsum = tsumv;
      is_top = 0;
    } else {
      *rec_tsum = tsumv;
    }
    rec_tsum         = &tsumv->rec_tsum;

    tsumv->num_dim   = order_tot;
    tsumv->virt_dim  = virt_dim;
    tsumv->blk_sz_A  = vrt_sz_A;
    tsumv->blk_sz_B  = vrt_sz_B;

    int * new_sym_A, * new_sym_B;
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, A->sym, sizeof(int)*A->order);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, B->sym, sizeof(int)*B->order);

    seq_tsr_spsum * tsumseq = new seq_tsr_spsum(this);
    tsumseq->is_inner = 0;
    tsumseq->edge_len_A  = virt_blk_len_A;
    tsumseq->sym_A       = new_sym_A;
    tsumseq->edge_len_B  = virt_blk_len_B;
    tsumseq->sym_B       = new_sym_B;
    tsumseq->is_custom   = is_custom;
    if (is_custom){
      tsumseq->is_inner  = 0;
      tsumseq->func      = func;
    } else tsumseq->func = NULL;

    if (is_top) {
      htsum = tsumseq;
      is_top = 0;
    } else {
      *rec_tsum = tsumseq;
    }

    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(blk_len_A);
    CTF_int::cdealloc(blk_len_B);
    return htsum;
  }

  tsum * summation::construct_dense_sum(int         inner_stride,
                                        int const * phys_mapped){
    int i, iA, iB, order_tot, is_top, sA, sB, need_rep, i_A, i_B, j, k;
    int64_t blk_sz_A, blk_sz_B, vrt_sz_A, vrt_sz_B;
    int nphys_dim, nvirt;
    int * idx_arr, * virt_dim;
    int64_t * virt_blk_len_A, * virt_blk_len_B;
    int64_t * blk_len_A, * blk_len_B;
    tsum * htsum = NULL , ** rec_tsum = NULL;
    mapping * map;
    strp_tsr * str_A, * str_B;

    is_top = 1;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &order_tot, &idx_arr);

    nphys_dim = A->topo->order;

    CTF_int::alloc_ptr(sizeof(int)*order_tot,   (void**)&virt_dim);
    CTF_int::alloc_ptr(sizeof(int64_t)*A->order,    (void**)&blk_len_A);
    CTF_int::alloc_ptr(sizeof(int64_t)*B->order,    (void**)&blk_len_B);
    CTF_int::alloc_ptr(sizeof(int64_t)*A->order,    (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int64_t)*B->order,    (void**)&virt_blk_len_B);

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
      strp_sum * ssum = new strp_sum(this);
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

    need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[2*i+0] == 0 ||
          phys_mapped[2*i+1] == 0){
        need_rep = 1;
        break;
      }
    }

    if (need_rep){
/*      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Replicating tensor\n");*/

      tsum_replicate * rtsum = new tsum_replicate(this, phys_mapped, blk_sz_A, blk_sz_B);

      if (is_top){
        htsum = rtsum;
        is_top = 0;
      } else {
        *rec_tsum = rtsum;
      }
      rec_tsum      = &rtsum->rec_tsum;
    }

    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
      tsum_virt * tsumv = new tsum_virt(this);
      if (is_top) {
        htsum = tsumv;
        is_top = 0;
      } else {
        *rec_tsum = tsumv;
      }
      rec_tsum         = &tsumv->rec_tsum;

      tsumv->num_dim   = order_tot;
      tsumv->virt_dim  = virt_dim;
      tsumv->blk_sz_A  = vrt_sz_A;
      tsumv->blk_sz_B  = vrt_sz_B;
    } else CTF_int::cdealloc(virt_dim);
    int * new_sym_A, * new_sym_B;
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&new_sym_A);
    memcpy(new_sym_A, A->sym, sizeof(int)*A->order);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&new_sym_B);
    memcpy(new_sym_B, B->sym, sizeof(int)*B->order);

    seq_tsr_sum * tsumseq = new seq_tsr_sum(this);
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
    tsumseq->edge_len_A  = virt_blk_len_A;
    tsumseq->sym_A       = new_sym_A;
    tsumseq->edge_len_B  = virt_blk_len_B;
    tsumseq->sym_B       = new_sym_B;
    tsumseq->is_custom   = is_custom;
    if (is_custom){
      tsumseq->is_inner  = 0;
      tsumseq->func      = func;
    } else tsumseq->func = NULL;
    if (is_top) {
      htsum = tsumseq;
      is_top = 0;
    } else {
      *rec_tsum = tsumseq;
    }

    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(blk_len_A);
    CTF_int::cdealloc(blk_len_B);
    return htsum;

  }


  tsum * summation::construct_sum(int inner_stride){
    int i;
    int nphys_dim;
    int * phys_mapped;
    tsum * htsum;
    mapping * map;

    nphys_dim = A->topo->order;

    CTF_int::alloc_ptr(sizeof(int)*nphys_dim*2, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*2);



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
    if (A->is_sparse || B->is_sparse){
      htsum = construct_sparse_sum(phys_mapped);
    } else {
      htsum = construct_dense_sum(inner_stride, phys_mapped);
    }

    CTF_int::cdealloc(phys_mapped);

    return htsum;
  }

  int summation::home_sum_tsr(bool run_diag, bool handle_sym){
    int ret, was_home_A, was_home_B;
    tensor * tnsr_A, * tnsr_B;
    // code below turns summations into scaling, but never seems to be invoked in AQ or test_suite, so commenting it out for now
/*    if (A==B && !is_custom){
      bool is_scal = true;
      for (int i=0; i<A->order; i++){
        if (idx_A[i] != idx_B[i]) is_scal = false;
      }
      if (is_scal){
        if (alpha == NULL && beta == NULL){
          scaling scl = scaling(A, idx_A, NULL);
          scl.execute();
        } else {
          char nalpha[A->sr->el_size];
          if (alpha == NULL) A->sr->copy(nalpha, A->sr->mulid());
          else A->sr->copy(nalpha, alpha);

          if (beta == NULL) A->sr->add(nalpha, A->sr->mulid(), nalpha);
          else A->sr->add(nalpha, beta, nalpha);

          scaling scl = scaling(A, idx_A, nalpha);
          scl.execute();
        }
        return SUCCESS;
      }
    }*/

    summation osum = summation(*this);
   
    A->unfold();
    B->unfold();
    // FIXME: if custom function, we currently don't know whether its odd, even or neither, so unpack everything
    /*if (is_custom){
      bool is_nonsym=true;
      for (int i=0; i<A->order; i++){
        if (A->sym[i] != NS){
          is_nonsym = false;
        }
      }
      if (!is_nonsym){
        int sym_A[A->order];
        std::fill(sym_A, sym_A+A->order, NS);
        int idx_A[A->order];
        for (int i=0; i<A->order; i++){
          idx_A[i] = i;
        }
        tensor tA(A->sr, A->order, A->lens, sym_A, A->wrld, 1);
        tA.is_home = 0;
        tA.has_home = 0;
        summation st(A, idx_A, A->sr->mulid(), &tA, idx_A, A->sr->mulid());
        st.execute();
        summation stme(*this);
        stme.A = &tA;
        stme.execute();
        return SUCCESS;
      }
    }
    if (is_custom){
      bool is_nonsym=true;
      for (int i=0; i<B->order; i++){
        if (B->sym[i] != NS){
          is_nonsym = false;
        }
      }
      if (!is_nonsym){
        int sym_B[B->order];
        std::fill(sym_B, sym_B+B->order, NS);
        int idx_B[B->order];
        for (int i=0; i<B->order; i++){
          idx_B[i] = i;
        }
        tensor tB(B->sr, B->order, B->lens, sym_B, B->wrld, 1);
        tB.is_home = 0;
        tB.has_home = 0;
        if (!B->sr->isequal(B->sr->addid(), beta)){
          summation st(B, idx_B, B->sr->mulid(), &tB, idx_B, B->sr->mulid());
          st.execute();
        }
        summation stme(*this);
        stme.B = &tB;
        stme.execute();
        summation stme2(&tB, idx_B, B->sr->mulid(), B, idx_B, B->sr->addid());
        stme2.execute();
        return SUCCESS;
      }
    }*/

  #ifndef HOME_CONTRACT
    #ifdef USE_SYM_SUM
      if (handle_sym)
        ret = sym_sum_tsr(run_diag);
      else
        ret = sum_tensors(run_diag);
      return ret;
    #else
      ret = sum_tensors(run_diag);
      return ret;
    #endif
  #else
    if (A->has_zero_edge_len || 
        B->has_zero_edge_len){
      if (!B->sr->isequal(beta,B->sr->mulid()) && !B->has_zero_edge_len){ 
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
      delete cpy_tsr_A;
      return SUCCESS;
    }
    if (A->order == 0 && B->order == 0){
      if (B->wrld->rank != 0){
        if (B->nnz_tot == 0 && A->nnz_tot == 0){
          int64_t nnz_blk = 0;
          B->set_new_nnz_glb(&nnz_blk);
        }
      } else {
        char tmp_A[A->sr->el_size];
        char * op_A;
        char * op_B;
        bool has_op = true;
        if (A->is_sparse){
          if (A->nnz_tot == 0){
            has_op=false;
          } else {
            PairIterator prs(A->sr,A->data);
            op_A = prs[0].d();
          }
        } else {
          op_A = A->data;
        }
        if (alpha != NULL){
          A->sr->mul(op_A, alpha, tmp_A);
          op_A = tmp_A;
        }
        if (B->is_sparse){
          if (B->nnz_tot == 0){
            if (!has_op) return SUCCESS;
            B->data = B->sr->pair_alloc(1);
            B->sr->set_pair(B->data, 0, B->sr->addid());
            int64_t nnz_blk = 1;
            B->set_new_nnz_glb(&nnz_blk);
          }
          PairIterator prs(B->sr,B->data);
          op_B = prs[0].d();
        } else {
          op_B = B->data;
        }

        if (beta != NULL)
          B->sr->mul(op_B, beta, op_B);
        if (has_op){
          if (is_custom){
            func->acc_f(op_A,op_B,B->sr); 
          } else {
            B->sr->add(op_A,op_B,B->data); 
          }
        }
      }
      return SUCCESS;
    }
    was_home_A = A->is_home;
    was_home_B = B->is_home;
    if (was_home_A){
      tnsr_A              = new tensor(A,0,0);
      tnsr_A->data        = A->data;
      tnsr_A->home_buffer = A->home_buffer;
      tnsr_A->is_home     = 1;
      tnsr_A->has_home    = 1;
      tnsr_A->home_size   = A->home_size;
      tnsr_A->is_mapped   = 1;
      tnsr_A->topo        = A->topo;
      copy_mapping(A->order, A->edge_map, tnsr_A->edge_map);
      tnsr_A->set_padding();
      if (A->is_sparse){
        CTF_int::alloc_ptr(tnsr_A->calc_nvirt()*sizeof(int64_t), (void**)&tnsr_A->nnz_blk);
        tnsr_A->set_new_nnz_glb(A->nnz_blk);
      }
      osum.A              = tnsr_A;
    } else tnsr_A = NULL;     
    if (was_home_B){
      tnsr_B              = new tensor(B,0,0);
      tnsr_B->data        = B->data;
      tnsr_B->home_buffer = B->home_buffer;
      tnsr_B->is_home     = 1;
      tnsr_B->has_home    = 1;
      tnsr_B->home_size   = B->home_size;
      tnsr_B->is_mapped   = 1;
      tnsr_B->topo        = B->topo;
      copy_mapping(B->order, B->edge_map, tnsr_B->edge_map);
      tnsr_B->set_padding();
      if (B->is_sparse){
        CTF_int::alloc_ptr(tnsr_B->calc_nvirt()*sizeof(int64_t), (void**)&tnsr_B->nnz_blk);
        tnsr_B->set_new_nnz_glb(B->nnz_blk);
      }
      osum.B              = tnsr_B;
    } else tnsr_B = NULL;
  #if DEBUG >= 2
    if (A->wrld->cdt.rank == 0)
      printf("Start head sum:\n");
  #endif
    
    #ifdef USE_SYM_SUM
    if (handle_sym)
      ret = osum.sym_sum_tsr(run_diag);
    else
      ret = osum.sum_tensors(run_diag);
    #else
    ret = osum.sum_tensors(run_diag);
    #endif
  #if DEBUG >= 2
    if (A->wrld->cdt.rank == 0)
      printf("End head sum:\n");
  #endif

    if (ret!= SUCCESS) return ret;
    if (was_home_A) tnsr_A->unfold(); 
    else A->unfold();
    if (was_home_B){
      tnsr_B->unfold();
      if (B->is_sparse){
        cdealloc(B->nnz_blk);
        //do below manually rather than calling set_new_nnz_glb since virt factor may be different
        CTF_int::alloc_ptr(tnsr_B->calc_nvirt()*sizeof(int64_t), (void**)&B->nnz_blk);
        for (int i=0; i<tnsr_B->calc_nvirt(); i++){
          B->nnz_blk[i] = tnsr_B->nnz_blk[i];
        }
        B->nnz_loc = tnsr_B->nnz_loc;
        B->nnz_tot = tnsr_B->nnz_tot;
      } 
      B->data = tnsr_B->data;
    } else B->unfold();

    if (was_home_B && !tnsr_B->is_home){
      if (A->wrld->cdt.rank == 0)
        DPRINTF(1,"Migrating tensor %s back to home\n", B->name);
      distribution odst(tnsr_B);
      B->is_home = 0;
      B->has_home = 0;
      TAU_FSTART(redistribute_for_sum_home);
      B->redistribute(odst);
      TAU_FSTOP(redistribute_for_sum_home);
      if (!B->is_sparse){
        B->sr->copy(B->home_buffer, B->data, B->size);
        B->sr->dealloc(B->data);
        B->data = B->home_buffer;
      } else if (tnsr_B->home_buffer != NULL) {
        tnsr_B->sr->pair_dealloc(tnsr_B->home_buffer);
        tnsr_B->home_buffer = NULL;
      }
      tnsr_B->is_data_aliased = 1;
      tnsr_B->is_home = 0;
      tnsr_B->has_home = 0;
      B->is_home = 1;
      B->has_home = 1;
      delete tnsr_B;
    } else if (was_home_B){
      if (!B->is_sparse){
        if (tnsr_B->data != B->data){
          printf("Tensor %s is a copy of %s and did not leave home but buffer is %p was %p\n", tnsr_B->name, B->name, tnsr_B->data, B->data);
          ABORT;
        }
      } else
        B->data = tnsr_B->data;
      tnsr_B->has_home = 0;
      tnsr_B->is_home = 0;
      tnsr_B->is_data_aliased = 1;
      delete tnsr_B;
    }
    if (was_home_A && !tnsr_A->is_home){
      if (A->is_sparse){
        A->data = tnsr_A->home_buffer;
        tnsr_A->home_buffer = NULL;
        tnsr_A->has_home = 1;
        tnsr_A->is_home = 1;
      } else {
        tnsr_A->has_home = 0;
        tnsr_A->is_home = 0;
      }
      delete tnsr_A;
    } else if (was_home_A) {
      tnsr_A->has_home = 0;
      tnsr_A->is_home = 0;
      tnsr_A->is_data_aliased = 1;
      if (A->is_sparse)
        A->data = tnsr_A->data;
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
  #if (DEBUG >= 2)
    print();
  #endif
    bool is_cons = check_consistency();
    if (!is_cons) return ERROR;

    A->unfold();
    B->unfold();
    if (A->has_zero_edge_len || B->has_zero_edge_len){
      if (!B->sr->isequal(beta, B->sr->mulid()) && !B->has_zero_edge_len){ 
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



    int * new_idx_A, * new_idx_B;
    if (!is_custom || func->is_distributive){
      tensor * new_tsr_A = A->self_reduce(idx_A, &new_idx_A, B->order, idx_B, &new_idx_B);
      if (new_tsr_A != A) {
        summation s(new_tsr_A, new_idx_A, alpha, B, new_idx_B, beta, func);
        s.execute();
        delete new_tsr_A;
        cdealloc(new_idx_A);
        cdealloc(new_idx_B);
        return SUCCESS;
      }
    }

    // If we have sparisity, use separate mechanism
    /*if (A->is_sparse || B->is_sparse){
      sp_sum();
      return SUCCESS;
    }*/
    tnsr_A = A;
    tnsr_B = B;
    char * new_alpha = (char*)alloc(tnsr_B->sr->el_size);
    CTF_int::alloc_ptr(sizeof(int)*tnsr_A->order,     (void**)&map_A);
    CTF_int::alloc_ptr(sizeof(int)*tnsr_B->order,     (void**)&map_B);
    CTF_int::alloc_ptr(sizeof(int*)*tnsr_B->order,    (void**)&dstack_map_B);
    CTF_int::alloc_ptr(sizeof(tensor*)*tnsr_B->order, (void**)&dstack_tsr_B);
    memcpy(map_A, idx_A, tnsr_A->order*sizeof(int));
    memcpy(map_B, idx_B, tnsr_B->order*sizeof(int));
    while (!run_diag && tnsr_A->extract_diag(map_A, 1, new_tsr, &new_idx_map) == SUCCESS){
      if (tnsr_A != A) delete tnsr_A;
      CTF_int::cdealloc(map_A);
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

    summation new_sum = summation(*this);
    new_sum.A = tnsr_A;
    new_sum.B = tnsr_B;
    memcpy(new_sum.idx_A, map_A, sizeof(int)*tnsr_A->order);
    memcpy(new_sum.idx_B, map_B, sizeof(int)*tnsr_B->order);
    if (tnsr_A == tnsr_B){
      tensor nnew_tsr = tensor(tnsr_A);
      new_sum.A = &nnew_tsr;
      new_sum.B = tnsr_B;
      new_sum.sym_sum_tsr(run_diag);
      
      /*clone_tensor(ntid_A, 1, &new_tid);
      new_type = *type;
      new_type.tid_A = new_tid;
      stat = sym_sum_tsr(alpha_, beta, &new_type, ftsr, felm, run_diag);
      del_tsr(new_tid);
      return stat;*/
    } else {
      
  /*    new_type.tid_A = ntid_A;
      new_type.tid_B = ntid_B;
      new_type.idx_map_A = map_A;
      new_type.idx_map_B = map_B;*/
  
      //FIXME: make these typefree...
      int sign = align_symmetric_indices(tnsr_A->order,
                                         new_sum.idx_A,
                                         tnsr_A->sym,
                                         tnsr_B->order,
                                         new_sum.idx_B,
                                         tnsr_B->sym);
      int ocfact = overcounting_factor(tnsr_A->order,
                                       new_sum.idx_A,
                                       tnsr_A->sym,
                                       tnsr_B->order,
                                       new_sum.idx_B,
                                       tnsr_B->sym);
  
      if (ocfact != 1 || sign != 1){
        if (ocfact != 1){
          tnsr_B->sr->safecopy(new_alpha, tnsr_B->sr->addid());
          
          for (int i=0; i<ocfact; i++){
            tnsr_B->sr->add(new_alpha, alpha, new_alpha);
          }
          alpha = new_alpha;
        }
        if (sign == -1){
          tnsr_B->sr->safeaddinv(alpha, new_alpha);
          alpha = new_alpha;
        }
      }
  
      /*if (A->sym[0] == SY && B->sym[0] == AS){
        print();
        ASSERT(0); 
      }*/
      if (new_sum.unfold_broken_sym(NULL) != -1){
        if (A->wrld->cdt.rank == 0)
          DPRINTF(1, "Permutational symmetry is broken\n");
  
        summation * unfold_sum;
        sidx = new_sum.unfold_broken_sym(&unfold_sum);
        int sy;
        sy = 0;
        int sidx2 = unfold_sum->unfold_broken_sym(NULL);
        if (sidx%2 == 0 && (A->sym[sidx/2] == SY || unfold_sum->A->sym[sidx/2] == SY)) sy = 1;
        if (sidx%2 == 1 && (B->sym[sidx/2] == SY || unfold_sum->B->sym[sidx/2] == SY)) sy = 1;
        if ((sidx2 != -1 || 
            (sy && (sidx%2 == 0  || !tnsr_B->sr->isequal(new_sum.beta, tnsr_B->sr->addid()))))){
          if (sidx%2 == 0){
            if (unfold_sum->A->sym[sidx/2] == NS){
              if (A->wrld->cdt.rank == 0)
                DPRINTF(1,"Performing operand desymmetrization for summation of A idx=%d\n",sidx/2);
              desymmetrize(tnsr_A, unfold_sum->A, 0);
            } else {
              if (A->wrld->cdt.rank == 0)
                DPRINTF(1,"Performing operand symmetrization for summation\n");
              symmetrize(unfold_sum->A, tnsr_A);
            }
            //unfold_sum->B = tnsr_B;
            unfold_sum->sym_sum_tsr(run_diag);
    //        sym_sum_tsr(alpha, beta, &unfold_type, ftsr, felm, run_diag);
            if (tnsr_A != unfold_sum->A){
              unfold_sum->A->unfold();
              tnsr_A->pull_alias(unfold_sum->A);
              delete unfold_sum->A;
            }
          } else {
            //unfold_sum->A = tnsr_A;
            if (A->wrld->cdt.rank == 0)
              DPRINTF(1,"Performing product desymmetrization for summation\n");
            desymmetrize(tnsr_B, unfold_sum->B, 1);
            unfold_sum->sym_sum_tsr(run_diag);
            if (A->wrld->cdt.rank == 0)
              DPRINTF(1,"Performing product symmetrization for summation\n");
            if (tnsr_B->data != unfold_sum->B->data && !tnsr_B->sr->isequal(tnsr_B->sr->mulid(), unfold_sum->beta)){
              int sidx_B[tnsr_B->order];
              for (int iis=0; iis<tnsr_B->order; iis++){
                sidx_B[iis] = iis;
              }
              scaling sscl = scaling(tnsr_B, sidx_B, unfold_sum->beta);
              sscl.execute();
            }
            symmetrize(tnsr_B, unfold_sum->B);

    //        sym_sum_tsr(alpha, beta, &unfold_type, ftsr, felm, run_diag);
            if (tnsr_B != unfold_sum->B){
              unfold_sum->B->unfold();
              tnsr_B->pull_alias(unfold_sum->B);
              delete unfold_sum->B;
            }
          }
        } else {
          if (sidx != -1 && sidx%2 == 1){
            delete unfold_sum->B;
          } else if (sidx != -1 && sidx%2 == 0){
            delete unfold_sum->A;
          }
          //get_sym_perms(&new_type, alpha, perm_types, signs);
          get_sym_perms(new_sum, perm_types, signs);
          if (A->wrld->cdt.rank == 0)
            DPRINTF(1,"Performing %d summation permutations\n",
                    (int)perm_types.size());
          dbeta = beta;
          char * new_alpha = (char*)alloc(tnsr_B->sr->el_size);

          tensor * inv_tsr_A = NULL;
          bool need_inv = false;
          // if we have no multiplicative operator, must inverse sign manually
          if (tnsr_B->sr->mulid() == NULL){
            for (i=0; i<(int)perm_types.size(); i++){
              if (signs[i] == -1)
                need_inv = true;
            }
            if (need_inv){
              inv_tsr_A = new tensor(tnsr_A);
              inv_tsr_A->addinv();
            }
          }
          for (i=0; i<(int)perm_types.size(); i++){
            // if group apply additive inverse manually
            if (signs[i] == -1 && need_inv){
              perm_types[i].A = inv_tsr_A;
            } else {
              if (signs[i] == 1)
                tnsr_B->sr->safecopy(new_alpha, alpha);
              else 
                tnsr_B->sr->safeaddinv(alpha, new_alpha);
              perm_types[i].alpha = new_alpha;
            }
            perm_types[i].beta = dbeta;
            perm_types[i].sum_tensors(run_diag);
            dbeta = new_sum.B->sr->mulid();
          }
          cdealloc(new_alpha);
          if (need_inv){
            delete inv_tsr_A;
          }
  /*        for (i=0; i<(int)perm_types.size(); i++){
            free_type(&perm_types[i]);
          }*/
          perm_types.clear();
          signs.clear();
        }
        delete unfold_sum;
      } else {
        new_sum.alpha = alpha;
        new_sum.sum_tensors(run_diag);
  /*      sum_tensors(alpha, beta, new_type.tid_A, new_type.tid_B, new_type.idx_map_A,
                    new_type.idx_map_B, ftsr, felm, run_diag);*/
      }
    }
    if (tnsr_A != A) delete tnsr_A;
    for (i=nst_B-1; i>=0; i--){
//      extract_diag(dstack_tid_B[i], dstack_map_B[i], 0, &ntid_B, &new_idx_map);
      dstack_tsr_B[i]->extract_diag(dstack_map_B[i], 0, tnsr_B, &new_idx_map);
      //del_tsr(ntid_B);
      delete tnsr_B;
      cdealloc(dstack_map_B[i]);
      cdealloc(new_idx_map);
      tnsr_B = dstack_tsr_B[i];
    }
    ASSERT(tnsr_B == B);
    CTF_int::cdealloc(new_alpha);
    CTF_int::cdealloc(map_A);
    CTF_int::cdealloc(map_B);
    CTF_int::cdealloc(dstack_map_B);
    CTF_int::cdealloc(dstack_tsr_B);

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
    TAU_FSTART(sum_preprocessing);
    check_consistency();
    A->unfold();
    B->unfold();
    if (A->has_zero_edge_len || B->has_zero_edge_len){
      if (!B->sr->isequal(beta,B->sr->mulid()) && !B->has_zero_edge_len){ 
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
      TAU_FSTOP(sum_preprocessing);
      return SUCCESS;
    }


    //FIXME: remove all of the below, sum_tensors should never be called without sym_sum
    CTF_int::alloc_ptr(sizeof(int)*A->order,  (void**)&map_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order,  (void**)&map_B);
    CTF_int::alloc_ptr(sizeof(int*)*B->order, (void**)&dstack_map_B);
    CTF_int::alloc_ptr(sizeof(tensor*)*B->order, (void**)&dstack_tsr_B);
    tnsr_A = A;
    tnsr_B = B;
    memcpy(map_A, idx_A, tnsr_A->order*sizeof(int));
    memcpy(map_B, idx_B, tnsr_B->order*sizeof(int));
    while (!run_diag && tnsr_A->extract_diag(map_A, 1, new_tsr, &new_idx_map) == SUCCESS){
      if (tnsr_A != A) delete tnsr_A;
      CTF_int::cdealloc(map_A);
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

    if (!tnsr_A->is_sparse && tnsr_B->is_sparse){
      tensor * stnsr_A = tnsr_A;
      tnsr_A = new tensor(stnsr_A);
      tnsr_A->sparsify(); 
      if (A != stnsr_A) delete stnsr_A;

    }

    TAU_FSTOP(sum_preprocessing);

    // FIXME: if A has self indices and function is distributive, presum A first, otherwise if it is dense and B is sparse, sparsify A
    summation new_sum = summation(*this);
    new_sum.A = tnsr_A;
    new_sum.B = tnsr_B;
    if (tnsr_A == tnsr_B){
      tensor * nnew_tsr = new tensor(tnsr_A);
      new_sum.A = nnew_tsr;
      new_sum.B = tnsr_B;
    } else{ 
     //FIXME: remove the below, sum_tensors should never be called without sym_sum
     int sign = align_symmetric_indices(tnsr_A->order,
                                        new_sum.idx_A,
                                        tnsr_A->sym,
                                        tnsr_B->order,
                                        new_sum.idx_B,
                                        tnsr_B->sym);

      #if DEBUG >= 1
        new_sum.print();
      #endif

      ASSERT(sign == 1);
/*        if (sign == -1){
          char * new_alpha = (char*)malloc(tnsr_B->sr->el_size);
          tnsr_B->sr->addinv(alpha, new_alpha);
          alpha = new_alpha;
        }*/

  #if 0 //VERIFY
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
      
      TAU_FSTART(sum_tensors_map);

      /* Check if the current tensor mappings can be summed on */
  #if REDIST
      if (1) {
  #else
      if (new_sum.check_mapping() == 0) {
  #endif
  #if DEBUG == 2
        if (A->wrld->cdt.rank == 0){
          printf("Remapping tensors for sum:\n");
        }
  #endif
        /* remap if necessary */
        stat = new_sum.map();
        if (stat == ERROR) {
          printf("Failed to map tensors to physical grid\n");
          TAU_FSTOP(sum_tensors);
          TAU_FSTOP(sum_tensors_map);
          return ERROR;
        }
      } else {
  #if DEBUG > 2
        if (A->wrld->cdt.rank == 0){
          printf("Keeping mappings:\n");
        }
        tnsr_A->print_map(stdout);
        tnsr_B->print_map(stdout);
  #endif
      }
      /* Construct the tensor algorithm we would like to use */
      ASSERT(new_sum.check_mapping());
  #if FOLD_TSR
      if (is_custom == false && new_sum.can_fold()){
        //FIXME bit of a guess, no?
        //double est_time_nofold = 4.*(A->size + B->size)*COST_MEMBW;
        //if (new_sum.est_time_fold() + (A->size + B->size)*COST_MEMBW < est_time_nofold){
        if (true){
          /*if (A->wrld->cdt.rank == 0)
            printf("Decided to fold\n");*/
          int inner_stride;
          TAU_FSTART(map_fold);
          inner_stride = new_sum.map_fold();
          TAU_FSTOP(map_fold);
          sumf = new_sum.construct_sum(inner_stride);
        } else {
          /*if (A->wrld->cdt.rank == 0)
            printf("Decided not to fold\n");*/
        
          sumf = new_sum.construct_sum();
        }
      } else{
  #if DEBUG >= 1
        if (A->wrld->cdt.rank == 0){
          printf("Could not fold summation, is_custom = %d, new_sum.can_fold = %d\n", is_custom, new_sum.can_fold());
        }
  #endif
        sumf = new_sum.construct_sum();
      }
        /*sumf = construct_sum(alpha, beta, ntid_A, map_A, ntid_B, map_B,
                             ftsr, felm);*/
  #else
      sumf = new_sum.construct_sum();
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
      /*if (A->wrld->cdt.rank == 0){
        for (int i=0; i<tensors[ntid_A]->order; i++){
          printf("padding[%d] = %d\n",i, tensors[ntid_A]->padding[i]);
        }
        for (int i=0; i<tensors[ntid_B]->order; i++){
          printf("padding[%d] = %d\n",i, tensors[ntid_B]->padding[i]);
        }
      }*/
  #endif

      TAU_FSTOP(sum_tensors_map);
  #if DEBUG >= 2
      if (tnsr_B->wrld->rank==0)
        sumf->print();
      tnsr_A->print_map();
      tnsr_B->print_map();
  #endif
      /* Invoke the contraction algorithm */
#ifdef PROFILE
      TAU_FSTART(pre_sum_func_barrier);
      MPI_Barrier(tnsr_B->wrld->comm);
      TAU_FSTOP(pre_sum_func_barrier);
#endif
      TAU_FSTART(activate_topo);
      tnsr_A->topo->activate();
      TAU_FSTOP(activate_topo);
      TAU_FSTART(sum_func);
      sumf->run();
      TAU_FSTOP(sum_func);
      if (tnsr_B->is_sparse){
        tspsum * spsumf = (tspsum*)sumf;
        if (tnsr_B->data != spsumf->new_B){
          tnsr_B->sr->pair_dealloc(tnsr_B->data);
          tnsr_B->data = spsumf->new_B;
          //tnsr_B->nnz_loc = spsumf->new_nnz_B;
        }
        tnsr_B->set_new_nnz_glb(tnsr_B->nnz_blk);
        ASSERT(tnsr_B->nnz_loc == spsumf->new_nnz_B);
      }
      /*tnsr_B->unfold();
      tnsr_B->print();
      MPI_Barrier(tnsr_B->wrld->comm);
      if (tnsr_B->wrld->rank==1){
      for (int i=0; i<tnsr_B->size; i++){
        printf("[%d] %dth element  ",tnsr_B->wrld->rank,i);
        tnsr_B->sr->print(tnsr_B->data+i*tnsr_B->sr->el_size);
        printf("\n");
      }
      }*/
#ifdef PROFILE
      TAU_FSTART(post_sum_func_barrier);
      MPI_Barrier(tnsr_B->wrld->comm);
      TAU_FSTOP(post_sum_func_barrier);
#endif
      TAU_FSTART(sum_postprocessing);
      tnsr_A->topo->deactivate();
      tnsr_A->unfold();
      tnsr_B->unfold();
#ifndef SEQ
      //FIXME: when is this actually needed? can we do it in sym_sum instead?
      stat = tnsr_B->zero_out_padding();
#endif

  #if 0 //VERIFY
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
      for (i=0; (int64_t)i<nA; i++){
        if (fabs(uA[i] - sA[i]) > 1.E-6){
          printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
        }
      }

      cpy_sym_sum(alpha, uA, order_A, edge_len_A, edge_len_A, sym_A, map_A,
                  beta, sB, order_B, edge_len_B, edge_len_B, sym_B, map_B);
      assert(stat == SUCCESS);

      for (i=0; (int64_t)i<nB; i++){
        if (fabs(uB[i] - sB[i]) > 1.E-6){
          printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
        }
        assert(fabs(sB[i] - uB[i]) < 1.E-6);
      }
      CTF_int::cdealloc(uA);
      CTF_int::cdealloc(uB);
      CTF_int::cdealloc(sA);
      CTF_int::cdealloc(sB);
  #endif

      delete sumf;
      if (tnsr_A != A){
        delete tnsr_A;
      }
      for (int i=nst_B-1; i>=0; i--){
        int ret = dstack_tsr_B[i]->extract_diag(dstack_map_B[i], 0, tnsr_B, &new_idx_map);
        ASSERT(ret == SUCCESS);
        delete tnsr_B;
        tnsr_B = dstack_tsr_B[i];
      }
      ASSERT(tnsr_B == B);
    }
  //#ifndef SEQ
    //stat = B->zero_out_padding();
  //#endif
    CTF_int::cdealloc(map_A);
    CTF_int::cdealloc(map_B);
    CTF_int::cdealloc(dstack_map_B);
    CTF_int::cdealloc(dstack_tsr_B);
    TAU_FSTOP(sum_postprocessing);

    TAU_FSTOP(sum_tensors);
    return SUCCESS;
  }

  int summation::unfold_broken_sym(summation ** nnew_sum){
    int sidx, i, num_tot, iA, iA2, iB;
    int * idx_arr;
    int bsym = NS;
    summation * new_sum;
   
    if (nnew_sum != NULL){
      new_sum = new summation(*this);
      *nnew_sum = new_sum;
    } else new_sum = NULL;

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);

    sidx = -1;
    for (i=0; i<A->order; i++){
      if (A->sym[i] != NS){
        iA = idx_A[i];
        if (idx_arr[2*iA+1] != -1){
          if (B->sym[idx_arr[2*iA+1]] == NS ||
              ((B->sym[idx_arr[2*iA+1]] == AS) ^ (A->sym[i] == AS)) ||
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
                ((A->sym[idx_arr[2*iB+0]] == AS) ^ (B->sym[i] == AS)) ||
                idx_arr[2*idx_B[i+1]+0] == -1 ||
                idx_B[i+1] != idx_A[idx_arr[2*iB+0]+1]){
              sidx = 2*i+1;
              break;
            } else if (A->sym[idx_arr[2*iB+0]] == NS){
              sidx = 2*i;
              bsym = B->sym[i];
              break;
            }
          } else if (idx_arr[2*idx_B[i+1]+0] != -1){
            sidx = 2*i+1;
            break;
          }
        }
      }
    }
    //if we have e.g. b[""] = A["ij"] with SY A, symmetry preserved bu t need to account for diagonal, this just unpacks (FIXME: suboptimal)
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
        int nA_sym[A->order];
        memcpy(nA_sym, new_sum->A->sym, sizeof(int)*new_sum->A->order);
        nA_sym[sidx/2] = bsym;
        new_sum->A->set_sym(nA_sym);
      } else {
        new_sum->B = new tensor(B, 0, 0);
        int nB_sym[B->order];
        memcpy(nB_sym, new_sum->B->sym, sizeof(int)*new_sum->B->order);
        nB_sym[sidx/2] = NS;
        new_sum->B->set_sym(nB_sym);
      }
    }
    CTF_int::cdealloc(idx_arr);
    return sidx;
  }

  bool summation::check_consistency(){
    int i, num_tot;
    int iA, iB;
    int * idx_arr;
    int64_t len;
       
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
          printf("i = %d Error in sum call: The %dth edge length (%ld) of tensor %s does not",
                  i, iA, len, A->name);
          printf("match the %dth edge length (%ld) of tensor %s.\n",
                  iB, B->lens[iB], B->name);
        }
        return false;
      }
    }
    CTF_int::cdealloc(idx_arr);
    return true;

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
    //mapping * map;

    TAU_FSTART(check_sum_mapping);
    pass = 1;
   
    ASSERT(A->is_mapped);
    ASSERT(B->is_mapped);
 
    if (A->is_mapped == 0) pass = 0;
    if (B->is_mapped == 0) pass = 0;
    
    
    if (A->is_folded == 1) pass = 0;
    if (B->is_folded == 1) pass = 0;
    
    if (A->topo != B->topo) pass = 0;

    if (pass==0){
      DPRINTF(4,"failed confirmation here\n");
      TAU_FSTOP(check_sum_mapping);
      return 0;
    }
    
    CTF_int::alloc_ptr(sizeof(int)*A->topo->order, (void**)&phys_map);
    memset(phys_map, 0, sizeof(int)*A->topo->order);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
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
      if (iA != -1 && iB == -1) {
        mapping * map = &A->edge_map[iA];
        while (map->type == PHYSICAL_MAP){
          phys_map[map->cdt]++;
          if (map->has_child) map = map->child;
          else break;
        }
      }
      if (iB != -1 && iA == -1){
        mapping * map = &B->edge_map[iB];
        while (map->type == PHYSICAL_MAP){
          phys_map[map->cdt]++;
          if (map->has_child) map = map->child;
          else break;
        }
      }
    }
    /* Ensure that a replicated and a reduced mode are not mapped to processor grid dimensions not used by the other tensor */
    for (i=0; i<A->topo->order; i++){
      if (phys_map[i] > 1) {
        pass = 0;
        DPRINTF(3,"failed confirmation here i=%d\n",i);
      }
    }

    CTF_int::cdealloc(phys_map);
    CTF_int::cdealloc(idx_arr);

    //if we have don't have an additive id we can't replicate
    if (B->sr->addid() == NULL || B->is_sparse){
      int ndim_mapped = 0;
      for (int j=0; j<B->order; j++){
        mapping * map = &B->edge_map[j];
        if (map->type == PHYSICAL_MAP) ndim_mapped++;
        while (map->has_child) {
          map = map->child;
          if (map->type == PHYSICAL_MAP)
            ndim_mapped++;
        }
      }
      if (ndim_mapped < B->topo->order) pass = 0;
    }
       
    TAU_FSTOP(check_sum_mapping);

    return pass;
  }

  int summation::map_sum_indices(topology const * topo){
    int tsr_order, isum, iA, iB, i, j, jsum, jX, stat;
    int64_t * tsr_edge_len;
    int * tsr_sym_table, * restricted;
    int * idx_arr, * idx_sum;
    int num_sum, num_tot, idx_num;
    idx_num = 2;
    mapping * sum_map;

    TAU_FSTART(map_sum_indices);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            &num_tot, &idx_arr);

    CTF_int::alloc_ptr(sizeof(int)*num_tot, (void**)&idx_sum);
    
    num_sum = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[2*i] != -1 && idx_arr[2*i+1] != -1){
        idx_sum[num_sum] = i;
        num_sum++;
      }
    }

    tsr_order = num_sum;


    CTF_int::alloc_ptr(tsr_order*sizeof(int),           (void**)&restricted);
    CTF_int::alloc_ptr(tsr_order*sizeof(int64_t),       (void**)&tsr_edge_len);
    CTF_int::alloc_ptr(tsr_order*tsr_order*sizeof(int), (void**)&tsr_sym_table);
    CTF_int::alloc_ptr(tsr_order*sizeof(mapping),       (void**)&sum_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));

    for (i=0; i<tsr_order; i++){ 
      sum_map[i].type             = NOT_MAPPED; 
      sum_map[i].has_child        = 0;
      sum_map[i].np               = 1;
    }
    for (i=0; i<num_sum; i++){
      isum = idx_sum[i];
      iA = idx_arr[isum*2+0];
      iB = idx_arr[isum*2+1];

      if (A->edge_map[iA].type != NOT_MAPPED){
        ASSERT(B->edge_map[iB].type == NOT_MAPPED);
        copy_mapping(1, &A->edge_map[iA], &sum_map[i]);
      } else if (B->edge_map[iB].type != NOT_MAPPED){
        copy_mapping(1, &B->edge_map[iB], &sum_map[i]);
      }
    }

    /* Map a tensor of dimension.
     * Set the edge lengths and symmetries according to those in sum dims of A and B.
     * This gives us a mapping for the common mapped dimensions of tensors A and B. */
    for (i=0; i<num_sum; i++){
      isum = idx_sum[i];
      iA = idx_arr[isum*idx_num+0];
      iB = idx_arr[isum*idx_num+1];

      tsr_edge_len[i] = A->pad_edge_len[iA];

      /* Check if A has symmetry among the dimensions being contracted over.
       * Ignore symmetry with non-contraction dimensions.
       * FIXME: this algorithm can be more efficient but should not be a bottleneck */
      if (A->sym[iA] != NS){
        for (j=0; j<num_sum; j++){
          jsum = idx_sum[j];
          jX = idx_arr[jsum*idx_num+0];
          if (jX == iA+1){
            tsr_sym_table[i*tsr_order+j] = 1;
            tsr_sym_table[j*tsr_order+i] = 1;
          }
        }
      }
      if (B->sym[iB] != NS){
        for (j=0; j<num_sum; j++){
          jsum = idx_sum[j];
          jX = idx_arr[jsum*idx_num+1];
          if (jX == iB+1){
            tsr_sym_table[i*tsr_order+j] = 1;
            tsr_sym_table[j*tsr_order+i] = 1;
          }
        }
      }
    }
    /* Run the mapping algorithm on this construct */
    stat = map_tensor(topo->order,        tsr_order, 
                      tsr_edge_len,       tsr_sym_table,
                      restricted,         topo->dim_comm,
                      NULL,               0,
                      sum_map);

    if (stat == ERROR){
      TAU_FSTOP(map_sum_indices);
      return ERROR;
    }
    
    /* define mapping of tensors A and B according to the mapping of sum dims */
    if (stat == SUCCESS){
      for (i=0; i<num_sum; i++){
        isum = idx_sum[i];
        iA = idx_arr[isum*idx_num+0];
        iB = idx_arr[isum*idx_num+1];

        copy_mapping(1, &sum_map[i], &A->edge_map[iA]);
        copy_mapping(1, &sum_map[i], &B->edge_map[iB]);
      }
    }
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(tsr_edge_len);
    CTF_int::cdealloc(tsr_sym_table);
    for (i=0; i<num_sum; i++){
      sum_map[i].clear();
    }
    CTF_int::cdealloc(sum_map);
    CTF_int::cdealloc(idx_sum);
    CTF_int::cdealloc(idx_arr);

    TAU_FSTOP(map_sum_indices);
    return stat;

  }

  int summation::map(){
    int i, ret, need_remap;
    int need_remap_A, need_remap_B;
    int d;
    topology * old_topo_A, * old_topo_B;
    int btopo;
    int gtopo;

    //ASSERT(A->wrld->cdt.cm == B->wrld->cdt.cm);
    ASSERT(A->wrld->cdt.np == B->wrld->cdt.np);
    World * wrld = A->wrld;
   
    TAU_FSTART(map_tensor_pair);
  #if DEBUG >= 2
    if (wrld->rank == 0)
      printf("Initial mappings:\n");
    A->print_map(stdout);
    B->print_map(stdout);
  #endif

    //FIXME: try to avoid unfolding immediately, as its not always necessary
    A->unfold();
    B->unfold();
    A->set_padding();
    B->set_padding();


    distribution dA(A);
    distribution dB(B);
    old_topo_A = A->topo;
    old_topo_B = B->topo;
    mapping * old_map_A = new mapping[A->order];
    mapping * old_map_B = new mapping[B->order];
    copy_mapping(A->order, A->edge_map, old_map_A);
    copy_mapping(B->order, B->edge_map, old_map_B);
    btopo = -1;
    int64_t size;
    int64_t min_size = INT64_MAX;
    /* Attempt to map to all possible permutations of processor topology */
    for (i=A->wrld->cdt.rank; i<2*(int)A->wrld->topovec.size(); i+=A->wrld->cdt.np){
  //  for (i=global_comm.rank*topovec.size(); i<2*(int)topovec.size(); i++){
      A->clear_mapping();
      B->clear_mapping();
      A->set_padding();
      B->set_padding();

      A->topo = wrld->topovec[i/2];
      B->topo = wrld->topovec[i/2];
      A->is_mapped = 1;
      B->is_mapped = 1;

      if (i%2 == 0){
        ret = map_self_indices(A, idx_A);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(B, idx_B);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }
      ret = map_sum_indices(A->topo);
      if (ret == NEGATIVE) continue;
      else if (ret != SUCCESS){
        return ret;
      }
      if (i%2 == 0){
        ret = map_self_indices(A, idx_A);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(B, idx_B);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }

      if (i%2 == 0){
        ret = map_self_indices(A, idx_A);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        ret = A->map_tensor_rem(A->topo->order, 
                                A->topo->dim_comm);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        copy_mapping(A->order, B->order,
                     idx_A, A->edge_map, 
                     idx_B, B->edge_map,0);
        ret = B->map_tensor_rem(B->topo->order, 
                                B->topo->dim_comm);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(B, idx_B);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        ret = B->map_tensor_rem(B->topo->order, 
                                B->topo->dim_comm);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
        copy_mapping(B->order, A->order,
                     idx_B, B->edge_map, 
                     idx_A, A->edge_map,0);
        ret = A->map_tensor_rem(A->topo->order, 
                                A->topo->dim_comm);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }
      if (i%2 == 0){
        ret = map_self_indices(B, idx_B);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      } else {
        ret = map_self_indices(A, idx_A);
        if (ret == NEGATIVE) continue;
        else if (ret != SUCCESS) return ret;
      }

  /*    ret = map_symtsr(A->order, A->sym_table, A->edge_map);
      ret = map_symtsr(B->order, B->sym_table, B->edge_map);
      if (ret!=SUCCESS) return ret;
      return SUCCESS;*/

  #if DEBUG >= 4
      A->print_map(stdout,0);
      B->print_map(stdout,0);
  #endif
      if (!check_mapping()) continue;
      A->set_padding();
      B->set_padding();
      size = A->size + B->size;

      need_remap_A = 0;
      need_remap_B = 0;

      if (A->topo == old_topo_A){
        for (d=0; d<A->order; d++){
          if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
            need_remap_A = 1;
        }
      } else
        need_remap_A = 1;
      if (need_remap_A){
        if (!A->is_sparse && can_block_reshuffle(A->order, dA.phase, A->edge_map)){
          size += A->size*std::max(1.0,log2(wrld->cdt.np));
        } else {
          if (A->is_sparse){
            double nnz_frac_A = std::min(2,(int)A->calc_npe())*((double)A->nnz_tot)/(A->size*A->calc_npe());
            size += 25.*nnz_frac_A*A->size*std::max(1.0,log2(wrld->cdt.np));
          } else
            size += 5.*A->size*std::max(1.0,log2(wrld->cdt.np));
        }
      }
      if (B->topo == old_topo_B){
        for (d=0; d<B->order; d++){
          if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
            need_remap_B = 1;
        }
      } else
        need_remap_B = 1;
      if (need_remap_B){
        if (!B->is_sparse && can_block_reshuffle(B->order, dB.phase, B->edge_map)){
          size += B->size*std::max(1.0,log2(wrld->cdt.np));
        } else {
          double pref = 1.0;
          if (B->is_home)
            pref = 2.0;
          if (B->is_sparse){
            double nnz_frac_A = std::min(2,(int)A->calc_npe())*((double)A->nnz_tot)/(A->size*A->calc_npe());
            double nnz_frac_B = std::min(2,(int)B->calc_npe())*((double)B->nnz_tot)/(B->size*B->calc_npe());
            nnz_frac_B = std::max(nnz_frac_B, nnz_frac_A);
            size += 25.*pref*nnz_frac_B*B->size*std::max(1.0,log2(wrld->cdt.np));
          } else
            size += 5.*pref*B->size*std::max(1.0,log2(wrld->cdt.np));
        }
      }

      /*nvirt = (int64_t)calc_nvirt(A);
      tnvirt = nvirt*(int64_t)calc_nvirt(B);
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
      min_size = INT64_MAX;
    /* pick lower dimensional mappings, if equivalent */
    gtopo = get_best_topo(min_size, btopo, wrld->cdt);
    TAU_FSTOP(map_tensor_pair);
    if (gtopo == -1){
      printf("CTF ERROR: Failed to map pair!\n");
      ABORT;
      return ERROR;
    }
    
    A->clear_mapping();
    B->clear_mapping();
    A->set_padding();
    B->set_padding();

    A->topo = wrld->topovec[gtopo/2];
    B->topo = wrld->topovec[gtopo/2];

    A->is_mapped = 1;
    B->is_mapped = 1;
      
    if (gtopo%2 == 0){
      ret = map_self_indices(A, idx_A);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(B, idx_B);
      ASSERT(ret == SUCCESS);
    }
    ret = map_sum_indices(A->topo);
    ASSERT(ret == SUCCESS);
    if (gtopo%2 == 0){
      ret = map_self_indices(A, idx_A);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(B, idx_B);
      ASSERT(ret == SUCCESS);
    }

    if (gtopo%2 == 0){
      ret = map_self_indices(A, idx_A);
      ASSERT(ret == SUCCESS);
      ret = A->map_tensor_rem(A->topo->order, 
                              A->topo->dim_comm);
      ASSERT(ret == SUCCESS);
      copy_mapping(A->order, B->order,
                   idx_A, A->edge_map, 
                   idx_B, B->edge_map,0);
      ret = B->map_tensor_rem(B->topo->order, 
                              B->topo->dim_comm);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(B, idx_B);
      ASSERT(ret == SUCCESS);
      ret = B->map_tensor_rem(B->topo->order, 
                              B->topo->dim_comm);
      ASSERT(ret == SUCCESS);
      copy_mapping(B->order, A->order,
                   idx_B, B->edge_map, 
                   idx_A, A->edge_map,0);
      ret = A->map_tensor_rem(A->topo->order, 
                              A->topo->dim_comm);
      ASSERT(ret == SUCCESS);
    }
    if (gtopo%2 == 0){
      ret = map_self_indices(B, idx_B);
      ASSERT(ret == SUCCESS);
    } else {
      ret = map_self_indices(A, idx_A);
      ASSERT(ret == SUCCESS);
    }

    ASSERT(check_mapping());

    A->set_padding();
    B->set_padding();
  #if DEBUG > 2
    if (wrld->cdt.rank == 0)
      printf("New mappings:\n");
    A->print_map(stdout);
    B->print_map(stdout);
  #endif

    TAU_FSTART(redistribute_for_sum);
   
    A->is_cyclic = 1;
    B->is_cyclic = 1;
    need_remap = 0;
    if (A->topo == old_topo_A){
      for (d=0; d<A->order; d++){
        if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap){
      A->redistribute(dA);
    }
    need_remap = 0;
    if (B->topo == old_topo_B){
      for (d=0; d<B->order; d++){
        if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap){
      B->redistribute(dB);
    }

    TAU_FSTOP(redistribute_for_sum);
    delete [] old_map_A;
    delete [] old_map_B;

    return SUCCESS;
  }

  void summation::print(){
    int i;
    //max = A->order+B->order;

    CommData global_comm = A->wrld->cdt;
    MPI_Barrier(global_comm.cm);
    if (global_comm.rank == 0){
      char sname[200];
      sname[0] = '\0';
      sprintf(sname, "%s", B->name);
      sprintf(sname+strlen(sname),"[");
      for (i=0; i<B->order; i++){
        if (i>0)
          sprintf(sname+strlen(sname)," %d",idx_B[i]);
        else
          sprintf(sname+strlen(sname),"%d",idx_B[i]);
      }
      sprintf(sname+strlen(sname),"] <- ");
      sprintf(sname+strlen(sname), "%s", A->name);
      sprintf(sname+strlen(sname),"[");
      for (i=0; i<A->order; i++){
        if (i>0)
          sprintf(sname+strlen(sname)," %d",idx_A[i]);
        else
          sprintf(sname+strlen(sname),"%d",idx_A[i]);
      }
      sprintf(sname+strlen(sname),"]");
      printf("CTF: Summation %s\n",sname);


/*      printf("Summing Tensor %s into %s\n", A->name, B->name);
      printf("alpha is "); 
      if (alpha != NULL) A->sr->print(alpha);
      else printf("NULL");
      printf("\nbeta is "); 
      if (beta != NULL) B->sr->print(beta);
      else printf("NULL"); 
      printf("\n");
      printf("Summation index table:\n");
      printf("     A     B\n");
      for (i=0; i<max; i++){
        ex_A=0;
        ex_B=0;
        printf("%d:   ",i);
        for (j=0; j<A->order; j++){
          if (idx_A[j] == i){
            ex_A++;
            if (A->sym[j] == SY)
              printf("%dY ",j);
            else if (A->sym[j] == SH)
              printf("%dH ",j);
            else if (A->sym[j] == AS)
              printf("%dS ",j);
            else
              printf("%d  ",j);
          }
        }
        if (ex_A == 0)
          printf("      ");
        if (ex_A == 1)
          printf("   ");
        for (j=0; j<B->order; j++){
          if (idx_B[j] == i){
            ex_B=1;
            if (B->sym[j] == SY)
              printf("%dY ",j);
            else if (B->sym[j] == SH)
              printf("%dH ",j);
            else if (B->sym[j] == AS)
              printf("%dS ",j);
            else
              printf("%d  ",j);
          }
        }
        printf("\n");
        if (ex_A + ex_B== 0) break;
      }*/
    }
  }
              

  void summation::sp_sum(){
    int64_t num_pair;
    char * mapped_data;
    
    bool is_idx_matched = true;
    if (A->order != B->order)
      is_idx_matched = false;
    else {
      for (int o=0; o<A->order; o++){
        if (idx_A[o] != idx_B[o]){
          is_idx_matched = false;
        }
      }
    }


    //read data from A    
    A->read_local_nnz(&num_pair, &mapped_data);

    if (!is_idx_matched){
      int64_t lda_A[A->order];
      int64_t lda_B[B->order];
      lda_A[0] = 1;
      for (int o=1; o<A->order; o++){
        lda_A[o] = lda_A[o-1]*A->lens[o];
      }
      lda_B[0] = 1;
      for (int o=1; o<B->order; o++){
        lda_B[o] = lda_B[o-1]*B->lens[o];
      }
      PairIterator pi(A->sr, mapped_data);
#ifdef USE_OMP
      #pragma omp parallel for
#endif
      for (int i=0; i<num_pair; i++){
        int64_t k = pi[i].k();
        int64_t k_new = 0;
        for (int o=0; o<A->order; o++){
          int64_t kpart = (k/lda_A[o])%A->lens[o];
          //FIXME: slow, but handles diagonal indexing, probably worth having separate versions
          for (int q=0; q<B->order; q++){
            if (idx_A[o] == idx_B[q]){
              k_new += kpart*lda_B[q];
            }
          }
        }
        ((int64_t*)(pi[i].ptr))[0] = k_new;
      }

      // when idx_A has indices idx_B does not, we need to reduce, which can be done partially here since the elements of A should be sorted
      bool is_reduce = false;
      for (int oA=0; oA<A->order; oA++){
        bool inB = false;
        for (int oB=0; oB<B->order; oB++){
          if (idx_A[oA] == idx_B[oB]){
            inB = true;
          }
        }
        if (!inB) is_reduce = true;
      }
  
      if (is_reduce && num_pair > 0){
        pi.sort(num_pair);
        int64_t nuniq=1;
        for (int64_t i=1; i<num_pair; i++){
          if (pi[i].k() != pi[i-1].k()) nuniq++;
        }
        if (nuniq != num_pair){
          char * swap_data = mapped_data;
          alloc_ptr(A->sr->pair_size()*nuniq, (void**)&mapped_data);
          PairIterator pi_new(A->sr, mapped_data);
          int64_t cp_st = 0;
          int64_t acc_st = -1;
          int64_t pfx = 0;
          for (int64_t i=1; i<num_pair; i++){
            if (pi[i].k() == pi[i-1].k()){
              if (cp_st < i){ 
                memcpy(pi_new[pfx].ptr, pi[cp_st].ptr, A->sr->pair_size()*(i-cp_st));
                pfx += i-cp_st;
              }
              cp_st = i+1;

              if (acc_st == -1) acc_st = i;
            } else {
              if (acc_st != -1){
                for (int64_t j=acc_st; j<i; j++){
                  A->sr->add(pi_new[pfx-1].d(), pi[j].d(), pi_new[pfx-1].d());
                }
              }
              acc_st = -1;
            }           
          }
          if (cp_st < num_pair)
            memcpy(pi_new[pfx].ptr, pi[cp_st].ptr, A->sr->pair_size()*(num_pair-cp_st));
          if (acc_st != -1){
            for (int64_t j=acc_st; j<num_pair; j++){
              A->sr->add(pi_new[pfx-1].d(), pi[j].d(), pi_new[pfx-1].d());
            }
          }
          cdealloc(swap_data);
          num_pair = nuniq;
        }
      }

      // if applying custom function, apply immediately on reduced form
      if (is_custom && !func->is_accumulator()){
        char * swap_data = mapped_data;
        alloc_ptr(B->sr->pair_size()*num_pair, (void**)&mapped_data);
        PairIterator pi_new(B->sr, mapped_data);
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<num_pair; i++){
          if (alpha == NULL)
            func->apply_f(pi[i].d(), pi_new[i].d());
          else  {
            char tmp_A[A->sr->el_size];
            A->sr->mul(pi[i].d(), alpha, tmp_A);
            func->apply_f(tmp_A, pi_new[i].d());
          }
        }
        cdealloc(swap_data);
        alpha = NULL;
      }
  
      // when idx_B has indices idx_A does not, we need to map, which we do by replicating the key value pairs of B
      // FIXME this is probably not most efficient, but not entirely stupid, as at least the set of replicated pairs is not expected to be bigger than B
      int nmap_idx = 0;
      int64_t map_idx_len[B->order];
      int64_t map_idx_lda[B->order];
      int map_idx_rev[B->order];
      for (int oB=0; oB<B->order; oB++){
        bool inA = false;
        for (int oA=0; oA<A->order; oA++){
          if (idx_A[oA] == idx_B[oB]){
            inA = true;
          }
        }
        if (!inA){ 
          bool is_rep=false;
          for (int ooB=0; ooB<oB; ooB++){
            if (idx_B[ooB] == idx_B[oB]){
              is_rep = true;
              map_idx_lda[map_idx_rev[ooB]] += lda_B[oB];
              break;
            }
          }
          if (!is_rep){
            map_idx_len[nmap_idx] = B->lens[oB];
            map_idx_lda[nmap_idx] = lda_B[oB];
            map_idx_rev[nmap_idx] = oB;
            nmap_idx++;
          }
        }
      }
      if (nmap_idx > 0){
        int64_t tot_rep=1;
        for (int midx=0; midx<nmap_idx; midx++){
          tot_rep *= map_idx_len[midx];
        }
        char * swap_data = mapped_data;
        alloc_ptr(A->sr->pair_size()*num_pair*tot_rep, (void**)&mapped_data);
        PairIterator pi_new(A->sr, mapped_data);
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<num_pair; i++){
          for (int64_t r=0; r<tot_rep; r++){
            memcpy(pi_new[i*tot_rep+r].ptr, pi[i].ptr, A->sr->pair_size());
          }
        }
#ifdef USE_OMP
        #pragma omp parallel for
#endif
        for (int64_t i=0; i<num_pair; i++){
          int64_t phase=1;
          for (int midx=0; midx<nmap_idx; midx++){
            int64_t stride=phase;
            phase *= map_idx_len[midx];
            for (int64_t r=0; r<tot_rep/phase; r++){
              for (int64_t m=1; m<map_idx_len[midx]; m++){
                for (int64_t s=0; s<stride; s++){
                  ((int64_t*)(pi_new[i*tot_rep + r*phase + m*stride + s].ptr))[0] += m*map_idx_lda[midx];
                }
              }
            }
          }
        }
        cdealloc(swap_data);
        num_pair *= tot_rep;
      }
    }
    
    B->write(num_pair, alpha, beta, mapped_data, 'w');
    cdealloc(mapped_data);

  }

}
