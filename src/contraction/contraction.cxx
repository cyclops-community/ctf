#include "contraction.h"
#include "../redistribution/nosym_transp.h"
#include "../scaling/strp_tsr.h"
#include "../mapping/mapping.h"
#include "../mapping/distribution.h"
#include "../tensor/untyped_tensor.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "sym_seq_ctr.h"
#include "spctr_comm.h"
#include "ctr_tsr.h"
#include "ctr_offload.h"
#include "ctr_2d_general.h"
#include "spctr_2d_general.h"
#include "../symmetry/sym_indices.h"
#include "../symmetry/symmetrization.h"
#include "../redistribution/nosym_transp.h"
#include "../redistribution/redist.h"
#include "../sparse_formats/coo.h"
#include "../sparse_formats/csr.h"
#include <cfloat>
#include <limits>

namespace CTF_int {

  using namespace CTF;

  double de_flops = 0;
  double sp_flops = 0;
  double skip_flops = 0;

  struct sgroup {
    int n, s, t, v;
    std::vector<int> inds;
  };

  static int64_t choose(int n, int m){
    int64_t f = 1;
    for (int i=0; i<m; i++){
      f*=(n-i);
    }
    for (int i=2; i<=m; i++){
      f=f/i;
    }
    return f;
  }

  static double get_de_cost(std::vector<sgroup> & vec){
    double flops = 2;
    for (int i=0; i<vec.size(); i++){
      flops *= choose(vec[i].n,vec[i].s)*choose(vec[i].n,vec[i].t)*choose(vec[i].n,vec[i].v);
/*      int dim = vec[i].s + vec[i].t + vec[i].v;
      assert(dim <= 2);
      int div = std::max(std::max(vec[i].s, vec[i].t),vec[i].v);
      if (div == 2){
        flops *= vec[i].n-1;
        flops = flops / div;
      } else if (dim == 2)
        flops *= vec[i].n;*/
    }
    return flops;
  }


  static double get_sp_cost(std::vector<sgroup> & vec){
    double flops;
    int64_t size_A=1, size_B=1, size_C=1;
    double rec_flops;
    if (vec.size()==1){
      rec_flops = 1;
    } else {
      std::vector<sgroup> rec_vec;
      for (int j=1; j<vec.size(); j++){
        rec_vec.push_back(vec[j]);
        size_A *= choose(vec[j].n, vec[j].s+vec[j].v);
        size_B *= choose(vec[j].n, vec[j].v+vec[j].t);
        size_C *= choose(vec[j].n, vec[j].s+vec[j].t);
      } 
      rec_flops = get_sp_cost(rec_vec);
    }
    printf("rec_flops = %1.2E sA = %1.2E sB = %1.2E sC = %1.2E\n",rec_flops,(double)size_A,(double)size_B,(double)size_C);
    int dim = vec[0].s + vec[0].t + vec[0].v;
    if (dim == 1){
      if (vec.size()==1)
        flops = 2*vec[0].n;
      else
        flops = rec_flops*vec[0].n;
    } else {
      int n = vec[0].n;
      int s = vec[0].s;
      int t = vec[0].t;
      int v = vec[0].v;
      if (v>0){
        flops = choose(n,s+t+v)*(rec_flops+(choose(s+t+v,t)-1)*size_A+(choose(s+t+v,s)-1)*size_B+choose(s+t+v,v)*size_C);
        if (t>0)
          flops += choose(n,s+v)*size_A;
        if (s>0)
          flops += choose(n,v+t)*size_B;
        
        //flops += choose(n,s+v)*size_A+choose(n,v+t)*size_B+choose(n,s+t)*size_C;
      } else {
        flops = choose(n,s+t+v)*(rec_flops+(choose(s+t+v,t)-1)*size_A+(choose(s+t+v,s)-1)*size_B+(choose(s+t+v,v)-1)*size_C);
        for (int r=1; r<=std::min(s,t); r++){
          flops += choose(s+t,r)*choose(s+t-r,r)*choose(n,s+t)*size_C;
        }
        /*if (s>0 && t>0){
          flops += .5*choose(2*(s+t),s+t)*choose(n,s+t)*size_C;
        }*/
      }
    }
    printf("flops = %1.2E\n",flops);
    return flops;
  }

  static void count_cost(contraction const * c){
    int num_tot;
    int * idx_arr;
    c->print();
    c->A->print_map();
    c->B->print_map();
    c->C->print_map();
      
    inv_idx(c->A->order, c->idx_A,
            c->B->order, c->idx_B,
            c->C->order, c->idx_C,
            &num_tot, &idx_arr);

    bool skip = false;
    for (int i=0; i<num_tot; i++){
      int ni = 0;
      if (idx_arr[3*i+0] != -1) ni++;
      if (idx_arr[3*i+1] != -1) ni++;
      if (idx_arr[3*i+2] != -1) ni++;
      if (ni != 2) skip = true;
    }

    if (skip || num_tot < 6){
      double flops = 1;
      for (int i=0; i<num_tot; i++){
        if (idx_arr[3*i+0] != -1) flops *= c->A->lens[idx_arr[3*i+0]];
        else if (idx_arr[3*i+1] != -1) flops *= c->B->lens[idx_arr[3*i+1]];
        else if (idx_arr[3*i+2] != -1) flops *= c->C->lens[idx_arr[3*i+2]];
      }
      skip_flops += flops;
    } else {
      //assumes there are no sgroup with s,t,v>0
      std::vector<sgroup> symgrps;
      for (int iA=0; iA<c->A->order; iA++){
        int nsym = 0;
        while (c->A->sym[iA+nsym] != NS){
          nsym++;
          sgroup s;
          s.n = c->A->lens[iA];
          s.t = 0;
          s.s = 0;
          for (int j=0; j<=nsym; j++){
            if (idx_arr[3*c->idx_A[iA+j]+2] != -1) s.s++;
            s.inds.push_back(c->idx_A[iA+j]);
          }
          s.v = nsym+1-s.s;
          symgrps.push_back(s);
        }
      }
      for (int iB=0; iB<c->B->order; iB++){
        int nsym = 0;
        while (c->B->sym[iB+nsym] != NS){
          nsym++;
          sgroup s;
          s.n = c->B->lens[iB];
          s.s = 0;
          s.t = 0;
          for (int j=0; j<=nsym; j++){
            if (idx_arr[3*c->idx_B[iB+j]+2] != -1) s.t++;
            s.inds.push_back(c->idx_B[iB+j]);
          }
          s.v = nsym+1-s.t;
          if (s.t != 0)
            symgrps.push_back(s);
        }
      }

      for (int iC=0; iC<c->C->order; iC++){
        int nsym = 0;
        while (c->C->sym[iC+nsym] != NS){
          nsym++;
          sgroup s;
          s.n = c->C->lens[iC];
          s.v = 0;
          s.s = 0;
          for (int j=0; j<=nsym; j++){
            if (idx_arr[3*c->idx_C[iC+j]] != -1) s.s++;
            s.inds.push_back(c->idx_C[iC+j]);
          }
          s.t = nsym+1-s.s;
          if (s.t != 0 && s.s != 0)
            symgrps.push_back(s);
        }
      }

      int nsubgrps = 1<<symgrps.size();

      double de_cost = DBL_MAX;
      double sp_cost = DBL_MAX;
  
      for (int i=0; i<nsubgrps; i++){
        std::vector<sgroup> subgrp;
        int m = i;
        for (int j=0; j<symgrps.size(); j++){
          if (m%2 == 1) subgrp.push_back(symgrps[j]);
          m = m>>1;
        }
        bool used[num_tot];
        std::fill(used, used+num_tot, 0);
        bool skip = false;
        for (int k=0; k<subgrp.size(); k++){
          for (int l=0; l<subgrp[k].s+subgrp[k].t+subgrp[k].v; l++){
            if (used[subgrp[k].inds[l]]) skip = true;
            else used[subgrp[k].inds[l]] = 1;
          }
        }
        if (!skip){
          std::vector< std::pair<int,sgroup> > psubgrp;
          for (int k=0; k<subgrp.size(); k++){
            psubgrp.push_back(std::pair<int,sgroup>(k,subgrp[k]));
          }
          for (int j=0; j<num_tot; j++){
            if (!used[j]){
              sgroup s;
              if (idx_arr[3*j+0] == -1){
                s.n = c->B->lens[idx_arr[3*j+1]];
                s.s = 0;
                s.t = 1;
                s.v = 0;
              }
              if (idx_arr[3*j+1] == -1){
                s.n = c->C->lens[idx_arr[3*j+2]];
                s.s = 1;
                s.t = 0;
                s.v = 0;
              }
              if (idx_arr[3*j+2] == -1){
                s.n = c->B->lens[idx_arr[3*j+1]];
                s.s = 0;
                s.t = 0;
                s.v = 1;
              }
              psubgrp.push_back(std::pair<int,sgroup>(psubgrp.size(),s));
            }
          }
          do {
            std::vector<sgroup> nsubgrp;
            for (int k=0; k<psubgrp.size(); k++){
              nsubgrp.push_back(psubgrp[k].second);
              printf("n = %d s = %d t = %d v = %d\n",nsubgrp[k].n,nsubgrp[k].s,nsubgrp[k].t,nsubgrp[k].v);
            }
            double tde_cost = get_de_cost(nsubgrp); 
            double tsp_cost = get_sp_cost(nsubgrp); 
            printf("de cost = %1.2E sp cost = %1.2E\n", tde_cost, tsp_cost);
            de_cost = std::min(de_cost,tde_cost); 
            sp_cost = std::min(sp_cost,tsp_cost); 
          } while (std::next_permutation(&(psubgrp[0]), &(psubgrp[subgrp.size()]), 
                      [](std::pair<int,sgroup> p1, std::pair<int,sgroup> p2){ 
                        return p1.first < p2.first; 
                      }));
        }
      }

      de_flops += de_cost;
      sp_flops += sp_cost;
      printf("Contraction requires %1.2E/%1.2E de flops, %1.2E/%1.2E sp flops (%1.2E skipped flops)\n",de_cost, de_flops, sp_cost, sp_flops, skip_flops);
    }
  }

  contraction::~contraction(){
    if (idx_A != NULL) cdealloc(idx_A);
    if (idx_B != NULL) cdealloc(idx_B);
    if (idx_C != NULL) cdealloc(idx_C);
  }

  contraction::contraction(contraction const & other){
    A     = other.A;
    idx_A = (int*)alloc(sizeof(int)*other.A->order);
    memcpy(idx_A, other.idx_A, sizeof(int)*other.A->order);
    B     = other.B;
    idx_B = (int*)alloc(sizeof(int)*other.B->order);
    memcpy(idx_B, other.idx_B, sizeof(int)*other.B->order);
    C     = other.C;
    idx_C = (int*)alloc(sizeof(int)*other.C->order);
    memcpy(idx_C, other.idx_C, sizeof(int)*other.C->order);
    if (other.is_custom) is_custom = 1;
    else is_custom = 0;
    func      = other.func;
    alpha = other.alpha;
    beta  = other.beta;
  }
 
  contraction::contraction(tensor *               A_,
                           int const *            idx_A_,
                           tensor *               B_,
                           int const *            idx_B_,
                           char const *           alpha_,
                           tensor *               C_,
                           int const *            idx_C_,
                           char const *           beta_,
                           bivar_function const * func_){
    A = A_;
    B = B_;
    C = C_;
    if (func_ == NULL) is_custom = 0;
    else is_custom = 1;
    func = func_;
    alpha = alpha_;
    beta  = beta_;
    
    idx_A = (int*)alloc(sizeof(int)*A->order);
    idx_B = (int*)alloc(sizeof(int)*B->order);
    idx_C = (int*)alloc(sizeof(int)*C->order);
    memcpy(idx_A, idx_A_, sizeof(int)*A->order);
    memcpy(idx_B, idx_B_, sizeof(int)*B->order);
    memcpy(idx_C, idx_C_, sizeof(int)*C->order);
  }
 
  contraction::contraction(tensor *               A_,
                           char const *           cidx_A,
                           tensor *               B_,
                           char const *           cidx_B,
                           char const *           alpha_,
                           tensor *               C_,
                           char const *           cidx_C,
                           char const *           beta_,
                           bivar_function const * func_){
    A = A_;
    B = B_;
    C = C_;
    if (func_ == NULL) is_custom = 0;
    else is_custom = 1;
    func = func_;
    alpha = alpha_;
    beta  = beta_;
    
    conv_idx(A->order, cidx_A, &idx_A, B->order, cidx_B, &idx_B, C->order, cidx_C, &idx_C);
  }

  void contraction::execute(){
#if DEBUG >= 2
    if (A->wrld->cdt.rank == 0) printf("Contraction::execute (head):\n");
    print();
#endif

    count_cost(this);

    //if (A->wrld->cdt.cm == MPI_COMM_WORLD){
      update_all_models(A->wrld->cdt.cm);
    //}
    
    int stat = home_contract();
    assert(stat == SUCCESS); 
  }
  
  template<typename ptype>
  void get_perm(int     perm_order,
                ptype   A,
                ptype   B,
                ptype   C,
                ptype & tA,
                ptype & tB,
                ptype & tC){
    switch (perm_order){
      case 0:
        tA = A;
        tB = B;
        tC = C;
        break;
      case 1:
        tA = A;
        tB = C;
        tC = B;
        break;
      case 2:
        tA = B;
        tB = A;
        tC = C;
        break;
      case 3:
        tA = B;
        tB = C;
        tC = A;
        break;
      case 4:
        tA = C;
        tB = A;
        tC = B;
        break;
      case 5:
        tA = C;
        tB = B;
        tC = A;
        break;
      default:
        assert(0);
        break;
    }
  }


  double contraction::estimate_time(){
    assert(0); //FIXME
    return 0.0;
  }

  int contraction::is_equal(contraction const & os){
    if (this->A != os.A) return 0;
    if (this->B != os.B) return 0;
    if (this->C != os.C) return 0;
    
    for (int i=0; i<A->order; i++){
      if (idx_A[i] != os.idx_A[i]) return 0;
    }
    for (int i=0; i<B->order; i++){
      if (idx_B[i] != os.idx_B[i]) return 0;
    }
    for (int i=0; i<C->order; i++){
      if (idx_C[i] != os.idx_C[i]) return 0;
    }
    return 1;
  }

  /**
   * \brief calculate the dimensions of the matrix the contraction gets reduced to
   *  (A, B, and C may be permuted)
   *
   * \param[in] A tensor 1
   * \param[in] B tensor 2
   * \param[in] C tensor 3
   * \param[in] idx_A indices of tensor 1
   * \param[in] idx_B indices of tensor 2
   *
   * \param[in] ordering_A the dimensional-ordering of the inner mapping of A
   * \param[in] ordering_B the dimensional-ordering of the inner mapping of B
   * \param[out] inner_prm parameters includng n,m,k
   */
  void calc_fold_nmk(
                     tensor const * A,
                     tensor const * B,
                     tensor const * C,
                     int const *    idx_A,
                     int const *    idx_B,
                     int const *    idx_C,
                     int const *    ordering_A,
                     int const *    ordering_B,
                     iparam *       inner_prm){
    int i, num_ctr, num_tot;
    int * idx_arr;
      
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    num_ctr = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        num_ctr++;
      } 
    }
    inner_prm->m = 1;
    inner_prm->n = 1;
    inner_prm->k = 1;
    for (i=0; i<A->order; i++){
      if (i >= num_ctr)
        inner_prm->m = inner_prm->m * A->pad_edge_len[ordering_A[i]];
      else 
        inner_prm->k = inner_prm->k * A->pad_edge_len[ordering_A[i]];
    }
    for (i=0; i<B->order; i++){
      if (i >= num_ctr)
        inner_prm->n = inner_prm->n * B->pad_edge_len[ordering_B[i]];
    }
    /* This gets set later */
    inner_prm->sz_C = 0;
    CTF_int::cdealloc(idx_arr);
  }

  void contraction::get_fold_indices(int *  num_fold,
                                     int ** fold_idx){
    int i, in, num_tot, nfold, broken;
    int iA, iB, iC, inA, inB, inC, iiA, iiB, iiC;
    int * idx_arr, * idx;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    CTF_int::alloc_ptr(num_tot*sizeof(int), (void**)&idx);

    for (i=0; i<num_tot; i++){
      idx[i] = 1;
    }

    for (iA=0; iA<A->order; iA++){
      i = idx_A[iA];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      broken = 0;
      inA = iA;
      do {
        in = idx_A[inA];
        inB = idx_arr[3*in+1];
        inC = idx_arr[3*in+2];
        if (((inA>=0) + (inB>=0) + (inC>=0) != 2) ||
            ((inB == -1) ^ (iB == -1)) ||
            ((inC == -1) ^ (iC == -1)) ||
            (iB != -1 && inB - iB != in-i) ||
            (iC != -1 && inC - iC != in-i) ||
            (iB != -1 && A->sym[inA] != B->sym[inB]) ||
            (iC != -1 && A->sym[inA] != C->sym[inC])){
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
    
    for (iC=0; iC<C->order; iC++){
      i = idx_C[iC];
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      broken = 0;
      inC = iC;
      do {
        in = idx_C[inC];
        inA = idx_arr[3*in+0];
        inB = idx_arr[3*in+1];
        if (((inC>=0) + (inA>=0) + (inB>=0) != 2) ||
            ((inA == -1) ^ (iA == -1)) ||
            ((inB == -1) ^ (iB == -1)) ||
            (iA != -1 && inA - iA != in-i) ||
            (iB != -1 && inB - iB != in-i) ||
            (iA != -1 && C->sym[inC] != A->sym[inA]) ||
            (iB != -1 && C->sym[inC] != B->sym[inB])){
          broken = 1;
        }
        inC++;
      } while (C->sym[inC-1] != NS);
      if (broken){
        for (iiC=iC;iiC<inC;iiC++){
          idx[idx_C[iiC]] = 0;
        }
      }
    }
    
    for (iB=0; iB<B->order; iB++){
      i = idx_B[iB];
      iC = idx_arr[3*i+2];
      iA = idx_arr[3*i+0];
      broken = 0;
      inB = iB;
      do {
        in = idx_B[inB];
        inC = idx_arr[3*in+2];
        inA = idx_arr[3*in+0];
        if (((inB>=0) + (inC>=0) + (inA>=0) != 2) ||
            ((inC == -1) ^ (iC == -1)) ||
            ((inA == -1) ^ (iA == -1)) ||
            (iC != -1 && inC - iC != in-i) ||
            (iA != -1 && inA - iA != in-i) ||
            (iC != -1 && B->sym[inB] != C->sym[inC]) ||
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

  int contraction::can_fold(){
    int nfold, * fold_idx, i, j;
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
    for (i=0; i<C->order; i++){
      for (j=i+1; j<C->order; j++){
        if (idx_C[i] == idx_C[j]) return 0;
      }
    }
    get_fold_indices(&nfold, &fold_idx);
    if (A->is_sparse){
      //when A is sparse we must fold all indices and reduce block contraction entirely to coomm
      if ((A->order+B->order+C->order)%2 == 1 ||
          (A->order+B->order+C->order)/2 < nfold){
        return 0;
      }
    }
    CTF_int::cdealloc(fold_idx);
    /* FIXME: 1 folded index is good enough for now, in the future model */
    return nfold > 0;
  }


  /**
   * \brief find ordering of indices of tensor to reduce to DGEMM (A, B, and C may be permuted
   *
   * \param[in] A tensor 1
   * \param[in] B tensor 2
   * \param[in] C tensor 3
   * \param[in] idx_A indices of tensor 1
   * \param[in] idx_B indices of tensor 2
   * \param[in] idx_C indices of tensor 3
   * \param[out] new_ordering_A the new ordering for indices of A
   * \param[out] new_ordering_B the new ordering for indices of B
   * \param[out] new_ordering_C the new ordering for indices of C
   */
  void get_len_ordering(
            tensor const * A,
            tensor const * B,
            tensor const * C,
            int const * idx_A,
            int const * idx_B,
            int const * idx_C,
            int ** new_ordering_A,
            int ** new_ordering_B,
            int ** new_ordering_C){
    int i, num_tot, num_ctr, idx_ctr, num_no_ctr_A;
    int idx_no_ctr_A, idx_no_ctr_B;
    int * ordering_A, * ordering_B, * ordering_C, * idx_arr;
    
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&ordering_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&ordering_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&ordering_C);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    num_ctr = 0, num_no_ctr_A = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        num_ctr++;
      } else if (idx_arr[3*i] != -1){
        num_no_ctr_A++;
      }
    }
    /* Put all contraction indices up front, put A indices in front for C */
    idx_ctr = 0, idx_no_ctr_A = 0, idx_no_ctr_B = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        ordering_A[idx_ctr] = idx_arr[3*i];
        ordering_B[idx_ctr] = idx_arr[3*i+1];
        idx_ctr++;
      } else {
        if (idx_arr[3*i] != -1){
          ordering_A[num_ctr+idx_no_ctr_A] = idx_arr[3*i];
          ordering_C[idx_no_ctr_A] = idx_arr[3*i+2];
          idx_no_ctr_A++;
        }
        if (idx_arr[3*i+1] != -1){
          ordering_B[num_ctr+idx_no_ctr_B] = idx_arr[3*i+1];
          ordering_C[num_no_ctr_A+idx_no_ctr_B] = idx_arr[3*i+2];
          idx_no_ctr_B++;
        }
      }
    }
    CTF_int::cdealloc(idx_arr);
    *new_ordering_A = ordering_A;
    *new_ordering_B = ordering_B;
    *new_ordering_C = ordering_C;
    
    //iparam iprm;
    //calc_fold_nmk(A, B, C, idx_A, idx_B, idx_C, *new_ordering_A, *new_ordering_B, &iprm);
    //return iprm;
  }

  
  void contraction::get_fold_ctr(contraction *& fold_ctr,
                                 int &          all_fdim_A,
                                 int &          all_fdim_B,
                                 int &          all_fdim_C,
                                 int *&         all_flen_A,
                                 int *&         all_flen_B,
                                 int *&         all_flen_C){
    int i, j, nfold, nf;
    int * fold_idx, * fidx_A, * fidx_B, * fidx_C;
    tensor * fA, * fB, * fC;

    get_fold_indices(&nfold, &fold_idx);
    if (nfold == 0) {
      CTF_int::cdealloc(fold_idx);
      assert(0); //return ERROR;
    }

    /* overestimate this space to not bother with it later */
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_A);
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_B);
    CTF_int::alloc_ptr(nfold*sizeof(int), (void**)&fidx_C);

    A->fold(nfold, fold_idx, idx_A,
            &all_fdim_A, &all_flen_A);
    B->fold(nfold, fold_idx, idx_B,
            &all_fdim_B, &all_flen_B);
    C->fold(nfold, fold_idx, idx_C,
            &all_fdim_C, &all_flen_C);

//    printf("rec tsr C order is %d\n",C->rec_tsr->order);

    nf = 0;
    for (i=0; i<A->order; i++){
      for (j=0; j<nfold; j++){
        if (A->sym[i] == NS && idx_A[i] == fold_idx[j]){
          fidx_A[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<B->order; i++){
      for (j=0; j<nfold; j++){
        if (B->sym[i] == NS && idx_B[i] == fold_idx[j]){
          fidx_B[nf] = j;
          nf++;
        }
      }
    }
    nf = 0;
    for (i=0; i<C->order; i++){
      for (j=0; j<nfold; j++){
        if (C->sym[i] == NS && idx_C[i] == fold_idx[j]){
          fidx_C[nf] = j;
          nf++;
        }
      }
    }

    fA = A->rec_tsr;
    fB = B->rec_tsr;
    fC = C->rec_tsr;

    int * sidx_A, * sidx_B, * sidx_C;
    CTF_int::conv_idx<int>(fA->order, fidx_A, &sidx_A,
                           fB->order, fidx_B, &sidx_B,
                           fC->order, fidx_C, &sidx_C);

    fold_ctr = new contraction(fA, sidx_A, fB, sidx_B, alpha, fC, sidx_C, beta, func);

    CTF_int::cdealloc(sidx_A);
    CTF_int::cdealloc(sidx_B);
    CTF_int::cdealloc(sidx_C);

    CTF_int::cdealloc(fidx_A);
    CTF_int::cdealloc(fidx_B);
    CTF_int::cdealloc(fidx_C);
    CTF_int::cdealloc(fold_idx);
  }

  void contraction::select_ctr_perm(contraction const * fold_ctr,
                                    int                 all_fdim_A,
                                    int                 all_fdim_B,
                                    int                 all_fdim_C,
                                    int const *         all_flen_A,
                                    int const *         all_flen_B,
                                    int const *         all_flen_C,
                                    int &               bperm_order,
                                    double &            btime,
                                    iparam &            iprm){
    bperm_order = -1;
    btime = DBL_MAX;
    int tall_fdim_A, tall_fdim_B, tall_fdim_C;
    int const * tall_flen_A, * tall_flen_B, * tall_flen_C;
    tensor * tA, * tB, * tC;
    tensor * tfA, * tfB, * tfC;
    int * tidx_A, * tidx_B, * tidx_C;
    int * tfnew_ord_A, * tfnew_ord_B, * tfnew_ord_C;
    int * tAiord, * tBiord, * tCiord;

    //iterate over permutations of {A,B,C}
    for (int iord=0; iord<6; iord++){
      get_perm<tensor*>(iord, A, B, C, 
                       tA, tB, tC);
      get_perm<tensor*>(iord, fold_ctr->A, fold_ctr->B, fold_ctr->C, 
                       tfA, tfB, tfC);
      get_perm<int*>(iord, fold_ctr->idx_A, fold_ctr->idx_B, fold_ctr->idx_C,
                           tidx_A, tidx_B, tidx_C);
      get_perm<int const*>(iord, all_flen_A, all_flen_B, all_flen_C,
                           tall_flen_A, tall_flen_B, tall_flen_C);
      get_perm<int>(iord, all_fdim_A, all_fdim_B, all_fdim_C,
                          tall_fdim_A, tall_fdim_B, tall_fdim_C);
      get_len_ordering(tfA, tfB, tfC, tidx_A, tidx_B, tidx_C, 
                        &tfnew_ord_A, &tfnew_ord_B, &tfnew_ord_C); 
      // m,n,k should be invarient to what transposes are done
      if (iord == 0){
        calc_fold_nmk(tfA, tfB, tfC, tidx_A, tidx_B, tidx_C, tfnew_ord_A, tfnew_ord_B, &iprm);
      }

      CTF_int::alloc_ptr(tall_fdim_A*sizeof(int), (void**)&tAiord);
      CTF_int::alloc_ptr(tall_fdim_B*sizeof(int), (void**)&tBiord);
      CTF_int::alloc_ptr(tall_fdim_C*sizeof(int), (void**)&tCiord);

      memcpy(tAiord, tA->inner_ordering, tall_fdim_A*sizeof(int));
      memcpy(tBiord, tB->inner_ordering, tall_fdim_B*sizeof(int));
      memcpy(tCiord, tC->inner_ordering, tall_fdim_C*sizeof(int));

      permute_target(tfA->order, tfnew_ord_A, tAiord);
      permute_target(tfB->order, tfnew_ord_B, tBiord);
      permute_target(tfC->order, tfnew_ord_C, tCiord);
    
      double time_est = 0.0;
      if (tA->is_sparse)
        time_est += tA->nnz_tot/(tA->size*tA->calc_npe())*tA->calc_nvirt()*est_time_transp(tall_fdim_A, tAiord, tall_flen_A, 1, tA->sr);
      else
        time_est += tA->calc_nvirt()*est_time_transp(tall_fdim_A, tAiord, tall_flen_A, 1, tA->sr);
      if (tB->is_sparse)
        time_est += tB->nnz_tot/(tB->size*tB->calc_npe())*tB->calc_nvirt()*est_time_transp(tall_fdim_B, tBiord, tall_flen_B, 1, tB->sr);
      else
        time_est += tB->calc_nvirt()*est_time_transp(tall_fdim_B, tBiord, tall_flen_B, 1, tB->sr);
      if (tC->is_sparse)
        time_est += 2.*tC->nnz_tot/(tC->size*tC->calc_npe())*tC->calc_nvirt()*est_time_transp(tall_fdim_C, tCiord, tall_flen_C, 1, tC->sr);
      else
        time_est += 2.*tC->calc_nvirt()*est_time_transp(tall_fdim_C, tCiord, tall_flen_C, 1, tC->sr);
      if (A->is_sparse || B->is_sparse || C->is_sparse){
        if (iord == 1){
          if (time_est <= btime){
            btime = time_est;
            bperm_order = iord;
          }
        }
      } else {
        if (time_est <= btime){
          btime = time_est;
          bperm_order = iord;
        }
      }
      cdealloc(tAiord);
      cdealloc(tBiord);
      cdealloc(tCiord);
      cdealloc(tfnew_ord_A);
      cdealloc(tfnew_ord_B);
      cdealloc(tfnew_ord_C);
    }

    switch (bperm_order){
      case 0: // A B C
        //index order : 1. AB 2. AC 3. BC
        iprm.tA = 'T';
        iprm.tB = 'N';
        iprm.tC = 'N';
        break;
      case 1: // A C B
        //index order : 1. AC 2. AB 3. BC
        iprm.tA = 'N';
        iprm.tB = 'N';
        iprm.tC = 'N';
        //calc_fold_nmk(fold_ctr->A, fold_ctr->B, fold_ctr->C, fold_ctr->idx_A, fold_ctr->idx_B, fold_ctr->idx_C, fnew_ord_A, fnew_ord_C, &iprm);
        break;
      case 2: // B A C
        //index order : 1. AB 2. BC 3. AC
        //C^T=B^T*A^T
        iprm.tA = 'N';
        iprm.tB = 'T';
        iprm.tC = 'T';
        break;
      case 3: // B C A
        //index order : 1. BC 2. AB 3. AC
        //C^T=B^T*A^T
        iprm.tA = 'N';
        iprm.tB = 'N';
        iprm.tC = 'T';
        break;
      case 4: // C A B
        //index order : 1. CA 2. BC 3. AB
        iprm.tA = 'N';
        iprm.tB = 'T';
        iprm.tC = 'N';
        //calc_fold_nmk(fold_ctr->A, fold_ctr->B, fold_ctr->C, fold_ctr->idx_A, fold_ctr->idx_B, fold_ctr->idx_C, fnew_ord_B, fnew_ord_C, &iprm);
        break;
      case 5: // C B A
        //index order : 1. BC 2. AC 3. AB
        //C^T=B^T*A^T
        iprm.tA = 'T';
        iprm.tB = 'N';
        iprm.tC = 'T';
        break;
      default:
        assert(0);
        break;
    }

  }

  iparam contraction::map_fold(bool do_transp){
    int i, all_fdim_A, all_fdim_B, all_fdim_C;
    int nvirt_A, nvirt_B, nvirt_C;
    int * fnew_ord_A, * fnew_ord_B, * fnew_ord_C;
    int * all_flen_A, * all_flen_B, * all_flen_C;
    iparam iprm;
    contraction * fold_ctr;
    int bperm_order = -1;
    double btime = DBL_MAX;
    int tall_fdim_A, tall_fdim_B, tall_fdim_C;
    tensor * tA, * tB, * tC;
    tensor * tfA, * tfB, * tfC;
    int * tidx_A, * tidx_B, * tidx_C;

    get_fold_ctr(fold_ctr, all_fdim_A, all_fdim_B, all_fdim_C,
                           all_flen_A, all_flen_B, all_flen_C);

  #if DEBUG>=2
    if (do_transp){
      CommData global_comm = A->wrld->cdt;
      if (global_comm.rank == 0){
        printf("Folded contraction type:\n");
      }
      fold_ctr->print();
    }
  #endif
    select_ctr_perm(fold_ctr, all_fdim_A, all_fdim_B, all_fdim_C,
                              all_flen_A, all_flen_B, all_flen_C,
                    bperm_order, btime, iprm);
    if (A->is_sparse || B->is_sparse || C->is_sparse){
      bperm_order = 1;
      iprm.tA = 'N';
      iprm.tB = 'N';
      iprm.tC = 'N';
    }
//    printf("bperm_order = %d\n", bperm_order);
    get_perm<tensor*>(bperm_order, A, B, C, 
                     tA, tB, tC);
    get_perm<tensor*>(bperm_order, fold_ctr->A, fold_ctr->B, fold_ctr->C, 
                     tfA, tfB, tfC);
    get_perm<int*>(bperm_order, fold_ctr->idx_A, fold_ctr->idx_B, fold_ctr->idx_C,
                         tidx_A, tidx_B, tidx_C);
    get_perm<int>(bperm_order, all_fdim_A, all_fdim_B, all_fdim_C,
                        tall_fdim_A, tall_fdim_B, tall_fdim_C);

    get_len_ordering(tfA, tfB, tfC, tidx_A, tidx_B, tidx_C, 
                     &fnew_ord_A, &fnew_ord_B, &fnew_ord_C); 

    permute_target(tfA->order, fnew_ord_A, tA->inner_ordering);
    permute_target(tfB->order, fnew_ord_B, tB->inner_ordering);
    permute_target(tfC->order, fnew_ord_C, tC->inner_ordering);
    
    if (do_transp){
      nvirt_A = A->calc_nvirt();
      if (!A->is_sparse){
        for (i=0; i<nvirt_A; i++){
          nosym_transpose(all_fdim_A, A->inner_ordering, all_flen_A,
                          A->data + A->sr->el_size*i*(A->size/nvirt_A), 1, A->sr);
        }
      } else {
        int64_t new_sz_A = 0;
        A->rec_tsr->is_sparse = 1;
        A->rec_tsr->nnz_blk = (int64_t*)alloc(nvirt_A*sizeof(int64_t));
        for (i=0; i<nvirt_A; i++){
          if (A->sr->has_csrmm)
            A->rec_tsr->nnz_blk[i] = get_csr_size(A->nnz_blk[i], iprm.m, A->sr->el_size); 
          else
            A->rec_tsr->nnz_blk[i] = get_coo_size(A->nnz_blk[i], A->sr->el_size); 
          new_sz_A += A->rec_tsr->nnz_blk[i];
        }
        A->rec_tsr->data = (char*)alloc(new_sz_A);
        A->rec_tsr->is_data_aliased = false;
        int nrow_idx = 0;
        int phase[A->order];
        for (i=0; i<A->order; i++){
          phase[i] = A->edge_map[i].calc_phase();
          for (int j=0; j<C->order; j++){
            if (idx_A[i] == idx_C[j]) nrow_idx++;
          }
        }
        char * data_ptr_out = A->rec_tsr->data;
        char const * data_ptr_in = A->data;
        for (i=0; i<nvirt_A; i++){
          if (A->sr->has_csrmm){
            COO_Matrix cm(A->nnz_blk[i], A->sr);
            cm.set_data(A->nnz_blk[i], A->order, A->lens, A->inner_ordering, nrow_idx, data_ptr_in, A->sr, phase);
            CSR_Matrix cs(cm, iprm.m, A->sr, data_ptr_out);
            cdealloc(cm.all_data);
          } else {
            COO_Matrix cm(data_ptr_out);
            cm.set_data(A->nnz_blk[i], A->order, A->lens, A->inner_ordering, nrow_idx, data_ptr_in, A->sr, phase);
          }
          data_ptr_in += A->nnz_blk[i]*A->sr->pair_size();
          data_ptr_out += A->rec_tsr->nnz_blk[i];
        }
      }
      nvirt_B = B->calc_nvirt();
      for (i=0; i<nvirt_B; i++){
        nosym_transpose(all_fdim_B, B->inner_ordering, all_flen_B,
                        B->data + B->sr->el_size*i*(B->size/nvirt_B), 1, B->sr);
      }
      nvirt_C = C->calc_nvirt();
      for (i=0; i<nvirt_C; i++){
        nosym_transpose(all_fdim_C, C->inner_ordering, all_flen_C,
                        C->data + C->sr->el_size*i*(C->size/nvirt_C), 1, C->sr);
      }
    }

    CTF_int::cdealloc(fnew_ord_A);
    CTF_int::cdealloc(fnew_ord_B);
    CTF_int::cdealloc(fnew_ord_C);
    CTF_int::cdealloc(all_flen_A);
    CTF_int::cdealloc(all_flen_B);
    CTF_int::cdealloc(all_flen_C);
    delete fold_ctr;

    return iprm;
  }


  double contraction::est_time_fold(){
    int all_fdim_A, all_fdim_B, all_fdim_C;
    int * all_flen_A, * all_flen_B, * all_flen_C;
    iparam iprm;
    contraction * fold_ctr;
    int bperm_order = -1;
    double btime = DBL_MAX;

    get_fold_ctr(fold_ctr, all_fdim_A, all_fdim_B, all_fdim_C,
                           all_flen_A, all_flen_B, all_flen_C);
 
    select_ctr_perm(fold_ctr, all_fdim_A, all_fdim_B, all_fdim_C,
                              all_flen_A, all_flen_B, all_flen_C,
                    bperm_order, btime, iprm);

    CTF_int::cdealloc(all_flen_A);
    CTF_int::cdealloc(all_flen_B);
    CTF_int::cdealloc(all_flen_C);
    delete fold_ctr;
    A->remove_fold();
    B->remove_fold();
    C->remove_fold();

    return btime;
  }



  int contraction::unfold_broken_sym(contraction ** new_contraction){
    int i, num_tot, iA, iB, iC;
    int * idx_arr;
    tensor * nA, * nB, * nC;
   
    contraction * nctr; 
    
    if (new_contraction != NULL){
      nA = new tensor(A, 0, 0);
      nB = new tensor(B, 0, 0);
      nC = new tensor(C, 0, 0);
      nctr = new contraction(nA, idx_A, nB, idx_B, alpha, nC, idx_C, beta, func);
      *new_contraction = nctr;

      nA->clear_mapping();
      nA->set_padding();
      copy_mapping(A->order, A->edge_map, nA->edge_map);
      nA->is_mapped = 1;
      nA->topo      = A->topo;
      nA->set_padding();

      nB->clear_mapping();
      nB->set_padding();
      copy_mapping(B->order, B->edge_map, nB->edge_map);
      nB->is_mapped = 1;
      nB->topo      = B->topo;
      nB->set_padding();

      nC->clear_mapping();
      nC->set_padding();
      copy_mapping(C->order, C->edge_map, nC->edge_map);
      nC->is_mapped = 1;
      nC->topo      = C->topo;
      nC->set_padding();
    } else {
      nA = NULL;
      nB = NULL;
      nC = NULL;
    }

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    int nA_sym[A->order];
    if (new_contraction != NULL)
      memcpy(nA_sym, nA->sym, sizeof(int)*nA->order);
    for (i=0; i<A->order; i++){
      if (A->sym[i] != NS){
        iA = idx_A[i];
        if (idx_arr[3*iA+1] != -1){
          if (B->sym[idx_arr[3*iA+1]] != A->sym[i] ||
              idx_A[i+1] != idx_B[idx_arr[3*iA+1]+1]){
            if (new_contraction != NULL){
              nA_sym[i] = NS;
              nA->set_sym(nA_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*idx_A[i+1]+1] != -1){
            if (new_contraction != NULL){
              nA_sym[i] = NS;
              nA->set_sym(nA_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }       
        }
        if (idx_arr[3*iA+2] != -1){
          if (C->sym[idx_arr[3*iA+2]] != A->sym[i] ||
              idx_A[i+1] != idx_C[idx_arr[3*iA+2]+1]){
            if (new_contraction != NULL){
              nA_sym[i] = NS;
              nA->set_sym(nA_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }
        } else {
          if (idx_arr[3*idx_A[i+1]+2] != -1){
            if (new_contraction != NULL){
              nA_sym[i] = NS;
              nA->set_sym(nA_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i;
          }       
        }
      }
    }

   
    int nB_sym[B->order];
    if (new_contraction != NULL)
      memcpy(nB_sym, nB->sym, sizeof(int)*nB->order);
    for (i=0; i<B->order; i++){
      if (B->sym[i] != NS){
        iB = idx_B[i];
        if (idx_arr[3*iB+0] != -1){
          if (A->sym[idx_arr[3*iB+0]] != B->sym[i] ||
              idx_B[i+1] != idx_A[idx_arr[3*iB+0]+1]){
            if (new_contraction != NULL){
              nB_sym[i] = NS;
              nB->set_sym(nB_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*idx_B[i+1]+0] != -1){
            if (new_contraction != NULL){
              nB_sym[i] = NS;
              nB->set_sym(nB_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }       
        }
        if (idx_arr[3*iB+2] != -1){
          if (C->sym[idx_arr[3*iB+2]] != B->sym[i] || 
              idx_B[i+1] != idx_C[idx_arr[3*iB+2]+1]){
            if (new_contraction != NULL){
              nB_sym[i] = NS;
              nB->set_sym(nB_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }
        } else {
          if (idx_arr[3*idx_B[i+1]+2] != -1){
            if (new_contraction != NULL){
              nB_sym[i] = NS;
              nB->set_sym(nB_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i+1;
          }       
        }
      }
    } 
    //if A=B, output symmetry may still be preserved, so long as all indices in A and B are proper
    bool is_preserv = true;
    if (A != B) is_preserv = false; 
    else {
      for (int j=0; j<A->order; j++){
        if (idx_A[j] != idx_B[j]){
          iA = idx_A[j];
          iB = idx_B[j];
          if (idx_arr[3*iA+2] == -1 || idx_arr[3*iB+2] == -1) is_preserv = false;
          else {
            for (int k=MIN(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k<MAX(idx_arr[3*iA+2],idx_arr[3*iB+2]);
                     k++){
               if (C->sym[k] != SY) is_preserv = false;
            }
          }
        }
      }
    }
    if (!is_preserv){
      int nC_sym[C->order];
      if (new_contraction != NULL)
        memcpy(nC_sym, nC->sym, sizeof(int)*nC->order);
      for (i=0; i<C->order; i++){
        if (C->sym[i] != NS){
          iC = idx_C[i];
          if (idx_arr[3*iC+1] != -1){
            if (B->sym[idx_arr[3*iC+1]] != C->sym[i] ||
                idx_C[i+1] != idx_B[idx_arr[3*iC+1]+1]){
              if (new_contraction != NULL){
                nC_sym[i] = NS;
                nC->set_sym(nC_sym);
              }
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*idx_C[i+1]+1] != -1){
            if (new_contraction != NULL){
              nC_sym[i] = NS;
              nC->set_sym(nC_sym);
            }
            CTF_int::cdealloc(idx_arr); 
            return 3*i+2;
          }       
          if (idx_arr[3*iC+0] != -1){
            if (A->sym[idx_arr[3*iC+0]] != C->sym[i] ||
                idx_C[i+1] != idx_A[idx_arr[3*iC+0]+1]){
              if (new_contraction != NULL){
                nC_sym[i] = NS;
                nC->set_sym(nC_sym);
              }
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }
          } else if (idx_arr[3*iC+0] == -1){
            if (idx_arr[3*idx_C[i+1]] != -1){
              if (new_contraction != NULL){
                nC_sym[i] = NS;
                nC->set_sym(nC_sym);
              }
              CTF_int::cdealloc(idx_arr); 
              return 3*i+2;
            }       
          }
        }
      }
    }
    CTF_int::cdealloc(idx_arr);
    return -1;
  }

  void contraction::check_consistency(){
    int i, num_tot, len;
    int iA, iB, iC;
    int * idx_arr;
       
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    for (i=0; i<num_tot; i++){
      len = -1;
      iA = idx_arr[3*i+0];
      iB = idx_arr[3*i+1];
      iC = idx_arr[3*i+2];
      if (iA != -1){
        len = A->lens[iA];
      }
      if (len != -1 && iB != -1 && len != B->lens[iB]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s does not",
                  iA, A->name);
          printf("match the %dth edge length of tensor %s.\n",
                  iB, B->name);
        }
        ABORT;
      }
      if (len != -1 && iC != -1 && len != C->lens[iC]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s (%d) does not",
                  iA, A->name, len);
          printf("match the %dth edge length of tensor %s (%d).\n",
                  iC, C->name, C->lens[iC]);
        }
        ABORT;
      }
      if (iB != -1){
        len = B->lens[iB];
      }
      if (len != -1 && iC != -1 && len != C->lens[iC]){
        if (A->wrld->cdt.rank == 0){
          printf("Error in contraction call: The %dth edge length of tensor %s does not",
                  iB, B->name);
          printf("match the %dth edge length of tensor %s.\n",
                  iC, C->name);
        }
        ABORT;
      }
    }
    CTF_int::cdealloc(idx_arr);
  }

    
  int contraction::check_mapping(){

    int num_tot, i, ph_A, ph_B, iA, iB, iC, pass, order, topo_order;
    int * idx_arr;
    int * phys_mismatched, * phys_mapped;
    mapping * map;
    tensor * pA, * pB;

    pass = 1;

    if (A->is_mapped == 0) pass = 0;
    if (B->is_mapped == 0) pass = 0;
    if (C->is_mapped == 0) pass = 0;
    ASSERT(pass==1);
    
    if (A->is_folded == 1) pass = 0;
    if (B->is_folded == 1) pass = 0;
    if (C->is_folded == 1) pass = 0;
    
    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    if (A->topo != B->topo) pass = 0;
    if (A->topo != C->topo) pass = 0;

    if (pass==0){
      DPRINTF(3,"failed confirmation here\n");
      return 0;
    }

    topo_order = A->topo->order;
    CTF_int::alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mismatched);
    CTF_int::alloc_ptr(sizeof(int)*topo_order, (void**)&phys_mapped);
    memset(phys_mismatched, 0, sizeof(int)*topo_order);
    memset(phys_mapped, 0, sizeof(int)*topo_order);


    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    
    if (!check_self_mapping(A, idx_A))
      pass = 0;
    if (!check_self_mapping(B, idx_B))
      pass = 0;
    if (!check_self_mapping(C, idx_C))
      pass = 0;
    if (pass == 0){
      DPRINTF(3,"failed confirmation here\n");
    }


    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] != -1 &&
          idx_arr[3*i+1] != -1 &&
          idx_arr[3*i+2] != -1){
        iA = idx_arr[3*i+0];
        iB = idx_arr[3*i+1];
        iC = idx_arr[3*i+2];
  //      printf("A[%d].np = %d\n", iA, A->edge_map[iA].np);
        //printf("B[%d].np = %d\n", iB, B->edge_map[iB].np);
        //printf("C[%d].np = %d\n", iC, C->edge_map[iC].np);
        if (0 == comp_dim_map(&B->edge_map[iB], &A->edge_map[iA]) || 
            0 == comp_dim_map(&B->edge_map[iB], &C->edge_map[iC])){
          DPRINTF(3,"failed confirmation here %d %d %d\n",iA,iB,iC);
          pass = 0;
          break;
        } else {
          map = &A->edge_map[iA];
          for (;;){
            if (map->type == PHYSICAL_MAP){
              if (phys_mapped[map->cdt] == 1){
                DPRINTF(3,"failed confirmation here %d\n",iA);
                pass = 0;
                break;
              } else {
                phys_mapped[map->cdt] = 1;
                phys_mismatched[map->cdt] = 1;
              }
            } else break;
            if (map->has_child) map = map->child;
            else break;
          } 
        }
      }
    }
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] == -1 ||
          idx_arr[3*i+1] == -1 ||
          idx_arr[3*i+2] == -1){
        for (order=0; order<3; order++){
          switch (order){
            case 0:
              pA = A;
              pB = B;
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+2];
              break;
            case 1:
              pA = A;
              pB = C;
              iA = idx_arr[3*i+0];
              iB = idx_arr[3*i+2];
              iC = idx_arr[3*i+1];
              break;
            case 2:
              pA = C;
              pB = B;
              iA = idx_arr[3*i+2];
              iB = idx_arr[3*i+1];
              iC = idx_arr[3*i+0];
              break;
          }
          if (iC == -1){
            if (iB == -1){
              if (iA != -1) {
                map = &pA->edge_map[iA];
                for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mapped[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here %d\n",iA);
                      pass = 0;
                      break;
                    } else
                      phys_mapped[map->cdt] = 1;
                  } else break;
                  if (map->has_child) map = map->child;
                  else break;
                } 
              }
            } else if (iA == -1){
              map = &pB->edge_map[iB];
              for (;;){
                if (map->type == PHYSICAL_MAP){
                if (phys_mapped[map->cdt] == 1){
                  DPRINTF(3,"failed confirmation here %d\n",iA);
                  pass = 0;
                  break;
                } else
                  phys_mapped[map->cdt] = 1;
              } else break;
              if (map->has_child) map = map->child;
              else break;
              } 
            } else { 
              /* Confirm that the phases of A and B 
                 over which we are contracting are the same */
              ph_A = pA->edge_map[iA].calc_phase();
              ph_B = pB->edge_map[iB].calc_phase();

              if (ph_A != ph_B){
                //if (global_comm.rank == 0) 
                  DPRINTF(3,"failed confirmation here iA=%d iB=%d\n",iA,iB);
                pass = 0;
                break;
              }
              /* If the mapping along this dimension is the same make sure
                 its mapped to a onto a unique free dimension */
              if (comp_dim_map(&pB->edge_map[iB], &pA->edge_map[iA])){
                map = &pB->edge_map[iB];
              if (map->type == PHYSICAL_MAP){
                if (phys_mapped[map->cdt] == 1){
                  DPRINTF(3,"failed confirmation here %d\n",iB);
                  pass = 0;
                } else
                  phys_mapped[map->cdt] = 1;
                } 
                /*if (map->has_child) {
                  if (map->child->type == PHYSICAL_MAP){
                    DPRINTF(3,"failed confirmation here %d, matched and folded physical mapping not allowed\n",iB);
                    pass = 0;
                  }
                }*/
              } else {
                /* If the mapping along this dimension is different, make sure
                   the mismatch is mapped onto unqiue physical dimensions */
                map = &pA->edge_map[iA];
                for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mismatched[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here i=%d iA=%d iB=%d\n",i,iA,iB);
                      pass = 0;
                      break;
                    } else
                      phys_mismatched[map->cdt] = 1;
                    if (map->has_child) 
                      map = map->child;
                    else break;
                  } else break;
                      }
                      map = &pB->edge_map[iB];
                            for (;;){
                  if (map->type == PHYSICAL_MAP){
                    if (phys_mismatched[map->cdt] == 1){
                      DPRINTF(3,"failed confirmation here i=%d iA=%d iB=%d\n",i,iA,iB);
                      pass = 0;
                      break;
                    } else
                      phys_mismatched[map->cdt] = 1;
                    if (map->has_child) 
                      map = map->child;
                    else break;
                  } else break;
                }
              }
            }
          }
        }
      }
    }
    for (i=0; i<topo_order; i++){
      if (phys_mismatched[i] == 1 && phys_mapped[i] == 0){
        DPRINTF(3,"failed confirmation here i=%d\n",i);
        pass = 0;
        break;
      }
  /*   if (phys_mismatched[i] == 0 && phys_mapped[i] == 0){
        DPRINTF(3,"failed confirmation here i=%d\n",i);
        pass = 0;
        break;
      }    */
    }


    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(phys_mismatched);
    CTF_int::cdealloc(phys_mapped);
    return pass;
  }

  /**
   * \brief map the indices over which we will be weighing
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C corresponding to every global index
   * \param idx_weigh specification of which indices are being contracted
   * \param num_tot total number of indices
   * \param num_weigh number of indices being contracted over
   * \param topo topology to map to
   */
  static int
      map_weigh_indices(int const *      idx_arr,
                        int const *      idx_weigh,
                        int              num_tot,
                        int              num_weigh,
                        topology const * topo,
                        tensor *         A,
                        tensor *         B,
                        tensor *         C){
    int tsr_order, iweigh, iA, iB, iC, i, j, k, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * weigh_map;

    TAU_FSTART(map_weigh_indices);

    tsr_order = num_weigh;

    
    for (i=0; i<num_weigh; i++){
      iweigh = idx_weigh[i];
      iA = idx_arr[iweigh*3+0];
      iB = idx_arr[iweigh*3+1];
      iC = idx_arr[iweigh*3+2];

      if (A->edge_map[iA].type == PHYSICAL_MAP ||
          B->edge_map[iB].type == PHYSICAL_MAP ||
          C->edge_map[iC].type == PHYSICAL_MAP)
        return NEGATIVE; 
    }  
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_int::alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_int::alloc_ptr(tsr_order*sizeof(mapping),            (void**)&weigh_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));
    extract_free_comms(topo, A->order, A->edge_map,
                             B->order, B->edge_map,
                       num_sub_phys_dims, &sub_phys_comm, &comm_idx);

    for (i=0; i<tsr_order; i++){ 
      weigh_map[i].type             = VIRTUAL_MAP; 
      weigh_map[i].has_child        = 0; 
      weigh_map[i].np               = 1; 
    }
    for (i=0; i<num_weigh; i++){
      iweigh = idx_weigh[i];
      iA = idx_arr[iweigh*3+0];
      iB = idx_arr[iweigh*3+1];
      iC = idx_arr[iweigh*3+2];

      
      weigh_map[i].np = lcm(weigh_map[i].np,A->edge_map[iA].np);
      weigh_map[i].np = lcm(weigh_map[i].np,B->edge_map[iB].np);
      weigh_map[i].np = lcm(weigh_map[i].np,C->edge_map[iC].np);

      tsr_edge_len[i] = A->pad_edge_len[iA];

      for (j=i+1; j<num_weigh; j++){
        jX = idx_arr[idx_weigh[j]*3+0];

        for (k=MIN(iA,jX); k<MAX(iA,jX); k++){
          if (A->sym[k] == NS)
            break;
        }
        if (k==MAX(iA,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+1];

        for (k=MIN(iB,jX); k<MAX(iB,jX); k++){
          if (B->sym[k] == NS)
            break;
        }
        if (k==MAX(iB,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }

        jX = idx_arr[idx_weigh[j]*3+2];

        for (k=MIN(iC,jX); k<MAX(iC,jX); k++){
          if (C->sym[k] == NS)
            break;
        }
        if (k==MAX(iC,jX)){ 
          tsr_sym_table[i*tsr_order+j] = 1;
          tsr_sym_table[j*tsr_order+i] = 1;
        }
      }
    }
    stat = map_tensor(num_sub_phys_dims,  tsr_order,
                      tsr_edge_len,       tsr_sym_table,
                      restricted,         sub_phys_comm,
                      comm_idx,           0,
                      weigh_map);

    if (stat == ERROR)
      return ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == SUCCESS){
      for (i=0; i<num_weigh; i++){
        iweigh = idx_weigh[i];
        iA = idx_arr[iweigh*3+0];
        iB = idx_arr[iweigh*3+1];
        iC = idx_arr[iweigh*3+2];

        copy_mapping(1, &weigh_map[i], &A->edge_map[iA]);
        copy_mapping(1, &weigh_map[i], &B->edge_map[iB]);
        copy_mapping(1, &weigh_map[i], &C->edge_map[iC]);
      }
    }
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(tsr_edge_len);
    CTF_int::cdealloc(tsr_sym_table);
    for (i=0; i<num_weigh; i++){
      weigh_map[i].clear();
    }
    CTF_int::cdealloc(weigh_map);
    //if (num_sub_phys_dims > 0)
    CTF_int::cdealloc(sub_phys_comm);
    CTF_int::cdealloc(comm_idx);

    TAU_FSTOP(map_weigh_indices);
    return stat;
  }
  /**
   * \brief map the indices over which we will be contracting
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_ctr specification of which indices are being contracted
   * \param num_tot total number of indices
   * \param num_ctr number of indices being contracted over
   * \param topo topology to map to
   */
  static int
  map_ctr_indices(int const *      idx_arr,
                  int const *      idx_ctr,
                  int              num_tot,
                  int              num_ctr,
                  topology const * topo,
                  tensor *         A,
                  tensor *         B){
    int tsr_order, ictr, iA, iB, i, j, jctr, jX, stat, num_sub_phys_dims;
    int * tsr_edge_len, * tsr_sym_table, * restricted, * comm_idx;
    CommData  * sub_phys_comm;
    mapping * ctr_map;

    TAU_FSTART(map_ctr_indices);

    tsr_order = num_ctr*2;

    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&restricted);
    CTF_int::alloc_ptr(tsr_order*sizeof(int),                (void**)&tsr_edge_len);
    CTF_int::alloc_ptr(tsr_order*tsr_order*sizeof(int),       (void**)&tsr_sym_table);
    CTF_int::alloc_ptr(tsr_order*sizeof(mapping),            (void**)&ctr_map);

    memset(tsr_sym_table, 0, tsr_order*tsr_order*sizeof(int));
    memset(restricted, 0, tsr_order*sizeof(int));

    for (i=0; i<tsr_order; i++){ 
      ctr_map[i].type             = VIRTUAL_MAP; 
      ctr_map[i].has_child        = 0; 
      ctr_map[i].np               = 1; 
    }
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

      copy_mapping(1, &A->edge_map[iA], &ctr_map[2*i+0]);
      copy_mapping(1, &B->edge_map[iB], &ctr_map[2*i+1]);
    }
  /*  for (i=0; i<tsr_order; i++){ 
      if (ctr_map[i].type == PHYSICAL_MAP) is_premapped = 1;
    }*/

    extract_free_comms(topo, A->order, A->edge_map,
                             B->order, B->edge_map,
                       num_sub_phys_dims, &sub_phys_comm, &comm_idx);
    

    /* Map a tensor of dimension 2*num_ctr, with symmetries among each pair.
     * Set the edge lengths and symmetries according to those in ctr dims of A and B.
     * This gives us a mapping for the contraction dimensions of tensors A and B. */
    for (i=0; i<num_ctr; i++){
      ictr = idx_ctr[i];
      iA = idx_arr[ictr*3+0];
      iB = idx_arr[ictr*3+1];

      tsr_edge_len[2*i+0] = A->pad_edge_len[iA];
      tsr_edge_len[2*i+1] = A->pad_edge_len[iA];

      tsr_sym_table[2*i*tsr_order+2*i+1] = 1;
      tsr_sym_table[(2*i+1)*tsr_order+2*i] = 1;

      /* Check if A has symmetry among the dimensions being contracted over.
       * Ignore symmetry with non-contraction dimensions.
       * FIXME: this algorithm can be more efficient but should not be a bottleneck */
      if (A->sym[iA] != NS){
        for (j=0; j<num_ctr; j++){
          jctr = idx_ctr[j];
          jX = idx_arr[jctr*3+0];
          if (jX == iA+1){
            tsr_sym_table[2*i*tsr_order+2*j] = 1;
            tsr_sym_table[2*i*tsr_order+2*j+1] = 1;
            tsr_sym_table[2*j*tsr_order+2*i] = 1;
            tsr_sym_table[2*j*tsr_order+2*i+1] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j+1] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i+1] = 1;
          }
        }
      }
      if (B->sym[iB] != NS){
        for (j=0; j<num_ctr; j++){
          jctr = idx_ctr[j];
          jX = idx_arr[jctr*3+1];
          if (jX == iB+1){
            tsr_sym_table[2*i*tsr_order+2*j] = 1;
            tsr_sym_table[2*i*tsr_order+2*j+1] = 1;
            tsr_sym_table[2*j*tsr_order+2*i] = 1;
            tsr_sym_table[2*j*tsr_order+2*i+1] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j] = 1;
            tsr_sym_table[(2*i+1)*tsr_order+2*j+1] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i] = 1;
            tsr_sym_table[(2*j+1)*tsr_order+2*i+1] = 1;
          }
        }
      }
    }
    /* Run the mapping algorithm on this construct */
    /*if (is_premapped){
      stat = map_symtsr(tsr_order, tsr_sym_table, ctr_map);
    } else {*/
      stat = map_tensor(num_sub_phys_dims,  tsr_order,
                        tsr_edge_len,       tsr_sym_table,
                        restricted,         sub_phys_comm,
                        comm_idx,           0,
                        ctr_map);

    //}
    if (stat == ERROR)
      return ERROR;
    
    /* define mapping of tensors A and B according to the mapping of ctr dims */
    if (stat == SUCCESS){
      for (i=0; i<num_ctr; i++){
        ictr = idx_ctr[i];
        iA = idx_arr[ictr*3+0];
        iB = idx_arr[ictr*3+1];

  /*      A->edge_map[iA] = ctr_map[2*i+0];
        B->edge_map[iB] = ctr_map[2*i+1];*/
        copy_mapping(1, &ctr_map[2*i+0], &A->edge_map[iA]);
        copy_mapping(1, &ctr_map[2*i+1], &B->edge_map[iB]);
      }
    }
    CTF_int::cdealloc(restricted);
    CTF_int::cdealloc(tsr_edge_len);
    CTF_int::cdealloc(tsr_sym_table);
    for (i=0; i<2*num_ctr; i++){
      ctr_map[i].clear();
    }
    CTF_int::cdealloc(ctr_map);
    CTF_int::cdealloc(sub_phys_comm);
    CTF_int::cdealloc(comm_idx);

    TAU_FSTOP(map_ctr_indices);
    return stat;
  }

  /**
   * \brief map the indices over which we will not be contracting
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_noctr specification of which indices are not being contracted
   * \param num_tot total number of indices
   * \param num_noctr number of indices not being contracted over
   * \param topo topology to map to
   */
  static int
  map_no_ctr_indices(int const *      idx_arr,
                     int const *      idx_no_ctr,
                     int              num_tot,
                     int              num_no_ctr,
                     topology const * topo,
                     tensor *         A,
                     tensor *         B,
                     tensor *         C){
    int stat, i, inoctr, iA, iB, iC;

    TAU_FSTART(map_noctr_indices);

  /*  for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iC != -1 && iA != -1){
        copy_mapping(1, C->edge_map + iC, A->edge_map + iA); 
      } 
      if (iB != -1 && iA != -1){
        copy_mapping(1, C->edge_map + iB, A->edge_map + iA); 
      }
    }*/
    /* Map remainders of A and B to remainders of phys grid */
    stat = A->map_tensor_rem(topo->order,  topo->dim_comm, 1);
    if (stat != SUCCESS){
      if (A->order != 0 || B->order != 0 || C->order != 0){
        TAU_FSTOP(map_noctr_indices);
        return stat;
      }
    }
    for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iA != -1 && iC != -1){
        copy_mapping(1, A->edge_map + iA, C->edge_map + iC); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, B->edge_map + iB, C->edge_map + iC); 
      } 
    }
    stat = C->map_tensor_rem(topo->order,  topo->dim_comm, 0);
    if (stat != SUCCESS){
      TAU_FSTOP(map_noctr_indices);
      return stat;
    }
    for (i=0; i<num_no_ctr; i++){
      inoctr = idx_no_ctr[i];
      iA = idx_arr[3*inoctr+0];
      iB = idx_arr[3*inoctr+1];
      iC = idx_arr[3*inoctr+2];

      
      if (iA != -1 && iC != -1){
        copy_mapping(1, C->edge_map + iC, A->edge_map + iA); 
      } 
      if (iB != -1 && iC != -1){
        copy_mapping(1, C->edge_map + iC, B->edge_map + iB); 
      }
    }
    TAU_FSTOP(map_noctr_indices);

    return SUCCESS;
  }


  /**
   * \brief map the indices which are indexed only for A or B or C
   *
   * \param idx_arr array of index mappings of size order*3 that
   *        lists the indices (or -1) of A,B,C 
   *        corresponding to every global index
   * \param idx_extra specification of which indices are not being contracted
   * \param num_extra number of indices not being contracted over
   */
  static int
      map_extra_indices(int const * idx_arr,
                        int const * idx_extra,
                        int         num_extra,
                        tensor *    A,
                        tensor *    B,
                        tensor *    C){
    int i, iA, iB, iC, iextra;


    for (i=0; i<num_extra; i++){
      iextra = idx_extra[i];
      iA = idx_arr[3*iextra+0];
      iB = idx_arr[3*iextra+1];
      iC = idx_arr[3*iextra+2];

      if (iA != -1){
        //FIXME handle extra indices via reduction
        if (A->edge_map[iA].type == PHYSICAL_MAP)
          return NEGATIVE;
        if (A->edge_map[iA].type == NOT_MAPPED){
          A->edge_map[iA].type = VIRTUAL_MAP;
          A->edge_map[iA].np = 1;
          A->edge_map[iA].has_child = 0;
        }
      } else {
        if (iB != -1) {
          if (B->edge_map[iB].type == PHYSICAL_MAP)
            return NEGATIVE;
          if (B->edge_map[iB].type == NOT_MAPPED){
            B->edge_map[iB].type = VIRTUAL_MAP;
            B->edge_map[iB].np = 1;
            B->edge_map[iB].has_child = 0;
          }
        } else {
          ASSERT(iC != -1);
          if (C->edge_map[iC].type == PHYSICAL_MAP)
            return NEGATIVE;
          if (C->edge_map[iC].type == NOT_MAPPED){
            C->edge_map[iC].type = VIRTUAL_MAP;
            C->edge_map[iC].np = 1;
            C->edge_map[iC].has_child = 0;
          }
        }
      }
    }
    return SUCCESS;
  }

  int contraction::
      get_num_map_variants(topology const * topo){
    int nAB, nAC, nBC, nmax_ctr_2d; 
    return get_num_map_variants(topo, nmax_ctr_2d, nAB, nAC, nBC);
  }

  int contraction::
      get_num_map_variants(topology const * topo,
                           int &            nmax_ctr_2d,
                           int &            nAB,
                           int &            nAC,
                           int &            nBC){
    TAU_FSTART(get_num_map_vars);
    int num_tot;
    int * idx_arr;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    nAB=0;
    nAC=0;
    nBC=0;
  
    for (int i=0; i<num_tot; i++){
      if (idx_arr[3*i+0] != -1 && idx_arr[3*i+1] != -1 && idx_arr[3*i+2] == -1)
        nAB++;
      if (idx_arr[3*i+0] != -1 && idx_arr[3*i+1] == -1 && idx_arr[3*i+2] != -1)
        nAC++;
      if (idx_arr[3*i+0] == -1 && idx_arr[3*i+1] != -1 && idx_arr[3*i+2] != -1)
        nBC++;
    }
    nmax_ctr_2d = std::min(topo->order/2,std::min(nAC,std::min(nAB,nBC)));
    int nv=0;
    for (int nctr_2d=0; nctr_2d<=nmax_ctr_2d; nctr_2d++){
      if (topo->order-2*nctr_2d <= num_tot){
        nv += std::pow(3,nctr_2d)*choose(num_tot,topo->order-2*nctr_2d)*choose(nAB,nctr_2d)*choose(nAC,nctr_2d)*choose(nBC,nctr_2d);
      }
    }
    cdealloc(idx_arr);
    TAU_FSTOP(get_num_map_vars);
    return nv;
  }
  
  bool contraction::switch_topo_perm(){
    ASSERT(A->topo == B->topo && B->topo == C->topo);
    topology const * topo = A->topo;
    int new_order[topo->order*sizeof(int)];
    std::fill(new_order, new_order+topo->order, -1);
    int il=0;
    for (int i=0; i<A->order; i++){
      mapping const * map = A->edge_map+i;
      if (map->type == PHYSICAL_MAP && map->has_child && map->child->type == PHYSICAL_MAP){
        new_order[map->cdt] = il;
        il++;
        new_order[map->child->cdt] = il;
        il++;
      }
    }
    for (int i=0; i<B->order; i++){
      mapping const * map = B->edge_map+i;
      if (map->type == PHYSICAL_MAP && map->has_child && map->child->type == PHYSICAL_MAP){
        if (new_order[map->cdt] != -1 || new_order[map->child->cdt] != -1){
          if (new_order[map->child->cdt] != new_order[map->cdt]+1)
            return false;
        } else {
          new_order[map->cdt] = il;
          il++;
          new_order[map->child->cdt] = il;
          il++;
        }
      }
    }
    for (int i=0; i<C->order; i++){
      mapping const * map = C->edge_map+i;
      if (map->type == PHYSICAL_MAP && map->has_child && map->child->type == PHYSICAL_MAP){
        if (new_order[map->cdt] != -1 || new_order[map->child->cdt] != -1){
          if (new_order[map->child->cdt] != new_order[map->cdt]+1)
            return false;
        } else {
          new_order[map->cdt] = il;
          il++;
          new_order[map->child->cdt] = il;
          il++;
        }
      }
    }

    for (int i=0; i<topo->order; i++){
      if (new_order[i] == -1){
        new_order[i] = il;
        il++;
      }
    }
    int new_lens[topo->order];
    for (int i=0; i<topo->order; i++){
      new_lens[new_order[i]] = topo->lens[i];
//      printf("new_order[%d/%d] = %d, new_lens[%d] = %d\n", i, topo->order, new_order[i], new_order[i], new_lens[new_order[i]]);
    }
    topology * new_topo = NULL;
    for (int i=0; i<(int)A->wrld->topovec.size(); i++){
      if (A->wrld->topovec[i]->order == topo->order){
        bool has_same_len = true;
        for (int j=0; j<topo->order; j++){
          if (A->wrld->topovec[i]->lens[j] != new_lens[j]) has_same_len = false;
        }
        if (has_same_len){
          new_topo = A->wrld->topovec[i];
          break;
        }
      }
    }
    ASSERT(new_topo != NULL);
    A->topo = new_topo;
    B->topo = new_topo;
    C->topo = new_topo;
    for (int i=0; i<A->order; i++){
      mapping * map = A->edge_map + i;
      while (map != NULL && map->type == PHYSICAL_MAP){
        map->cdt = new_order[map->cdt];
        if (map->has_child) map = map->child;
        else map = NULL;
      }
    }
    for (int i=0; i<B->order; i++){
      mapping * map = B->edge_map + i;
      while (map != NULL && map->type == PHYSICAL_MAP){
        map->cdt = new_order[map->cdt];
        if (map->has_child) map = map->child;
        else map = NULL;
      }
    }
    for (int i=0; i<C->order; i++){
      mapping * map = C->edge_map + i;
      while (map != NULL && map->type == PHYSICAL_MAP){
        map->cdt = new_order[map->cdt];
        if (map->has_child) map = map->child;
        else map = NULL;
      }
    }
    return true;

  }
 
  bool contraction::
      exh_map_to_topo(topology const * topo,
                      int              variant){
   
    int num_tot;
    int * idx_arr;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    int nAB, nAC, nBC, nmax_ctr_2d; 
    int nvar_tot = get_num_map_variants(topo, nmax_ctr_2d, nAB, nAC, nBC);
    ASSERT(variant<nvar_tot);

    int nv=0;
    for (int nctr_2d=0; nctr_2d<=nmax_ctr_2d; nctr_2d++){
      int nv0 = nv;
      if (topo->order-2*nctr_2d <= num_tot){
        nv += std::pow(3,nctr_2d)*choose(num_tot,topo->order-2*nctr_2d)*choose(nAB,nctr_2d)*choose(nAC,nctr_2d)*choose(nBC,nctr_2d);
      }
      if (nv > variant){
        int v = variant - nv0;
        int rep_choices = choose(num_tot,topo->order-2*nctr_2d);
        int rep_ch = v%rep_choices;
        int rep_inds[topo->order-2*nctr_2d];
        get_choice(num_tot,topo->order-2*nctr_2d,rep_ch,rep_inds);
        for (int i=2*nctr_2d; i<topo->order; i++){
          int r = rep_inds[i-2*nctr_2d];
          if (idx_arr[3*r+0] != -1)
            A->edge_map[idx_arr[3*r+0]].aug_phys(topo, i);
          if (idx_arr[3*r+1] != -1)
            B->edge_map[idx_arr[3*r+1]].aug_phys(topo, i);
          if (idx_arr[3*r+2] != -1)
            C->edge_map[idx_arr[3*r+2]].aug_phys(topo, i);
        }
        int iAB[nctr_2d];
        int iAC[nctr_2d];
        int iBC[nctr_2d];
        int ord[nctr_2d];
        v = v/rep_choices;
        for (int i=0; i<nctr_2d; i++){
          ord[i] = v%3;
          v = v/3;
        }
        get_choice(nAB,nctr_2d,v%choose(nAB,nctr_2d),iAB);
        v = v/choose(nAB,nctr_2d);
        get_choice(nAC,nctr_2d,v%choose(nAC,nctr_2d),iAC);
        v = v/choose(nAC,nctr_2d);
        get_choice(nBC,nctr_2d,v%choose(nBC,nctr_2d),iBC);
        v = v/choose(nBC,nctr_2d);
        
        for (int i=0; i<nctr_2d; i++){
         // printf("iAB[%d] = %d iAC[%d] = %d iBC[%d] = %d ord[%d] = %d\n", i, iAB[i], i, iAC[i], i, iBC[i], i, ord[i]);
          int iiAB=0;
          int iiAC=0;
          int iiBC=0;
          for (int j=0; j<num_tot; j++){
            if (idx_arr[3*j+0] != -1 && idx_arr[3*j+1] != -1 && idx_arr[3*j+2] == -1){
              if (iAB[i] == iiAB){
                switch (ord[i]){
                  case 0:
                    A->edge_map[idx_arr[3*j+0]].aug_phys(topo, 2*i);
                    B->edge_map[idx_arr[3*j+1]].aug_phys(topo, 2*i+1);
                    break;
                  case 1:
                  case 2:
                    A->edge_map[idx_arr[3*j+0]].aug_phys(topo, 2*i);
                    B->edge_map[idx_arr[3*j+1]].aug_phys(topo, 2*i);
                    break;
                }
              }
              iiAB++;
            }
            if (idx_arr[3*j+0] != -1 && idx_arr[3*j+1] == -1 && idx_arr[3*j+2] != -1){
              if (iAC[i] == iiAC){
                switch (ord[i]){
                  case 0:
                    A->edge_map[idx_arr[3*j+0]].aug_phys(topo, 2*i+1);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i+1);
                  case 1:
                    break;
                    A->edge_map[idx_arr[3*j+0]].aug_phys(topo, 2*i+1);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i);
                    break;
                  case 2:
                    A->edge_map[idx_arr[3*j+0]].aug_phys(topo, 2*i+1);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i+1);
                    break;
                }
              }
              iiAC++;
            }
            if (idx_arr[3*j+0] == -1 && idx_arr[3*j+1] != -1 && idx_arr[3*j+2] != -1){
              if (iBC[i] == iiBC){
                switch (ord[i]){
                  case 2:
                    B->edge_map[idx_arr[3*j+1]].aug_phys(topo, 2*i+1);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i);
                    break;
                  case 0:
                    B->edge_map[idx_arr[3*j+1]].aug_phys(topo, 2*i);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i);
                    break;
                  case 1:
                    B->edge_map[idx_arr[3*j+1]].aug_phys(topo, 2*i+1);
                    C->edge_map[idx_arr[3*j+2]].aug_phys(topo, 2*i+1);
                    break;
                }
              }
              iiBC++;
            }
          }
        }
        break;
      }
    }
  /*  int num_totp = num_tot+1;
    int num_choices = num_totp*num_totp;
    int nv = variant;
    // go in reverse order to make aug_phys have potential to be correct order with multiple phys dims
    for (int idim=topo->order-1; idim>=0; idim--){
      int i = (nv%num_choices)/num_totp;
      int j = (nv%num_choices)%num_totp;
      nv = nv/num_choices;
      int iA = -1;
      int iB = -1;
      int iC = -1;
      if (i!=num_tot){
        iA = idx_arr[3*i+0];
        iB = idx_arr[3*i+1];
        iC = idx_arr[3*i+2];
        if (i==j){
          if (iA != -1 || iB != -1 || iC != -1) return false;
          else {
            A->edge_map[iA].aug_phys(topo, idim);
            B->edge_map[iB].aug_phys(topo, idim);
            C->edge_map[iC].aug_phys(topo, idim);
          }
        }
      } 
      int jA = -1;
      int jB = -1;
      int jC = -1;
      if (j!= num_tot && j!= i){
        jA = idx_arr[3*j+0];
        jB = idx_arr[3*j+1];
        jC = idx_arr[3*j+2];
      }
      if (i!=j){
        if (iA != -1) A->edge_map[iA].aug_phys(topo, idim);
        else if (jA != -1) A->edge_map[jA].aug_phys(topo, idim);
        if (iB != -1) B->edge_map[iB].aug_phys(topo, idim);
        else if (jB != -1) B->edge_map[jB].aug_phys(topo, idim);
        if (iC != -1) C->edge_map[iC].aug_phys(topo, idim);
        else if (jC != -1) C->edge_map[jC].aug_phys(topo, idim);
      }
    }*/
    
    
    //A->order*B->order*C->order+A->order*B->order+A->order*C->order+B->order*C->order+A->order+B->order+C->order+1;
/*    int nv = variant;
    for (int idim=0; idim<topo->order; idim++){
      int iv = nv%num_choices;
      nv = nv/num_choices;
      if (iv == 0) continue;
      iv--;
      if (iv < A->order+B->order+C->order){
        if (iv < A->order){
          A->edge_map[iv].aug_phys(topo, idim);
          continue;
        }
        iv -= A->order;
        if (iv < B->order){
          B->edge_map[iv].aug_phys(topo, idim);
          continue;
        }
        iv -= B->order;
        if (iv < C->order){
          C->edge_map[iv].aug_phys(topo, idim);
          continue;
        }
      }
      iv -= A->order+B->order+C->order;
      if (iv < A->order*B->order+A->order*C->order+B->order*C->order){
        if (iv < A->order*B->order){
          A->edge_map[iv%A->order].aug_phys(topo, idim);
          B->edge_map[iv/A->order].aug_phys(topo, idim);
          continue;
        }
        iv -= A->order*B->order;
        if (iv < A->order*C->order){
          A->edge_map[iv%A->order].aug_phys(topo, idim);
          C->edge_map[iv/A->order].aug_phys(topo, idim);
          continue;
        }
        iv -= A->order*C->order;
        if (iv < B->order*C->order){
          B->edge_map[iv%B->order].aug_phys(topo, idim);
          C->edge_map[iv/B->order].aug_phys(topo, idim);
          continue;
        }
      } 
      iv -= A->order*B->order+A->order*C->order+B->order*C->order;
      A->edge_map[iv%A->order].aug_phys(topo, idim);
      iv = iv/A->order;
      B->edge_map[iv%B->order].aug_phys(topo, idim);
      iv = iv/B->order;
      C->edge_map[iv%C->order].aug_phys(topo, idim);
    }   */ 
    bool is_mod = true;
    while (is_mod){
      is_mod = false;
      int ret;
      ret = map_symtsr(A->order, A->sym_table, A->edge_map);
      ASSERT(ret == SUCCESS); //if (ret!=SUCCESS) return ret;
      ret = map_symtsr(B->order, B->sym_table, B->edge_map);
      ASSERT(ret == SUCCESS); //if (ret!=SUCCESS) return ret;
      ret = map_symtsr(C->order, C->sym_table, C->edge_map);
      ASSERT(ret == SUCCESS); //if (ret!=SUCCESS) return ret;

      for (int i=0; i<num_tot; i++){
        int iA = idx_arr[3*i+0];
        int iB = idx_arr[3*i+1];
        int iC = idx_arr[3*i+2];
        int lcm_phase = 1;
        if (iA != -1) lcm_phase = lcm(lcm_phase,A->edge_map[iA].calc_phase());
        if (iB != -1) lcm_phase = lcm(lcm_phase,B->edge_map[iB].calc_phase());
        if (iC != -1) lcm_phase = lcm(lcm_phase,C->edge_map[iC].calc_phase());
        if (iA != -1 && lcm_phase != A->edge_map[iA].calc_phase()){
          A->edge_map[iA].aug_virt(lcm_phase);
          is_mod = true;
        }
        if (iB != -1 && lcm_phase != B->edge_map[iB].calc_phase()){
          B->edge_map[iB].aug_virt(lcm_phase);
          is_mod = true;
        }
        if (iC != -1 && lcm_phase != C->edge_map[iC].calc_phase()){
          C->edge_map[iC].aug_virt(lcm_phase);
          is_mod = true;
        }
        //printf("i=%d lcm_phase=%d\n",i,lcm_phase);
      }
    }
    cdealloc(idx_arr);
    return true;
  }

  int contraction::
      map_to_topology(topology * topo,
                      int        order){
      /*                int *      idx_ctr,
                      int *      idx_extra,
                      int *      idx_no_ctr,
                      int *      idx_weigh){*/
    int num_tot, num_ctr, num_no_ctr, num_weigh, num_extra, i, ret;
    int const * tidx_A, * tidx_B, * tidx_C;
    int * idx_arr, * idx_extra, * idx_no_ctr, * idx_weigh, * idx_ctr;

    tensor * tA, * tB, * tC;
    get_perm<tensor*>(order, A, B, C, tA, tB, tC);
    get_perm<const int*>(order, idx_A, idx_B, idx_C, tidx_A, tidx_B, tidx_C);
   
    inv_idx(tA->order, tidx_A,
            tB->order, tidx_B,
            tC->order, tidx_C,
            &num_tot, &idx_arr);

    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_no_ctr);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_extra);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_weigh);
    CTF_int::alloc_ptr(sizeof(int)*num_tot,         (void**)&idx_ctr);
    num_ctr = 0, num_no_ctr = 0, num_extra = 0, num_weigh = 0;
    for (i=0; i<num_tot; i++){
      if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1 && idx_arr[3*i+2] != -1){
        idx_weigh[num_weigh] = i;
        num_weigh++;
      } else if (idx_arr[3*i] != -1 && idx_arr[3*i+1] != -1){
        idx_ctr[num_ctr] = i;
        num_ctr++;
      } else if (idx_arr[3*i+2] != -1 &&  
                  ((idx_arr[3*i+0] != -1) || (idx_arr[3*i+1] != -1))){
        idx_no_ctr[num_no_ctr] = i;
        num_no_ctr++;
      } else {
        idx_extra[num_extra] = i;
        num_extra++;
      }
    }
    tA->topo = topo;
    tB->topo = topo;
    tC->topo = topo;
    
    /* Map the weigh indices of A, B, and C*/


    int stat;
    ret = map_weigh_indices(idx_arr, idx_weigh, num_tot, num_weigh, topo, tA, tB, tC);
    do {
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }

      
      /* Map the contraction indices of A and B */
      ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, topo, tA, tB);
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }


    /*  ret = map_self_indices(tA, tidx_A);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }
      ret = map_self_indices(tB, tidx_B);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }
      ret = map_self_indices(tC, tidx_C);
      if (ret == NEGATIVE) {
        CTF_int::cdealloc(idx_arr);
        return NEGATIVE;
      }
      if (ret == ERROR) {
        CTF_int::cdealloc(idx_arr);
        return ERROR;
      }*/
      ret = map_extra_indices(idx_arr, idx_extra, num_extra, tA, tB, tC);
      if (ret == NEGATIVE) {
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        stat = ret;
        break;
      }


      /* Map C or equivalently, the non-contraction indices of A and B */
      ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, topo, tA, tB, tC);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }
      ret = map_symtsr(tA->order, tA->sym_table, tA->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tB->order, tB->sym_table, tB->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tC->order, tC->sym_table, tC->edge_map);
      if (ret!=SUCCESS) return ret;

      /* Do it again to make sure everything is properly mapped. FIXME: loop */
      ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr, topo, tA ,tB);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }
      ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr, topo, tA, tB, tC);
      if (ret == NEGATIVE){
        stat = ret;
        break;
      }
      if (ret == ERROR) {
        return ERROR;
      }

      /*ret = map_ctr_indices(idx_arr, idx_ctr, num_tot, num_ctr,
                                tA, tB, topo);*/
      /* Map C or equivalently, the non-contraction indices of A and B */
      /*ret = map_no_ctr_indices(idx_arr, idx_no_ctr, num_tot, num_no_ctr,
                                    tA, tB, tC, topo);*/
      ret = map_symtsr(tA->order, tA->sym_table, tA->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tB->order, tB->sym_table, tB->edge_map);
      if (ret!=SUCCESS) return ret;
      ret = map_symtsr(tC->order, tC->sym_table, tC->edge_map);
      if (ret!=SUCCESS) return ret;
      

      stat = SUCCESS;
    } while(0);

    cdealloc(idx_arr); cdealloc(idx_ctr); cdealloc(idx_extra); cdealloc(idx_no_ctr); cdealloc(idx_weigh);
    return stat;
  }

  int contraction::try_topo_morph(){
    topology * tA, * tB, * tC;
    int ret;
    tensor * tsr_keep, * tsr_change_A, * tsr_change_B;
    
    tA = A->topo;
    tB = B->topo;
    tC = C->topo;

    if (tA == tB && tB == tC){
      return SUCCESS;
    }

    if (tA->order >= tB->order){
      if (tA->order >= tC->order){
        tsr_keep = A;
        tsr_change_A = B;
        tsr_change_B = C;
      } else {
        tsr_keep = C;
        tsr_change_A = A;
        tsr_change_B = B;
      } 
    } else {
      if (tB->order >= tC->order){
        tsr_keep = B;
        tsr_change_A = A;
        tsr_change_B = C;
      } else {
        tsr_keep = C;
        tsr_change_A = A;
        tsr_change_B = B;
      }
    }
    
    tA = tsr_change_A->topo;
    tB = tsr_change_B->topo;
    tC = tsr_keep->topo;

    if (tA != tC){
      ret = can_morph(tC, tA);
      if (!ret)
        return NEGATIVE;
    }
    if (tB != tC){
      ret = can_morph(tC, tB);
      if (!ret)
        return NEGATIVE;
    }
    
    if (tA != tC){
      morph_topo(tC, tA,
                 tsr_change_A->order, tsr_change_A->edge_map);
      tsr_change_A->topo = tC;
    }
    if (tB != tC){
      morph_topo(tC, tB,
                 tsr_change_B->order, tsr_change_B->edge_map);
      tsr_change_B->topo = tC;
    }
    return SUCCESS;

  }

  void contraction::get_best_sel_map(distribution const * dA, distribution const * dB, distribution const * dC, topology * old_topo_A, topology * old_topo_B, topology * old_topo_C, mapping const * old_map_A, mapping const * old_map_B, mapping const * old_map_C, int & idx, double & time){
    int ret, j, d;
    int need_remap_A, need_remap_B, need_remap_C;
    int64_t memuse;//, bmemuse;
    double est_time, best_time;
    int btopo;
    bool is_ctr_sparse = A->is_sparse || B->is_sparse || C->is_sparse;
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;
    btopo = -1;
    best_time = DBL_MAX;
    int num_tot;
    int * idx_arr; 
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    cdealloc(idx_arr);
    int64_t max_memuse = proc_bytes_available();
    for (j=0; j<6; j++){
      // Attempt to map to all possible permutations of processor topology 
  #if DEBUG < 3 
      for (int t=global_comm.rank; t<(int)wrld->topovec.size()+3; t+=global_comm.np){
  #else
      for (int t=0; t<(int)wrld->topovec.size()+3; t++){
  #endif
        A->clear_mapping();
        B->clear_mapping();
        C->clear_mapping();
        A->set_padding();
        B->set_padding();
        C->set_padding();
      
        topology * topo_i = NULL;
        if (t < 3){
          switch (t){
            case 0:
            if (old_topo_A == NULL) continue;
            topo_i = old_topo_A;
            copy_mapping(A->order, old_map_A, A->edge_map);
            break;
          
            case 1:
            if (old_topo_B == NULL) continue;
            topo_i = old_topo_B;
            copy_mapping(B->order, old_map_B, B->edge_map);
            break;

            case 2:
            if (old_topo_C == NULL) continue;
            topo_i = old_topo_C;
            copy_mapping(C->order, old_map_C, C->edge_map);
            break;
          }
        } else topo_i = wrld->topovec[t-3];
      
        TAU_FSTART(map_ctr_to_topo);
        ret = map_to_topology(topo_i, j);
        TAU_FSTOP(map_ctr_to_topo);

        if (ret == ERROR) {
          TAU_FSTOP(select_ctr_map);
        }
        if (ret == NEGATIVE){
          //printf("map_to_topology returned negative\n");
          continue;
        }
    
        A->is_mapped = 1;
        B->is_mapped = 1;
        C->is_mapped = 1;
        A->topo = topo_i;
        B->topo = topo_i;
        C->topo = topo_i;
        
        TAU_FSTART(check_ctr_mapping);
        if (check_mapping() == 0){ 
          TAU_FSTOP(check_ctr_mapping);
          continue;
        }
        TAU_FSTOP(check_ctr_mapping);
        est_time = 0.0;
        TAU_FSTART(est_ctr_map_time);
        A->set_padding();
        B->set_padding();
        C->set_padding();
  #if DEBUG >= 4
        printf("\nTest mappings:\n");
        A->print_map(stdout, 0);
        B->print_map(stdout, 0);
        C->print_map(stdout, 0);
  #endif
        ctr * sctr = construct_ctr();
        double nnz_frac_A = 1.0;
        double nnz_frac_B = 1.0;
        double nnz_frac_C = 1.0;
        if (A->is_sparse) nnz_frac_A = std::min(1.0, (std::max(5.,log2(A->calc_npe()))*A->nnz_tot)/(A->size*A->calc_npe()));
        if (B->is_sparse) nnz_frac_B = std::min(1.0, (std::max(5.,log2(B->calc_npe()))*B->nnz_tot)/(B->size*B->calc_npe()));
        if (C->is_sparse) nnz_frac_C = std::min(1.0, (std::max(5.,log2(C->calc_npe()))*C->nnz_tot)/(C->size*C->calc_npe()));
        if (is_ctr_sparse){
          est_time = ((spctr*)sctr)->est_time_rec(sctr->num_lyr, nnz_frac_A, nnz_frac_B, nnz_frac_C);
        } else { 
          est_time = sctr->est_time_rec(sctr->num_lyr);
        }
  #if FOLD_TSR
        if (!is_custom && can_fold()){
          est_time = est_time_fold();
          iparam prm = map_fold(false);
          ctr * sctrf = construct_ctr(1, &prm);
          est_time += sctrf->est_time_rec(sctrf->num_lyr);
          delete sctrf;
          A->remove_fold();
          B->remove_fold();
          C->remove_fold();
        }
  #endif
  #if DEBUG >= 4
        printf("mapping passed contr est_time = %E sec\n", est_time);
  #endif 
        ASSERT(est_time >= 0.0);
        memuse = 0;
        need_remap_A = 0;
        need_remap_B = 0;
        need_remap_C = 0;
        if (topo_i == old_topo_A){
          for (d=0; d<A->order; d++){
            if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
              need_remap_A = 1;
          }
        } else
          need_remap_A = 1;
        if (need_remap_A) {
          est_time += A->est_redist_time(*dA, nnz_frac_A); 
          memuse = A->get_redist_mem(*dA, nnz_frac_A);
        } else
          memuse = 0;
        if (topo_i == old_topo_B){
          for (d=0; d<B->order; d++){
            if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
              need_remap_B = 1;
          }
        } else
          need_remap_B = 1;
        if (need_remap_B) {
          est_time += B->est_redist_time(*dB, nnz_frac_B); 
          memuse = std::max(memuse,B->get_redist_mem(*dB, nnz_frac_B));
        }
        if (topo_i == old_topo_C){
          for (d=0; d<C->order; d++){
            if (!comp_dim_map(&C->edge_map[d],&old_map_C[d]))
              need_remap_C = 1;
          }
        } else
          need_remap_C = 1;
        if (need_remap_C) {
          est_time += 2.*C->est_redist_time(*dC, nnz_frac_C); 
          memuse = 2.*std::max(memuse,C->get_redist_mem(*dC, nnz_frac_C));
        }
        memuse = MAX((int64_t)sctr->mem_rec(), memuse);
  #if DEBUG >= 4
        printf("total (with redistribution and transp) est_time = %E\n", est_time);
  #endif
        ASSERT(est_time >= 0.0);

        TAU_FSTOP(est_ctr_map_time);
        TAU_FSTART(get_avail_res);
        if ((int64_t)memuse >= max_memuse){
          DPRINTF(2,"Not enough memory available for topo %d with order\n", j);
          TAU_FSTOP(get_avail_res);
          delete sctr;
          continue;
        } 
        TAU_FSTOP(get_avail_res);
        if (A->size > INT_MAX || B->size > INT_MAX || C->size > INT_MAX){
          DPRINTF(2,"MPI does not handle enough bits for topo %d with order\n", j);
          delete sctr;
          continue;
        }

        if (est_time < best_time) {
          best_time = est_time;
          //bmemuse = memuse;
          btopo = 6*t+j;
        }  
        delete sctr;
      }
    }
    TAU_FSTOP(select_ctr_map);
    TAU_FSTART(all_select_ctr_map);
    double gbest_time;
    MPI_Allreduce(&best_time, &gbest_time, 1, MPI_DOUBLE, MPI_MIN, global_comm.cm);
    if (best_time != gbest_time){
      btopo = INT_MAX;
    }
    int ttopo;
    MPI_Allreduce(&btopo, &ttopo, 1, MPI_INT, MPI_MIN, global_comm.cm);
    TAU_FSTOP(all_select_ctr_map);

    idx=ttopo;
    time=gbest_time;

  }

  void contraction::get_best_exh_map(distribution const * dA, distribution const * dB, distribution const * dC, topology * old_topo_A, topology * old_topo_B, topology * old_topo_C, mapping const * old_map_A, mapping const * old_map_B, mapping const * old_map_C, int & idx, double & time, double init_best_time=DBL_MAX){
    int d;
    int need_remap_A, need_remap_B, need_remap_C;
    int64_t memuse;//, bmemuse;
    double est_time, best_time;
    int btopo;
    bool is_ctr_sparse = A->is_sparse || B->is_sparse || C->is_sparse;
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;
    btopo = -1;
    best_time = init_best_time;
    int num_tot;
    int * idx_arr; 
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    cdealloc(idx_arr);
    int64_t tot_num_choices = 0;
    for (int i=0; i<(int)wrld->topovec.size(); i++){
     // tot_num_choices += pow(num_choices,(int)wrld->topovec[i]->order);
      tot_num_choices += get_num_map_variants(wrld->topovec[i]);
    }
    int64_t valid_mappings = 0;
    int64_t choice_offset = 0;
    int64_t max_memuse = proc_bytes_available();
    TAU_FSTOP(init_select_ctr_map);
    for (int i=0; i<(int)wrld->topovec.size(); i++){
//      int tnum_choices = pow(num_choices,(int) wrld->topovec[i]->order);
      int tnum_choices = get_num_map_variants(wrld->topovec[i]);

      int64_t old_off = choice_offset;
      choice_offset += tnum_choices;
      for (int j=0; j<tnum_choices; j++){
        if ((old_off + j)%global_comm.np != global_comm.rank)
          continue;
        A->clear_mapping();
        B->clear_mapping();
        C->clear_mapping();
        A->set_padding();
        B->set_padding();
        C->set_padding();
        topology * topo_i = wrld->topovec[i];
        TAU_FSTART(exh_map);
        bool br = exh_map_to_topo(topo_i, j);
        TAU_FSTOP(exh_map);
        if (!br) DPRINTF(3,"exh_map_to_topo returned false\n");
        if (!br) continue;
        A->is_mapped = 1;
        B->is_mapped = 1;
        C->is_mapped = 1;
        A->topo = topo_i;
        B->topo = topo_i;
        C->topo = topo_i;
        
        TAU_FSTART(switch_topo_perm);
        br = switch_topo_perm();
        TAU_FSTOP(switch_topo_perm);
        if (!br){ DPRINTF(3,"switch topo perm returned false\n"); }
        if (!br) continue;
        TAU_FSTART(check_ctr_mapping);
        if (check_mapping() == 0){ 
          TAU_FSTOP(check_ctr_mapping);
          continue;
        }
        TAU_FSTOP(check_ctr_mapping);
        valid_mappings++;
        est_time = 0.0;
        
        TAU_FSTART(est_ctr_map_time);
        A->set_padding();
        B->set_padding();
        C->set_padding();
  #if DEBUG >= 4
        printf("\nTest mappings:\n");
        A->print_map(stdout, 0);
        B->print_map(stdout, 0);
        C->print_map(stdout, 0);
  #endif

        double nnz_frac_A = 1.0;
        double nnz_frac_B = 1.0;
        double nnz_frac_C = 1.0;
        if (A->is_sparse) nnz_frac_A = std::min(1.0, (std::max(5.,log2(A->calc_npe()))*A->nnz_tot)/(A->size*A->calc_npe()));
        if (B->is_sparse) nnz_frac_B = std::min(1.0, (std::max(5.,log2(B->calc_npe()))*B->nnz_tot)/(B->size*B->calc_npe()));
        if (C->is_sparse) nnz_frac_C = std::min(1.0, (std::max(5.,log2(C->calc_npe()))*C->nnz_tot)/(C->size*C->calc_npe()));
        memuse = 0;
        need_remap_A = 0;
        need_remap_B = 0;
        need_remap_C = 0;
        if (topo_i == old_topo_A){
          for (d=0; d<A->order; d++){
            if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
              need_remap_A = 1;
          }
        } else
          need_remap_A = 1;
        if (need_remap_A) {
          est_time += A->est_redist_time(*dA, nnz_frac_A); 
          memuse = A->get_redist_mem(*dA, nnz_frac_A);
        } else
          memuse = 0;
        if (topo_i == old_topo_B){
          for (d=0; d<B->order; d++){
            if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
              need_remap_B = 1;
          }
        } else
          need_remap_B = 1;
        if (need_remap_B) {
          est_time += B->est_redist_time(*dB, nnz_frac_B); 
          memuse = std::max(memuse,B->get_redist_mem(*dB, nnz_frac_B));
        }
        if (topo_i == old_topo_C){
          for (d=0; d<C->order; d++){
            if (!comp_dim_map(&C->edge_map[d],&old_map_C[d]))
              need_remap_C = 1;
          }
        } else
          need_remap_C = 1;
        if (need_remap_C) {
          est_time += 2.*C->est_redist_time(*dC, nnz_frac_C); 
          memuse = 2.*std::max(memuse,C->get_redist_mem(*dC, nnz_frac_C));
        }
 
        if (est_time >= best_time) continue;

        ctr * sctr = construct_ctr();
        if (is_ctr_sparse){
          est_time = ((spctr*)sctr)->est_time_rec(sctr->num_lyr, nnz_frac_A, nnz_frac_B, nnz_frac_C);
        } else { 
          est_time = sctr->est_time_rec(sctr->num_lyr);
        }
  #if FOLD_TSR
        if (!is_custom && can_fold()){
          est_time = est_time_fold();
          iparam prm = map_fold(false);
          ctr * sctrf = construct_ctr(1, &prm);
          est_time += sctrf->est_time_rec(sctrf->num_lyr);
          delete sctrf;
          A->remove_fold();
          B->remove_fold();
          C->remove_fold();
        }
  #endif
  #if DEBUG >= 4
        printf("mapping passed contr est_time = %E sec\n", est_time);
  #endif 
        memuse = MAX((int64_t)sctr->mem_rec(), memuse);
  #if DEBUG >= 4
        printf("total (with redistribution and transp) est_time = %E\n", est_time);
  #endif
        ASSERT(est_time >= 0.0);

        TAU_FSTOP(est_ctr_map_time);
        TAU_FSTART(get_avail_res);
        if ((int64_t)memuse >= max_memuse){
          DPRINTF(2,"Not enough memory available for topo %d with order %d\n", i, j);
          TAU_FSTOP(get_avail_res);
          delete sctr;
          continue;
        } 
        TAU_FSTOP(get_avail_res);
        if (A->size > INT_MAX || B->size > INT_MAX || C->size > INT_MAX){
          DPRINTF(2,"MPI does not handle enough bits for topo %d with order %d \n", i, j);
          delete sctr;
          continue;
        }
        if (est_time < best_time) {
          best_time = est_time;
          //bmemuse = memuse;
          btopo = old_off+j;
        }  
        delete sctr;
      }
    }
    TAU_FSTOP(select_ctr_map);
#if DEBUG >= 2 
    int64_t tot_valid_mappings;
    MPI_Allreduce(&valid_mappings, &tot_valid_mappings, 1, MPI_INT64_T, MPI_SUM, global_comm.cm);
    if (A->wrld->rank == 0) DPRINTF(2,"number valid mappings was %ld/%ld\n", tot_valid_mappings, tot_num_choices);
#endif
    TAU_FSTART(all_select_ctr_map);
    double gbest_time;
    MPI_Allreduce(&best_time, &gbest_time, 1, MPI_DOUBLE, MPI_MIN, global_comm.cm);
    if (best_time != gbest_time){
      btopo = INT_MAX;
    }
    int ttopo;
    MPI_Allreduce(&btopo, &ttopo, 1, MPI_INT, MPI_MIN, global_comm.cm);
    TAU_FSTOP(all_select_ctr_map);

    idx=ttopo;
    time=gbest_time;


  }

  int contraction::map(ctr ** ctrf, bool do_remap){
    int ret, j, need_remap, d;
    int * old_phase_A, * old_phase_B, * old_phase_C;
    topology * old_topo_A, * old_topo_B, * old_topo_C;
    distribution * dA, * dB, * dC;
    old_topo_A = A->topo;
    old_topo_B = B->topo;
    old_topo_C = C->topo;
    mapping * old_map_A = new mapping[A->order];
    mapping * old_map_B = new mapping[B->order];
    mapping * old_map_C = new mapping[C->order];
    copy_mapping(A->order, A->edge_map, old_map_A);
    copy_mapping(B->order, B->edge_map, old_map_B);
    copy_mapping(C->order, C->edge_map, old_map_C);

    ASSERT(A->wrld->comm == B->wrld->comm && B->wrld->comm == C->wrld->comm);
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;
    
    TAU_FSTART(select_ctr_map);
    TAU_FSTART(init_select_ctr_map);
  #if BEST_VOL
    CTF_int::alloc_ptr(sizeof(int)*A->order,     (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order,     (void**)&virt_blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order,     (void**)&virt_blk_len_C);
  #endif
    
    ASSERT(A->is_mapped);
    ASSERT(B->is_mapped);
    ASSERT(C->is_mapped);
    if (do_remap){
    #if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("Initial mappings:\n");
      A->print_map();
      B->print_map();
      C->print_map();
    #endif
    }
    A->unfold();
    B->unfold();
    C->unfold();
    A->set_padding();
    B->set_padding();
    C->set_padding();
    /* Save the current mappings of A, B, C */
    dA = new distribution(A);
    dB = new distribution(B);
    dC = new distribution(C);
    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&old_phase_A);
    for (j=0; j<A->order; j++){
      old_phase_A[j]   = A->edge_map[j].calc_phase();
    }
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&old_phase_B);
    for (j=0; j<B->order; j++){
      old_phase_B[j]   = B->edge_map[j].calc_phase();
    }
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&old_phase_C);
    for (j=0; j<C->order; j++){
      old_phase_C[j]   = C->edge_map[j].calc_phase();
    }

    //bmemuse = UINT64_MAX;
    int ttopo, ttopo_sel, ttopo_exh;
    double gbest_time_sel, gbest_time_exh;
  
    TAU_FSTART(get_best_sel_map);
    get_best_sel_map(dA, dB, dC, old_topo_A, old_topo_B, old_topo_C, old_map_A, old_map_B, old_map_C, ttopo_sel, gbest_time_sel);
    TAU_FSTOP(get_best_sel_map);
    if (gbest_time_sel < 1.){
      gbest_time_exh = gbest_time_sel+1.;
      ttopo_exh = ttopo_sel;
    } else {
      TAU_FSTART(get_best_exh_map);
      get_best_exh_map(dA, dB, dC, old_topo_A, old_topo_B, old_topo_C, old_map_A, old_map_B, old_map_C, ttopo_exh, gbest_time_exh, gbest_time_sel);
      TAU_FSTOP(get_best_exh_map);
    }
    if (gbest_time_sel <= gbest_time_exh){
      ttopo = ttopo_sel;
    } else {
      ttopo = ttopo_exh;
    }

    A->clear_mapping();
    B->clear_mapping();
    C->clear_mapping();
    A->set_padding();
    B->set_padding();
    C->set_padding();
    
    if (!do_remap || ttopo == INT_MAX || ttopo == -1){
      CTF_int::cdealloc(old_phase_A);
      CTF_int::cdealloc(old_phase_B);
      CTF_int::cdealloc(old_phase_C);
      delete [] old_map_A;
      delete [] old_map_B;
      delete [] old_map_C;
      delete dA;
      delete dB;
      delete dC;

      if (ttopo == INT_MAX || ttopo == -1){
        printf("ERROR: Failed to map contraction!\n");
        ASSERT(0);
        //ABORT;
        return ERROR;
      }
      return SUCCESS;
    }
    topology * topo_g;
    int j_g;
    if (gbest_time_sel <= gbest_time_exh){
      j_g = ttopo%6;
      if (ttopo < 18){
        switch (ttopo/6){
          case 0:
          topo_g = old_topo_A;
          copy_mapping(A->order, old_map_A, A->edge_map);
          break;
        
          case 1:
          topo_g = old_topo_B;
          copy_mapping(B->order, old_map_B, B->edge_map);
          break;

          case 2:
          topo_g = old_topo_C;
          copy_mapping(C->order, old_map_C, C->edge_map);
          break;
          
          default:
          topo_g = NULL;
          assert(0);
          break;
        }
      } else topo_g = wrld->topovec[(ttopo-18)/6];
    } else {
      int64_t choice_offset = 0;
      int i=0;
      int64_t old_off = 0;
      for (i=0; i<(int)wrld->topovec.size(); i++){
        //int tnum_choices = pow(num_choices,(int) wrld->topovec[i]->order);
        int tnum_choices = get_num_map_variants(wrld->topovec[i]);
        old_off = choice_offset;
        choice_offset += tnum_choices;
        if (choice_offset > ttopo) break;
      }
      topo_g = wrld->topovec[i];
      j_g = ttopo-old_off;
    }

    A->topo = topo_g;
    B->topo = topo_g;
    C->topo = topo_g;
    A->is_mapped = 1;
    B->is_mapped = 1;
    C->is_mapped = 1;
    
    if (gbest_time_sel <= gbest_time_exh){
      ret = map_to_topology(topo_g, j_g);
      if (ret == NEGATIVE || ret == ERROR) {
        printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
        return ERROR;
      }
    } else {
      exh_map_to_topo(topo_g, j_g);
      switch_topo_perm();
    }
  #if DEBUG > 2
    if (!check_mapping())
      printf("ERROR ON FINAL MAP ATTEMPT, THIS SHOULD NOT HAPPEN\n");
  //  else if (global_comm.rank == 0) printf("Mapping successful estimated execution time = %lf sec\n",best_time);
  #endif
    ASSERT(check_mapping());
    A->set_padding();
    B->set_padding();
    C->set_padding();
    *ctrf = construct_ctr();
    #if DEBUG > 2
    if (global_comm.rank == 0)
      printf("New mappings:\n");
    A->print_map(stdout);
    B->print_map(stdout);
    C->print_map(stdout);
    MPI_Barrier(global_comm.cm);
    #endif
     
    //FIXME: adhoc? 
    /*memuse = MAX((int64_t)(*ctrf)->mem_rec(), (int64_t)(A->size*A->sr->el_size+B->size*B->sr->el_size+C->size*C->sr->el_size)*3);
    if (global_comm.rank == 0)
      VPRINTF(1,"Contraction will use %E bytes per processor out of %E available memory and take an estimated of %lf sec\n",
              (double)memuse,(double)proc_bytes_available(),gbest_time);*/

    if (A->is_cyclic == 0 &&
        B->is_cyclic == 0 &&
        C->is_cyclic == 0){
      A->is_cyclic = 0;
      B->is_cyclic = 0;
      C->is_cyclic = 0;
    } else {
      A->is_cyclic = 1;
      B->is_cyclic = 1;
      C->is_cyclic = 1;
    }
    /* redistribute tensor data */
    TAU_FSTART(redistribute_for_contraction);
    need_remap = 0;
    if (A->topo == old_topo_A){
      for (d=0; d<A->order; d++){
        if (!comp_dim_map(&A->edge_map[d],&old_map_A[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      A->redistribute(*dA);
    need_remap = 0;
    if (B->topo == old_topo_B){
      for (d=0; d<B->order; d++){
        if (!comp_dim_map(&B->edge_map[d],&old_map_B[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      B->redistribute(*dB);
    need_remap = 0;
    if (C->topo == old_topo_C){
      for (d=0; d<C->order; d++){
        if (!comp_dim_map(&C->edge_map[d],&old_map_C[d]))
          need_remap = 1;
      }
    } else
      need_remap = 1;
    if (need_remap)
      C->redistribute(*dC);
                   
    TAU_FSTOP(redistribute_for_contraction);
    
    CTF_int::cdealloc( old_phase_A );
    CTF_int::cdealloc( old_phase_B );
    CTF_int::cdealloc( old_phase_C );
    
    delete [] old_map_A;
    delete [] old_map_B;
    delete [] old_map_C;

    
    delete dA;
    delete dB;
    delete dC;

    return SUCCESS;
  }


  ctr * contraction::construct_dense_ctr(int            is_inner,
                                         iparam const * inner_params,
                                         int *          nvirt_all,
                                         int            is_used,
                                         int const *    phys_mapped){
    int num_tot, i, i_A, i_B, i_C, is_top, nphys_dim;
    int64_t nvirt;
    int64_t blk_sz_A, blk_sz_B, blk_sz_C;
    int64_t vrt_sz_A, vrt_sz_B, vrt_sz_C;
    //int sA, sB, sC, 
    bool need_rep;
    int * blk_len_A, * virt_blk_len_A, * blk_len_B;
    int * virt_blk_len_B, * blk_len_C, * virt_blk_len_C;
    int * idx_arr, * virt_dim;
    //strp_tsr * str_A, * str_B, * str_C;
    mapping * map;
    ctr * hctr = NULL;
    ctr ** rec_ctr = NULL;
    ASSERT(A->wrld->comm == B->wrld->comm && B->wrld->comm == C->wrld->comm);
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;

    is_top = 1;
    nphys_dim = A->topo->order;

    TAU_FSTART(construct_contraction);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);


    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&virt_blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&virt_blk_len_C);

    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&blk_len_C);
    CTF_int::alloc_ptr(sizeof(int)*num_tot, (void**)&virt_dim);

    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = A->size;
    blk_sz_B = B->size;
    blk_sz_C = C->size;
    calc_dim(A->order, blk_sz_A, A->pad_edge_len, A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(B->order, blk_sz_B, B->pad_edge_len, B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);
    calc_dim(C->order, blk_sz_C, C->pad_edge_len, C->edge_map,
             &vrt_sz_C, virt_blk_len_C, blk_len_C);

    /* Strip out the relevant part of the tensor if we are contracting over diagonal */
/*    sA = strip_diag( A->order, num_tot, idx_A, vrt_sz_A,
                     A->edge_map, A->topo, A->sr,
                     blk_len_A, &blk_sz_A, &str_A);
    sB = strip_diag( B->order, num_tot, idx_B, vrt_sz_B,
                     B->edge_map, B->topo, B->sr,
                     blk_len_B, &blk_sz_B, &str_B);
    sC = strip_diag( C->order, num_tot, idx_C, vrt_sz_C,
                     C->edge_map, C->topo, C->sr,
                     blk_len_C, &blk_sz_C, &str_C);

    if (sA || sB || sC){
      ASSERT(0);//this is always done via sum now
      if (global_comm.rank == 0)
        DPRINTF(1,"Stripping tensor\n");
      strp_ctr * sctr = new strp_ctr;
      hctr = sctr;
      is_top = 0;
      rec_ctr = &sctr->rec_ctr;

      sctr->rec_strp_A = str_A;
      sctr->rec_strp_B = str_B;
      sctr->rec_strp_C = str_C;
      sctr->strip_A = sA;
      sctr->strip_B = sB;
      sctr->strip_C = sC;
    }*/

    need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[3*i+0] == 0 ||
        phys_mapped[3*i+1] == 0 ||
        phys_mapped[3*i+2] == 0){
        /*ASSERT((phys_mapped[3*i+0] == 0 && phys_mapped[3*i+1] == 0) ||
        (phys_mapped[3*i+0] == 0 && phys_mapped[3*i+2] == 0) ||
        (phys_mapped[3*i+1] == 0 && phys_mapped[3*i+2] == 0));*/
        need_rep = 1;
        break;
      }
    }
    if (need_rep){
      if (global_comm.rank == 0)
        DPRINTF(2,"Replicating tensor\n");

      ctr_replicate * rctr = new ctr_replicate(this, phys_mapped, blk_sz_A, blk_sz_B, blk_sz_C);
      if (is_top){
        hctr = rctr;
        is_top = 0;
      } else {
        *rec_ctr = rctr;
      }
      rec_ctr = &rctr->rec_ctr;
    }

  //#ifdef OFFLOAD
    int total_iter = 1;
    int upload_phase_A = 1;
    int upload_phase_B = 1;
    int download_phase_C = 1;
  //#endif
    nvirt = 1;

    ctr_2d_general * bottom_ctr_gen = NULL;
  /*  if (nvirt_all != NULL)
      *nvirt_all = 1;*/
    for (i=0; i<num_tot; i++){
      virt_dim[i] = 1;
      i_A = idx_arr[3*i+0];
      i_B = idx_arr[3*i+1];
      i_C = idx_arr[3*i+2];
      /* If this index belongs to exactly two tensors */
      if ((i_A != -1 && i_B != -1 && i_C == -1) ||
          (i_A != -1 && i_B == -1 && i_C != -1) ||
          (i_A == -1 && i_B != -1 && i_C != -1)) {
        ctr_2d_general * ctr_gen = new ctr_2d_general(this);
  #ifdef OFFLOAD
        ctr_gen->alloc_host_buf = false;
  #endif
        int is_built = 0;
        if (i_A == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C);
        }
        if (i_B == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A);
        }
        if (i_C == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      virt_blk_len_A,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B);
        }
        if (is_built){
          if (is_top){
            hctr = ctr_gen;
            is_top = 0;
          } else {
            *rec_ctr = ctr_gen;
          }
          if (bottom_ctr_gen == NULL)
            bottom_ctr_gen = ctr_gen;
          rec_ctr = &ctr_gen->rec_ctr;
        } else {
          ctr_gen->rec_ctr = NULL;
          delete ctr_gen;
        }
      } else {
        if (i_A != -1){
          map = &A->edge_map[i_A];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_B != -1){
          map = &B->edge_map[i_B];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_C != -1){
          map = &C->edge_map[i_C];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        }
      }
      /*if (sA && i_A != -1){
        nvirt = virt_dim[i]/str_A->strip_dim[i_A];
      } else if (sB && i_B != -1){
        nvirt = virt_dim[i]/str_B->strip_dim[i_B];
      } else if (sC && i_C != -1){
        nvirt = virt_dim[i]/str_C->strip_dim[i_C];
      }*/
      
      nvirt = nvirt * virt_dim[i];
    }
    if (nvirt_all != NULL)
      *nvirt_all = nvirt;
    bool do_offload = false;
  #ifdef OFFLOAD
    if (!is_custom && is_inner > 0 && C->sr->is_offloadable()){
      do_offload = true;
      if (bottom_ctr_gen != NULL)
        bottom_ctr_gen->alloc_host_buf = true;
      ctr_offload * ctroff = new ctr_offload(this, blk_sz_A, blk_sz_B, blk_sz_C, total_iter, upload_phase_A, upload_phase_B, download_phase_C);
      if (is_top){
        hctr = ctroff;
        is_top = 0;
      } else {
        *rec_ctr = ctroff;
      }
      rec_ctr = &ctroff->rec_ctr;
    }
  #endif



    ASSERT(blk_sz_A >= vrt_sz_A);
    ASSERT(blk_sz_B >= vrt_sz_B);
    ASSERT(blk_sz_C >= vrt_sz_C);
    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
      ctr_virt * ctrv = new ctr_virt(this, num_tot, virt_dim, vrt_sz_A, vrt_sz_B, vrt_sz_C);
      if (is_top) {
        hctr = ctrv;
        is_top = 0;
      } else {
        *rec_ctr = ctrv;
      }
      rec_ctr = &ctrv->rec_ctr;
    } else
      CTF_int::cdealloc(virt_dim);

    iparam const * iptr = NULL;
    iparam inp_cpy;
    if (is_inner){
      inp_cpy = *inner_params;
      inp_cpy.offload = do_offload;
      iptr = &inp_cpy;
    }
    seq_tsr_ctr * ctrseq = new seq_tsr_ctr(this, is_inner, iptr, virt_blk_len_A, virt_blk_len_B, virt_blk_len_C, vrt_sz_C);
    if (is_top) {
      hctr = ctrseq;
      is_top = 0;
    } else {
      *rec_ctr = ctrseq;
    }

/*    hctr->beta = this->beta;
    hctr->A    = A->data;
    hctr->B    = B->data;
    hctr->C    = C->data;*/
  /*  if (global_comm.rank == 0){
      int64_t n,m,k;
      dtype old_flops;
      dtype new_flops;
      ggg_sym_nmk(A->order, A->pad_edge_len, idx_A, A->sym,
      B->order, B->pad_edge_len, idx_B, B->sym,
      C->order, &n, &m, &k);
      old_flops = 2.0*(dtype)n*(dtype)m*(dtype)k;
      new_flops = A->calc_nvirt();
      new_flops *= B->calc_nvirt();
      new_flops *= C->calc_nvirt();
      new_flops *= global_comm.np;
      new_flops = sqrt(new_flops);
      new_flops *= global_comm.np;
      ggg_sym_nmk(A->order, virt_blk_len_A, idx_A, A->sym,
      B->order, virt_blk_len_B, idx_B, B->sym,
      C->order, &n, &m, &k);
      printf("Each subcontraction is a " PRId64 " by " PRId64 " by " PRId64 " DGEMM performing %E flops\n",n,m,k,
        2.0*(dtype)n*(dtype)m*(dtype)k);
      new_flops *= 2.0*(dtype)n*(dtype)m*(dtype)k;
      printf("Contraction performing %E flops rather than %E, a factor of %lf more flops due to padding\n",
        new_flops, old_flops, new_flops/old_flops);

    }*/

    CTF_int::cdealloc(idx_arr);
    CTF_int::cdealloc(blk_len_A);
    CTF_int::cdealloc(blk_len_B);
    CTF_int::cdealloc(blk_len_C);

    return hctr;
  }


  ctr * contraction::construct_sparse_ctr(int            is_inner,
                                          iparam const * inner_params,
                                          int *          nvirt_all,
                                          int            is_used,
                                          int const *    phys_mapped){
    int num_tot, i, i_A, i_B, i_C, nphys_dim, is_top, nvirt;
    int * idx_arr, * virt_dim;
    int64_t blk_sz_A, blk_sz_B, blk_sz_C;
    int64_t vrt_sz_A, vrt_sz_B, vrt_sz_C;
    int * blk_len_A, * virt_blk_len_A, * blk_len_B;
    int * virt_blk_len_B, * blk_len_C, * virt_blk_len_C;
    mapping * map;
    spctr * hctr = NULL;
    spctr ** rec_ctr = NULL;
    ASSERT(A->wrld->cdt.cm == B->wrld->cdt.cm && B->wrld->cdt.cm == C->wrld->cdt.cm);
    World * wrld = A->wrld;
    CommData global_comm = wrld->cdt;

    is_top = 1;
    nphys_dim = A->topo->order;

    TAU_FSTART(construct_contraction);

    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);

    nphys_dim = A->topo->order;

    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&virt_blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&virt_blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&virt_blk_len_C);

    CTF_int::alloc_ptr(sizeof(int)*A->order, (void**)&blk_len_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order, (void**)&blk_len_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order, (void**)&blk_len_C);
    CTF_int::alloc_ptr(sizeof(int)*num_tot, (void**)&virt_dim);

    /* Determine the block dimensions of each local subtensor */
    blk_sz_A = A->size;
    blk_sz_B = B->size;
    blk_sz_C = C->size;
    calc_dim(A->order, blk_sz_A, A->pad_edge_len, A->edge_map,
             &vrt_sz_A, virt_blk_len_A, blk_len_A);
    calc_dim(B->order, blk_sz_B, B->pad_edge_len, B->edge_map,
             &vrt_sz_B, virt_blk_len_B, blk_len_B);
    calc_dim(C->order, blk_sz_C, C->pad_edge_len, C->edge_map,
             &vrt_sz_C, virt_blk_len_C, blk_len_C);

    if (!is_inner){
      if (A->is_sparse && A->wrld->np > 1){
        spctr_pin_keys * skctr = new spctr_pin_keys(this, 0);
        if (is_top){
          hctr = skctr;
          is_top = 0;
        } else {
          *rec_ctr = skctr;
        }
        rec_ctr = &skctr->rec_ctr;
      }
  
      if (B->is_sparse && B->wrld->np > 1){
        spctr_pin_keys * skctr = new spctr_pin_keys(this, 1);
        if (is_top){
          hctr = skctr;
          is_top = 0;
        } else {
          *rec_ctr = skctr;
        }
        rec_ctr = &skctr->rec_ctr;
      }
  
      if (C->is_sparse && C->wrld->np > 1){
        spctr_pin_keys * skctr = new spctr_pin_keys(this, 1);
        if (is_top){
          hctr = skctr;
          is_top = 0;
        } else {
          *rec_ctr = skctr;
        }
        rec_ctr = &skctr->rec_ctr;
      }
    }


    bool need_rep = 0;
    for (i=0; i<nphys_dim; i++){
      if (phys_mapped[3*i+0] == 0 ||
        phys_mapped[3*i+1] == 0 ||
        phys_mapped[3*i+2] == 0){
        /*ASSERT((phys_mapped[3*i+0] == 0 && phys_mapped[3*i+1] == 0) ||
        (phys_mapped[3*i+0] == 0 && phys_mapped[3*i+2] == 0) ||
        (phys_mapped[3*i+1] == 0 && phys_mapped[3*i+2] == 0));*/
        need_rep = 1;
        break;
      }
    }
    if (need_rep){
      if (global_comm.rank == 0)
        DPRINTF(2,"Replicating tensor\n");

      spctr_replicate * rctr = new spctr_replicate(this, phys_mapped, blk_sz_A, blk_sz_B, blk_sz_C);
      if (is_top){
        hctr = rctr;
        is_top = 0;
      } else {
        *rec_ctr = rctr;
      }
      rec_ctr = &rctr->rec_ctr;
    }


  //#ifdef OFFLOAD
    int total_iter = 1;
    int upload_phase_A = 1;
    int upload_phase_B = 1;
    int download_phase_C = 1;
  //#endif
    nvirt = 1;


    spctr_2d_general * bottom_ctr_gen = NULL;
    ASSERT(A->is_sparse);
    blk_sz_A = A->calc_nvirt();
    for (int a=0; a<A->order; a++){
      blk_len_A[a] = A->edge_map[a].calc_phase()/A->edge_map[a].calc_phys_phase();
    }
    int ones[A->order];
    std::fill(ones, ones+A->order, 1);
    for (i=0; i<num_tot; i++){
      virt_dim[i] = 1;
      i_A = idx_arr[3*i+0];
      i_B = idx_arr[3*i+1];
      i_C = idx_arr[3*i+2];
      /* If this index belongs to exactly two tensors */
      if ((i_A != -1 && i_B != -1 && i_C == -1) ||
          (i_A != -1 && i_B == -1 && i_C != -1) ||
          (i_A == -1 && i_B != -1 && i_C != -1)) {
        spctr_2d_general * ctr_gen = new spctr_2d_general(this);
  #ifdef OFFLOAD
        ctr_gen->alloc_host_buf = false;
  #endif
        int is_built = 0;
        if (i_A == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      ones,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C);
        }
        if (i_B == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      ones,
                                      upload_phase_A);
        }
        if (i_C == -1){
          is_built = ctr_2d_gen_build(is_used,
                                      global_comm,
                                      i,
                                      virt_dim,
                                      ctr_gen->edge_len,
                                      total_iter,
                                      C,
                                      i_C,
                                      ctr_gen->cdt_C,
                                      ctr_gen->ctr_lda_C,
                                      ctr_gen->ctr_sub_lda_C,
                                      ctr_gen->move_C,
                                      blk_len_C,
                                      blk_sz_C,
                                      virt_blk_len_C,
                                      download_phase_C,
                                      A,
                                      i_A,
                                      ctr_gen->cdt_A,
                                      ctr_gen->ctr_lda_A,
                                      ctr_gen->ctr_sub_lda_A,
                                      ctr_gen->move_A,
                                      blk_len_A,
                                      blk_sz_A,
                                      ones,
                                      upload_phase_A,
                                      B,
                                      i_B,
                                      ctr_gen->cdt_B,
                                      ctr_gen->ctr_lda_B,
                                      ctr_gen->ctr_sub_lda_B,
                                      ctr_gen->move_B,
                                      blk_len_B,
                                      blk_sz_B,
                                      virt_blk_len_B,
                                      upload_phase_B);
        }
        if (is_built){
          if (is_top){
            hctr = ctr_gen;
            is_top = 0;
          } else {
            *rec_ctr = ctr_gen;
          }
          if (bottom_ctr_gen == NULL)
            bottom_ctr_gen = ctr_gen;
          rec_ctr = &ctr_gen->rec_ctr;
        } else {
          ctr_gen->rec_ctr = NULL;
          delete ctr_gen;
        }
      } else {
        if (i_A != -1){
          map = &A->edge_map[i_A];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_B != -1){
          map = &B->edge_map[i_B];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        } else if (i_C != -1){
          map = &C->edge_map[i_C];
          while (map->has_child) map = map->child;
          if (map->type == VIRTUAL_MAP)
            virt_dim[i] = map->np;
        }
      }
      /*if (sA && i_A != -1){
        nvirt = virt_dim[i]/str_A->strip_dim[i_A];
      } else if (sB && i_B != -1){
        nvirt = virt_dim[i]/str_B->strip_dim[i_B];
      } else if (sC && i_C != -1){
        nvirt = virt_dim[i]/str_C->strip_dim[i_C];
      }*/
      
      nvirt = nvirt * virt_dim[i];
    }

    if (nvirt_all != NULL)
      *nvirt_all = nvirt;

    ASSERT(blk_sz_A >= 1);
    ASSERT(blk_sz_B >= 1);
    ASSERT(blk_sz_C >= 1);

    /* Multiply over virtual sub-blocks */
    if (nvirt > 1){
      spctr_virt * ctrv = new spctr_virt(this, num_tot, virt_dim, vrt_sz_A, vrt_sz_B, vrt_sz_C);
      if (is_top) {
        hctr = ctrv;
        is_top = 0;
      } else {
        *rec_ctr = ctrv;
      }
      rec_ctr = &ctrv->rec_ctr;
    } else
      CTF_int::cdealloc(virt_dim);

    int krnl_type = is_inner;
    if (krnl_type == 1 && A->sr->has_csrmm) krnl_type = 2;
    seq_tsr_spctr * ctrseq = new seq_tsr_spctr(this, krnl_type, inner_params, virt_blk_len_A, virt_blk_len_B, virt_blk_len_C, vrt_sz_C);
    if (is_top) {
      hctr = ctrseq;
      is_top = 0;
    } else {
      *rec_ctr = ctrseq;
    }

    CTF_int::cdealloc(blk_len_A);
    CTF_int::cdealloc(blk_len_B);
    CTF_int::cdealloc(blk_len_C);
    CTF_int::cdealloc(idx_arr);

    return hctr;
  }

  ctr * contraction::construct_ctr(int            is_inner,
                                   iparam const * inner_params,
                                   int *          nvirt_all,
                                   int            is_used){
    int i;
    mapping * map;
    int * phys_mapped;

    int nphys_dim = A->topo->order;
  
    CTF_int::alloc_ptr(sizeof(int)*nphys_dim*3, (void**)&phys_mapped);
    memset(phys_mapped, 0, sizeof(int)*nphys_dim*3);

    for (i=0; i<A->order; i++){
      map = &A->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+0] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+0] = 1;
        }
      }
    }
    for (i=0; i<B->order; i++){
      map = &B->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+1] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+1] = 1;
        }
      }
    }
    for (i=0; i<C->order; i++){
      map = &C->edge_map[i];
      if (map->type == PHYSICAL_MAP){
        phys_mapped[3*map->cdt+2] = 1;
      }
      while (map->has_child) {
        map = map->child;
        if (map->type == PHYSICAL_MAP){
          phys_mapped[3*map->cdt+2] = 1;
        }
      }
    }
    ctr * hctr;
    if (A->is_sparse || B->is_sparse || B->is_sparse){
      hctr = construct_sparse_ctr(is_inner, inner_params, nvirt_all, is_used, phys_mapped);
    } else {
      hctr = construct_dense_ctr(is_inner, inner_params, nvirt_all, is_used, phys_mapped);
    }
    CTF_int::cdealloc(phys_mapped);
    TAU_FSTOP(construct_contraction);
    return hctr;
  }

  int contraction::contract(){
    int stat;
    ctr * ctrf;
    CommData global_comm = C->wrld->cdt;

    if (A->has_zero_edge_len || B->has_zero_edge_len
        || C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }
    //FIXME: create these tensors without home
    if (A == B || A == C){
      tensor * new_tsr = new tensor(A);
      contraction new_ctr = contraction(*this);
      new_ctr.A = new_tsr;
      stat = new_ctr.contract();
      delete new_tsr;
      return stat;
    }
    if (B == C){
      tensor * new_tsr = new tensor(B);
      contraction new_ctr = contraction(*this);
      new_ctr.B = new_tsr;
      stat = new_ctr.contract();
      delete new_tsr;
      return stat;
    }
    for (int i=0; i<A->order; i++){
      int iA = idx_A[i];
      bool has_match = false;
      for (int j=0; j<B->order; j++){
        if (idx_B[j] == iA) has_match = true;
      }
      for (int j=0; j<C->order; j++){
        if (idx_C[j] == iA) has_match = true;
      }
      if (false && !has_match){
        int new_len[A->order-1];
        int new_sym[A->order-1];
        int new_idx[A->order-1];
        for (int j=0; j<A->order; j++){
          if (j==iA) continue;
          if (j<iA){
            new_len[j] = A->lens[j];
            new_sym[j] = A->sym[j];
            new_idx[j] = idx_A[j];
          } else {
            new_len[j-1] = A->lens[j];
            new_sym[j-1] = A->sym[j];
            new_idx[j-1] = idx_A[j];
          }
        }
        tensor * new_tsr = new tensor(A->sr, A->order-1, new_len, new_sym, A->wrld, 1, A->name, 1, A->is_sparse);
        summation s(A, idx_A, A->sr->mulid(), new_tsr, new_idx, A->sr->mulid());
        s.execute();
        contraction ctr(new_tsr, new_idx, B, idx_B, alpha, C, idx_C, beta, func);
        ctr.execute();
        delete new_tsr;
        return SUCCESS;
      }
    }


    ASSERT(!C->is_sparse);
    if (B->is_sparse){
      ASSERT(!A->is_sparse);
      //FIXME ASSERT that commitative
      contraction CBA(B,idx_B,A,idx_A,alpha,C,idx_C,beta,func);
      CBA.contract();
      return SUCCESS;
    } 
    if (A->is_sparse){
      /*bool A_inds_ordered = true;
      for (int i=1; i<A->order; i++){
        if (idx_A[i] < idx_A[i-1]) A_inds_ordered = false;
      }*/
      int new_idx_A[A->order];
      int new_idx_B[B->order];
      int new_idx_C[C->order];

      int num_tot, * idx_arr;
      inv_idx(A->order, idx_A,
              B->order, idx_B,
              C->order, idx_C,
              &num_tot, &idx_arr);
      bool used[num_tot];
      memset(used,0,sizeof(bool)*num_tot);
      for (int i=0; i<num_tot; i++){
        int la=-2;
        int ila=-1;
        for (int j=num_tot-1; j>=0; j--){
          if (used[j] == 0 && idx_arr[3*j]>la){
            ila = j;
            la = idx_arr[3*j];
          }
        }
        ASSERT(ila>=0);
        used[ila] = 1;
        if (idx_arr[3*ila] != -1) 
          new_idx_A[idx_arr[3*ila]]=num_tot-i-1;
        if (idx_arr[3*ila+1] != -1) 
          new_idx_B[idx_arr[3*ila+1]]=num_tot-i-1;
        if (idx_arr[3*ila+2] != -1) 
          new_idx_C[idx_arr[3*ila+2]]=num_tot-i-1;
      }
      cdealloc(idx_arr);
      bool is_chngd = false;
      for (int i=0; i<A->order; i++){
        if (idx_A[i] != new_idx_A[i]) is_chngd=true;
      }
      for (int i=0; i<B->order; i++){
        if (idx_B[i] != new_idx_B[i]) is_chngd=true;
      }
      for (int i=0; i<C->order; i++){
        if (idx_C[i] != new_idx_C[i]) is_chngd=true;
      }
      if (is_chngd){
        contraction CBA(A,new_idx_A,B,new_idx_B,alpha,C,new_idx_C,beta,func);
        CBA.contract();
        return SUCCESS;
      }
      
    }


  #if DEBUG >= 1 //|| VERBOSE >= 1)
//    if (global_comm.rank == 0)
  //    printf("Contraction permutation:\n");
    print();
  #endif

    TAU_FSTART(contract);

    prescale_operands();
  #if 0 //VERIFY
    int64_t nsA, nsB;
    int64_t nA, nB, nC, up_nC;
    dtype * sA, * sB, * ans_C;
    dtype * uA, * uB, * uC;
    dtype * up_C, * up_ans_C, * pup_C;
    int order_A, order_B, order_C, i, pass;
    int * edge_len_A, * edge_len_B, * edge_len_C;
    int * sym_A, * sym_B, * sym_C;
    int * sym_tmp;
    stat = allread_tsr(type->tid_A, &nsA, &sA);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_B, &nsB, &sB);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_C, &nC, &ans_C);
    assert(stat == SUCCESS);
  #endif
    /* Check if the current tensor mappings can be contracted on */
    /*fseq_tsr_ctr fftsr=ftsr;
    if (ftsr.func_ptr == NULL){
      fftsr.func_ptr = &sym_seq_ctr_ref;
  #ifdef OFFLOAD
      fftsr.is_offloadable = 0;
  #endif
    }*/
  #if REDIST
    //stat = map_tensors(type, fftsr, felm, alpha, beta, &ctrf);
    stat = map(&ctrf);
    if (stat == ERROR) {
      printf("Failed to map tensors to physical grid\n");
      return ERROR;
    }
  #else
    if (check_mapping() == 0) {
      /* remap if necessary */
      stat = map(&ctrf);
      if (stat == ERROR) {
        printf("Failed to map tensors to physical grid\n");
        return ERROR;
      }
    } else {
      /* Construct the tensor algorithm we would like to use */
  #if DEBUG > 2
      if (global_comm.rank == 0)
        printf("Keeping mappings:\n");
/*      A->print_map(stdout);
      B->print_map(stdout);
      C->print_map(stdout);*/
  #endif
      ctrf = construct_ctr();
    }
#if DEBUG >= 1
    if (global_comm.rank == 0){
      int64_t memuse = ctrf->mem_rec();
      printf("CTF: Contraction does not require redistribution, will use %E bytes per processor out of %E available memory and take an estimated of %E sec\n",
              (double)memuse,(double)proc_bytes_available(),ctrf->est_time_rec(1));
      A->print_map();
      B->print_map();
      C->print_map();
    }
#endif

  #endif
    ASSERT(check_mapping());
    bool is_inner = false;
  #if FOLD_TSR
    if (!is_custom) is_inner = can_fold();
    if (is_inner){
      iparam prm;
      TAU_FSTART(map_fold);
      prm = map_fold();
      TAU_FSTOP(map_fold);
      delete ctrf;
      ctrf = construct_ctr(1, &prm);
    } 
  #endif
  #if DEBUG >=2
    ctrf->print();
  #endif
  #if VERBOSE >= 1
  double dtt = MPI_Wtime();
  #endif
  #ifdef DEBUG
    if (global_comm.rank == 0){
      //DPRINTF(1,"[%d] performing contraction\n",
        //  global_comm.rank);
     // DPRINTF(1,"%E bytes of buffer space will be needed for this contraction\n",
       // (double)ctrf->mem_rec());
      DPRINTF(2,"%E bytes needed, System memory = %E bytes total, %E bytes used, %E bytes available.\n",
        (double)ctrf->mem_rec(),
        (double)proc_bytes_total(),
        (double)proc_bytes_used(),
        (double)proc_bytes_available());
    }
  #endif
  #if DEBUG >=2
    A->print_map();
    B->print_map();
    C->print_map();
  #endif
  //  stat = zero_out_padding(type->tid_A);
  //  stat = zero_out_padding(type->tid_B);
  #ifdef PROFILE
    TAU_FSTART(pre_ctr_func_barrier);
    MPI_Barrier(global_comm.cm);
    TAU_FSTOP(pre_ctr_func_barrier);
  #endif
    TAU_FSTART(ctr_func);
    /* Invoke the contraction algorithm */
    A->topo->activate();
    if (A->is_sparse || B->is_sparse || C->is_sparse){
      int64_t * size_blk_A = NULL;
      int64_t * size_blk_B = NULL;
      int64_t * size_blk_C = NULL;
      char * data_A;
      char * data_B;
      char * data_C;
      if (!is_inner){
        data_A = A->data;
        data_B = B->data;
        data_C = C->data;
        if (A->nnz_blk != NULL){
          alloc_ptr(A->calc_nvirt()*sizeof(int64_t), (void**)&size_blk_A);
          for (int i=0; i<A->calc_nvirt(); i++){
            size_blk_A[i] = A->nnz_blk[i]*A->sr->pair_size();
          }
        }
        if (B->nnz_blk != NULL){
          alloc_ptr(B->calc_nvirt()*sizeof(int64_t), (void**)&size_blk_B);
          for (int i=0; i<B->calc_nvirt(); i++){
            size_blk_B[i] = B->nnz_blk[i]*B->sr->pair_size();
          }
        }
        if (C->nnz_blk != NULL){
          alloc_ptr(C->calc_nvirt()*sizeof(int64_t), (void**)&size_blk_C);
          for (int i=0; i<C->calc_nvirt(); i++){
            size_blk_C[i] = C->nnz_blk[i]*C->sr->pair_size();
          }
        }
      } else {
        if (A->is_sparse) data_A = A->rec_tsr->data;
        else              data_A = A->data;
        if (B->is_sparse) data_B = B->rec_tsr->data;
        else              data_B = B->data;
        if (C->is_sparse) data_C = C->rec_tsr->data;
        else              data_C = C->data;
        size_blk_B = B->rec_tsr->nnz_blk;
        size_blk_A = A->rec_tsr->nnz_blk;
        size_blk_C = C->rec_tsr->nnz_blk;
      }

      ((spctr*)ctrf)->run(data_A, A->calc_nvirt(), size_blk_A,
                          data_B, B->calc_nvirt(), size_blk_B,
                          data_C, C->calc_nvirt(), size_blk_C,
                          data_C);

      //FIXME: adjust C->nnz_blk
      if (!is_inner){
        if (size_blk_A != NULL) cdealloc(size_blk_A);
        if (size_blk_B != NULL) cdealloc(size_blk_B);
        if (size_blk_C != NULL) cdealloc(size_blk_C);
      }
    } else
      ctrf->run(A->data, B->data, C->data);
    A->topo->deactivate();

  #ifdef PROFILE
    TAU_FSTART(post_ctr_func_barrier);
    MPI_Barrier(global_comm.cm);
    TAU_FSTOP(post_ctr_func_barrier);
  #endif
    TAU_FSTOP(ctr_func);
  #ifndef SEQ
    if (C->is_cyclic)
      stat = C->zero_out_padding();
  #endif
    A->unfold();
    B->unfold();
    if (A->wrld->rank == 0){
      VPRINTF(1, "Contraction permutation completed in %lf sec.\n",MPI_Wtime()-dtt);
    }


  #if 0 //VERIFY
    stat = allread_tsr(type->tid_A, &nA, &uA);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_A, &order_A, &edge_len_A, &sym_A);
    assert(stat == SUCCESS);

    stat = allread_tsr(type->tid_B, &nB, &uB);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_B, &order_B, &edge_len_B, &sym_B);
    assert(stat == SUCCESS);

    if (nsA != nA) { printf("nsA = " PRId64 ", nA = " PRId64 "\n",nsA,nA); ABORT; }
    if (nsB != nB) { printf("nsB = " PRId64 ", nB = " PRId64 "\n",nsB,nB); ABORT; }
    for (i=0; (int64_t)i<nA; i++){
      if (fabs(uA[i] - sA[i]) > 1.E-6){
        printf("A[i] = %lf, sA[i] = %lf\n", uA[i], sA[i]);
      }
    }
    for (i=0; (int64_t)i<nB; i++){
      if (fabs(uB[i] - sB[i]) > 1.E-6){
        printf("B[%d] = %lf, sB[%d] = %lf\n", i, uB[i], i, sB[i]);
      }
    }

    stat = allread_tsr(type->tid_C, &nC, &uC);
    assert(stat == SUCCESS);
    stat = get_tsr_info(type->tid_C, &order_C, &edge_len_C, &sym_C);
    assert(stat == SUCCESS);
    DEBUG_PRINTF("packed size of C is " PRId64 " (should be " PRId64 ")\n", nC,
      sy_packed_size(order_C, edge_len_C, sym_C));

    pup_C = (dtype*)CTF_int::alloc(nC*sizeof(dtype));

    cpy_sym_ctr(alpha,
          uA, order_A, edge_len_A, edge_len_A, sym_A, idx_A,
          uB, order_B, edge_len_B, edge_len_B, sym_B, idx_B,
          beta,
      ans_C, order_C, edge_len_C, edge_len_C, sym_C, idx_C);
    assert(stat == SUCCESS);

  #if ( DEBUG>=5)
    for (i=0; i<nC; i++){
  //    if (fabs(C[i]-ans_C[i]) > 1.E-6){
        printf("PACKED: C[%d] = %lf, ans_C[%d] = %lf\n",
         i, C[i], i, ans_C[i]);
  //     }
    }
  #endif

    punpack_tsr(uC, order_C, edge_len_C,
          sym_C, 1, &sym_tmp, &up_C);
    punpack_tsr(ans_C, order_C, edge_len_C,
          sym_C, 1, &sym_tmp, &up_ans_C);
    punpack_tsr(up_ans_C, order_C, edge_len_C,
          sym_C, 0, &sym_tmp, &pup_C);
    for (i=0; (int64_t)i<nC; i++){
      assert(fabs(pup_C[i] - ans_C[i]) < 1.E-6);
    }
    pass = 1;
    up_nC = 1;
    for (i=0; i<order_C; i++){ up_nC *= edge_len_C[i]; };

    for (i=0; i<(int)up_nC; i++){
      if (fabs((up_C[i]-up_ans_C[i])/up_ans_C[i]) > 1.E-6 &&
    fabs((up_C[i]-up_ans_C[i])) > 1.E-6){
        printf("C[%d] = %lf, ans_C[%d] = %lf\n",
         i, up_C[i], i, up_ans_C[i]);
        pass = 0;
      }
    }
    if (!pass) ABORT;

  #endif

    delete ctrf;

    TAU_FSTOP(contract);
    return SUCCESS;
  }


  int contraction::sym_contract(){
    int i;
    //int ** scl_idxs_C;
    //dtype * scl_alpha_C;
    int stat = SUCCESS;
    int * new_idx;
    int * map_A, * map_B, * map_C;
    tensor ** dstack_tsr_C;
    int ** dstack_map_C;
    int nst_C;
    std::vector<contraction> perm_types;
    std::vector<int> signs;
    char const * dbeta;
    ctr * ctrf;
    tensor * tnsr_A, * tnsr_B, * tnsr_C;
  
    this->check_consistency();
  
    CommData global_comm = A->wrld->cdt;
  
    A->unfold();
    B->unfold();
    C->unfold();
    if (A->has_zero_edge_len || B->has_zero_edge_len
        || C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=j-num_diag;
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }
    CTF_int::alloc_ptr(sizeof(int)*A->order,          (void**)&map_A);
    CTF_int::alloc_ptr(sizeof(int)*B->order,          (void**)&map_B);
    CTF_int::alloc_ptr(sizeof(int)*C->order,          (void**)&map_C);
    CTF_int::alloc_ptr(sizeof(int*)*C->order,         (void**)&dstack_map_C);
    CTF_int::alloc_ptr(sizeof(tensor*)*C->order,      (void**)&dstack_tsr_C);
    memcpy(map_A, idx_A, A->order*sizeof(int));
    memcpy(map_B, idx_B, B->order*sizeof(int));
    memcpy(map_C, idx_C, C->order*sizeof(int));

    tnsr_A = A;
    tnsr_B = B;
    tnsr_C = C;
    
    tensor * new_tsr;
    while (tnsr_A->extract_diag(map_A, 1, new_tsr, &new_idx) == SUCCESS){
      if (tnsr_A != A) delete tnsr_A;
      CTF_int::cdealloc(map_A);
      tnsr_A = new_tsr;
      map_A = new_idx;
    }
    while (tnsr_B->extract_diag(map_B, 1, new_tsr, &new_idx) == SUCCESS){
      if (tnsr_B != B) delete tnsr_B;
      CTF_int::cdealloc(map_B);
      tnsr_B = new_tsr;
      map_B = new_idx;
    }
    nst_C = 0;
    while (tnsr_C->extract_diag(map_C, 1, new_tsr, &new_idx) == SUCCESS){
      dstack_map_C[nst_C] = map_C;
      dstack_tsr_C[nst_C] = tnsr_C;
      nst_C++;
      tnsr_C = new_tsr;
      map_C = new_idx;
    }

    bivar_function const * fptr;
    if (is_custom) fptr = func;
    else fptr = NULL;

    contraction new_ctr = contraction(tnsr_A, map_A, tnsr_B, map_B, alpha, tnsr_C, map_C, beta, fptr);
    tnsr_A->unfold();
    tnsr_B->unfold();
    tnsr_C->unfold();
    /*if (ntid_A == ntid_B || ntid_A == ntid_C){*/
    if (tnsr_A == tnsr_C){
      tensor * nnew_tsr = new tensor(tnsr_A);
      contraction nnew_ctr = contraction(new_ctr);
      nnew_ctr.A = nnew_tsr;
      stat = nnew_ctr.sym_contract();
      delete nnew_tsr;
    } else if (tnsr_B == tnsr_C){
      tensor * nnew_tsr = new tensor(tnsr_B);
      contraction nnew_ctr = contraction(new_ctr);
      nnew_ctr.B = nnew_tsr;
      stat = nnew_ctr.sym_contract();
      delete nnew_tsr;
    } else {

      int sign = align_symmetric_indices(tnsr_A->order,
                                         new_ctr.idx_A,
                                         tnsr_A->sym,
                                         tnsr_B->order,
                                         new_ctr.idx_B,
                                         tnsr_B->sym,
                                         tnsr_C->order,
                                         new_ctr.idx_C,
                                         tnsr_C->sym);

      /*
       * Apply a factor of n! for each set of n symmetric indices which are contracted over
       */
      int ocfact = overcounting_factor(tnsr_A->order,
                                       new_ctr.idx_A,
                                       tnsr_A->sym,
                                       tnsr_B->order,
                                       new_ctr.idx_B,
                                       tnsr_B->sym,
                                       tnsr_C->order,
                                       new_ctr.idx_C,
                                       tnsr_C->sym);
      char const * align_alpha = alpha;
      if (sign != 1){
        char * u_align_alpha = (char*)alloc(tnsr_C->sr->el_size);
        if (sign == -1){
          tnsr_C->sr->addinv(alpha, u_align_alpha);
//          alpha = new_alpha;
        }
        align_alpha = u_align_alpha;
        //FIXME free new_alpha
      }

      char * oc_align_alpha = (char*)alloc(tnsr_C->sr->el_size);
      tnsr_C->sr->safecopy(oc_align_alpha, align_alpha);
      if (ocfact != 1){
        if (ocfact != 1){
          tnsr_B->sr->safecopy(oc_align_alpha, tnsr_B->sr->addid());
          
          for (int i=0; i<ocfact; i++){
            tnsr_B->sr->add(oc_align_alpha, align_alpha, oc_align_alpha);
          }
//          alpha = new_alpha;
        }
      }
      //new_ctr.alpha = alpha;


      //std::cout << alpha << ' ' << alignfact << ' ' << ocfact << std::endl;

      if (new_ctr.unfold_broken_sym(NULL) != -1){
        if (global_comm.rank == 0)
          DPRINTF(2,"Contraction index is broken\n");

        contraction * unfold_ctr;
        new_ctr.unfold_broken_sym(&unfold_ctr);
        if (unfold_ctr->map(&ctrf, 0) == SUCCESS){
/*  #else
        int sy = 0;
        for (i=0; i<A->order; i++){
          if (A->sym[i] == SY) sy = 1;
        }
        for (i=0; i<B->order; i++){
          if (B->sym[i] == SY) sy = 1;
        }
        for (i=0; i<C->order; i++){
          if (C->sym[i] == SY) sy = 1;
        }
        if (sy && unfold_ctr->map(&ctrf, 0) == SUCCESS)
  #endifi*/
          if (tnsr_A == tnsr_B){
            tnsr_A = new tensor(tnsr_B);
          }
          desymmetrize(tnsr_A, unfold_ctr->A, 0);
          desymmetrize(tnsr_B, unfold_ctr->B, 0);
          desymmetrize(tnsr_C, unfold_ctr->C, 1);
          if (global_comm.rank == 0)
            DPRINTF(1,"%d Performing index desymmetrization\n",tnsr_A->wrld->rank);
          unfold_ctr->alpha = align_alpha;
          stat = unfold_ctr->sym_contract();
          if (!unfold_ctr->C->is_data_aliased && !tnsr_C->sr->isequal(tnsr_C->sr->mulid(), unfold_ctr->beta)){
            int sidx_C[tnsr_C->order];
            for (int iis=0; iis<tnsr_C->order; iis++){
              sidx_C[iis] = iis;
            }
            scaling sscl = scaling(tnsr_C, sidx_C, unfold_ctr->beta);
            sscl.execute();
          }
          symmetrize(tnsr_C, unfold_ctr->C);
          if (tnsr_A != unfold_ctr->A){
            unfold_ctr->A->unfold();
            tnsr_A->pull_alias(unfold_ctr->A);
            delete unfold_ctr->A;
          }
          if (tnsr_B != unfold_ctr->B){
            unfold_ctr->B->unfold();
            tnsr_B->pull_alias(unfold_ctr->B);
            delete unfold_ctr->B;
          }
          if (tnsr_C != unfold_ctr->C){
            unfold_ctr->C->unfold();
            tnsr_C->pull_alias(unfold_ctr->C);
            delete unfold_ctr->C;
          }
        } else {
            DPRINTF(1,"%d Not Performing index desymmetrization\n",tnsr_A->wrld->rank);
          get_sym_perms(new_ctr, perm_types, signs);
                        //&nscl_C, &scl_maps_C, &scl_alpha_C);
          dbeta = beta;
          char * new_alpha = (char*)alloc(tnsr_B->sr->el_size);
          for (i=0; i<(int)perm_types.size(); i++){
            if (signs[i] == 1)
              C->sr->copy(new_alpha, oc_align_alpha);
            else {
              ASSERT(signs[i]==-1);
              tnsr_C->sr->addinv(oc_align_alpha, new_alpha);
            }
            perm_types[i].alpha = new_alpha;
            perm_types[i].beta = dbeta;
            stat = perm_types[i].contract();
            dbeta = new_ctr.C->sr->mulid();
          }
          perm_types.clear();
          signs.clear();
        }
        delete unfold_ctr;
      } else {
        new_ctr.alpha = oc_align_alpha;
        stat = new_ctr.contract();
      }
      if (tnsr_A != A) delete tnsr_A;
      if (tnsr_B != B) delete tnsr_B;
      for (int i=nst_C-1; i>=0; i--){
        dstack_tsr_C[i]->extract_diag(dstack_map_C[i], 0, tnsr_C, &new_idx);
        delete tnsr_C;
        tnsr_C = dstack_tsr_C[i];
      }
      ASSERT(tnsr_C == C);
      CTF_int::cdealloc(oc_align_alpha);
    }

    CTF_int::cdealloc(map_A);
    CTF_int::cdealloc(map_B);
    CTF_int::cdealloc(map_C);
    CTF_int::cdealloc(dstack_map_C);
    CTF_int::cdealloc(dstack_tsr_C);

    return stat;
  }

  int contraction::home_contract(){
  #ifndef HOME_CONTRACT
    return sym_contract();
  #else
    int ret;
    int was_home_A, was_home_B, was_home_C;
    A->unfold();
    B->unfold();
    C->unfold();
    
    if (A->has_zero_edge_len || 
        B->has_zero_edge_len || 
        C->has_zero_edge_len){
      if (!C->sr->isequal(beta,C->sr->mulid()) && !C->has_zero_edge_len){ 
        int * new_idx_C; 
        int num_diag = 0;
        new_idx_C = (int*)CTF_int::alloc(sizeof(int)*C->order);
        for (int i=0; i<C->order; i++){
          new_idx_C[i]=i-num_diag;
          for (int j=0; j<i; j++){
            if (idx_C[i] == idx_C[j]){
              new_idx_C[i]=new_idx_C[j];
              num_diag++;
              break;
            }
          }
        }
        scaling scl = scaling(C, new_idx_C, beta);
        scl.execute();
        CTF_int::cdealloc(new_idx_C);
      }
      return SUCCESS;
    }

    CTF_int::contract_mst();

    //if (stype->tid_A == stype->tid_B || stype->tid_A == stype->tid_C){
    /*if (stype->tid_A == stype->tid_C){
      clone_tensor(stype->tid_A, 1, &new_tid);
      CTF_ctr_type_t new_type = *stype;
      new_type.tid_A = new_tid;
      ret = home_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return ret;
    } else if (stype->tid_B == stype->tid_C){
      clone_tensor(stype->tid_B, 1, &new_tid);
      CTF_ctr_type_t new_type = *stype;
      new_type.tid_B = new_tid;
      ret = home_contract(&new_type, ftsr, felm, alpha, beta);
      del_tsr(new_tid);
      return ret;
    }*/ 

    //CTF_ctr_type_t ntype = *stype;
    contraction new_ctr = contraction(*this);

    was_home_A = A->is_home;
    was_home_B = B->is_home;
    was_home_C = C->is_home;

    if (was_home_A){
//      clone_tensor(stype->tid_A, 0, &ntype.tid_A, 0);
      new_ctr.A = new tensor(A, 0, 0); //tensors[ntype.tid_A];
      new_ctr.A = new_ctr.A;
      new_ctr.A->data = A->data;
      new_ctr.A->home_buffer = A->home_buffer;
      new_ctr.A->is_home = 1;
      new_ctr.A->is_mapped = 1;
      new_ctr.A->topo = A->topo;
      copy_mapping(A->order, A->edge_map, new_ctr.A->edge_map);
      new_ctr.A->set_padding();
    }     
    if (was_home_B){
      if (A == B){ //stype->tid_A == stype->tid_B){
        new_ctr.B = new_ctr.A; //tensors[ntype.tid_B];
      } else {
        new_ctr.B = new tensor(B, 0, 0); //tensors[ntype.tid_A];
/*        clone_tensor(stype->tid_B, 0, &ntype.tid_B, 0);
        new_ctr.B = tensors[ntype.tid_B];*/
        new_ctr.B->data = B->data;
        new_ctr.B->home_buffer = B->home_buffer;
        new_ctr.B->is_home = 1;
        new_ctr.B->is_mapped = 1;
        new_ctr.B->topo = B->topo;
        copy_mapping(B->order, B->edge_map, new_ctr.B->edge_map);
        new_ctr.B->set_padding();
      }
    }
    if (was_home_C){
      if (C == A){ //stype->tid_C == stype->tid_A){
        new_ctr.C = new_ctr.A; //tensors[ntype.tid_C];
      } else if (C == B){ //stype->tid_C == stype->tid_B){
        new_ctr.C = new_ctr.B; //tensors[ntype.tid_C];
      } else {
        new_ctr.C = new tensor(C, 0, 0); //tensors[ntype.tid_C];
        /*clone_tensor(stype->tid_C, 0, &ntype.tid_C, 0);
        new_ctr.C = tensors[ntype.tid_C];*/
        new_ctr.C->data = C->data;
        new_ctr.C->home_buffer = C->home_buffer;
        new_ctr.C->is_home = 1;
        new_ctr.C->is_mapped = 1;
        new_ctr.C->topo = C->topo;
        copy_mapping(C->order, C->edge_map, new_ctr.C->edge_map);
        new_ctr.C->set_padding();
      }
    }

    ret = new_ctr.sym_contract();//&ntype, ftsr, felm, alpha, beta);
    if (ret!= SUCCESS) return ret;
    if (was_home_A) new_ctr.A->unfold();
    if (was_home_B && A != B) new_ctr.B->unfold();
    if (was_home_C) new_ctr.C->unfold();

    if (was_home_C && !new_ctr.C->is_home){
      if (C->wrld->rank == 0)
        DPRINTF(2,"Migrating tensor %s back to home\n", C->name);
      distribution dC = distribution(new_ctr.C);
/*      save_mapping(new_ctr.C,
                   &old_phase_C, &old_rank_C,
                   &old_virt_dim_C, &old_pe_lda_C,
                   &old_size_C,
                   &was_cyclic_C, &old_padding_C,
                   &old_edge_len_C, &topovec[new_ctr.C->itopo]);*/
      C->data = new_ctr.C->data;
      C->is_home = 0;
      TAU_FSTART(redistribute_for_ctr_home);
      C->redistribute(dC);
/*      remap_tensor(stype->tid_C, C, C->topo, old_size_C,
                   old_phase_C, old_rank_C, old_virt_dim_C,
                   old_pe_lda_C, was_cyclic_C,
                   old_padding_C, old_edge_len_C, global_comm);*/
      TAU_FSTOP(redistribute_for_ctr_home);
      memcpy(C->home_buffer, C->data, C->size*C->sr->el_size);
      CTF_int::cdealloc(C->data);
      C->data = C->home_buffer;
      C->is_home = 1;
      new_ctr.C->is_data_aliased = 1;
      delete new_ctr.C;
    } else if (was_home_C) {
  /*    C->itopo = new_ctr.C->itopo;
      copy_mapping(C->order, new_ctr.C->edge_map, C->edge_map);
      C->set_padding();*/
      ASSERT(new_ctr.C->data == C->data);
      new_ctr.C->is_data_aliased = 1;
      delete new_ctr.C;
    }
    if (new_ctr.A != new_ctr.C){ //ntype.tid_A != ntype.tid_C){
      if (was_home_A && !new_ctr.A->is_home){
        new_ctr.A->has_home = 0;
        delete new_ctr.A;
      } else if (was_home_A) {
        new_ctr.A->is_data_aliased = 1;
        delete new_ctr.A;
      }
    }
    if (new_ctr.B != new_ctr.A && new_ctr.B != new_ctr.C){
      if (was_home_B && A != B && !new_ctr.B->is_home){
        new_ctr.B->has_home = 0;
        delete new_ctr.B;
      } else if (was_home_B && A != B) {
        new_ctr.B->is_data_aliased = 1;
        delete new_ctr.B;
      }
    }
    return SUCCESS;
  #endif
  }



  bool contraction::need_prescale_operands(){
    int num_tot, * idx_arr;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    tensor * T, * V;
    int fT, fV;
    int * idx_T, * idx_V;
    if (A->size <= B->size){ 
      T = A;
      V = B;
      fT = 0;
      fV = 1;
      idx_T = idx_A;
      idx_V = idx_B;
    } else {
      T = B;
      V = A;
      fT = 1;
      fV = 0;
      idx_T = idx_B;
      idx_V = idx_A;
    }
    for (int iT=0; iT<T->order; iT++){
      int i = idx_T[iT];
      int iV = idx_arr[3*i+fV];
      int iiV = iV;
      int iiC = idx_arr[3*i+2];
      ASSERT(iT == idx_arr[3*i+fT]);
      int npres = 0;
      while (T->sym[iT+npres] == SY && iiC == -1 &&
              ( (iV == -1 && iiV == -1) || 
                (iV != -1 && iiV != -1 && V->sym[iiV] == SY)
              )
            ){
        npres++;
        int ii = idx_T[iT+npres];
        iiV = idx_arr[3*ii+fV];
        iiC = idx_arr[3*ii+2];
      }
      if (T->sym[iT+npres] == NS){ 
        if (iiC == -1 &&
            ( (iV == -1 && iiV == -1) || 
              (iV != -1 && iiV != -1 && V->sym[iiV] == NS)
            ) )
          npres++;
      }
      
      if (npres > 1)
        return true;
    }

    for (int iV=0; iV<V->order; iV++){
      int i = idx_V[iV];
      int iiT = idx_arr[3*i+fT];
      int iiC = idx_arr[3*i+2];
      ASSERT(iV == idx_arr[3*i+fV]);
      int npres = 0;
      while (V->sym[iV+npres] == SY && iiC == -1 && iiT == -1){
        npres++;
        int ii = idx_V[iV+npres];
        iiT = idx_arr[3*ii+fT];
        iiC = idx_arr[3*i+2];
      }
      if (V->sym[iV+npres] == NS && iiC == -1 && iiT == -1){
        npres++;
      }
      if (npres > 1)
        return true;
    }
    return false;
  }

  void contraction::prescale_operands(){
    int num_tot, * idx_arr;
    inv_idx(A->order, idx_A,
            B->order, idx_B,
            C->order, idx_C,
            &num_tot, &idx_arr);
    tensor * T, * V;
    int fT, fV;
    int * idx_T, * idx_V;
    if (A->size <= B->size){ 
      T = A;
      V = B;
      fT = 0;
      fV = 1;
      idx_T = idx_A;
      idx_V = idx_B;
    } else {
      T = B;
      V = A;
      fT = 1;
      fV = 0;
      idx_T = idx_B;
      idx_V = idx_A;
    }
    for (int iT=0; iT<T->order; iT++){
      int i = idx_T[iT];
      int iV = idx_arr[3*i+fV];
      int iiV = iV;
      int iiC = idx_arr[3*i+2];
      ASSERT(iT == idx_arr[3*i+fT]);
      int npres = 0;
      while (T->sym[iT+npres] == SY && iiC == -1 &&
              ( (iV == -1 && iiV == -1) || 
                (iV != -1 && iiV != -1 && V->sym[iiV] == SY)
              )
            ){
        npres++;
        int ii = idx_T[iT+npres];
        iiV = idx_arr[3*ii+fV];
        iiC = idx_arr[3*ii+2];
      }
      if (T->sym[iT+npres] == NS){ 
        if (iiC == -1 &&
            ( (iV == -1 && iiV == -1) || 
              (iV != -1 && iiV != -1 && V->sym[iiV] == NS)
            ) )
          npres++;
      }
      
      if (npres > 1){
        int sym_mask[T->order];
        std::fill(sym_mask, sym_mask+T->order, 0);
        std::fill(sym_mask+iT, sym_mask+iT+npres, 1);
        /*for (int k=0; k<T->order; k++){
          printf("sym_mask[%d]=%d\n",k,sym_mask[k]);
        }*/
        
        if (T->is_home){
          if (T->wrld->cdt.rank == 0)
            DPRINTF(2,"Tensor %s leaving home\n", T->name);
          T->data = (char*)CTF_int::mst_alloc(T->size*T->sr->el_size);
          memcpy(T->data, T->home_buffer, T->size*T->sr->el_size);
          T->is_home = 0;
        }
        T->scale_diagonals(sym_mask);
      }
      iT += std::max(npres-1, 0);
    }

    for (int iV=0; iV<V->order; iV++){
      int i = idx_V[iV];
      int iiT = idx_arr[3*i+fT];
      int iiC = idx_arr[3*i+2];
      ASSERT(iV == idx_arr[3*i+fV]);
      int npres = 0;
      while (V->sym[iV+npres] == SY && iiC == -1 && iiT == -1){
        npres++;
        int ii = idx_V[iV+npres];
        iiT = idx_arr[3*ii+fT];
        iiC = idx_arr[3*i+2];
      }
      if (V->sym[iV+npres] == NS && iiC == -1 && iiT == -1){
        npres++;
      }
      if (V->is_home){
        if (V->wrld->cdt.rank == 0)
          DPRINTF(2,"Tensor %s leaving home\n", V->name);
        V->data = (char*)CTF_int::mst_alloc(V->size*V->sr->el_size);
        memcpy(V->data, V->home_buffer, V->size*V->sr->el_size);
        V->is_home = 0;
      }

      if (npres > 1){
        int sym_mask[V->order];
        std::fill(sym_mask, sym_mask+V->order, 0);
        std::fill(sym_mask+iV, sym_mask+iV+npres, 1);
        V->scale_diagonals(sym_mask);
      }
      iV += std::max(npres-1, 0);
    }
    cdealloc(idx_arr);
  }

  void contraction::print() const {
    int i;
    //max = A->order+B->order+C->order;
    CommData global_comm = A->wrld->cdt;
    MPI_Barrier(global_comm.cm);
    if (global_comm.rank == 0){
//      printf("Contracting Tensor %s with %s into %s\n", A->name, B->name, C->name);
      char cname[200];
      cname[0] = '\0';
      sprintf(cname, "%s", C->name);
      sprintf(cname+strlen(cname),"[");
      for (i=0; i<C->order; i++){
        if (i>0)
          sprintf(cname+strlen(cname)," %d",idx_C[i]);
        else
          sprintf(cname+strlen(cname),"%d",idx_C[i]);
      }
      sprintf(cname+strlen(cname),"] <- ");
      sprintf(cname+strlen(cname), "%s", A->name);
      sprintf(cname+strlen(cname),"[");
      for (i=0; i<A->order; i++){
        if (i>0)
          sprintf(cname+strlen(cname)," %d",idx_A[i]);
        else
          sprintf(cname+strlen(cname),"%d",idx_A[i]);
      }
      sprintf(cname+strlen(cname),"]*");
      sprintf(cname+strlen(cname), "%s", B->name);
      sprintf(cname+strlen(cname),"[");
      for (i=0; i<B->order; i++){
        if (i>0)
          sprintf(cname+strlen(cname)," %d",idx_B[i]);
        else
          sprintf(cname+strlen(cname),"%d",idx_B[i]);
      }
      sprintf(cname+strlen(cname),"]");
      printf("CTF: Contraction %s\n",cname);

/*
      if (alpha != NULL){
        printf("alpha is "); 
        A->sr->print(alpha);
        printf("\nbeta is "); 
        B->sr->print(beta);
        printf("\n");
      }

      printf("Contraction index table:\n");
      printf("     A      B      C\n");
      for (i=0; i<max; i++){
        ex_A=0;
        ex_B=0;
        ex_C=0;
        printf("%d:   ",i);
        for (j=0; j<A->order; j++){
          if (idx_A[j] == i){
            ex_A++;
            if (A->sym[j] == SY)
              printf("%dSY ",j);
            else if (A->sym[j] == SH)
              printf("%dSH ",j);
            else if (A->sym[j] == AS)
              printf("%dAS ",j);
            else
              printf("%d   ",j);
          }
        }
        if (ex_A == 0)
          printf("       ");
        if (ex_A == 1)
          printf("    ");
        for (j=0; j<B->order; j++){
          if (idx_B[j] == i){
            ex_B=1;
            if (B->sym[j] == SY)
              printf("%dSY ",j);
            else if (B->sym[j] == SH)
              printf("%dSH ",j);
            else if (B->sym[j] == AS)
              printf("%dAS ",j);
            else
              printf("%d   ",j);
          }
        }
        if (ex_B == 0)
          printf("       ");
        if (ex_B == 1)
          printf("    ");
        for (j=0; j<C->order; j++){
          if (idx_C[j] == i){
            ex_C=1;
            if (C->sym[j] == SY)
              printf("%dSY ",j);
            else if (C->sym[j] == SH)
              printf("%dSH ",j);
            else if (C->sym[j] == AS)
              printf("%dAS ",j);
            else
              printf("%d  ",j);
          }
        }
        printf("\n");
        if (ex_A + ex_B + ex_C == 0) break;
      }*/
    }
  }
}
