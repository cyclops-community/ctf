#include "coo.h"
#include "csr.h"
#include "ccsr.h"
#include "../shared/util.h"
#include "../contraction/ctr_comm.h"

namespace CTF_int {
  int64_t get_coo_size(int64_t nnz, int val_size, bool is_int64){
    val_size = std::max(val_size,8*((val_size + 7)/8));
    if (is_int64){
      return nnz*(val_size+sizeof(int64_t)*2)+3*sizeof(int64_t);
    } else {
      return nnz*(val_size+sizeof(int)*2)+3*sizeof(int64_t);
    }
  }

  template <typename int_type>
  tCOO_Matrix<int_type>::tCOO_Matrix(int64_t nnz, algstrct const * sr){
    int64_t size = get_coo_size(nnz, sr->el_size, typeid(int_type)==typeid(int64_t));
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = typeid(int_type)==typeid(int64_t);
    ((int64_t*)all_data)[1] = nnz;
    ((int64_t*)all_data)[2] = sr->el_size;
    //printf("all_data %p vals %p\n",all_data,this->vals());
  }

  template <typename int_type>
  tCOO_Matrix<int_type>::tCOO_Matrix(char * all_data_){
    all_data = all_data_;
  }
 
  template <typename int_type>
  tCOO_Matrix<int_type>::tCOO_Matrix(CSR_Matrix const & csr, algstrct const * sr){
    int64_t nnz = csr.nnz(); 
    int64_t v_sz = csr.val_size(); 
    int const * csr_ja = csr.JA();
    int const * csr_ia = csr.IA();
    char const * csr_vs = csr.vals();

    int64_t size = get_coo_size(nnz, v_sz, typeid(int_type)==typeid(int64_t));
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = typeid(int_type)==typeid(int64_t);
    ((int64_t*)all_data)[1] = nnz;
    ((int64_t*)all_data)[2] = v_sz;
    
    char * vs = vals();
    int_type * coo_rs = rows();
    int_type * coo_cs = cols();

    sr->init_shell(nnz, vs);
  
    ASSERT(typeid(int_type) == typeid(int));
    sr->csr_to_coo(nnz, csr.nrow(), csr_vs, csr_ja, csr_ia, vs, (int*)coo_rs, (int*)coo_cs);
  }
 
  template <typename int_type>
  tCOO_Matrix<int_type>::tCOO_Matrix(CCSR_Matrix const & csr, algstrct const * sr){
    int64_t nnz = csr.nnz(); 
    int64_t nnz_row = csr.nnz_row(); 
    int64_t v_sz = csr.val_size(); 
    int const * csr_ja = csr.JA();
    int const * csr_ia = csr.IA();
    int64_t const * row_enc = csr.nnz_row_encoding();
    char const * csr_vs = csr.vals();

    int64_t size = get_coo_size(nnz, v_sz, typeid(int_type)==typeid(int64_t));
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = typeid(int_type)==typeid(int64_t);
    ((int64_t*)all_data)[1] = nnz;
    ((int64_t*)all_data)[2] = v_sz;
    
    char * vs = vals();
    int_type * coo_rs = rows();
    int_type * coo_cs = cols();

    sr->init_shell(nnz, vs);
  
    ASSERT(typeid(int_type) == typeid(int64_t));
    sr->ccsr_to_coo(nnz, nnz_row, csr_vs, csr_ja, csr_ia, row_enc, vs, (int64_t*)coo_rs, (int64_t*)coo_cs);
  }


  template <typename int_type>
  int64_t tCOO_Matrix<int_type>::nnz() const {
    return ((int64_t*)all_data)[1];
  }

  template <typename int_type>
  int tCOO_Matrix<int_type>::val_size() const {
    return ((int64_t*)all_data)[2];
  }

  template <typename int_type>
  int64_t tCOO_Matrix<int_type>::size() const {
    return get_coo_size(nnz(),val_size(), typeid(int_type)==typeid(int64_t));
  }
  
  template <typename int_type>
  char * tCOO_Matrix<int_type>::vals() const {
    return all_data + 3*sizeof(int64_t);
  }

  template <typename int_type>
  int_type * tCOO_Matrix<int_type>::rows() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();

    return (int_type*)(all_data + n*v_sz+3*sizeof(int64_t));
  } 

  template <typename int_type>
  int_type * tCOO_Matrix<int_type>::cols() const {
    int64_t n = this->nnz();
    int v_sz = ((int64_t*)all_data)[2];

    return (int_type*)(all_data + n*(v_sz+sizeof(int_type))+3*sizeof(int64_t));
  } 

  template <typename int_type>
  void tCOO_Matrix<int_type>::set_data(int64_t nz, int order, int const * sym, int_type const * lens, int_type const * pad_edge_len, int all_fdim, int_type const * all_flen, int const * rev_ordering, int nrow_idx, char const * tsr_data, algstrct const * sr, int const * phase){
    TAU_FSTART(convert_to_COO);
    ((int64_t*)all_data)[0] = typeid(int_type)==typeid(int64_t);
    ((int64_t*)all_data)[1] = nz;
    ((int64_t*)all_data)[2] = sr->el_size;
    int v_sz = sr->el_size;

    int * rev_ord_lens = (int*)alloc(sizeof(int)*all_fdim);
    int * ordering = (int*)alloc(sizeof(int)*all_fdim);
    int64_t * lda_col = (int64_t*)alloc(sizeof(int64_t)*(all_fdim-nrow_idx));
    int64_t * lda_row = (int64_t*)alloc(sizeof(int64_t)*nrow_idx);

    for (int i=0; i<all_fdim; i++){
      ordering[rev_ordering[i]]=i;
    }
    for (int i=0; i<all_fdim; i++){
      //printf("[%d] %d -> %d\n", all_flen[i], i, ordering[i]);
      rev_ord_lens[ordering[i]] = all_flen[i];//lens[i]/phase[i];
      //assert(lens[i]%phase[i] == 0);
      //int ii = lens[i]/phase[i];
      //if (lens[i]%phase[i] > 0) ii++;
      //assert(ii==all_flen[i]);
      //if (lens[i]%phase[i] > 0) rev_ord_lens[ordering[i]]++;
    }
    for (int i=0; i<all_fdim; i++){
      if (i==0 && i<nrow_idx){
        lda_row[0] = 1;
      }
      if (i>0 && i<nrow_idx){
        lda_row[i] = lda_row[i-1]*rev_ord_lens[i-1];
      }
      if (i==nrow_idx){
        lda_col[0] = 1;
      }
      if (i>nrow_idx){
        lda_col[i-nrow_idx] = lda_col[i-nrow_idx-1]*rev_ord_lens[i-1];
      //  printf("lda_col[%d] = %ld len[%d] = %d\n",i-nrow_idx, lda_col[i-nrow_idx], i, rev_ord_lens[i]);
      }
      //printf("nrow_idx = %d, all_fdim = %d order = %d\n",nrow_idx,all_fdim,order);
    }
 
    int_type * rs = rows();
    int_type * cs = cols();
    char * vs = vals();

    //printf("nz=%ld\n",nz);

#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      ConstPairIterator pi(sr, tsr_data);
      int64_t k = pi[i].k();
      cs[i] = 1;
      rs[i] = 1;
      if (all_fdim != order){
        // above if means we have symmetry and are folding multiple dimensions into one longer one
        // when this is the case, first need to transform global index to a local index
        int64_t k_new = 0;
        int64_t sup_lda = 1;
        int64_t sub_tot_lda = 1;
        int last_index = -1;
        int findex = 0;
        for (int j=0; j<order; j++){
          int64_t kpart = (k%lens[j])/phase[j];
          k = k/lens[j];
          if (j == 0){
            k_new += kpart;
            if (sym[0] == NS){
              last_index = 0;
              sup_lda = pad_edge_len[0]/phase[0];
              findex = 1;
            } else {
              sub_tot_lda = pad_edge_len[0]/phase[0];
            }
          } else {
            int64_t sub_lda = 1;
            for (int l=0; l<j-last_index; l++){
              sub_lda *= kpart+l;
              sub_lda /= (l+1);
            }
            k_new += sub_lda*sup_lda;
            sub_tot_lda *= (pad_edge_len[j]/phase[j] +(j-last_index-1));
            sub_tot_lda /= (j-last_index);

            //printf("j=%d last_index = %d, mul %d\n",j,last_index,(pad_edge_len[j]/phase[j] +(j-last_index-1))/(j-last_index));
            //printf("hlll %d %d %d %d %ld %d\n", last_index,j-1, pad_edge_len[j]/phase[j], all_flen[findex], sub_tot_lda , all_flen[findex]);
            //if symmetric group of indices is of same size as next folded length, increment
            if (sym[j] == NS){
              sub_tot_lda = 1;
              sup_lda *= all_flen[findex];
              findex++;
              last_index = j;
            }
            //printf("k_orig=%ld k=%ld lens[%d] = %d phase =%d, k_new =%ld, kpart =%ld, sub_lda = %ld, sup_lda=%ld\n",pi[i].k(),k,j,lens[j],phase[j],k_new,kpart,sub_lda,sup_lda);
          }
        }
        //printf("all_fdim = %d, nrow_idx = %d\n",all_fdim,nrow_idx);
        for (int j=0; j<all_fdim; j++){
          int64_t kpart = k_new%all_flen[j];
          if (ordering[j] < nrow_idx){
            rs[i] += kpart*lda_row[ordering[j]];
          } else {
            cs[i] += kpart*lda_col[ordering[j]-nrow_idx];
          //  printf("%d %ld %d %d %ld\n",j,kpart,ordering[j],nrow_idx,lda_col[ordering[j]-nrow_idx]);
          }
          //printf("ordering[%d] = %d, kpart = %ld rs[%d] = %d cs[%d] = %d nrow_idx=%d\n",j,ordering[j],kpart,i,rs[i],i,cs[i],nrow_idx);
          k_new=k_new/all_flen[j];
        }
      } else {
        for (int j=0; j<order; j++){
          int64_t kpart = (k%lens[j])/phase[j];
          if (ordering[j] < nrow_idx){
            rs[i] += kpart*lda_row[ordering[j]];
          } else {
            cs[i] += kpart*lda_col[ordering[j]-nrow_idx];
          //  printf("%d %ld %d %d %ld\n",j,kpart,ordering[j],nrow_idx,lda_col[ordering[j]-nrow_idx]);
          }
          k=k/lens[j];
        }
      }
      pi[i].read_val(vs+v_sz*i);

      //printf("k=%ld col = %d row = %d\n", pi[i].k(), cs[i], rs[i]);
      //printf("wrote value ");
      //sr->print(pi[i].d());
      //printf(" at %p v_Sz = %d\n",vs+v_sz*i, v_sz);
    }
    cdealloc(ordering);
    cdealloc(rev_ord_lens);
    cdealloc(lda_col);
    cdealloc(lda_row);
    TAU_FSTOP(convert_to_COO);
  }

  template <typename int_type>
  void tCOO_Matrix<int_type>::get_data(int64_t nz, int order, int_type const * lens, int const * rev_ordering, int nrow_idx, char * tsr_data, algstrct const * sr, int const * phase, int const * phase_rank){
    TAU_FSTART(convert_to_COO);
    ASSERT(((int64_t*)all_data)[0] == (typeid(int_type)==typeid(int64_t)));
    ASSERT(((int64_t*)all_data)[1] == nz);
    ASSERT(((int64_t*)all_data)[2] == sr->el_size);
    int v_sz = sr->el_size;

    int * rev_ord_lens = (int*)alloc(sizeof(int)*order);
    int * ordering = (int*)alloc(sizeof(int)*order);
    int64_t * lda_col = (int64_t*)alloc(sizeof(int64_t)*(order-nrow_idx));
    int64_t * lda_row = (int64_t*)alloc(sizeof(int64_t)*nrow_idx);

#if DEBUG >= 1
    int64_t tot_sz = 1;
#endif
    //FIXME: handle symmetric folded indices as in set_data
    for (int i=0; i<order; i++){
#if DEBUG >= 1
      tot_sz *= lens[i];
#endif
      ordering[rev_ordering[i]]=i;
    }
    for (int i=0; i<order; i++){
    //  printf("[%d] %d -> %d\n", lens[i], i, ordering[i]);
      rev_ord_lens[ordering[i]] = lens[i]/phase[i];
      if (lens[i]%phase[i] > 0) rev_ord_lens[ordering[i]]++;
    }

    for (int i=0; i<order; i++){
      if (i==0 && i<nrow_idx){
        lda_row[0] = 1;
      }
      if (i>0 && i<nrow_idx){
        lda_row[i] = lda_row[i-1]*rev_ord_lens[i-1];
      }
      if (i==nrow_idx){
        lda_col[0] = 1;
      }
      if (i>nrow_idx){
        lda_col[i-nrow_idx] = lda_col[i-nrow_idx-1]*rev_ord_lens[i-1];
      //  printf("lda_col[%d] = %ld len[%d] = %d\n",i-nrow_idx, lda_col[i-nrow_idx], i, rev_ord_lens[i]);
      }
    }
 
    int_type * rs = rows();
    int_type * cs = cols();
    char * vs = vals();

#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      PairIterator pi(sr, tsr_data);
      int64_t k = 0;
      int64_t lda_k = 1;
      for (int j=0; j<order; j++){
        int64_t kpart;
        if (ordering[j] < nrow_idx){
          kpart = ((rs[i]-1)/lda_row[ordering[j]])%rev_ord_lens[ordering[j]];
        } else {
          kpart = ((cs[i]-1)/lda_col[ordering[j]-nrow_idx])%rev_ord_lens[ordering[j]];
        }
        //  printf("%d %ld %d %d %ld\n",j,kpart,ordering[j],nrow_idx,lda_col[ordering[j]-nrow_idx]);
       // if (j>0){ kpart *= lens[j-1]; }
        k+=(kpart*phase[j]+phase_rank[j])*lda_k;
/*        if (k>=tot_sz){
          printf("%d kpart %ld k1 %d phase[j-1] %d phase_rank[j-1] %d lda_k %ld\n",j-1,((k-(kpart*phase[j]+phase_rank[j])*lda_k)/(lda_k/lens[j-1])-phase_rank[j-1])/phase[j-1],k-(kpart*phase[j]+phase_rank[j])*lda_k,phase[j-1],phase_rank[j-1],lda_k/lens[j-1]);
          printf("%d kpart %ld k2 %ld phase[j] %d phase_rank[j] %d lda_k %ld\n",j,kpart,(kpart*phase[j]+phase_rank[j])*lda_k,phase[j],phase_rank[j],lda_k);
        }*/
        lda_k *= lens[j];
      }
#if DEBUG >= 1
      if (k>=tot_sz) printf("k=%ld tot_sz=%ld c = %ld r = %ld\n",k,tot_sz,(int64_t)cs[i],(int64_t)rs[i]);
      ASSERT(k<tot_sz);
#endif
      //printf("p[%d %d] [%d,%d]->%ld\n",phase_rank[0],phase_rank[1],rs[i],cs[i],k);
      pi[i].write_key(k);
      pi[i].write_val(vs+v_sz*i);
      //printf("k=%ld col = %d row = %d\n", pi[i].k(), cs[i], rs[i]);
      //sr->print(pi[i].d());
//      memcpy(pi[i].d(), vs+v_sz*i, v_sz);
    }
    PairIterator pi2(sr, tsr_data);
    TAU_FSTART(COO_to_kvpair_sort);
    pi2.sort(nz);
    TAU_FSTOP(COO_to_kvpair_sort);
    cdealloc(ordering);
    cdealloc(rev_ord_lens);
    cdealloc(lda_col);
    cdealloc(lda_row);
    TAU_FSTOP(convert_to_COO);
  }


  template <typename int_type>
  void tCOO_Matrix<int_type>::coomm(char const * A, algstrct const * sr_A, int_type m, int_type n, int_type k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func){
    ASSERT(0);
    assert(0); // COOMM not available for this int_type
  }

  template <>
  void tCOO_Matrix<int>::coomm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func){
    tCOO_Matrix cA((char*)A);
    int64_t nz = cA.nnz(); 
    int const * rs = cA.rows();
    int const * cs = cA.cols();
    char const * vs = cA.vals();
    if (func != NULL){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      func->ccoomm(m,n,k,vs,rs,cs,nz,B,C);
    } else {
      ASSERT(sr_B->el_size == sr_A->el_size);
      ASSERT(sr_C->el_size == sr_A->el_size);
      sr_A->coomm(m,n,k,alpha,vs,rs,cs,nz,B,beta,C,func);
    }
  }

  bool is_COO_int64(char const * all_data){
    return (bool)((int64_t*)all_data)[0];
  }

  template class tCOO_Matrix<int>;
  template class tCOO_Matrix<int64_t>;
}
