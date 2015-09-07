#include "coo.h"
#include "../shared/util.h"

namespace CTF_int {
  int64_t get_coo_size(int64_t nnz, int val_size){
    return nnz*(val_size+sizeof(int)*2)+2*sizeof(int64_t);
  }

  COO_Matrix::COO_Matrix(int64_t nnz, algstrct const * sr){
    int64_t size = get_coo_size(nnz, sr->el_size);
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = nnz;
    ((int64_t*)all_data)[1] = sr->el_size;
  }

  COO_Matrix::COO_Matrix(char * all_data_){
    all_data = all_data_;
  }

  int64_t COO_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int COO_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }

  int64_t COO_Matrix::size() const {
    return get_coo_size(nnz(),val_size());
  }
  
  char * COO_Matrix::vals() const {
    return all_data + 2*sizeof(int64_t);
  }

  int * COO_Matrix::rows() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();

    return (int*)(all_data + n*v_sz+2*sizeof(int64_t));
  } 

  int * COO_Matrix::cols() const {
    int64_t n = this->nnz();
    int v_sz = ((int64_t*)all_data)[1];

    return (int*)(all_data + n*(v_sz+sizeof(int))+2*sizeof(int64_t));
  } 

  void COO_Matrix::set_data(int64_t nz, int order, int const * lens, int const * rev_ordering, int nrow_idx, char const * tsr_data, algstrct const * sr, int const * phase){
    TAU_FSTART(convert_to_COO);
    ((int64_t*)all_data)[0] = nz;
    ((int64_t*)all_data)[1] = sr->el_size;
    int v_sz = sr->el_size;

    int * rev_ord_lens = (int*)alloc(sizeof(int)*order);
    int * ordering = (int*)alloc(sizeof(int)*order);
    int64_t * lda_col = (int64_t*)alloc(sizeof(int64_t)*(order-nrow_idx));
    int64_t * lda_row = (int64_t*)alloc(sizeof(int64_t)*nrow_idx);

    for (int i=0; i<order; i++){
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
 
    int * rs = rows();
    int * cs = cols();
    char * vs = vals();

    ConstPairIterator pi(sr, tsr_data);
    for (int64_t i=0; i<nz; i++){
      int64_t k = pi[i].k();
      cs[i] = 1;
      rs[i] = 1;
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
    //  printf("k=%ld col = %d row = %d\n", pi[i].k(), cs[i], rs[i]);
      memcpy(vs+v_sz*i, pi[i].d(), v_sz);
    }
    cdealloc(ordering);
    cdealloc(rev_ord_lens);
    cdealloc(lda_col);
    cdealloc(lda_row);
    TAU_FSTOP(convert_to_COO);
  }


  void COO_Matrix::coomm(algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func){
    int64_t nz = nnz(); 
    int const * rs = rows();
    int const * cs = cols();
    char const * vs = vals();
    ASSERT(sr_B->el_size == sr_A->el_size);
    ASSERT(sr_C->el_size == sr_A->el_size);
    sr_A->coomm(m,n,k,alpha,vs,rs,cs,nz,B,beta,C,func);
  }
}
