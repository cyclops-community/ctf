#include "csr.h"
#include "../contraction/ctr_comm.h"
#include "../shared/util.h"

#define ALIGN 256

namespace CTF_int {
  int64_t get_csr_size(int64_t nnz, int nrow, int val_size){
    int offset = 3*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += nnz*val_size;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nrow+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return offset + sizeof(int)*nnz;
  }

  CSR_Matrix::CSR_Matrix(int64_t nnz, int nrow, algstrct const * sr){
    ASSERT(ALIGN >= 16);
    int64_t size = get_csr_size(nnz, nrow, sr->el_size);
    all_data = (char*)alloc(size);
    ((int64_t*)all_data)[0] = nnz;
    ((int64_t*)all_data)[1] = sr->el_size;
    ((int64_t*)all_data)[2] = nrow;
  }

  CSR_Matrix::CSR_Matrix(char * all_data_){
    ASSERT(ALIGN >= 16);
    all_data = all_data_;
  }

  CSR_Matrix::CSR_Matrix(COO_Matrix const & coom, int nrow, algstrct const * sr, char * data){
    ASSERT(ALIGN >= 16);
    int64_t nz = coom.nnz(); 
    int64_t v_sz = coom.val_size(); 
    int const * coo_rs = coom.rows();
    int const * coo_cs = coom.cols();
    char const * vs = coom.vals();

    int64_t size = get_csr_size(nz, nrow, v_sz);
    if (data == NULL)
      all_data = (char*)alloc(size);
    else
      all_data = data;
    ((int64_t*)all_data)[0] = nz;
    ((int64_t*)all_data)[1] = v_sz;
    ((int64_t*)all_data)[2] = nrow;

    char * csr_vs = vals();
    int * csr_rs = rows();
    int * csr_cs = cols();

    //memcpy(csr_vs, vs, nz*v_sz);
    //memset(csr_rs

    sr->coo_to_csr(nz, nrow, csr_vs, csr_cs, csr_rs, vs, coo_rs, coo_cs);
/*    for (int i=0; i<nrow; i++){
      printf("csr_rs[%d] = %d\n",i,csr_rs[i]);
    }
    for (int i=0; i<nz; i++){
      printf("csr_cs[%d] = %d\n",i,csr_cs[i]);
    }*/
    
  }

  int64_t CSR_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int CSR_Matrix::val_size() const {
    return ((int64_t*)all_data)[1];
  }


  int64_t CSR_Matrix::size() const {
    return get_csr_size(nnz(),nrow(),val_size());
  }
  
  int CSR_Matrix::nrow() const {
    return ((int64_t*)all_data)[2];
  }
  
  char * CSR_Matrix::vals() const {
    int offset = 3*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    return all_data + offset;
  }

  int * CSR_Matrix::rows() const {
    int64_t n = this->nnz();
    int v_sz = this->val_size();

    int offset = 3*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);

    return (int*)(all_data + offset);
  } 

  int * CSR_Matrix::cols() const {
    int64_t n = this->nnz();
    int64_t nr = this->nrow();
    int v_sz = this->val_size();

    int offset = 3*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += n*v_sz;
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    offset += (nr+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    //return (int*)(all_data + n*v_sz+(nr+1)*sizeof(int)+3*sizeof(int64_t));
    return (int*)(all_data + offset);
  } 

  void CSR_Matrix::csrmm(algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload){
    if (func != NULL && func->has_gemm && do_offload){
      assert(sr_C->isequal(beta, sr_C->mulid()));
      assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
      func->coffload_csrmm(m,n,k,all_data,B,C);
    } else {
      int64_t nz = nnz(); 
      int const * rs = rows();
      int const * cs = cols();
      char const * vs = vals();
      if (func != NULL && func->has_gemm){
        assert(sr_C->isequal(beta, sr_C->mulid()));
        assert(alpha == NULL || sr_C->isequal(alpha, sr_C->mulid()));
        func->ccsrmm(m,n,k,vs,rs,cs,nz,B,C);
      } else {
        ASSERT(sr_B->el_size == sr_A->el_size);
        ASSERT(sr_C->el_size == sr_A->el_size);
        assert(!do_offload);
        sr_A->csrmm(m,n,k,alpha,vs,rs,cs,nz,B,beta,C,func);
      }
    }
  }

}
