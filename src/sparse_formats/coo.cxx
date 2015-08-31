#include "coo.h"
#include "../shared/util.h"

namespace CTF_int {
  int64_t get_coo_size(int64_t nnz, int val_size){
    return nnz*(val_size+sizeof(int)*2)+2*sizeof(int64_t);
  }

  COO_Matrix::COO_Matrix(int64_t nnz, algstrct const * sr){
    int64_t size = get_coo_size(nnz, sr->el_size);
    all_data = (char*)alloc(size);
  }

  COO_Matrix::COO_Matrix(char * all_data_){
    all_data = all_data_;
  }

  int64_t COO_Matrix::nnz() const {
    return ((int64_t*)all_data)[0];
  }

  int64_t COO_Matrix::size() const {
    return nnz()*((int64_t*)all_data)[1];
  }
  
  char * COO_Matrix::vals() const {
    return all_data + 2*sizeof(int64_t);
  }

  int * COO_Matrix::rows() const {
    int64_t n = this->nnz();
    int val_size = ((int64_t*)all_data)[1];

    return (int*)(all_data + n*val_size+2*sizeof(int64_t));
  } 

  int * COO_Matrix::cols() const {
    int64_t n = this->nnz();
    int val_size = ((int64_t*)all_data)[1];

    return (int*)(all_data + n*(val_size+sizeof(int))+2*sizeof(int64_t));
  } 

  void COO_Matrix::coomm(algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, Bivar_Function const * func){
    int64_t nz = nnz(); 
    int const * rs = rows();
    int const * cs = cols();
    char const * vs = vals();
    ASSERT(sr_B == sr_A);
    ASSERT(sr_C == sr_A);
    sr_A->coomm(m,n,k,alpha,vs,rs,cs,nz,B,beta,C,func);
  }
}
