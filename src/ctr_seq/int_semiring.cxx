/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/
#include "../shared/util.h"
#include "../../include/ctf.hpp"

template <typename dtype>
void dgemm(dtype         tA,
           dtype         tB,
           int           m,
           int           n,
           int           k,
           dtype         alpha,
           dtype const * A,
           dtype const * B,
           dtype         beta,
           dtype *       C){
  int lda_A, lda_B, lda_C;
  lda_C = m;
  if (tA == 'n' || tA == 'N'){
    lda_A = m;
  } else {
    lda_A = k;
  }
  if (tB == 'n' || tB == 'N'){
    lda_B = k;
  } else {
    lda_B = n;
  }
  cxgemm<dtype>(tA,tB,m,n,k,alpha,A,lda_A,B,lda_B,beta,C,lda_C);
}

typedef sgemm template <> dgemm<float>;
typedef dgemm template <> dgemm<double>;
typedef cgemm template <> dgemm< std::complex<float> >;
typedef zgemm template <> dgemm< std::complex<double> >;

Int_Semiring::Int_Semiring(Int_Semiring const & other){
  el_size = other.el_size;
  addid = (char*)alloc(el_size);
  memcpy(addid,other.addid,el_size);
  mulid = (char*)alloc(el_size);
  memcpy(mulid,other.mulid,el_size);
  add = other.add;
  mul = other.mul;
  gemm = other.gemm;
}

Int_Semiring::Int_Semiring(
                 int          el_size_, 
                 char const * addid_,
                 char const * mulid_,
                 MPI_Op       addmop_,
                 void (*add_)(char const * a,
                              char const * b,
                              char       * c),
                 void (*mul_)(char const * a,
                              char const * b,
                              char       * c),
                 void (*gemm_)(char         tA,
                               char         tB,
                               int          m,
                               int          n,
                               int          k,
                               char const * alpha,
                               char const * A,
                               char const * B,
                               char const * beta,
                               char *       C)){
  el_size = el_size_;
  addid = (char*)alloc(el_size);
  memcpy(addid,addid_,el_size);
  mulid = (char*)alloc(el_size);
  memcpy(mulid,mulid_,el_size);
  add = add_;
  mul = mul_;
  gemm = gemm_;
}

Int_Semiring::~Int_Semiring(){
  free(addid);
  free(mulid);
}

