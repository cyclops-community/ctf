/*Copyright (c) 2014, Edgar Solomonik, all rights reserved.*/
#include "../shared/util.h"
#include "untyped_tensor.h"
#include "untyped_semiring.h"

namespace CTF_int {

  template <typename dtype>
  void dgemm(char          tA,
             char          tB,
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

/*  typedef sgemm template <> dgemm<float>;
  typedef dgemm template <> dgemm<double>;
  typedef cgemm template <> dgemm< std::complex<float> >;
  typedef zgemm template <> dgemm< std::complex<double> >;
*/
  semiring::semiring(){}

  semiring::semiring(semiring const & other){
    el_size = other.el_size;
    addid = (char*)CTF_alloc(el_size);
    memcpy(addid,other.addid,el_size);
    mulid = (char*)CTF_alloc(el_size);
    memcpy(mulid,other.mulid,el_size);
    add = other.add;
    mul = other.mul;
    gemm = other.gemm;
  }

  semiring::semiring(
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
                   void (*addinv_)(char const * a,
                                char       * b),
                   void (*gemm_)(char         tA,
                                 char         tB,
                                 int          m,
                                 int          n,
                                 int          k,
                                 char const * alpha,
                                 char const * A,
                                 char const * B,
                                 char const * beta,
                                 char *       C),
                   void (*axpy_)(int          n,
                                char const * alpha,
                                char const * X,
                                int          incX,
                                char       * Y,
                                int          incY),
                   void (*scal_)(int          n,
                                char const * alpha,
                                char const * X,
                                int          incX)){
    el_size = el_size_;
    addid = (char*)CTF_alloc(el_size);
    memcpy(addid,addid_,el_size);
    mulid = (char*)CTF_alloc(el_size);
    memcpy(mulid,mulid_,el_size);
    add = add_;
    addinv = addinv_;
    mul = mul_;
    gemm = gemm_;
    axpy = axpy_;
    scal = scal_;
  }

  semiring::~semiring(){
    free(addid);
    free(mulid);
  }

   
  bool semiring::isequal(char const * a, char const * b) const {
    bool iseq = true;
    for (int i=0; i<el_size; i++) {
      if (a[i] != b[i]) iseq = false;
    }
    return iseq;
  }
      
  void semiring::copy(char * a, char const * b) const {
    memcpy(a, b, el_size);
  }
  
  void semiring::copy_pair(char * a, char const * b) const {
    memcpy(a, b, pair_size());
  }
      
  void semiring::copy(char * a, char const * b, int64_t n) const {
    memcpy(a, b, el_size*n);
  }
  
  void semiring::copy_pairs(char * a, char const * b, int64_t n) const {
    memcpy(a, b, pair_size()*n);
  }

  void semiring::set(char * a, char const * b, int64_t n) const {
    switch (el_size) {
      case 4: {
          float * ia = (float*)a;
          float ib = *((float*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      case 8: {
          double * ia = (double*)a;
          double ib = *((double*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      case 16: {
          std::complex<double> * ia = (std::complex<double>*)a;
          std::complex<double> ib = *((std::complex<double>*)b);
          std::fill(ia, ia+n, ib);
        }
        break;
      default: {
          for (int i=0; i<n; i++) {
            memcpy(a+i*el_size, b, el_size);
          }
        }
        break;
    }
  }

  void semiring::set_pair(char * a, int64_t key, char const * vb) const {
    memcpy(a, &key, sizeof(int64_t));
    memcpy(a+sizeof(int64_t), &vb, el_size);
  }

  void semiring::set_pairs(char * a, char const * b, int64_t n) const {
    for (int i=0; i<n; i++) {
      memcpy(a + n*(sizeof(int64_t)+el_size), b, el_size);
    }
  }
 
  int64_t semiring::get_key(char const * a) const {
    return (int64_t)*a;
  }
     
  char const * semiring::get_value(char const * a) const {
    return a+sizeof(int64_t);
  }

  ConstPairIterator::ConstPairIterator(semiring const * sr_, char const * ptr_){ 
    sr=sr_; ptr=ptr_; 
  }

  ConstPairIterator ConstPairIterator::operator[](int n) const { 
    return ConstPairIterator(sr,ptr+(sr->el_size+sizeof(int64_t))*n);
  }

  int64_t ConstPairIterator::k() const {
    return (int64_t)*ptr;
  }

  char const * ConstPairIterator::d() const {
    return ptr+sizeof(int64_t);
  }

  void ConstPairIterator::read(char * buf, int64_t n) const {
    memcpy(buf, ptr, (sizeof(int64_t)+sr->el_size)*n);
  }
  
  void ConstPairIterator::read_val(char * buf) const {
    memcpy(buf, ptr+sizeof(int64_t), sr->el_size);
  }
  
  PairIterator::PairIterator(semiring const * sr_, char * ptr_){
    sr=sr_;
    ptr=ptr_;
  }

  PairIterator PairIterator::operator[](int n) const { 
    return PairIterator(sr,ptr+(sr->el_size+sizeof(int64_t))*n);
  }

  int64_t PairIterator::k() const {
    return (int64_t)*ptr;
  }

  char const * PairIterator::d() const {
    return ptr+sizeof(int64_t);
  }

  void PairIterator::read(char * buf, int64_t n) const {
    memcpy(buf, ptr, (sizeof(int64_t)+sr->el_size)*n);
  }
  
  void PairIterator::read_val(char * buf) const {
    memcpy(buf, ptr+sizeof(int64_t), sr->el_size);
  }

  void PairIterator::write(char const * buf, int64_t n){
    memcpy(ptr, buf, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write(PairIterator const iter, int64_t n){
    memcpy(ptr, iter.ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write(ConstPairIterator const iter, int64_t n){
    memcpy(ptr, iter.ptr, (sizeof(int64_t)+sr->el_size)*n);
  }

  void PairIterator::write_val(char const * buf){
    memcpy(ptr+sizeof(int64_t), buf, sr->el_size);
  }

  void PairIterator::write_key(int64_t key){
    ((int64_t*)ptr)[0] = key;
  }

  template<int l>
  struct CompPair{
    int64_t key;
    char data[l];
    bool operator < (const CompPair& other) const {
      return (key < other.key);
    }
  };
  template struct CompPair<1>;
  template struct CompPair<2>;
  template struct CompPair<4>;
  template struct CompPair<8>;
  template struct CompPair<12>;
  template struct CompPair<16>;
  template struct CompPair<20>;
  template struct CompPair<24>;
  template struct CompPair<28>;
  template struct CompPair<32>;
  
  struct CompPtrPair{
    int64_t key;
    int64_t idx;
    bool operator < (const CompPtrPair& other) const {
      return (key < other.key);
    }
  };

  void PairIterator::sort(int64_t n){
    switch (sr->el_size){
      case 1:
        std::sort((CompPair<1>*)ptr,((CompPair<1>*)ptr)+n);
        break;
      case 2:
        std::sort((CompPair<2>*)ptr,((CompPair<2>*)ptr)+n);
        break;
      case 4:
        std::sort((CompPair<4>*)ptr,((CompPair<4>*)ptr)+n);
        break;
      case 8:
        std::sort((CompPair<8>*)ptr,((CompPair<8>*)ptr)+n);
        break;
      case 12:
        std::sort((CompPair<12>*)ptr,((CompPair<12>*)ptr)+n);
        break;
      case 16:
        std::sort((CompPair<16>*)ptr,((CompPair<16>*)ptr)+n);
        break;
      case 20:
        std::sort((CompPair<20>*)ptr,((CompPair<20>*)ptr)+n);
        break;
      case 24:
        std::sort((CompPair<24>*)ptr,((CompPair<24>*)ptr)+n);
        break;
      case 28:
        std::sort((CompPair<28>*)ptr,((CompPair<28>*)ptr)+n);
        break;
      case 32:
        std::sort((CompPair<32>*)ptr,((CompPair<32>*)ptr)+n);
        break;
      default:
        CompPtrPair ptr_pairs[n];
        #pragma omp parallel
        for (int64_t i=0; i<n; i++){
          ptr_pairs[i].key = *(int64_t*)(ptr+i*(sizeof(int64_t)+sr->el_size));
          ptr_pairs[i].idx = i;
        }
        //FIXME :(
        char swap_buffer[(sizeof(int64_t)+sr->el_size)*n];
    
        memcpy(swap_buffer, ptr, (sizeof(int64_t)+sr->el_size)*n);
        
        #pragma omp parallel
        for (int64_t i=0; i<n; i++){
          memcpy(ptr+i*(sizeof(int64_t)+sr->el_size), 
                 swap_buffer+ptr_pairs[i].idx*(sizeof(int64_t)+sr->el_size),
                 sizeof(int64_t)+sr->el_size);
        }
    }
  }
}

