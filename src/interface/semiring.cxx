#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/offload.h"
#include "../sparse_formats/csr.h"
#include "../shared/util.h"

#ifdef USE_MKL
#include "../shared/mkl_symbs.h"
#endif

using namespace CTF_int;

namespace CTF_int {

  template <typename dtype>
  void gemm_batch(
            char           taA,
            char           taB,
            int            l,
            int            m,
            int            n,
            int            k,
            dtype          alpha,
            dtype   const* A,
            dtype   const* B,
            dtype          beta,
            dtype   *      C){
    if (m == 1 && n == 1 && k == 1) {
      for (int i=0; i<l; i++){
        C[i]*=beta;
        C[i]+=alpha*A[i]*B[i];
      }
      return;
    }
    int lda, ldb, ldc;
    ldc = m;
    if (taA == 'n' || taA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (taB == 'n' || taB == 'N'){
      ldb = k;
    } else {
      ldb = n;
    }
    dtype ** ptrs_A = get_grp_ptrs(m*k,l,A);
    dtype ** ptrs_B = get_grp_ptrs(k*n,l,B);
    dtype ** ptrs_C = get_grp_ptrs(m*n,l,C);
#if USE_BATCH_GEMM
    int group_count = 1;
    int size_per_group = l;
    CTF_BLAS::gemm_batch<dtype>(&taA, &taB, &m, &n, &k, &alpha, ptrs_A, &lda, ptrs_B, &ldb, &beta, ptrs_C, &ldc, &group_count, &size_per_group);
#else 
    for (int i=0; i<l; i++){
      CTF_BLAS::gemm<dtype>(&taA,&taB,&m,&n,&k,&alpha, ptrs_A[i] ,&lda, ptrs_B[i] ,&ldb,&beta, ptrs_C[i] ,&ldc);
    }
#endif
    free(ptrs_A);
    free(ptrs_B);
    free(ptrs_C);
  }

#define INST_GEMM_BATCH(dtype)            \
  template void gemm_batch<dtype>( char , \
             char ,                       \
             int ,                        \
             int ,                        \
             int ,                        \
             int ,                        \
             dtype ,                      \
             dtype const *,               \
             dtype const *,               \
             dtype ,                      \
             dtype *);
  INST_GEMM_BATCH(float)
  INST_GEMM_BATCH(double)
  INST_GEMM_BATCH(std::complex<float>)
  INST_GEMM_BATCH(std::complex<double>)
#undef INST_GEMM_BATCH

  template <typename dtype>
  void gemm(char           tA,
            char           tB,
            int            m,
            int            n,
            int            k,
            dtype          alpha,
            dtype  const * A,
            dtype  const * B,
            dtype          beta,
            dtype  *       C){
    int lda, lda_B, lda_C;
    lda_C = m;
    if (tA == 'n' || tA == 'N'){
      lda = m;
    } else {
      lda = k;
    }
    if (tB == 'n' || tB == 'N'){
      lda_B = k;
    } else {
      lda_B = n;
    }
    CTF_BLAS::gemm<dtype>(&tA,&tB,&m,&n,&k,&alpha,A,&lda,B,&lda_B,&beta,C,&lda_C);
  }

#define INST_GEMM(dtype)            \
  template void gemm<dtype>( char , \
             char ,                 \
             int ,                  \
             int ,                  \
             int ,                  \
             dtype ,                \
             dtype const *,         \
             dtype const *,         \
             dtype ,                \
             dtype *);
  INST_GEMM(float)
  INST_GEMM(double)
  INST_GEMM(std::complex<float>)
  INST_GEMM(std::complex<double>)
#undef INST_GEMM



  template <>
  void default_axpy<float>
                   (int           n,
                    float         alpha,
                    float const * X,
                    int           incX,
                    float *       Y,
                    int           incY){
    CTF_BLAS::SAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY){
    CTF_BLAS::DAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY){
    CTF_BLAS::CAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY){
    CTF_BLAS::ZAXPY(&n,&alpha,X,&incX,Y,&incY);
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX){
    CTF_BLAS::SSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX){
    CTF_BLAS::DSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX){
    CTF_BLAS::CSCAL(&n,&alpha,X,&incX);
  }

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX){
    CTF_BLAS::ZSCAL(&n,&alpha,X,&incX);
  }

#if USE_MKL

  template <>
  void default_vec_mul<float>(float const * a, float const * b, float * c, int64_t n){
    int nn = n;
    CTF_BLAS::MKL_VSMUL(&nn, a, b, c);
  }

  template <>
  void default_vec_mul<double>(double const * a, double const * b, double * c, int64_t n){
    int nn = n;
    CTF_BLAS::MKL_VDMUL(&nn, a, b, c);
  }
  template <>
  void default_vec_mul<std::complex<float>>(std::complex<float> const * a, std::complex<float> const * b, std::complex<float> * c, int64_t n){
    int nn = n;
    CTF_BLAS::MKL_VCMUL(&nn, a, b, c);
  }

  template <>
  void default_vec_mul<std::complex<double>>(std::complex<double> const * a, std::complex<double> const * b, std::complex<double> * c, int64_t n){
    int nn = n;
    CTF_BLAS::MKL_VZMUL(&nn, a, b, c);
  }

#endif

#define DEF_COOMM_KERNEL()                                \
    for (int j=0; j<n; j++){                              \
      for (int i=0; i<m; i++){                            \
        C[j*m+i] *= beta;                                 \
      }                                                   \
    }                                                     \
    for (int i=0; i<nnz_A; i++){                          \
      int row_A = rows_A[i]-1;                            \
      int col_A = cols_A[i]-1;                            \
      for (int col_C=0; col_C<n; col_C++){                \
         C[col_C*m+row_A] += alpha*A[i]*B[col_C*k+col_A]; \
      }                                                   \
    }

  template <>
  void default_coomm< float >
          (int           m,
           int           n,
           int           k,
           float         alpha,
           float const * A,
           int const *   rows_A,
           int const *   cols_A,
           int           nnz_A,
           float const * B,
           float         beta,
           float *       C){
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_SCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (float*)A, rows_A, cols_A, &nnz_A,
               (float*)B, &k, &beta,
               (float*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }

  template <>
  void default_coomm< double >
          (int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           int const *    rows_A,
           int const *    cols_A,
           int            nnz_A,
           double const * B,
           double         beta,
           double *       C){
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    //TAU_FSTART(MKL_DCOOMM);
    CTF_BLAS::MKL_DCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (double*)A, rows_A, cols_A, &nnz_A,
               (double*)B, &k, &beta,
               (double*)C, &m);
    //TAU_FSTOP(MKL_DCOOMM);
#else
    DEF_COOMM_KERNEL();
#endif
  }


  template <>
  void default_coomm< std::complex<float> >
          (int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           int const *                 rows_A,
           int const *                 cols_A,
           int                         nnz_A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C){
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_CCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<float>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<float>*)B, &k, &beta,
               (std::complex<float>*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }

  template <>
  void default_coomm< std::complex<double> >
     (int                          m,
      int                          n,
      int                          k,
      std::complex<double>         alpha,
      std::complex<double> const * A,
      int const *                  rows_A,
      int const *                  cols_A,
      int                          nnz_A,
      std::complex<double> const * B,
      std::complex<double>         beta,
      std::complex<double> *       C){
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (std::complex<double>*)A, rows_A, cols_A, &nnz_A,
               (std::complex<double>*)B, &k, &beta,
               (std::complex<double>*)C, &m);
#else
    DEF_COOMM_KERNEL();
#endif
  }
/*
#if USE_MKL
  template <>
  bool get_def_has_csrmm<float>(){ return true; }
  template <>
  bool get_def_has_csrmm<double>(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<float> >(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<double> >(){ return true; }
#else
  template <>
  bool get_def_has_csrmm<float>(){ return true; }
  template <>
  bool get_def_has_csrmm<double>(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<float> >(){ return true; }
  template <>
  bool get_def_has_csrmm< std::complex<double> >(){ return true; }
#endif
*/
#if (USE_MKL!=1)
  template <typename dtype>
  void muladd_csrmm
                 (int           m,
                  int           n,
                  int           k,
                  dtype         alpha,
                  dtype const * A,
                  int const *   JA,
                  int const *   IA,
                  int           nnz_A,
                  dtype const * B,
                  dtype         beta,
                  dtype *       C){
    //TAU_FSTART(muladd_csrmm);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
//#ifdef USE_OMP
//      #pragma omp parallel for
//#endif
      for (int col_B=0; col_B<n; col_B++){
        C[col_B*m+row_A] *= beta;
        if (IA[row_A] < IA[row_A+1]){
          int i_A1 = IA[row_A]-1;
          int col_A1 = JA[i_A1]-1;
          dtype tmp = A[i_A1]*B[col_B*k+col_A1];
          for (int i_A=IA[row_A]; i_A<IA[row_A+1]-1; i_A++){
            int col_A = JA[i_A]-1;
            tmp += A[i_A]*B[col_B*k+col_A];
          }
          C[col_B*m+row_A] += alpha*tmp;
        }
      }
    }
    //TAU_FSTOP(muladd_csrmm);
  }

  template<typename dtype>
  void muladd_csrmultd
                 (int           m,
                  int           n,
                  int           k,
                  dtype const * A,
                  int const *   JA,
                  int const *   IA,
                  int           nnz_A,
                  dtype const * B,
                  int const *   JB,
                  int const *   IB,
                  int           nnz_B,
                  dtype *       C){
    //TAU_FSTART(muladd_csrmultd);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
      for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
        int row_B = JA[i_A]-1; //=col_A
        for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
          int col_B = JB[i_B]-1;
          C[col_B*m+row_A] += A[i_A]*B[i_B];
          //printf("Here coor=%d, %d/%d %d/%d row_A = %d row_B = %d val = %lf after adding %lf*%lf\n",col_B*m+row_A,i_A,IA[row_A+1]-1,i_B,IB[row_B+1]-1,row_A,row_B,C[col_B*m+row_A],A[i_A],B[i_B]);
        }
      }
    }
    //TAU_FSTOP(muladd_csrmultd);
  }
#endif
}

namespace CTF {

  template <>
  void CTF::Semiring<float,1>::default_csrmm
          (int           m,
           int           n,
           int           k,
           float         alpha,
           float const * A,
           int const *   JA,
           int const *   IA,
           int           nnz_A,
           float const * B,
           float         beta,
           float *       C) const {
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_SCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm<float>(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template <>
  void CTF::Semiring<double,1>::default_csrmm
          (int            m,
           int            n,
           int            k,
           double         alpha,
           double const * A,
           int const *    JA,
           int const *    IA,
           int            nnz_A,
           double const * B,
           double         beta,
           double *       C) const {
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    //TAU_FSTART(MKL_DCSRMM);
    CTF_BLAS::MKL_DCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
    //TAU_FSTOP(MKL_DCSRMM);
#else
    CTF_int::muladd_csrmm<double>(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }


  template <>
  void CTF::Semiring<std::complex<float>,0>::default_csrmm
          (int                         m,
           int                         n,
           int                         k,
           std::complex<float>         alpha,
           std::complex<float> const * A,
           int const *                 JA,
           int const *                 IA,
           int                         nnz_A,
           std::complex<float> const * B,
           std::complex<float>         beta,
           std::complex<float> *       C) const {
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_CCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm< std::complex<float> >(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template <>
  void CTF::Semiring<std::complex<double>,0>::default_csrmm
          (int                          m,
           int                          n,
           int                          k,
           std::complex<double>         alpha,
           std::complex<double> const * A,
           int const *                  JA,
           int const *                  IA,
           int                          nnz_A,
           std::complex<double> const * B,
           std::complex<double>         beta,
           std::complex<double> *       C) const {
#if USE_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm< std::complex<double> >(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }


#if USE_MKL 
  #define CSR_MULTD_DEF(dtype,is_ord,MKL_name) \
  template<> \
  void CTF::Semiring<dtype,is_ord>::default_csrmultd \
                 (int            m, \
                  int            n, \
                  int            k, \
                  dtype          alpha, \
                  dtype const *  A, \
                  int const *    JA, \
                  int const *    IA, \
                  int            nnz_A, \
                  dtype const *  B, \
                  int const *    JB, \
                  int const *    IB, \
                  int            nnz_B, \
                  dtype          beta, \
                  dtype *        C) const { \
    if (alpha == this->taddid){ \
      if (beta != this->tmulid) \
        CTF_int::default_scal<dtype>(m*n, beta, C, 1); \
      return; \
    } \
    char transa = 'N'; \
    if (beta == this->taddid){ \
      CTF_BLAS::MKL_name(&transa, &m, &k, &n, A, JA, IA, B, JB, IB, C, &m); \
      if (alpha != this->tmulid) \
        CTF_int::default_scal<dtype>(m*n, alpha, C, 1); \
    } else { \
      dtype * tmp_C_buf = (dtype*)this->alloc(m*n); \
      CTF_BLAS::MKL_name(&transa, &m, &k, &n, A, JA, IA, B, JB, IB, tmp_C_buf, &m); \
      if (beta != this->tmulid) \
        CTF_int::default_scal<dtype>(m*n, beta, C, 1); \
      CTF_int::default_axpy<dtype>(m*n, alpha, tmp_C_buf, 1, C, 1); \
      this->dealloc((char*)tmp_C_buf); \
    } \
  }
#else 
  #define CSR_MULTD_DEF(dtype,is_ord,MKL_name) \
  template<> \
  void CTF::Semiring<dtype,is_ord>::default_csrmultd \
                 (int           m, \
                  int           n, \
                  int           k, \
                  dtype         alpha, \
                  dtype const * A, \
                  int const *   JA, \
                  int const *   IA, \
                  int           nnz_A, \
                  dtype const * B, \
                  int const *   JB, \
                  int const *   IB, \
                  int           nnz_B, \
                  dtype         beta, \
                  dtype *       C) const { \
    if (alpha == this->taddid){ \
      if (beta != this->tmulid) \
        CTF_int::default_scal<dtype>(m*n, beta, C, 1); \
      return; \
    } \
    if (alpha != this->tmulid || beta != this->tmulid){ \
      CTF_int::default_scal<dtype>(m*n, beta/alpha, C, 1); \
    } \
    CTF_int::muladd_csrmultd<dtype>(m,n,k,A,JA,IA,nnz_A,B,JB,IB,nnz_B,C); \
    if (alpha != this->tmulid){ \
      CTF_int::default_scal<dtype>(m*n, alpha, C, 1); \
    } \
  } 
#endif

  CSR_MULTD_DEF(float,1,MKL_SCSRMULTD)
  CSR_MULTD_DEF(double,1,MKL_DCSRMULTD)
  CSR_MULTD_DEF(std::complex<float>,0,MKL_CCSRMULTD)
  CSR_MULTD_DEF(std::complex<double>,0,MKL_ZCSRMULTD)


#if USE_MKL
  #define CSR_MULTCSR_DEF(dtype,is_ord,MKL_name) \
  template<> \
  void CTF::Semiring<dtype,is_ord>::default_csrmultcsr \
                     (int           m, \
                      int           n, \
                      int           k, \
                      dtype         alpha, \
                      dtype const * A, \
                      int const *   JA, \
                      int const *   IA, \
                      int           nnz_A, \
                      dtype const * B, \
                      int const *   JB, \
                      int const *   IB, \
                      int           nnz_B, \
                      dtype         beta, \
                      char *&       C_CSR) const { \
    char transa = 'N'; \
    CSR_Matrix C_in(C_CSR); \
 \
    int * new_ic = (int*)CTF_int::alloc(sizeof(int)*(m+1)); \
  \
    int sort = 1;  \
    int req = 1; \
    int info; \
    CTF_BLAS::MKL_name(&transa, &req, &sort, &m, &k, &n, A, JA, IA, B, JB, IB, NULL, NULL, new_ic, &req, &info); \
 \
    CSR_Matrix C_add(new_ic[m]-1, m, n, this); \
    memcpy(C_add.IA(), new_ic, (m+1)*sizeof(int)); \
    cdealloc((char*)new_ic); \
    req = 2; \
    CTF_BLAS::MKL_name(&transa, &req, &sort, &m, &k, &n, A, JA, IA, B, JB, IB, (dtype*)C_add.vals(), C_add.JA(), C_add.IA(), &req, &info); \
 \
    if (beta == this->taddid){ \
      C_CSR = C_add.all_data; \
    } else { \
      if (C_CSR != NULL && beta != this->tmulid){ \
        this->scal(C_in.nnz(), (char const *)&beta, C_in.vals(), 1); \
      } \
      if (alpha != this->tmulid){ \
        this->scal(C_add.nnz(), (char const *)&alpha, C_add.vals(), 1); \
      } \
      if (C_CSR == NULL){ \
        C_CSR = C_add.all_data; \
      } else { \
        char * C_ret = csr_add(C_CSR, C_add.all_data, false); \
        cdealloc((char*)C_add.all_data); \
        C_CSR = C_ret; \
      } \
    } \
  }
#else
  #define CSR_MULTCSR_DEF(dtype,is_ord,MKL_name) \
  template<> \
  void CTF::Semiring<dtype,is_ord>::default_csrmultcsr \
                     (int           m, \
                      int           n, \
                      int           k, \
                      dtype         alpha, \
                      dtype const * A, \
                      int const *   JA, \
                      int const *   IA, \
                      int           nnz_A, \
                      dtype const * B, \
                      int const *   JB, \
                      int const *   IB, \
                      int           nnz_B, \
                      dtype         beta, \
                      char *&       C_CSR) const { \
    this->gen_csrmultcsr(m,n,k,alpha,A,JA,IA,nnz_A,B,JB,IB,nnz_B,beta,C_CSR); \
  }
#endif

  CSR_MULTCSR_DEF(float,1,MKL_SCSRMULTCSR)
  CSR_MULTCSR_DEF(double,1,MKL_DCSRMULTCSR)
  CSR_MULTCSR_DEF(std::complex<float>,0,MKL_CCSRMULTCSR)
  CSR_MULTCSR_DEF(std::complex<double>,0,MKL_ZCSRMULTCSR)

/*  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<float>;
  }*/
  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<float>;
  }

  template<> 
  bool CTF::Semiring<std::complex<float>,0>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm< std::complex<float> >;
  }


  template<> 
  bool CTF::Semiring<double,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<double>;
  }

  template<> 
  bool CTF::Semiring<std::complex<double>,0>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm< std::complex<double> >;
  }

  template<> 
  void CTF::Semiring<float,1>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<float>(tA, tB, m, n, k, ((float const*)alpha)[0], (float const *)A, lda_A, (float const *)B, lda_B, ((float const*)beta)[0], (float*)C, m);
  }

  template<> 
  void CTF::Semiring<std::complex<float>,0>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<std::complex<float>>(tA, tB, m, n, k, ((std::complex<float> const*)alpha)[0], (std::complex<float> const *)A, lda_A, (std::complex<float> const *)B, lda_B, ((std::complex<float> const*)beta)[0], (std::complex<float>*)C, m);
  }

  template<> 
  void CTF::Semiring<double,1>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<double>(tA, tB, m, n, k, ((double const*)alpha)[0], (double const *)A, lda_A, (double const *)B, lda_B, ((double const*)beta)[0], (double*)C, m);
  }

  template<> 
  void CTF::Semiring<std::complex<double>,0>::offload_gemm(
                        char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
    int lda_A = k;
    if (tA == 'n' || tA == 'N') lda_A = m;
    int lda_B = n;
    if (tB == 'N' || tB == 'N') lda_B = k;
    CTF_int::offload_gemm<std::complex<double>>(tA, tB, m, n, k, ((std::complex<double> const*)alpha)[0], (std::complex<double> const *)A, lda_A, (std::complex<double> const *)B, lda_B, ((std::complex<double> const*)beta)[0], (std::complex<double>*)C, m);
  }


  template<> 
  bool CTF::Semiring<double,1>::is_last_col_zero(int64_t m, int64_t n, double const * M) const {
    TAU_FSTART(is_last_col_zero);
    for (int64_t i=0; i<n; i++){
      if (M[m*(n-1)+i]!= this->taddid){
        TAU_FSTOP(is_last_col_zero);
        return false;
      }
    }
    TAU_FSTOP(is_last_col_zero);
    return true;
  }
   

}
