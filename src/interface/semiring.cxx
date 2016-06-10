#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/offload.h"
#include "../sparse_formats/csr.h"

using namespace CTF_int;

namespace CTF_int {
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
#if USE_SP_MKL
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
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    TAU_FSTART(MKL_DCOOMM);
    CTF_BLAS::MKL_DCOOMM(&transa, &m, &n, &k, &alpha,
               matdescra, (double*)A, rows_A, cols_A, &nnz_A,
               (double*)B, &k, &beta,
               (double*)C, &m);
    TAU_FSTOP(MKL_DCOOMM);
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
#if USE_SP_MKL
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
#if USE_SP_MKL
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
#if USE_SP_MKL
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
#if (USE_SP_MKL!=1)
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
    TAU_FSTART(muladd_csrmm);
#ifdef USE_OMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
#ifdef USE_OMP
      #pragma omp parallel for
#endif
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
    TAU_FSTOP(muladd_csrmm);
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
    TAU_FSTART(muladd_csrmultd);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int row_A=0; row_A<m; row_A++){
      for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
        int row_B = JA[i_A]-1; //=col_A
        for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
          int col_B = JB[i_B]-1;
          C[col_B*m+row_A] += A[i_A]*B[i_B];
        }
      }
    }
    TAU_FSTOP(muladd_csrmultd);
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
#if USE_SP_MKL
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
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    TAU_FSTART(MKL_DCSRMM);
    CTF_BLAS::MKL_DCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
    TAU_FSTOP(MKL_DCSRMM);
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
#if USE_SP_MKL
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
#if USE_SP_MKL
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    CTF_BLAS::MKL_ZCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, JA, IA, IA+1, B, &k, &beta, C, &m);
#else
    CTF_int::muladd_csrmm< std::complex<double> >(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
#endif
  }

  template<>
  void CTF::Semiring<double,1>::default_csrmultd
                 (int            m,
                  int            n,
                  int            k,
                  double         alpha,
                  double const * A,
                  int const *    JA,
                  int const *    IA,
                  int            nnz_A,
                  double const * B,
                  int const *    JB,
                  int const *    IB,
                  int            nnz_B,
                  double         beta,
                  double *       C) const {
    TAU_FSTART(csrmultd);
    if (alpha == 0.0){
      if (beta != 1.0)
        CTF_int::default_scal<double>(m*n, beta, C, 1);
      return;
    }
#if USE_SP_MKL
    char transa = 'N';
    if (beta == 0.0){
      TAU_FSTART(MKL_DCSRMULTD);
      CTF_BLAS::MKL_DCSRMULTD(&transa, &m, &k, &n, A, JA, IA, B, JB, IB, C, &m);
      TAU_FSTOP(MKL_DCSRMULTD);
      if (alpha != 1.0)
        CTF_int::default_scal<double>(m*n, alpha, C, 1);
    } else {
      double * tmp_C_buf = (double*)malloc(sizeof(double)*m*n);
      TAU_FSTART(MKL_DCSRMULTD);
      CTF_BLAS::MKL_DCSRMULTD(&transa, &m, &k, &n, A, JA, IA, B, JB, IB, tmp_C_buf, &m);
      TAU_FSTOP(MKL_DCSRMULTD);
      if (beta != 1.0)
        CTF_int::default_scal<double>(m*n, beta, C, 1);
      CTF_int::default_axpy<double>(m*n, alpha, tmp_C_buf, 1, C, 1);
      free(tmp_C_buf);
    }
#else
    if (alpha != 1.0 || beta != 1.0){
      CTF_int::default_scal<double>(m*n, beta/alpha, C, 1);
    }
    CTF_int::muladd_csrmultd<double>(m,n,k,A,JA,IA,nnz_A,B,JB,IB,nnz_B,C);
    if (alpha != 1.0){
      CTF_int::default_scal<double>(m*n, alpha, C, 1);
    }
#endif
    TAU_FSTOP(csrmultd);

  }


  template<>
  void CTF::Semiring<double,1>::default_csrmultcsr
                     (int            m,
                      int            n,
                      int            k,
                      double         alpha,
                      double const * A,
                      int const *    JA,
                      int const *    IA,
                      int            nnz_A,
                      double const * B,
                      int const *    JB,
                      int const *    IB,
                      int            nnz_B,
                      double         beta,
                      char *&        C_CSR) const {
#if USE_SP_MKL
    char transa = 'N';
    CSR_Matrix C_in(C_CSR);

    int * new_ic = (int*)alloc(sizeof(int)*(m+1));
 
    int sort = 1; 
    int req = 1;
    int info;
    CTF_BLAS::MKL_DCSRMULTCSR(&transa, &req, &sort, &m, &k, &n, A, JA, IA, B, JB, IB, NULL, NULL, new_ic, &req, &info);

    CSR_Matrix C_add(new_ic[m]-1, m, n, C_in.val_size());
    memcpy(C_add.IA(), new_ic, (m+1)*sizeof(int));
    cdealloc(new_ic);
    req = 2;
    CTF_BLAS::MKL_DCSRMULTCSR(&transa, &req, &sort, &m, &k, &n, A, JA, IA, B, JB, IB, (double*)C_add.vals(), C_add.JA(), C_add.IA(), &req, &info);

    if (beta == 0.0){
      cdealloc(C_CSR);
      C_CSR = C_add.all_data;
    } else {
      if (beta != 1.0){
        this->scal(C_in.nnz(), (char const *)&beta, C_in.vals(), 1);
      }
      if (alpha != 1.0){
        this->scal(C_add.nnz(), (char const *)&alpha, C_add.vals(), 1);
      }
      char * C_ret = csr_add(C_CSR, C_add.all_data);
      cdealloc(C_CSR);
      cdealloc(C_add.all_data);
      C_CSR = C_ret;
    }
#else
    this->gen_csrmultcsr(m,n,k,alpha,A,JA,IA,nnz_A,B,JB,IB,nnz_B,beta,C_CSR);
#endif
  }


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


}
