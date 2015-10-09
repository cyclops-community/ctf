#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/offload.h"


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
  bool get_def_has_csrmm<float>(){ return false; }
  template <>
  bool get_def_has_csrmm<double>(){ return false; }
  template <>
  bool get_def_has_csrmm< std::complex<float> >(){ return false; }
  template <>
  bool get_def_has_csrmm< std::complex<double> >(){ return false; }
#endif

#if USE_SP_MKL
  template <>  
  void def_coo_to_csr<float>(int64_t nz, int nrow, float * csr_vs, int * csr_cs, int * csr_rs, float const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_SCSRCOO(job, &nrow, csr_vs, csr_cs, csr_rs, &inz, (float*)coo_vs, coo_rs, coo_cs, &info);

  }
  template <>  
  void def_coo_to_csr<double>(int64_t nz, int nrow, double * csr_vs, int * csr_cs, int * csr_rs, double const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    TAU_FSTART(MKL_DCSRCOO);
    CTF_BLAS::MKL_DCSRCOO(job, &nrow, csr_vs, csr_cs, csr_rs, &inz, (double*)coo_vs, coo_rs, coo_cs, &info);
    TAU_FSTOP(MKL_DCSRCOO);
  }
  template <>  
  void def_coo_to_csr<std::complex<float>>(int64_t nz, int nrow, std::complex<float> * csr_vs, int * csr_cs, int * csr_rs, std::complex<float> const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_CCSRCOO(job, &nrow, csr_vs, csr_cs, csr_rs, &inz, (std::complex<float>*)coo_vs, coo_rs, coo_cs, &info);
  }
  template <>  
  void def_coo_to_csr<std::complex<double>>(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_cs, int * csr_rs, std::complex<double> const * coo_vs, int const * coo_rs, int const * coo_cs){
    int inz = nz;
    int job[8]={2,1,1,0,inz,0,0,0};

    int info = 1;
    CTF_BLAS::MKL_ZCSRCOO(job, &nrow, csr_vs, csr_cs, csr_rs, &inz, (std::complex<double>*)coo_vs, coo_rs, coo_cs, &info);
  }
#else
  template <> 
  void def_coo_to_csr<float>(int64_t nz, int nrow, float * csr_vs, int * csr_cs, int * csr_rs, float const * coo_vs, int const * coo_rs, int const * coo_cs){
    printf("CTF ERROR: MKL required for COO to CSR conversion, should not be here\n");
    ASSERT(0);
  }
  template <>  
  void def_coo_to_csr<double>(int64_t nz, int nrow, double * csr_vs, int * csr_cs, int * csr_rs, double const * coo_vs, int const * coo_rs, int const * coo_cs){
    printf("CTF ERROR: MKL required for COO to CSR conversion, should not be here\n");
    ASSERT(0);
  }
  template <>  
  void def_coo_to_csr<std::complex<float>>(int64_t nz, int nrow, std::complex<float> * csr_vs, int * csr_cs, int * csr_rs, std::complex<float> const * coo_vs, int const * coo_rs, int const * coo_cs){
    printf("CTF ERROR: MKL required for COO to CSR conversion, should not be here\n");
    ASSERT(0);
  }
  template <>  
  void def_coo_to_csr<std::complex<double>>(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_cs, int * csr_rs, std::complex<double> const * coo_vs, int const * coo_rs, int const * coo_cs){
    printf("CTF ERROR: MKL required for COO to CSR conversion, should not be here\n");
    ASSERT(0);
  }

#endif


#if USE_SP_MKL
  template <>
  void default_csrmm< float >
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
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_SCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, cols_A, rows_A, rows_A+1, B, &k, &beta, C, &m);

  }

  template <>
  void default_csrmm< double >
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
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    TAU_FSTART(MKL_DCSRMM);
    CTF_BLAS::MKL_DCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, cols_A, rows_A, rows_A+1, B, &k, &beta, C, &m);
    TAU_FSTOP(MKL_DCSRMM);
  }


  template <>
  void default_csrmm< std::complex<float> >
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
    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_CCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, cols_A, rows_A, rows_A+1, B, &k, &beta, C, &m);

  }

  template <>
  void default_csrmm< std::complex<double> >
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

    char transa = 'N';
    char matdescra[6] = {'G',0,0,'F',0,0};
    
    CTF_BLAS::MKL_ZCSRMM(&transa, &m, &n, &k, &alpha, matdescra, A, cols_A, rows_A, rows_A+1, B, &k, &beta, C, &m);

  }

#else
  template <>
  void default_csrmm< float >
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
    printf("CTF ERROR: MKL required for CSRMM, should not be here\n");
    ASSERT(0);
  }

  template <>
  void default_csrmm< double >
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
    printf("CTF ERROR: MKL required for CSRMM, should not be here\n");
    ASSERT(0);
  }


  template <>
  void default_csrmm< std::complex<float> >
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

    printf("CTF ERROR: MKL required for CSRMM, should not be here\n");
    ASSERT(0);
  }



  template <>
  void default_csrmm< std::complex<double> >
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
    printf("CTF ERROR: MKL required for CSRMM, should not be here\n");
    ASSERT(0);
  }
#endif

}
namespace CTF {
/*  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const {
    return fgemm == &CTF_int::default_gemm<float>;
  }*/

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

}
