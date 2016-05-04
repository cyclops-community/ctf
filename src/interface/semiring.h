#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "functions.h"

namespace CTF_int {


  template <typename dtype>
  dtype default_mul(dtype a, dtype b){
    return a*b;
  }

  template <typename dtype>
  void default_axpy(int           n,
                    dtype         alpha,
                    dtype const * X,
                    int           incX,
                    dtype *       Y,
                    int           incY){
    for (int i=0; i<n; i++){
      Y[incY*i] += alpha*X[incX*i];
    }
  }

  template <>
  void default_axpy<float>
                   (int           n,
                    float         alpha,
                    float const * X,
                    int           incX,
                    float *       Y,
                    int           incY);

  template <>
  void default_axpy<double>
                   (int            n,
                    double         alpha,
                    double const * X,
                    int            incX,
                    double *       Y,
                    int            incY);

  template <>
  void default_axpy< std::complex<float> >
                   (int                         n,
                    std::complex<float>         alpha,
                    std::complex<float> const * X,
                    int                         incX,
                    std::complex<float> *       Y,
                    int                         incY);

  template <>
  void default_axpy< std::complex<double> >
                   (int                          n,
                    std::complex<double>         alpha,
                    std::complex<double> const * X,
                    int                          incX,
                    std::complex<double> *       Y,
                    int                          incY);

  template <typename dtype>
  void default_scal(int           n,
                    dtype         alpha,
                    dtype *       X,
                    int           incX){
    for (int i=0; i<n; i++){
      X[incX*i] *= alpha;
    }
  }

  template <>
  void default_scal<float>(int n, float alpha, float * X, int incX);

  template <>
  void default_scal<double>(int n, double alpha, double * X, int incX);

  template <>
  void default_scal< std::complex<float> >
      (int n, std::complex<float> alpha, std::complex<float> * X, int incX);

  template <>
  void default_scal< std::complex<double> >
      (int n, std::complex<double> alpha, std::complex<double> * X, int incX);

  template<typename dtype>
  void default_gemm(char          tA,
                    char          tB,
                    int           m,
                    int           n,
                    int           k,
                    dtype         alpha,
                    dtype const * A,
                    dtype const * B,
                    dtype         beta,
                    dtype *       C){
    int i,j,l;
    int istride_A, lstride_A, jstride_B, lstride_B;
    TAU_FSTART(default_gemm);
    if (tA == 'N' || tA == 'n'){
      istride_A=1; 
      lstride_A=m; 
    } else {
      istride_A=k; 
      lstride_A=1; 
    }
    if (tB == 'N' || tB == 'n'){
      jstride_B=k; 
      lstride_B=1; 
    } else {
      jstride_B=1; 
      lstride_B=n; 
    }
    for (j=0; j<n; j++){
      for (i=0; i<m; i++){
        C[j*m+i] *= beta;
        for (l=0; l<k; l++){
          C[j*m+i] += A[istride_A*i+lstride_A*l]*B[lstride_B*l+jstride_B*j];
        }
      }
    }
    TAU_FSTOP(default_gemm);
  }

  template<>
  inline void default_gemm<float>
            (char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             float          alpha,
             float  const * A,
             float  const * B,
             float          beta,
             float  *       C){
    CTF_int::sgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  inline void default_gemm<double>
            (char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             double         alpha,
             double const * A,
             double const * B,
             double         beta,
             double *       C){
    CTF_int::cidgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  inline void default_gemm< std::complex<float> >
            (char                        tA,
             char                        tB,
             int                         m,
             int                         n,
             int                         k,
             std::complex<float>         alpha,
             std::complex<float> const * A,
             std::complex<float> const * B,
             std::complex<float>         beta,
             std::complex<float> *       C){
    CTF_int::cgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<>
  inline void default_gemm< std::complex<double> >
            (char                         tA,
             char                         tB,
             int                          m,
             int                          n,
             int                          k,
             std::complex<double>         alpha,
             std::complex<double> const * A,
             std::complex<double> const * B,
             std::complex<double>         beta,
             std::complex<double> *       C){
    CTF_int::zgemm(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template <typename dtype>
  void default_coomm
                 (int           m,
                  int           n,
                  int           k,
                  dtype         alpha,
                  dtype const * A,
                  int const *   rows_A,
                  int const *   cols_A,
                  int           nnz_A,
                  dtype const * B,
                  dtype         beta,
                  dtype *       C){
    TAU_FSTART(default_coomm);
    for (int j=0; j<n; j++){
      for (int i=0; i<m; i++){
        C[j*m+i] *= beta;
      }
    }
    for (int i=0; i<nnz_A; i++){
      int row_A = rows_A[i]-1;
      int col_A = cols_A[i]-1;
      for (int col_C=0; col_C<n; col_C++){
         C[col_C*m+row_A] += alpha*A[i]*B[col_C*k+col_A];
      }
    }
    TAU_FSTOP(default_coomm);
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
           float *       C);

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
           double *       C);

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
           std::complex<float> *       C);

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
      std::complex<double> *       C);



/*
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
           float *       C);

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
           double *       C);


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
           std::complex<float> *       C);



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
           std::complex<double> *       C);

*/

  template <typename type>
  bool get_def_has_csrmm(){ return true; }
  template <>
  bool get_def_has_csrmm<float>();
  template <>
  bool get_def_has_csrmm<double>();
  template <>
  bool get_def_has_csrmm< std::complex<float> >();
  template <>
  bool get_def_has_csrmm< std::complex<double> >();

  template <typename dtype>  
  void seq_coo_to_csr(int64_t nz, int nrow, dtype * csr_vs, int * csr_cs, int * csr_rs, dtype const * coo_vs, int const * coo_rs, int const * coo_cs){
    TAU_FSTART(seq_coo_to_csr);
    csr_rs[0] = 1;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i=1; i<nrow+1; i++){
      csr_rs[i] = 0;
    }
    for (int64_t i=0; i<nz; i++){
      csr_rs[coo_rs[i]]++;
    }
    for (int i=0; i<nrow; i++){
      csr_rs[i+1] += csr_rs[i];
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      csr_cs[i] = i;
    }

    class comp_ref {
      public:
        int const * a;
        comp_ref(int const * a_){ a = a_; }
        bool operator()(int u, int v){ 
          return a[u] < a[v];
        }
    };

    comp_ref crc(coo_cs);
    TAU_FSTART(sort_coo_to_csr);
    std::sort(csr_cs, csr_cs+nz, crc);
    TAU_FSTOP(sort_coo_to_csr);
    comp_ref crr(coo_rs);
    TAU_FSTART(stsort_coo_to_csr);
    std::stable_sort(csr_cs, csr_cs+nz, crr);
    TAU_FSTOP(stsort_coo_to_csr);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      csr_vs[i] = coo_vs[csr_cs[i]];
      csr_cs[i] = coo_cs[csr_cs[i]];
    }
    TAU_FSTOP(seq_coo_to_csr);
  }


  template <typename dtype>  
  void def_coo_to_csr(int64_t nz, int nrow, dtype * csr_vs, int * csr_cs, int * csr_rs, dtype const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<dtype>(nz, nrow, csr_vs, csr_cs, csr_rs, coo_vs, coo_rs, coo_cs);
  }
 
  template <>  
  void def_coo_to_csr<float>(int64_t nz, int nrow, float * csr_vs, int * csr_cs, int * csr_rs, float const * coo_vs, int const * coo_rs, int const * coo_cs);
  template <>  
  void def_coo_to_csr<double>(int64_t nz, int nrow, double * csr_vs, int * csr_cs, int * csr_rs, double const * coo_vs, int const * coo_rs, int const * coo_cs);
  template <>  
  void def_coo_to_csr<std::complex<float>>(int64_t nz, int nrow, std::complex<float> * csr_vs, int * csr_cs, int * csr_rs, std::complex<float> const * coo_vs, int const * coo_rs, int const * coo_cs);
  template <>  
  void def_coo_to_csr<std::complex<double>>(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_cs, int * csr_rs, std::complex<double> const * coo_vs, int const * coo_rs, int const * coo_cs);
}

namespace CTF {
  /**
   * \addtogroup algstrct 
   * @{
   */

  /**
   * \brief Semiring is a Monoid with an addition multiplicaton function
   *   addition must have an identity and be associative, does not need to be commutative
   *   multiplications must have an identity as well as be distributive and associative
   *   special case (parent) of a Ring (which also has an additive inverse)
   */
  template <typename dtype=double, bool is_ord=CTF_int::get_default_is_ord<dtype>()> 
  class Semiring : public Monoid<dtype, is_ord> {
    public:
      dtype tmulid;
      void (*fscal)(int,dtype,dtype*,int);
      void (*faxpy)(int,dtype,dtype const*,int,dtype*,int);
      dtype (*fmul)(dtype a, dtype b);
      void (*fgemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);
      void (*fcoomm)(int,int,int,dtype,dtype const*,int const*,int const*,int,dtype const*,dtype,dtype*);
    
      Semiring(Semiring const & other) : Monoid<dtype, is_ord>(other) { 
        this->tmulid = other.tmulid;
        this->fscal  = other.fscal;
        this->faxpy  = other.faxpy;
        this->fmul   = other.fmul;
        this->fgemm  = other.fgemm;
        this->fcoomm = other.fcoomm;
      }

      virtual CTF_int::algstrct * clone() const {
        return new Semiring<dtype, is_ord>(*this);
      }

      /**
       * \brief constructor for algstrct equipped with * and +
       * \param[in] addid_ additive identity
       * \param[in] fadd_ binary addition function
       * \param[in] addmop_ MPI_Op operation for addition
       * \param[in] mulid_ multiplicative identity
       * \param[in] fmul_ binary multiplication function
       * \param[in] gemm_ block matrix multiplication function
       * \param[in] axpy_ vector sum function
       * \param[in] scal_ vector scale function
       */
      Semiring(dtype        addid_,
               dtype (*fadd_)(dtype a, dtype b),
               MPI_Op       addmop_,
               dtype        mulid_,
               dtype (*fmul_)(dtype a, dtype b),
               void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=NULL,
               void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=NULL,
               void (*scal_)(int,dtype,dtype*,int)=NULL,
               void (*coomm_)(int,int,int,dtype,dtype const*,int const*,int const*,int,dtype const*,dtype,dtype*)=NULL)
                : Monoid<dtype, is_ord>(addid_, fadd_, addmop_) , tmulid(mulid_) {
        fmul   = fmul_;
        fgemm  = gemm_;
        faxpy  = axpy_;
        fscal  = scal_;
        fcoomm = coomm_;
      }

      /**
       * \brief constructor for algstrct equipped with + only
       */
      Semiring() : Monoid<dtype,is_ord>() {
        tmulid = dtype(1);
        fmul   = &CTF_int::default_mul<dtype>;
        fgemm  = &CTF_int::default_gemm<dtype>;
        faxpy  = &CTF_int::default_axpy<dtype>;
        fscal  = &CTF_int::default_scal<dtype>;
        fcoomm = &CTF_int::default_coomm<dtype>;
        this->has_csrmm = CTF_int::get_def_has_csrmm<dtype>();
      }

      void mul(char const * a, 
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = fmul(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void safemul(char const * a, 
                   char const * b,
                   char *&      c) const {
        if (a == NULL && b == NULL){
          if (c!=NULL) free(c);
          c = NULL;
        } else if (a == NULL) {
          if (c==NULL) c = (char*)malloc(this->el_size);
          memcpy(c,b,this->el_size);
        } else if (b == NULL) {
          if (c==NULL) c = (char*)malloc(this->el_size);
          memcpy(c,b,this->el_size);
        } else {
          if (c==NULL) c = (char*)malloc(this->el_size);
          ((dtype*)c)[0] = fmul(((dtype*)a)[0],((dtype*)b)[0]);
        }
      }
 
      char const * mulid() const {
        return (char const *)&tmulid;
      }


      /** \brief X["i"]=alpha*X["i"]; */
      void scal(int          n,
                char const * alpha,
                char       * X,
                int          incX)  const {
        if (fscal != NULL) fscal(n, ((dtype const *)alpha)[0], (dtype *)X, incX);
        else {
          dtype const a = ((dtype*)alpha)[0];
          dtype * dX    = (dtype*) X;
          for (int64_t i=0; i<n; i++){
            dX[i] = fmul(a,dX[i]);
          }
        }
      }

      /** \brief Y["i"]+=alpha*X["i"]; */
      void axpy(int          n,
                char const * alpha,
                char const * X,
                int          incX,
                char       * Y,
                int          incY)  const {
        if (faxpy != NULL) faxpy(n, ((dtype const *)alpha)[0], (dtype const *)X, incX, (dtype *)Y, incY);
        else {
          assert(incX==1);
          assert(incY==1);
          dtype a           = ((dtype*)alpha)[0];
          dtype const * dX = (dtype*) X;
          dtype * dY       = (dtype*) Y;
          for (int64_t i=0; i<n; i++){
            dY[i] = this->fadd(fmul(a,dX[i]), dY[i]);
          }
        }
      }

      /** \brief beta*C["ij"]=alpha*A^tA["ik"]*B^tB["kj"]; */
      void gemm(char         tA,
                char         tB,
                int          m,
                int          n,
                int          k,
                char const * alpha,
                char const * A,
                char const * B,
                char const * beta,
                char *       C)  const {
        if (fgemm != NULL) fgemm(tA, tB, m, n, k, ((dtype const *)alpha)[0], (dtype const *)A, (dtype const *)B, ((dtype const *)beta)[0], (dtype *)C);
        else {
          TAU_FSTART(sring_gemm);
          dtype const * dA = (dtype const *) A;
          dtype const * dB = (dtype const *) B;
          dtype * dC       = (dtype*) C;
          if (!this->isequal(beta, this->mulid())){
            scal(m*n, beta, C, 1);
          }  
          int lda_Cj, lda_Ci, lda_Al, lda_Ai, lda_Bj, lda_Bl;
          lda_Cj = m;
          lda_Ci = 1;
          if (tA == 'N'){
            lda_Al = m;
            lda_Ai = 1;
          } else {
            assert(tA == 'T');
            lda_Al = 1;
            lda_Ai = k;
          } 
          if (tB == 'N'){
            lda_Bj = k;
            lda_Bl = 1;
          } else {
            assert(tB == 'T');
            lda_Bj = 1;
            lda_Bl = n;
          } 
          if (!this->isequal(alpha, this->mulid())){
            dtype a          = ((dtype*)alpha)[0];
            for (int64_t j=0; j<n; j++){
              for (int64_t i=0; i<m; i++){
                for (int64_t l=0; l<k; l++){
                  //dC[j*m+i] = this->fadd(fmul(a,fmul(dA[l*m+i],dB[j*k+l])), dC[j*m+i]);
                  dC[j*lda_Cj+i*lda_Ci] = this->fadd(fmul(a,fmul(dA[l*lda_Al+i*lda_Ai],dB[j*lda_Bj+l*lda_Bl])), dC[j*lda_Cj+i*lda_Ci]);
                }
              }
            }
          } else {
            for (int64_t j=0; j<n; j++){
              for (int64_t i=0; i<m; i++){
                for (int64_t l=0; l<k; l++){
                  //dC[j*m+i] = this->fadd(fmul(a,fmul(dA[l*m+i],dB[j*k+l])), dC[j*m+i]);
                  dC[j*lda_Cj+i*lda_Ci] = this->fadd(fmul(dA[l*lda_Al+i*lda_Ai],dB[j*lda_Bj+l*lda_Bl]), dC[j*lda_Cj+i*lda_Ci]);
                }
              }
            }
          }
          TAU_FSTOP(sring_gemm);
        } 
      }

      void offload_gemm(char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C) const {
        printf("CTF ERROR: offload gemm not present for this semiring\n");
        ASSERT(0);
      }

      bool is_offloadable() const {
        return false;
      }


      void coomm(int m, int n, int k, char const * alpha, char const * A, int const * rows_A, int const * cols_A, int64_t nnz_A, char const * B, char const * beta, char * C, CTF_int::bivar_function const * func) const {
        if (func == NULL && alpha != NULL && fcoomm != NULL){
          fcoomm(m, n, k, ((dtype const *)alpha)[0], (dtype const *)A, rows_A, cols_A, nnz_A, (dtype const *)B, ((dtype const *)beta)[0], (dtype *)C);
          return;
        }
        if (func == NULL && alpha != NULL && this->isequal(beta,mulid())){
          TAU_FSTART(func_coomm);
          dtype const * dA = (dtype const*)A;
          dtype const * dB = (dtype const*)B;
          dtype * dC = (dtype*)C;
          dtype a = ((dtype*)alpha)[0];
          if (!this->isequal(beta, this->mulid())){
            scal(m*n, beta, C, 1);
          }  
          for (int64_t i=0; i<nnz_A; i++){
            int row_A = rows_A[i]-1;
            int col_A = cols_A[i]-1;
            for (int col_C=0; col_C<n; col_C++){
              dC[col_C*m+row_A] = this->fadd(fmul(a,fmul(dA[i],dB[col_C*k+col_A])), dC[col_C*m+row_A]);
            }
          }
          TAU_FSTOP(func_coomm);
        } else { assert(0); }
      }

      void coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_cs, int * csr_rs, char const * coo_vs, int const * coo_rs, int const * coo_cs) const {
        CTF_int::def_coo_to_csr(nz, nrow, (dtype *)csr_vs, csr_cs, csr_rs, (dtype const *) coo_vs, coo_rs, coo_cs);
      }

      void default_csrmm
                     (int           m,
                      int           n,
                      int           k,
                      dtype         alpha,
                      dtype const * A,
                      int const *   IA,
                      int const *   JA,
                      int           nnz_A,
                      dtype const * B,
                      dtype         beta,
                      dtype *       C) const {
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int row_A=0; row_A<m; row_A++){
#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (int col_B=0; col_B<n; col_B++){
            C[col_B*m+row_A] = this->fmul(beta,C[col_B*m+row_A]);
            if (IA[row_A] < IA[row_A+1]){
              int i_A1 = IA[row_A]-1;
              int col_A1 = JA[i_A1]-1;
              dtype tmp = this->fmul(A[i_A1],B[col_B*k+col_A1]);
              for (int i_A=IA[row_A]; i_A<IA[row_A+1]-1; i_A++){
                int col_A = JA[i_A]-1;
                tmp = this->fadd(tmp, this->fmul(A[i_A],B[col_B*k+col_A]));
              }
              C[col_B*m+row_A] = this->fadd(C[col_B*m+row_A], this->fmul(alpha,tmp));
            }
          }
        }
      }


      /** \brief sparse version of gemm using CSR format for A */
      void csrmm(int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  rows_A,
                 int const *  cols_A,
                 int64_t      nnz_A,
                 char const * B,
                 char const * beta,
                 char *       C,
                 CTF_int::bivar_function const * func) const {
        assert(this->has_csrmm);
        assert(func == NULL);
        this->default_csrmm(m,n,k,((dtype*)alpha)[0],(dtype*)A,rows_A,cols_A,nnz_A,(dtype*)B,((dtype*)beta)[0],(dtype*)C);
      }
  };
  /**
   * @}
   */
}
#include "ring.h"
#endif
