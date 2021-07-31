#ifndef __SEMIRING_H__
#define __SEMIRING_H__

#include "functions.h"
#include "../sparse_formats/csr.h"
#include "../sparse_formats/ccsr.h"
#include "../redistribution/nosym_transp.h"
#include <iostream>
using namespace std;

namespace CTF_int {




  template <typename dtype>
  dtype default_mul(dtype a, dtype b){
    return a*b;
  }

  template <>
  inline bool default_mul<bool>(bool a, bool b){
    return a&&b;
  }

  template <typename dtype>
  void default_vec_mul(dtype const * a, dtype const * b, dtype * c, int64_t n){
    for (int64_t i=0; i<n; i++){
      c[i] = default_mul<dtype>(a[i],b[i]);      }
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
                   (int,float,float const *,int,float *,int);

  template <>
  void default_axpy<double>
                   (int,double,double const *,int,double *,int);

  template <>
  void default_axpy< std::complex<float> >
                   (int,std::complex<float>,std::complex<float> const *,int,std::complex<float> *,int);

  template <>
  void default_axpy< std::complex<double> >
                   (int,std::complex<double>,std::complex<double> const *,int,std::complex<double> *,int);

  template <typename dtype>
  void default_scal(int           n,
                    dtype         alpha,
                    dtype *       X,
                    int           incX){
    for (int i=0; i<n; i++){
      X[incX*i] = default_mul<dtype>(alpha,X[incX*i]);
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
    //TAU_FSTART(default_gemm);
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
        C[j*m+i] = default_mul<dtype>(C[j*m+i],beta);
        for (l=0; l<k; l++){
          C[j*m+i] += default_mul<dtype>(alpha,default_mul<dtype>(A[istride_A*i+lstride_A*l],B[lstride_B*l+jstride_B*j]));
        }
      }
    }
    //TAU_FSTOP(default_gemm);
  }

  template<typename dtype>
  dtype ** get_grp_ptrs(int64_t          grp_sz,
                        int64_t          ngrp,
                        dtype const *    data){
    dtype ** data_ptrs = (dtype**)malloc(sizeof(dtype*)*ngrp);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i=0; i<ngrp; i++){
      data_ptrs[i] = ((dtype*)data)+i*grp_sz;
    }
    return data_ptrs;
  }

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
            dtype   *      C);

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
            dtype  *       C);

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
    CTF_int::gemm<float>(tA,tB,m,n,k,alpha,A,B,beta,C);
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
    CTF_int::gemm<double>(tA,tB,m,n,k,alpha,A,B,beta,C);
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
    CTF_int::gemm< std::complex<float> >(tA,tB,m,n,k,alpha,A,B,beta,C);
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
    CTF_int::gemm< std::complex<double> >(tA,tB,m,n,k,alpha,A,B,beta,C);
  }

  template<typename dtype>
  void default_gemm_batch
                   (char         taA,
                    char         taB,
                    int          l,
                    int          m,
                    int          n,
                    int          k,
                    dtype        alpha,
                    dtype const* A,
                    dtype const* B,
                    dtype        beta,
                    dtype *      C){
    if (m == 1 && n == 1 && k == 1){
      for (int i=0; i<l; i++){
        C[i] = C[i]*beta + alpha*A[i]*B[i];
      }
    } else {
      for (int i=0; i<l; i++){
        default_gemm<dtype>(taA, taB, m, n, k, alpha, A+i*m*k, B+i*k*n, beta, C+i*m*n);
      }
    }
  }
  
  template<>
  inline void default_gemm_batch<float> 
            (char          taA,
             char          taB,
             int           l,
             int           m,
             int           n,
             int           k,
             float         alpha,
             float  const* A,
             float  const* B,
             float         beta,
             float  *      C){
    CTF_int::gemm_batch<float>(taA, taB, l, m, n, k, alpha, A, B, beta, C);        
  } 

  template<>
  inline void default_gemm_batch<double> 
            (char          taA,
             char          taB,
             int           l,
             int           m,
             int           n,
             int           k,
             double        alpha,
             double const* A,
             double const* B,
             double        beta,
             double *      C){
    CTF_int::gemm_batch<double>(taA, taB, l, m, n, k, alpha, A, B, beta, C);        
  } 

  template<>
  inline void default_gemm_batch<std::complex<float>> 
            (char                        taA,
             char                        taB,
             int                         l,
             int                         m,
             int                         n,
             int                         k,
             std::complex<float>         alpha,
             std::complex<float>  const* A,
             std::complex<float>  const* B,
             std::complex<float>         beta,
             std::complex<float>  *       C){
    CTF_int::gemm_batch< std::complex<float> >(taA, taB, l, m, n, k, alpha, A, B, beta, C);        
  } 

  template<>
  inline void default_gemm_batch<std::complex<double>> 
          (char                        taA,
           char                        taB,
           int                         l,
           int                         m,
           int                         n,
           int                         k,
           std::complex<double>        alpha,
           std::complex<double> const* A,
           std::complex<double> const* B,
           std::complex<double>        beta,
           std::complex<double> *      C){
    CTF_int::gemm_batch< std::complex<double> >(taA, taB, l, m, n, k, alpha, A, B, beta, C);        
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
    //TAU_FSTART(default_coomm);
    for (int j=0; j<n; j++){
      for (int i=0; i<m; i++){
        C[j*m+i] = default_mul<dtype>(C[j*m+i],beta);
      }
    }
    for (int i=0; i<nnz_A; i++){
      int row_A = rows_A[i]-1;
      int col_A = cols_A[i]-1;
      for (int col_C=0; col_C<n; col_C++){
         C[col_C*m+row_A] += default_mul<dtype>(alpha,default_mul<dtype>(A[i],B[col_C*k+col_A]));
      }
    }
    //TAU_FSTOP(default_coomm);
  }

  template <>
  void default_coomm< float >
          (int,int,int,float,float const *,int const *,int const *,int,float const *,float,float *);

  template <>
  void default_coomm< double >
          (int,int,int,double,double const *,int const *,int const *,int,double const *,double,double *);

  template <>
  void default_coomm< std::complex<float> >
          (int,int,int,std::complex<float>,std::complex<float> const *,int const *,int const *,int,std::complex<float> const *,std::complex<float>,std::complex<float> *);

  template <>
  void default_coomm< std::complex<double> >
     (int,int,int,std::complex<double>,std::complex<double> const *,int const *,int const *,int,std::complex<double> const *,std::complex<double>,std::complex<double> *);


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
      bool is_def;
      dtype tmulid;
      void (*fscal)(int,dtype,dtype*,int);
      void (*faxpy)(int,dtype,dtype const*,int,dtype*,int);
      dtype (*fmul)(dtype a, dtype b);
      void (*fvmul)(dtype const * a, dtype const * b, dtype * c, int64_t n);
      void (*fgemm)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);
      void (*fcoomm)(int,int,int,dtype,dtype const*,int const*,int const*,int,dtype const*,dtype,dtype*);
      void (*fgemm_batch)(char,char,int,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*);
      //void (*fcsrmm)(int,int,int,dtype,dtype const*,int const*,int const*,dtype const*,dtype,dtype*);
       //csrmultd_ kernel for multiplying two sparse matrices into a dense output 
      //void (*fcsrmultd)(int,int,int,dtype const*,int const*,int const*,dtype const*,int const*, int const*,dtype*,int);
    
      Semiring(Semiring const & other) : Monoid<dtype, is_ord>(other) { 
        this->tmulid      = other.tmulid;
        this->fscal       = other.fscal;
        this->faxpy       = other.faxpy;
        this->fmul        = other.fmul;
        this->fvmul       = other.fvmul;
        this->fgemm       = other.fgemm;
        this->fcoomm      = other.fcoomm;
        this->is_def      = other.is_def;
        this->fgemm_batch = other.fgemm_batch;
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
       * \param[in] fvmul_ binary vector multiplication function
       * \param[in] gemm_ block matrix multiplication function
       * \param[in] axpy_ vector sum function
       * \param[in] scal_ vector scale function
       * \param[in] coomm_ kernel for multiplying sparse matrix in coordinate format with dense matrix
       */
      Semiring(dtype        addid_,
               dtype (*fadd_)(dtype a, dtype b),
               MPI_Op       addmop_,
               dtype        mulid_,
               dtype (*fmul_)(dtype a, dtype b),
               void (*gemm_)(char,char,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=NULL,
               void (*axpy_)(int,dtype,dtype const*,int,dtype*,int)=NULL,
               void (*scal_)(int,dtype,dtype*,int)=NULL,
               void (*coomm_)(int,int,int,dtype,dtype const*,int const*,int const*,int,dtype const*,dtype,dtype*)=NULL,
               void (*fgemm_batch_)(char,char,int,int,int,int,dtype,dtype const*,dtype const*,dtype,dtype*)=NULL,
               void (*fvmul_)(dtype const * a, dtype const * b, dtype * c, int64_t n)=NULL)
                : Monoid<dtype, is_ord>(addid_, fadd_, addmop_) {
        fmul        = fmul_;
        fvmul       = fvmul_;
        fgemm       = gemm_;
        faxpy       = axpy_;
        fscal       = scal_;
        fcoomm      = coomm_;
        fgemm_batch = fgemm_batch_;
        tmulid      = mulid_;
        // if provided a coordinate MM kernel, don't use CSR
        this->has_coo_ker = (coomm_ != NULL);
        is_def = false;
      }

      /**
       * \brief constructor for algstrct equipped with + only
       */
      Semiring() : Monoid<dtype,is_ord>()  {
        tmulid      = dtype(1);
        fmul        = &CTF_int::default_mul<dtype>;
        fvmul       = &CTF_int::default_vec_mul<dtype>;
        fgemm       = &CTF_int::default_gemm<dtype>;
        faxpy       = &CTF_int::default_axpy<dtype>;
        fscal       = &CTF_int::default_scal<dtype>;
        fcoomm      = &CTF_int::default_coomm<dtype>;
        fgemm_batch = &CTF_int::default_gemm_batch<dtype>;
        is_def = true;
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
          if (c!=NULL) CTF_int::cdealloc(c);
          c = NULL;
        } else if (a == NULL) {
          if (c==NULL) c = (char*)CTF_int::alloc(this->el_size);
          memcpy(c,b,this->el_size);
        } else if (b == NULL) {
          if (c==NULL) c = (char*)CTF_int::alloc(this->el_size);
          memcpy(c,b,this->el_size);
        } else {
          if (c==NULL) c = (char*)CTF_int::alloc(this->el_size);
          ((dtype*)c)[0] = fmul(((dtype*)a)[0],((dtype*)b)[0]);
        }
      }
 
      char const * mulid() const {
        return (char const *)&tmulid;
      }

      bool has_mul() const { return true; }

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
        if (fgemm != NULL) {
          fgemm(tA, tB, m, n, k, ((dtype const *)alpha)[0], (dtype const *)A, (dtype const *)B, ((dtype const *)beta)[0], (dtype *)C);
        } else {
          //TAU_FSTART(sring_gemm);
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
          //TAU_FSTOP(sring_gemm);
        } 
      }

      void gemm_batch(char         tA,
                      char         tB,
                      int          l,
                      int          m,
                      int          n,
                      int          k,
                      char const * alpha,
                      char const * A,
                      char const * B,
                      char const * beta,
                      char *       C) const {
        if (fgemm_batch != NULL) {
          fgemm_batch(tA, tB, l, m, n, k, ((dtype const *)alpha)[0], ((dtype const *)A), ((dtype const *)B), ((dtype const *)beta)[0], ((dtype *)C));
        } else {
          for (int i=0; i<l; i++){
            gemm(tA, tB, m, n, k, alpha, A+m*k*i*sizeof(dtype), B+k*n*i*sizeof(dtype), beta, C+m*n*i*sizeof(dtype));
          }
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
        assert(0);
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
          //TAU_FSTART(func_coomm);
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
          //TAU_FSTOP(func_coomm);
        } else { assert(0); }
      }


      void gen_csrmm
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

      void default_csrmm
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
                      dtype *       C) const {
        gen_csrmm(m,n,k,alpha,A,JA,IA,nnz_A,B,beta,C);
      }
//      void (*fcsrmultd)(int,int,int,dtype const*,int const*,int const*,dtype const*,int const*, int const*,dtype*,int);

      /** \brief sparse version of gemm using CSR format for A */
      void csrmm(int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 char const * beta,
                 char *       C,
                 CTF_int::bivar_function const * func) const {
        assert(!this->has_coo_ker);
        assert(func == NULL);
        if (is_def)
          this->default_csrmm(m,n,k,((dtype*)alpha)[0],(dtype*)A,JA,IA,nnz_A,(dtype*)B,((dtype*)beta)[0],(dtype*)C);
        else
          this->gen_csrmm(m,n,k,((dtype*)alpha)[0],(dtype*)A,JA,IA,nnz_A,(dtype*)B,((dtype*)beta)[0],(dtype*)C);
      }

      bool is_last_col_zero(int64_t m, int64_t n, dtype const * M) const {
        for (int64_t i=0; i<m; i++){
          if (!this->isequal((char*)(M+(m*(n-1)+i)), (char*)&this->taddid)) return false;
        }
        return true;
      }

      void gen_ccsrmm
                     (int64_t         m,
                      int64_t         n0,
                      int64_t         k,
                      int64_t         nnz_row,
                      dtype           alpha,
                      dtype const *   A,
                      int const *     JA,
                      int const *     IA,
                      int64_t const * row_enc,
                      int64_t         nnz_A,
                      dtype const *   B,
                      dtype           beta,
                      char *&         C_CCSR) const {
        CTF_int::CCSR_Matrix M;
        int64_t n = n0;
        if (this->is_last_col_zero(k, n, B)){
          n = n0-1;
        }
        if (n == 0){
          M = CTF_int::CCSR_Matrix(0, 0, m, 1, this);
          if (C_CCSR != NULL && !this->isequal((char const *)&beta, this->addid())){
            CTF_int::CCSR_Matrix C(C_CCSR);
            if (!this->isequal((char const *)&beta, this->mulid()))
              this->scal(C.nnz(), (char*)&beta, C.all_data, 1);
            C_CCSR = CTF_int::CCSR_Matrix::ccsr_add(C.all_data, M.all_data, this);
            CTF_int::cdealloc(M.all_data);
          } else {
            //CTF_int::cdealloc(C_CCSR);
            C_CCSR = M.all_data;
          }
          return;
        } 
        if (nnz_row == 0){
          M = CTF_int::CCSR_Matrix(nnz_row*n, nnz_row, m, n, this);
        } else {
          int new_order[2] = {1, 0};
          int64_t lens[2] = {(int64_t)nnz_row, (int64_t)n};
          bool use_hptt = CTF_int::hptt_is_applicable(2, new_order, this->el_size);
          //Note: if there is padding last column of dense matrix would be full of zeros and we don't want to generate nonzeros for this colum, as this will cause tricky bugs!
          if (use_hptt){
            char * data = this->alloc(((int64_t)nnz_row)*n);
            this->init_shell(((int64_t)nnz_row)*n, data);
            csrmm(nnz_row,n,k,(char const *)&alpha, (char const *)A, JA, IA, nnz_A, (char const*)B, this->mulid(), data, NULL);
            M = CTF_int::CCSR_Matrix(((int64_t)nnz_row)*n, nnz_row, m, n0, this);
            CTF_int::nosym_transpose_hptt(2, new_order, lens, 1, data, M.vals(), this);
            this->dealloc(data);
          } else {
            M = CTF_int::CCSR_Matrix(((int64_t)nnz_row)*n, nnz_row, m, n0, this);
            csrmm(nnz_row,n,k,(char const *)&alpha, (char const *)A, JA, IA, nnz_A, (char const*)B, this->mulid(), M.vals(), NULL);
            CTF_int::nosym_transpose(2,new_order,lens,M.vals(),1,this);
          }
        }
        memcpy(M.nnz_row_encoding(), row_enc, nnz_row*sizeof(int64_t));
        int * C_IA = M.IA();
        C_IA[0] = 1;
        for (int64_t row_A=1; row_A<nnz_row+1; row_A++){
          C_IA[row_A] = C_IA[row_A-1] + n;
        }
        int * C_JA = M.JA();
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int64_t row_C=0; row_C<nnz_row; row_C++){
#ifdef _OPENMP
          #pragma omp parallel for
#endif
          for (int64_t col_C=0; col_C<n; col_C++){
            C_JA[row_C*n+col_C] = col_C+1;
          }
        }
        if (C_CCSR != NULL && !this->isequal((char const *)&beta, this->addid())){
          CTF_int::CCSR_Matrix C(C_CCSR);
          if (!this->isequal((char const *)&beta, this->mulid()))
            this->scal(C.nnz(), (char*)&beta, C.all_data, 1);
          C_CCSR = CTF_int::CCSR_Matrix::ccsr_add(C.all_data, M.all_data, this);
          CTF_int::cdealloc(M.all_data);
        } else {
          //CTF_int::cdealloc(C_CCSR);
          C_CCSR = M.all_data;
        }
      }

      void default_ccsrmm
                     (int64_t         m,
                      int64_t         n,
                      int64_t         k,
                      int64_t         nnz_row,
                      dtype           alpha,
                      dtype const *   A,
                      int const *     JA,
                      int const *     IA,
                      int64_t const * row_enc,
                      int64_t         nnz_A,
                      dtype const *   B,
                      dtype           beta,
                      char *&         C) const {
        gen_ccsrmm(m,n,k,nnz_row,alpha,A,JA,IA,row_enc,nnz_A,B,beta,C);
      }
//      void (*fccsrmultd)(int,int,int,dtype const*,int const*,int const*,dtype const*,int const*, int const*,dtype*,int);

      /** \brief sparse version of gemm using CSR format for A */
      void ccsrmm(int64_t         m,
                  int64_t         n,
                  int64_t         k,
                  int64_t         nnz_row,
                  char const *    alpha,
                  char const *    A,
                  int const *     JA,
                  int const *     IA,
                  int64_t const * row_enc,
                  int64_t         nnz_A,
                  char const *    B,
                  char const *    beta,
                  char *&         C,
                  CTF_int::bivar_function const * func) const {
        assert(!this->has_coo_ker);
        assert(func == NULL);
        if (is_def)
          this->default_ccsrmm(m,n,k,nnz_row,((dtype*)alpha)[0],(dtype*)A,JA,IA,row_enc,nnz_A,(dtype*)B,((dtype*)beta)[0],C);
        else
          this->gen_ccsrmm(m,n,k,nnz_row,((dtype*)alpha)[0],(dtype*)A,JA,IA,row_enc,nnz_A,(dtype*)B,((dtype*)beta)[0],C);
      }


      void gen_csrmultd
                     (int           m,
                      int           n,
                      int           k,
                      dtype         alpha,
                      dtype const * A,
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype const * B,
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      dtype         beta,
                      dtype *       C) const {
        
        if (!this->isequal((char const*)&beta, this->mulid())){
          this->scal(m*n, (char const *)&beta, (char*)C, 1);
        }
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int row_A=0; row_A<m; row_A++){
          for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
            int row_B = JA[i_A]-1; //=col_A
            for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
              int col_B = JB[i_B]-1;
              if (!this->isequal((char const*)&alpha, this->mulid()))
                this->fadd(C[col_B*m+row_A], this->fmul(alpha,this->fmul(A[i_A],B[i_B])));
              else
                this->fadd(C[col_B*m+row_A], this->fmul(A[i_A],B[i_B]));
            }
          }
        }
      }

      void default_csrmultd
                     (int           m,
                      int           n,
                      int           k,
                      dtype         alpha,
                      dtype const * A,
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype const * B,
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      dtype         beta,
                      dtype *       C) const {
        gen_csrmultd(m,n,k,alpha,A,JA,IA,nnz_A,B,JB,IB,nnz_B,beta,C);
      } 


      void gen_csrmultcsr
                      (int          m, 
                      int           n,
                      int           k, 
                      dtype         alpha,
                      dtype const * A, // A m by k
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype const * B, // B k by n
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      dtype         beta,
                      char *&       C_CSR) const {
        int * IC = (int*)CTF_int::alloc(sizeof(int)*(m+1));
        memset(IC, 0, sizeof(int)*(m+1));
#ifdef _OPENMP
        #pragma omp parallel
        {
#endif
          int * has_col = (int*)CTF_int::alloc(sizeof(int)*(n+1)); //n is the num of col of B
          int nnz = 0;
#ifdef _OPENMP
          #pragma omp for schedule(dynamic) // TO DO test other strategies
#endif         
          for (int i=0; i<m; i++){
            memset(has_col, 0, sizeof(int)*(n+1)); 
            nnz = 0;
            for (int j=0; j<IA[i+1]-IA[i]; j++){
              int row_B = JA[IA[i]+j-1]-1;
              for (int kk=0; kk<IB[row_B+1]-IB[row_B]; kk++){
                int idx_B = IB[row_B]+kk-1;
                if (has_col[JB[idx_B]] == 0){
                  nnz++;
                  has_col[JB[idx_B]] = 1;
                }
              }
              IC[i+1]=nnz;
            }
          }
          CTF_int::cdealloc(has_col);
#ifdef _OPENMP
        } // END PARALLEL 
#endif
        int ic_prev = 1;
        for(int i=0;i < m+1; i++){
          ic_prev += IC[i];
          IC[i] = ic_prev;
        }
        CTF_int::CSR_Matrix C(IC[m]-1, m, n, this);
        dtype * vC = (dtype*)C.vals();
        this->set((char *)vC, this->addid(), IC[m]+1);
        int * JC = C.JA();
        memcpy(C.IA(), IC, sizeof(int)*(m+1));
        CTF_int::cdealloc(IC);
        IC = C.IA();
#ifdef _OPENMP        
        #pragma omp parallel
        {
#endif      
          int ins = 0;
          int *dcol = (int *) CTF_int::alloc(n*sizeof(int));
          dtype *acc_data = (dtype*)CTF_int::alloc(n*sizeof (dtype));
#ifdef _OPENMP
          #pragma omp for
#endif            
          for (int i=0; i<m; i++){
            std::fill(acc_data, acc_data+n, this->taddid); 
            memset(dcol, 0, sizeof(int)*(n));
            ins = 0;
            for (int j=0; j<IA[i+1]-IA[i]; j++){
              int row_b = JA[IA[i]+j-1]-1; // 1-based
              int idx_a = IA[i]+j-1;
              for (int ii = 0; ii < IB[row_b+1]-IB[row_b]; ii++){
                int col_b = IB[row_b]+ii-1;
                int col_c = JB[col_b]-1; // 1-based
                dtype val = fmul(A[idx_a], B[col_b]);
                if (dcol[col_c] == 0){
                  dcol[col_c] = JB[col_b];
                }
                //acc_data[col_c] += val;
                acc_data[col_c]= this->fadd(acc_data[col_c], val);
              }
            }
            for(int jj = 0; jj < n; jj++){
              if (dcol[jj] != 0){
                JC[IC[i]+ins-1] = dcol[jj];
                vC[IC[i]+ins-1] = acc_data[jj];
                ++ins;
              }
            }
          }
          CTF_int::cdealloc(dcol);
          CTF_int::cdealloc(acc_data);
#ifdef _OPENMP
        } //PRAGMA END
#endif  
        CTF_int::CSR_Matrix C_in(C_CSR);
        if (!this->isequal((char const *)&alpha, this->mulid())){
          this->scal(C.nnz(), (char const *)&alpha, C.vals(), 1);
        }
        if (C_CSR == NULL || C_in.nnz() == 0 || this->isequal((char const *)&beta, this->addid())){
          C_CSR = C.all_data;
        } else {
          if (!this->isequal((char const *)&beta, this->mulid())){
            this->scal(C_in.nnz(), (char const *)&beta, C_in.vals(), 1);
          }
          char * ans = this->csr_add(C_CSR, C.all_data, false);
          CTF_int::cdealloc(C.all_data);
          C_CSR = ans;
        }
      }


     /* void gen_csrmultcsr_old
                     (int           m,
                      int           n,
                      int           k,
                      dtype         alpha,
                      dtype const * A,
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype const * B,
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      dtype         beta,
                      char *&       C_CSR) const {
        int * IC = (int*)CTF_int::alloc(sizeof(int)*(m+1));
        int * has_col = (int*)CTF_int::alloc(sizeof(int)*n);
        IC[0] = 1;
        for (int i=0; i<m; i++){
          memset(has_col, 0, sizeof(int)*n);
          IC[i+1] = IC[i];
          CTF_int::CSR_Matrix::compute_has_col(JA, IA, JB, IB, i, has_col);
          for (int j=0; j<n; j++){
            IC[i+1] += has_col[j];
          }
        }
        CTF_int::CSR_Matrix C(IC[m]-1, m, n, sizeof(dtype));
        dtype * vC = (dtype*)C.vals();
        this->set((char *)vC, this->addid(), IC[m]-1);
        int * JC = C.JA();
        memcpy(C.IA(), IC, sizeof(int)*(m+1));
        CTF_int::cdealloc(IC);
        IC = C.IA();
        int64_t * rev_col = (int64_t*)CTF_int::alloc(sizeof(int64_t)*n);
        for (int i=0; i<m; i++){
          memset(has_col, 0, sizeof(int)*n);
          CTF_int::CSR_Matrix::compute_has_col(JA, IA, JB, IB, i, has_col);
          int vs = 0;
          for (int j=0; j<n; j++){
            if (has_col[j]){
              JC[IC[i]+vs-1] = j+1;
              rev_col[j] = IC[i]+vs-1;
              vs++;
            }
          }
          for (int j=0; j<IA[i+1]-IA[i]; j++){
            int row_B = JA[IA[i]+j-1]-1;
            int idx_A = IA[i]+j-1;
            for (int l=0; l<IB[row_B+1]-IB[row_B]; l++){
              int idx_B = IB[row_B]+l-1;
              dtype tmp = fmul(A[idx_A],B[idx_B]);
              vC[(rev_col[JB[idx_B]-1])] = this->fadd(vC[(rev_col[JB[idx_B]-1])], tmp);
            }
          }
        }
        CTF_int::CSR_Matrix C_in(C_CSR);
        if (!this->isequal((char const *)&alpha, this->mulid())){
          this->scal(C.nnz(), (char const *)&alpha, C.vals(), 1);
        }
        if (C_CSR == NULL || C_in.nnz() == 0 || this->isequal((char const *)&beta, this->addid())){
          C_CSR = C.all_data;
        } else {
          if (!this->isequal((char const *)&beta, this->mulid())){
            this->scal(C_in.nnz(), (char const *)&beta, C_in.vals(), 1);
          }
          char * ans = this->csr_add(C_CSR, C.all_data);
          CTF_int::cdealloc(C.all_data);
          C_CSR = ans;
        }
        CTF_int::cdealloc(has_col);
        CTF_int::cdealloc(rev_col);
      }*/


      void default_csrmultcsr
                     (int           m,
                      int           n,
                      int           k,
                      dtype         alpha,
                      dtype const * A,
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype const * B,
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      dtype         beta,
                      char *&       C_CSR) const {
        this->gen_csrmultcsr(m,n,k,alpha,A,JA,IA,nnz_A,B,JB,IB,nnz_B,beta,C_CSR);
      }


      void csrmultd
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *       C) const {
        if (is_def)
          this->default_csrmultd(m,n,k,((dtype const*)alpha)[0],(dtype const*)A,JA,IA,nnz_A,(dtype const*)B,JB,IB,nnz_B,((dtype const*)beta)[0],(dtype*)C);
        else
          this->gen_csrmultd(m,n,k,((dtype const*)alpha)[0],(dtype const*)A,JA,IA,nnz_A,(dtype const*)B,JB,IB,nnz_B,((dtype const*)beta)[0],(dtype*)C);
      }


      void csrmultcsr
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *&      C_CSR) const {

        if (is_def){
          this->default_csrmultcsr(m,n,k,((dtype const*)alpha)[0],(dtype const*)A,JA,IA,nnz_A,(dtype const*)B,JB,IB,nnz_B,((dtype const*)beta)[0],C_CSR);
        } else {
          this->gen_csrmultcsr(m,n,k,((dtype const*)alpha)[0],(dtype const*)A,JA,IA,nnz_A,(dtype const*)B,JB,IB,nnz_B,((dtype const*)beta)[0],C_CSR);
        }
      }

      void accumulate_local_slice(int order,
                                  int64_t * lens,
                                  int64_t * lens_slice,
                                  int const * sym,
                                  int64_t const * offsets,
                                  int64_t const * ends,
                                  char const * slice_data,
                                  char const * alpha,
                                  char * tensor_data,
                                  char const * beta) const {
        dtype const * sdata = (dtype const*)slice_data;
        dtype * tdata = (dtype*)tensor_data;
        if (order == 1){
          dtype a = ((dtype*)alpha)[0];
          dtype b = ((dtype*)beta)[0];
          for (int64_t i=offsets[0]; i<ends[0]; i++){
            tdata[i] = this->fadd(this->fmul(b,tdata[i]),this->fmul(a,sdata[i-offsets[0]]));
          }
        } else {
          int64_t lda_tensor = 1;
          int64_t lda_slice = 1;
          for (int64_t i=0; i<order-1; i++){
            lda_tensor *= lens[i];
            lda_slice *= lens_slice[i];
          }
          for (int64_t i=offsets[order-1]; i<ends[order-1]; i++){
            this->accumulate_local_slice(order-1, lens, lens_slice, sym, offsets, ends, (char const*)(sdata + (i-offsets[order-1])*lda_slice), alpha, (char *)(tdata + i*lda_tensor), beta);
          }
        }
      }

      void MTTKRP(int                      order,
                  int64_t *                lens,
                  int *                    phys_phase,
                  int64_t                  k,
                  int64_t                  nnz,
                  int                      out_mode,
                  bool                     aux_mode_first,
                  CTF::Pair<dtype> const * tsr_data,
                  dtype const * const *    op_mats,
                  dtype *                  out_mat){
        if (aux_mode_first){
          dtype * buffer = (dtype*)this->alloc(k);
          dtype * out_buffer;
          if (out_mode != 0)
            out_buffer = (dtype*)this->alloc(k);
          int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*(order-1));
          int64_t idx = 0;
          while (idx < nnz){
            int64_t fiber_idx = tsr_data[idx].k/lens[0];
            int64_t fi = fiber_idx;
            for (int i=0; i<order-1; i++){
              inds[i] = (fi % lens[i+1])/phys_phase[i+1];
              fi = fi / lens[i+1];
            }
            int64_t fiber_nnz = 1;
            while (idx+fiber_nnz < nnz && tsr_data[idx+fiber_nnz].k/lens[0] == fiber_idx)
              fiber_nnz++;
            if (out_mode == 0){
              memcpy(buffer, op_mats[1] + inds[0]*k, k*sizeof(dtype));
              for (int i=1; i<order-1; i++){
                fvmul(buffer, op_mats[i+1]+inds[i]*k, buffer, k);
              }
              for (int64_t i=idx; i<idx+fiber_nnz; i++){
                int64_t kk = (tsr_data[i].k%lens[0])/phys_phase[0];
                this->faxpy(k, tsr_data[i].d, buffer, 1, out_mat+kk*k, 1);
              }
            } else {
              int64_t ok = inds[out_mode-1];
              if (out_mode > 1)
                memcpy(buffer, op_mats[1] + inds[0]*k, k*sizeof(dtype));
              else if (order > 2)
                memcpy(buffer, op_mats[2] + inds[1]*k, k*sizeof(dtype));
              else
                std::fill(buffer, buffer+k, this->tmulid);
              for (int i=1+(out_mode==1); i<order-1; i++){
                if (out_mode != i+1)
                  fvmul(buffer, op_mats[i+1] + inds[i]*k, buffer, k);
              }
              std::fill(out_buffer, out_buffer+k, this->taddid);
              for (int64_t i=idx; i<idx+fiber_nnz; i++){
                int64_t kk = (tsr_data[i].k%lens[0])/phys_phase[0];
                this->faxpy(k, tsr_data[i].d, op_mats[0] + kk*k, 1, out_buffer, 1);
              }
              fvmul(out_buffer, buffer, out_buffer, k);
              this->faxpy(k, this->tmulid, out_buffer, 1, out_mat + ok*k, 1);
              //for (int j=0; j<k; j++){
              //  out_mat[j+ok*k] += out_buffer[j]*buffer[j];
              //}
            }
            idx += fiber_nnz;
          }
          if (out_mode != 0)
            this->dealloc((char*)out_buffer);
          this->dealloc((char*)buffer);
          free(inds);
        } else {
          IASSERT(0);
        }
      }

      void MTTKRP(int                      order,
                  int64_t *                lens,
                  int *                    phys_phase,
                  int64_t                  nnz,
                  int                      out_mode,
                  CTF::Pair<dtype> const * tsr_data,
                  dtype const * const *    op_vecs,
                  dtype *                  out_vec){
        int64_t * inds = (int64_t*)malloc(sizeof(int64_t)*(order-1));
        int64_t idx = 0;
        while (idx < nnz){
          int64_t fiber_idx = tsr_data[idx].k/lens[0];
          int64_t fi = fiber_idx;
          for (int i=0; i<order-1; i++){
            inds[i] = (fi % lens[i+1])/phys_phase[i+1];
            fi = fi / lens[i+1];
          }
          int64_t fiber_nnz = 1;
          while (idx+fiber_nnz < nnz && tsr_data[idx+fiber_nnz].k/lens[0] == fiber_idx)
            fiber_nnz++;
          if (out_mode == 0){
            dtype buf_val = op_vecs[1][inds[0]];
            for (int i=1; i<order-1; i++){
              buf_val *= op_vecs[i+1][inds[i]];
            }
            for (int64_t i=idx; i<idx+fiber_nnz; i++){
              int64_t kk = (tsr_data[i].k%lens[0])/phys_phase[0];
              out_vec[kk] += tsr_data[i].d*buf_val;
            }
          } else {
            int64_t ok = inds[out_mode-1];
            dtype buf_val = op_vecs[1][inds[0]];
            if (out_mode > 1)
              buf_val = op_vecs[1][inds[0]];
            else if (order > 2)
              buf_val = op_vecs[2][inds[1]];
            else
              buf_val = this->tmulid;
            for (int i=1+(out_mode==1); i<order-1; i++){
              if (out_mode != i+1)
                buf_val *= op_vecs[i+1][inds[i]];
            }
            dtype buf_val2 = this->taddid;
            for (int64_t i=idx; i<idx+fiber_nnz; i++){
              int64_t kk = (tsr_data[i].k%lens[0])/phys_phase[0];
              buf_val2 += tsr_data[i].d*op_vecs[0][kk];
            }
            out_vec[ok] += buf_val*buf_val2;
          }
          idx += fiber_nnz;
        }
        free(inds);
      }

  };
  /**
   * @}
   */
}
namespace CTF {
// TODO: add these with manual loop
//  template <>
//  bool CTF::Semiring<float,1>::is_last_col_zero(int64_t m, int64_t n, float const * M) const;
  template <>
  bool CTF::Semiring<double,1>::is_last_col_zero(int64_t m, int64_t n, double const * M) const;
//  template <>
//  bool CTF::Semiring<std::complex<float>,0>::is_last_col_zero(int64_t m, int64_t n, std::complex<float> const * M) const;
//  template <>
//  void CTF::Semiring<std::complex<double>,0>::is_last_col_zero(int64_t m, int64_t n, std::complex<double> const * M) const;


  template <>
  void CTF::Semiring<float,1>::default_csrmm(int,int,int,float,float const *,int const *,int const *,int,float const *,float,float *) const;
  template <>
  void CTF::Semiring<double,1>::default_csrmm(int,int,int,double,double const *,int const *,int const *,int,double const *,double,double *) const;
  template <>
  void CTF::Semiring<std::complex<float>,0>::default_csrmm(int,int,int,std::complex<float>,std::complex<float> const *,int const *,int const *,int,std::complex<float> const *,std::complex<float>,std::complex<float> *) const;
  template <>
  void CTF::Semiring<std::complex<double>,0>::default_csrmm(int,int,int,std::complex<double>,std::complex<double> const *,int const *,int const *,int,std::complex<double> const *,std::complex<double>,std::complex<double> *) const;


  template <>
  void CTF::Semiring<float,1>::default_csrmultd(int,int,int,float,float const *,int const *,int const *,int,float const *,int const *,int const *,int,float,float *) const;
  template <>
  void CTF::Semiring<double,1>::default_csrmultd(int,int,int,double,double const *,int const *,int const *,int,double const *,int const *,int const *,int,double,double *) const;
  template <>
  void CTF::Semiring<std::complex<float>,0>::default_csrmultd(int,int,int,std::complex<float>,std::complex<float> const *,int const *,int const *,int,std::complex<float> const *,int const *,int const *,int,std::complex<float>,std::complex<float> *) const;
  template <>
  void CTF::Semiring<std::complex<double>,0>::default_csrmultd(int,int,int,std::complex<double>,std::complex<double> const *,int const *,int const *,int,std::complex<double> const *,int const *,int const *,int,std::complex<double>,std::complex<double> *) const;

  template <>
  void CTF::Semiring<float,1>::default_csrmultcsr(int,int,int,float,float const *,int const *,int const *,int,float const *,int const *,int const *,int,float,char *&) const;
  template <>
  void CTF::Semiring<double,1>::default_csrmultcsr(int,int,int,double,double const *,int const *,int const *,int,double const *,int const *,int const *,int,double,char *&) const;
  template <>
  void CTF::Semiring<std::complex<float>,0>::default_csrmultcsr(int,int,int,std::complex<float>,std::complex<float> const *,int const *,int const *,int,std::complex<float> const *,int const *,int const *,int,std::complex<float>,char *&) const;
  template <>
  void CTF::Semiring<std::complex<double>,0>::default_csrmultcsr(int,int,int,std::complex<double>,std::complex<double> const *,int const *,int const *,int,std::complex<double> const *,int const *,int const *,int,std::complex<double>,char *&) const;


  template<> 
  bool CTF::Semiring<double,1>::is_offloadable() const;
  template<> 
  bool CTF::Semiring<float,1>::is_offloadable() const;
  template<> 
  bool CTF::Semiring<std::complex<float>,0>::is_offloadable() const;
  template<> 
  bool CTF::Semiring<std::complex<double>,0>::is_offloadable() const;

  template<> 
  void CTF::Semiring<double,1>::offload_gemm(char,char,int,int,int,char const *,char const *,char const *,char const *,char *) const;
  template<> 
  void CTF::Semiring<double,1>::offload_gemm(char,char,int,int,int,char const *,char const *,char const *,char const *,char *) const;
  template<> 
  void CTF::Semiring<std::complex<float>,0>::offload_gemm(char,char,int,int,int,char const *,char const *,char const *,char const *,char *) const;
  template<> 
  void CTF::Semiring<std::complex<double>,0>::offload_gemm(char,char,int,int,int,char const *,char const *,char const *,char const *,char *) const;
}

#include "ring.h"
#endif
