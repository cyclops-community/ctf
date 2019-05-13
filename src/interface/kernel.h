#ifndef __KERNEL_H__
#define __KERNEL_H__

#include "../sparse_formats/csr.h"
namespace CTF{
  #ifdef __CUDACC__
  #define NBLK 15
  #define NTRD 512
  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)>
  __global__ void cuda_gemmf(char            tA,
                             char            tB,
                             int             m,
                             int             n,
                             int             k,
                             dtype_A const * A,
                             dtype_B const * B,
                             dtype_C *       C){
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    int lda_A_m = tA == 'N' ? 1 : k;
    int lda_A_k = tA == 'N' ? m : 1;
    int lda_B_k = tB == 'N' ? 1 : n;
    int lda_B_n = tB == 'N' ? k : 1;
    for (int mi=bidx; mi<m; mi+=NBLK){
      for (int ni=tidx; ni<n; ni+=NTRD){
        for (int ki=0; ki<k; ki++){
          g(f(A[mi*lda_A_m+ki*lda_A_k],
              B[ki*lda_B_k+ni*lda_B_n]),
              C[mi        +ni*m]);
        }
      }
    }
  }

  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)>
  __device__ 
  void cuda_csrmmf(int             m,
                   int             n,
                   int             k,
                   dtype_A const * A,
                   int const *     JA,
                   int const *     IA,
                   dtype_B const * B,
                   dtype_C *       C){
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    for (int col_B=bidx; col_B<n; col_B+=NBLK){
      for (int row_A=tidx; row_A<m; row_A+=NTRD){
        for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
          int col_A = JA[i_A]-1;
          g(f(A[i_A],B[col_B*k+col_A]),C[col_B*m+row_A]);
        }
      }
    }
  }


  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)>
  __device__ 
  void cuda_csrmmf(int             m,
                   int             n,
                   int             k,
                   dtype_A const * A,
                   int const *     JA,
                   int const *     IA,
                   dtype_B const * B,
                   dtype_C *       C){
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    for (int col_B=bidx; col_B<n; col_B+=NBLK){
      for (int row_A=tidx; row_A<m; row_A+=NTRD){
        for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
          int col_A = JA[i_A]-1;
          g(f(A[i_A],B[col_B*k+col_A]),C[col_B*m+row_A]);
        }
      }
    }
  }
  
  //FIXME there is code replication here with ../sparse_foramts/csr.cxx
  #define ALIGN 256

  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)>
  __global__
  void offload_csrmm(int             m,
                     int             n,
                     int             k,
                     char const *    all_data,
                     dtype_B const * B,
                     dtype_C *       C){
    int64_t nnz_A = ((int64_t*)all_data)[0];
    int offset = 3*sizeof(int64_t);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    dtype_A const * A = (dtype_A const *)(all_data + offset);
    offset += nnz_A*sizeof(dtype_A);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    int const * IA = (int*)(all_data + offset); 
    offset += (m+1)*sizeof(int);
    if (offset % ALIGN != 0) offset += ALIGN-(offset%ALIGN);
    int const * JA = (int*)(all_data + offset);
    cuda_csrmmf<dtype_A,dtype_B,dtype_C,f,g>(m,n,k,A,JA,IA,B,C);
  }

  #undef ALIGN

  #endif

  template<typename dtype>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  void default_monoid(dtype a, dtype & b){ b = a+b; }

  template<typename dtype=double, void(*g)(dtype, dtype&)=default_monoid<dtype> >
  class Monoid_Kernel : public CTF_int::accumulatable {
    public:
    static MPI_Op get_MPI_Op(){
      MPI_Op moo;

      //FIXME: assumes monoid is commutative
      MPI_Op_create(
          [](void * a, void * b, int * n, MPI_Datatype*){ 
            for (int i=0; i<*n; i++){ 
              g(((dtype*)a)[i], ((dtype*)b)[i]);
            }
          },
          1, &moo);

      return moo;
    }

    Monoid_Kernel(){
      this->el_size = sizeof(dtype);
    }

    void accum(char const * a, 
               char * b) const { 
      g(((dtype const *)a)[0], ((dtype *)b)[0]); 
    }

    static void xpy(int           n,
                    dtype const * X,
                    int           incX,
                    dtype *       Y,
                    int           incY){

      for (int i=0; i<n; i++){
        g(X[incX*i],Y[incY*i]);
      }
    }
    /** \brief initialize n objects to zero
      * \param[in] n number of items
      * \param[in] arr array containing n items, to be set to zero
      */
    virtual void init_shell(int64_t n, char * arr) const {
      dtype dummy = dtype();
      for (int i=0; i<n; i++){
        memcpy(arr+i*el_size,(char*)&dummy,el_size);
      }
    }
  };
  


  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)=default_monoid<dtype_C> >
  class Bivar_Kernel : public Monoid_Kernel<dtype_C, g>, public Bivar_Function<dtype_A, dtype_B, dtype_C> {
    public:
    Bivar_Kernel() : Bivar_Function<dtype_A, dtype_B, dtype_C>(f) {
      this->has_kernel = true;
#ifdef __CUDACC__
      this->has_off_gemm = true;
#endif
      this->el_size = sizeof(dtype_C);
    }

    Bivar_Kernel(bool is_comm) : Bivar_Function<dtype_A, dtype_B, dtype_C>(f, is_comm) {
      this->has_kernel = true;
#ifdef __CUDACC__
      this->has_off_gemm = true;
#endif
    }



    static void gemm(char            tA,
                     char            tB,
                     int             m,
                     int             n,
                     int             k,
                     dtype_A const * A,
                     dtype_B const * B,
                     dtype_C *       C){
      int lda_A_m = tA == 'N' ? 1 : k;
      int lda_A_k = tA == 'N' ? m : 1;
      int lda_B_k = tB == 'N' ? 1 : n;
      int lda_B_n = tB == 'N' ? k : 1;
#ifdef _OPENMP
      #pragma omp parallel for
#endif 
      for (int mi=0; mi<m; mi++){
#ifdef _OPENMP
        #pragma omp parallel for
#endif 
        for (int ni=0; ni<n; ni++){
          for (int ki=0; ki<k; ki++){
            g(f(A[mi*lda_A_m+ki*lda_A_k],
                B[ki*lda_B_k+ni*lda_B_n]),
                C[mi        +ni*m]);
          }
        }
      }
    }


    static void coomm(int             m,
                      int             n,
                      int             k,
                      dtype_A const * A,
                      int const *     rows_A,
                      int const *     cols_A,
                      int             nnz_A,
                      dtype_B const * B,
                      dtype_C *       C){
      //TAU_FSTART(default_fcoomm);
      for (int i=0; i<nnz_A; i++){
        int row_A = rows_A[i]-1;
        int col_A = cols_A[i]-1;
        for (int col_C=0; col_C<n; col_C++){
          g(f(A[i],B[col_C*k+col_A]),C[col_C*m+row_A]);
        }
      }
      //TAU_FSTOP(default_fcoomm);
    }

    void ccoomm(int          m,
                 int          n,
                 int          k,
                 char const * A,
                 int const *  rows_A,
                 int const *  cols_A,
                 int64_t      nnz_A,
                 char const * B,
                 char *       C) const {
      coomm(m, n, k, (dtype_A const *)A, rows_A, cols_A, nnz_A, 
            (dtype_B const *)B, (dtype_C *)C);
    }

    static void csrmm(int             m,
                      int             n,
                      int             k,
                      dtype_A const * A,
                      int const *     JA,
                      int const *     IA,
                      int64_t         nnz_A,
                      dtype_B const * B,
                      dtype_C *       C){
      //TAU_FSTART(3type_csrmm);
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int row_A=0; row_A<m; row_A++){
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for (int col_B=0; col_B<n; col_B++){
          for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
            int col_A = JA[i_A]-1;
            g(f(A[i_A],B[col_B*k+col_A]),C[col_B*m+row_A]);
          }
        }
      }
      //TAU_FSTOP(3type_csrmm);
    }
    void cgemm(char         tA,
               char         tB,
               int          m,
               int          n,
               int          k,
               char const * A,
               char const * B,
               char *       C) const {
      gemm(tA, tB, m, n, k, 
           (dtype_A const *)A, (dtype_B const *)B, (dtype_C *)C);
    }

    // FIXME: below kernels replicate code from src/interface/semiring.h

    static void csrmultd
                 (int             m,
                  int             n,
                  int             k,
                  dtype_A const * A,
                  int const *     JA,
                  int const *     IA,
                  int64_t         nnz_A,
                  dtype_B const * B,
                  int const *     JB,
                  int const *     IB,
                  int64_t         nnz_B,
                  dtype_C *       C){
#ifdef _OPENMP
      #pragma omp parallel for
#endif
      for (int row_A=0; row_A<m; row_A++){
        for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
          int row_B = JA[i_A]-1; //=col_A
          for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
            int col_B = JB[i_B]-1;
            g(f(A[i_A],B[i_B]),C[col_B*m+row_A]);
          }
        }
      }
    }


    void csrmultcsr_old
              (int           m,
               int           n,
               int           k,
               dtype_A const * A,
               int const *   JA,
               int const *   IA,
               int           nnz_A,
               dtype_B const * B,
               int const *   JB,
               int const *   IB,
               int           nnz_B,
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
      CTF_int::CSR_Matrix C(IC[m]-1, m, n, this);
      dtype_C * vC = (dtype_C*)C.vals();
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
        memset(has_col, 0, sizeof(int)*n);
        for (int j=0; j<IA[i+1]-IA[i]; j++){
          int row_B = JA[IA[i]+j-1]-1;
          int idx_A = IA[i]+j-1;
          for (int l=0; l<IB[row_B+1]-IB[row_B]; l++){
            int idx_B = IB[row_B]+l-1;
            if (has_col[JB[idx_B]-1])
              g(f(A[idx_A],B[idx_B]), vC[rev_col[JB[idx_B]-1]]);  
            else
              vC[rev_col[JB[idx_B]-1]] = f(A[idx_A],B[idx_B]);
            has_col[JB[idx_B]-1] = 1;  
          }
        }
      }
      CTF_int::CSR_Matrix C_in(C_CSR);
      if (C_CSR == NULL || C_in.nnz() == 0){
        C_CSR = C.all_data;
      } else {
        char * ans = CTF_int::CSR_Matrix::csr_add(C_CSR, C.all_data, this);
        CTF_int::cdealloc(C.all_data);
        C_CSR = ans;
      }
      CTF_int::cdealloc(has_col);
      CTF_int::cdealloc(rev_col);
    }

    void csrmultcsr
                      (int          m, 
                      int           n,
                      int           k, 
                      dtype_A const * A, // A m by k
                      int const *   JA,
                      int const *   IA,
                      int           nnz_A,
                      dtype_B const * B, // B k by n
                      int const *   JB,
                      int const *   IB,
                      int           nnz_B,
                      char *&       C_CSR) const {
        //int *ic = (int*)Malloc(sizeof(int)*(m+1));
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
        dtype_C * vC = (dtype_C*)C.vals();
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
          dtype_C *acc_data = new dtype_C[n];
#ifdef _OPENMP
          #pragma omp for
#endif            
          for (int i=0; i<m; i++){
            memset(dcol, 0, sizeof(int)*(n));
            ins = 0;
            for (int j=0; j<IA[i+1]-IA[i]; j++){
              int row_b = JA[IA[i]+j-1]-1; // 1-based
              int idx_a = IA[i]+j-1;
              for (int ii = 0; ii < IB[row_b+1]-IB[row_b]; ii++){
                int col_b = IB[row_b]+ii-1;
                int col_c = JB[col_b]-1; // 1-based
//                    dtype_C val = fmul(A[idx_a], B[col_b]);
                if (dcol[col_c] == 0){
                    dcol[col_c] = JB[col_b];
                    acc_data[col_c] =f(A[idx_a],B[col_b]);
                } else {
                    g(f(A[idx_a],B[col_b]), acc_data[col_c]);
                }
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
          delete [] acc_data;
#ifdef _OPENMP
        } //PRAGMA END
#endif
      CTF_int::CSR_Matrix C_in(C_CSR);
      if (C_CSR == NULL || C_in.nnz() == 0){
        C_CSR = C.all_data;
      } else {
        char * ans = CTF_int::CSR_Matrix::csr_add(C_CSR, C.all_data, this);
        CTF_int::cdealloc(C.all_data);
        C_CSR = ans;
      }

    }



    void fcsrmultd
                 (int          m,
                  int          n,
                  int          k,
                  char const * A,
                  int const *  JA,
                  int const *  IA,
                  int64_t      nnz_A,
                  char const * B,
                  int const *  JB,
                  int const *  IB,
                  int64_t      nnz_B,
                  char *       C,
                  CTF_int::algstrct const * sr_C) const {
      csrmultd(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B,JB,IB,nnz_B,(dtype_C *)C);
    }

    void fcsrmultcsr
             (int          m,
              int          n,
              int          k,
              char const * A,
              int const *  JA,
              int const *  IA,
              int          nnz_A,
              char const * B,
              int const *  JB,
              int const *  IB,
              int          nnz_B,
              char *&      C_CSR,
              CTF_int::algstrct const * sr_C) const {
      csrmultcsr(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B, JB, IB, nnz_B, C_CSR);
    }

    void csrmm(int          m,
               int          n,
               int          k,
               char const * A,
               int const *  JA,
               int const *  IA,
               int64_t      nnz_A,
               char const * B,
               char *       C,
               CTF_int::algstrct const * sr_C) const {
      csrmm(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B, (dtype_C *)C);
    }



    static void offload_gemm(char            tA,
                             char            tB,
                             int             m,
                             int             n,
                             int             k,
                             dtype_A const * A,
                             dtype_B const * B,
                             dtype_C *       C){
#ifdef __CUDACC__
#ifdef PROFILE_CUGEMM
      //TAU_FSTART(3type_cugemm);
#endif
      cuda_gemmf<dtype_A,dtype_B,dtype_C,f,g><<<NBLK,NTRD>>>(tA, tB, m, n, k, A, B, C);
#ifdef PROFILE_CUGEMM
      cudaDeviceSynchronize();
      //TAU_FSTOP(3type_cugemm);
#endif
#else
      assert(0);
#endif
    }

    void coffload_gemm(char         tA,
                       char         tB,
                       int          m,
                       int          n,
                       int          k,
                       char const * A,
                       char const * B,
                       char *       C) const {
      offload_gemm(tA, tB, m, n, k, (dtype_A const *)A, (dtype_B const *)B, (dtype_C*)C);
    }

/*
    void coffload_csrmm(int          m,
                        int          n,
                        int          k,
                        char const * A,
                        int const *  JA,
                        int const *  IA,
                        int64_t      nnz_A,
                        char const * B,
                        char *       C) const {
      offload_csrmm(m, n, k, (dtype_A const *)A, JA, IA, nnz_A, (dtype_B const *)B, (dtype_C*)C);
    }*/

    void coffload_csrmm(int          m,
                        int          n,
                        int          k,
                        char const * all_data,
                        char const * B,
                        char *       C) const {
#ifdef __CUDACC__
#ifdef PROFILE_CUGEMM
      //TAU_FSTART(3type_cucsrmm);
#endif
      offload_csrmm<dtype_A,dtype_B,dtype_C,f,g><<<NBLK,NTRD>>>(m, n, k, all_data, (dtype_B const *)B, (dtype_C *)C);
#ifdef PROFILE_CUGEMM
      cudaDeviceSynchronize();
      //TAU_FSTOP(3type_cucsrmm);
#endif
#else
      assert(0);
#endif
//      offload_csrmm(m, n, k, (dtype_A const *)A, JA, IA, nnz_A, (dtype_B const *)B, (dtype_C*)C);
    }




/*    static void axpy(int             n,
                     dtype_C         alpha,
                     dtype_C const * X,
                     int             incX,
                     dtype_C *       Y
                     int           incY){

      for (int i=0; i<n; i++){
         g(f(alpha,X[incX*i]),Y[incY*i]);
      }
    }*/
  };

}
#endif
