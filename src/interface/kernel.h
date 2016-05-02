#ifndef __KERNEL_H__
#define __KERNEL_H__

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
  __global__ void cuda_csrmmf(int             m,
                              int             n,
                              int             k,
                              dtype_A const * A,
                              int const *     IA,
                              int const *     JA,
                              dtype_B const * B,
                              dtype_C *       C){
    int bidx = blockIdx.x;
    int tidx = threadIdx.x;
    for (int row_A=bidx; row_A<m; row_A+=NBLK){
      for (int col_B=tidx; col_B<n; col_B+=NTRD){
        for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
          int col_A = JA[i_A]-1;
          g(f(A[i_A],B[col_B*k+col_A]),C[col_B*m+row_A]);
        }
      }
    }
  }

  #endif

  template<typename dtype>
  #ifdef __CUDACC__
  __device__ __host__
  #endif
  void default_monoid(dtype a, dtype & b){ b = a+b; }

  template<typename dtype=double, void(*g)(dtype, dtype&)=default_monoid<dtype> >
  class Monoid_Kernel {
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


    static void xpy(int             n,
                    dtype const * X,
                    int             incX,
                    dtype *       Y,
                    int             incY){

      for (int i=0; i<n; i++){
        g(X[incX*i],Y[incY*i]);
      }
    }
  };
  


  template<typename dtype_A, typename dtype_B, typename dtype_C, dtype_C(*f)(dtype_A, dtype_B), void(*g)(dtype_C, dtype_C&)=default_monoid<dtype_C> >
  class Bivar_Kernel : public Monoid_Kernel<dtype_C, g>, public Bivar_Function<dtype_A, dtype_B, dtype_C> {
    public:
    Bivar_Kernel() : Bivar_Function<dtype_A, dtype_B, dtype_C>(f) {
      this->has_gemm = true;
#ifdef __CUDACC__
      this->has_off_gemm = true;
#endif
    }

    Bivar_Kernel(bool is_comm) : Bivar_Function<dtype_A, dtype_B, dtype_C>(f, is_comm) {
      this->has_gemm = true;
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
      TAU_FSTART(default_fcoomm);
      for (int i=0; i<nnz_A; i++){
        int row_A = rows_A[i]-1;
        int col_A = cols_A[i]-1;
        for (int col_C=0; col_C<n; col_C++){
          g(f(A[i],B[col_C*k+col_A]),C[col_C*m+row_A]);
        }
      }
      TAU_FSTOP(default_fcoomm);
    }

    void ccoomm(int                             m,
                int                             n,
                int                             k,
                char const *                    A,
                int const *                     rows_A,
                int const *                     cols_A,
                int64_t                         nnz_A,
                char const *                    B,
                char *                          C) const {
      int * ptr = NULL;
      ptr[0] = 3;
      coomm(m, n, k, (dtype_A const *)A, rows_A, cols_A, nnz_A, 
            (dtype_B const *)B, (dtype_C *)C);
    }

    void csrmm(int             m,
               int             n,
               int             k,
               dtype_A const * A,
               int const *     IA,
               int const *     JA,
               int64_t         nnz_A,
               dtype_B const * B,
               dtype_C *       C) const {
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
    }

    void ccsrmm(int          m,
                int          n,
                int          k,
                char const * A,
                int const *  IA,
                int const *  JA,
                int64_t      nnz_A,
                char const * B,
                char *       C) const {
      csrmm(m,n,k,(dtype_A const *)A,IA,JA,nnz_A,(dtype_B const *)B, (dtype_C *)C);
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



    static void offload_gemm(char            tA,
                             char            tB,
                             int             m,
                             int             n,
                             int             k,
                             dtype_A const * A,
                             dtype_B const * B,
                             dtype_C *       C){
#ifdef __CUDACC__
      cuda_gemmf<dtype_A,dtype_B,dtype_C,f,g><<<NBLK,NTRD>>>(tA, tB, m, n, k, A, B, C);
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
