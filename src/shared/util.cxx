/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include "assert.h"
#include "util.h"
#include "cblas.h"




//#if (defined BGP || defined BGQ)
//#define UTIL_SGEMM sgemm
//#define UTIL_DGEMM dgemm
//#define UTIL_ZGEMM zgemm
//#define UTIL_SAXPY saxpy
//#define UTIL_DAXPY daxpy
//#define UTIL_ZAXPY zaxpy
//#define UTIL_SCOPY scopy
//#define UTIL_DCOPY dcopy
//#define UTIL_ZCOPY zcopy
//#define UTIL_SSCAL sscal
//#define UTIL_DSCAL dscal
//#define UTIL_ZSCAL zscal
//#define UTIL_DDOT  ddot
//#else
//#define UTIL_SGEMM sgemm_
//#define UTIL_DGEMM dgemm_
//#define UTIL_ZGEMM zgemm_
//#define UTIL_SAXPY saxpy_
//#define UTIL_DAXPY daxpy_
//#define UTIL_ZAXPY zaxpy_
//#define UTIL_SCOPY scopy_
//#define UTIL_DCOPY dcopy_
//#define UTIL_ZCOPY zcopy_
//#define UTIL_SSCAL sscal_
//#define UTIL_DSCAL dscal_
//#define UTIL_ZSCAL zscal_
//#define UTIL_DDOT  ddot_
//#endif
//
//#ifdef USE_JAG
//extern "C"
//void jag_zgemm( char *,   char      *,
//                int *,    int       *,
//                int *,    double    *,
//                double *, int       *,
//                double *, int       *,
//                double *, double    *,
//                                int *);
//
//#endif
//
//extern "C"
//void UTIL_SGEMM(const char *,  const char *,
//                const int *,   const int *,
//                const int *,   const float *,
//                const float *, const int *,
//                const float *, const int *,
//                const float *, float *,
//                                const int *);
//
//
//extern "C"
//void UTIL_DGEMM(const char *,   const char *,
//                const int *,    const int *,
//                const int *,    const double *,
//                const double *, const int *,
//                const double *, const int *,
//                const double *, double *,
//                                const int *);
//
//extern "C"
//void UTIL_ZGEMM(const char *,                 const char *,
//                const int *,                  const int *,
//                const int *,                  const std::complex<double> *,
//                const std::complex<double> *, const int *,
//                const std::complex<double> *, const int *,
//                const std::complex<double> *, std::complex<double> *,
//                                const int *);
//
//extern "C"
//void UTIL_SAXPY(const int * n,    float * dA,
//                const float * dX, const int * incX,
//                float * dY,       const int * incY);
//
//extern "C"
//void UTIL_DAXPY(const int * n,     double * dA,
//                const double * dX, const int * incX,
//                double * dY,       const int * incY);
//
//extern "C"
//void UTIL_ZAXPY(const int * n,                   std::complex<double> * dA,
//                const std::complex<double> * dX, const int * incX,
//                std::complex<double> * dY,       const int * incY);
//
//extern "C"
//void UTIL_SCOPY(const int * n,
//                const float * dX, const int * incX,
//                float * dY,       const int * incY);
//
//extern "C"
//void UTIL_DCOPY(const int * n,
//                const double * dX, const int * incX,
//                double * dY,       const int * incY);
//
//extern "C"
//void UTIL_ZCOPY(const int * n,
//                const std::complex<double> * dX, const int * incX,
//                std::complex<double> * dY,       const int * incY);
//
//extern "C"
//void UTIL_SSCAL(const int *n, float *dA,
//                float * dX,   const int *incX);
//
//extern "C"
//void UTIL_DSCAL(const int *n, double *dA,
//                double * dX,  const int *incX);
//
//extern "C"
//void UTIL_ZSCAL(const int *n,              std::complex<double> *dA,
//                std::complex<double> * dX, const int *incX);
//
//extern "C"
//double UTIL_DDOT(const int * n,    const double * dX,
//                 const int * incX, const double * dY,
//                 const int * incY);
//
//void csgemm(const char transa, const char transb,
//            const int m,       const int n,
//            const int k,       const float a,
//            const float * A,   const int lda,
//            const float * B,   const int ldb,
//            const float b,     float * C,
//                                const int ldc){
//  UTIL_SGEMM(&transa, &transb, &m, &n, &k, &a, A,
//             &lda, B, &ldb, &b, C, &ldc);
//}
//
//void cdgemm(const char transa, const char transb,
//            const int m,       const int n,
//            const int k,       const double a,
//            const double * A,  const int lda,
//            const double * B,  const int ldb,
//            const double b,    double * C,
//                                const int ldc){
//  UTIL_DGEMM(&transa, &transb, &m, &n, &k, &a, A,
//             &lda, B, &ldb, &b, C, &ldc);
//}
//
//void czgemm(const char transa,              const char transb,
//            const int m,                    const int n,
//            const int k,                    const std::complex<double> a,
//            const std::complex<double> * A, const int lda,
//            const std::complex<double> * B, const int ldb,
//            const std::complex<double> b,   std::complex<double> * C,
//                                const int ldc){
//#ifdef USE_JAG
//  jag_zgemm((char*)&transa, (char*)&transb, (int*)&m, (int*)&n, (int*)&k, (double*)&a, (double*)A,
//             (int*)&lda, (double*)B, (int*)&ldb, (double*)&b, (double*)C, (int*)&ldc);
//#else
//  UTIL_ZGEMM(&transa, &transb, &m, &n, &k, &a, A,
//             &lda, B, &ldb, &b, C, &ldc);
//#endif
//}
//
//void csaxpy(const int n,       float  dA,
//            const float  * dX, const int incX,
//            float  * dY,       const int incY){
//  UTIL_SAXPY(&n, &dA, dX, &incX, dY, &incY);
//}
//
//
//void cdaxpy(const int n,       double dA,
//            const double * dX, const int incX,
//            double * dY,       const int incY){
//  UTIL_DAXPY(&n, &dA, dX, &incX, dY, &incY);
//}
//
//void czaxpy(const int n,
//            std::complex<double>         dA,
//            const std::complex<double> * dX,
//            const int                    incX,
//            std::complex<double> *       dY,
//            const int                    incY){
//  UTIL_ZAXPY(&n, &dA, dX, &incX, dY, &incY);
//}
//
//void cscopy(const int n,
//            const float  * dX,  const int incX,
//            float  * dY,        const int incY){
//  UTIL_DCOPY(&n, dX, &incX, dY, &incY);
//}
//
//void cdcopy(const int n,
//            const double * dX,  const int incX,
//            double * dY,        const int incY){
//  UTIL_DCOPY(&n, dX, &incX, dY, &incY);
//}
//
//void czcopy(const int                    n,
//            const std::complex<double> * dX,
//            const int                    incX,
//            std::complex<double> *       dY,
//            const int                    incY){
//  UTIL_ZCOPY(&n, dX, &incX, dY, &incY);
//}
//
//
//void cdscal(const int n,        double dA,
//            double * dX,  const int incX){
//  UTIL_DSCAL(&n, &dA, dX, &incX);
//}
//
//void czscal(const int n,        std::complex<double> dA,
//            std::complex<double> * dX,  const int incX){
//  UTIL_ZSCAL(&n, &dA, dX, &incX);
//}
//
//
//double cddot(const int n,       const double *dX,
//             const int incX,    const double *dY,
//             const int incY){
//  return UTIL_DDOT(&n, dX, &incX, dY, &incY);
//}


/**
 * \brief prints matrix in 2D
 * \param[in] M matrix
 * \param[in] n number of rows
 * \param[in] m number of columns
 */
void print_matrix(double *M, int n, int m){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%lf ", M[i+j*n]);
    }
    printf("\n");
  }
}

/* abomination */
double util_dabs(double x){
  if (x >= 0.0) return x;
  return -x;
}

#ifdef COMM_TIME
/** 
 * \brief ugliest timer implementation on Earth
 * \param[in] end the type of operation this function should do (oh god why?)
 * \param[in] cdt the communicator
 * \param[in] p the number of processors
 * \param[in] iter the number of iterations if relevant
 * \param[in] myRank your rank if relevant
 */
void __CM(const int     end, 
          const CommData cdt, 
          const int     p, 
          const int     iter, 
          const int     myRank){
  static volatile double __commTime     =0.0;
  static volatile double __commTimeDelta=0.0;
  static volatile double __idleTime     =0.0;
  static volatile double __idleTimeDelta=0.0;
  if (end == 0){
    __idleTimeDelta = TIME_SEC();       
    COMM_BARRIER(cdt); 
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  }
  else if (end == 1){
    __commTime += TIME_SEC() - __commTimeDelta;
  } else if (end == 2) {
    MPI_Reduce((void*)&__commTime, (void*)&__commTimeDelta, 1, COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt.cm); 
    __commTime = __commTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent doing communication on average per iteration\n", __commTime/iter); 

    MPI_Reduce((void*)&__idleTime, (void*)&__idleTimeDelta, 1,
            COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt.cm);
    __idleTime = __idleTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent idle per iteration\n", __idleTime/iter); 
  } else if (end == 3){
    __commTime =0.0;
    __idleTime =0.0;
  } else if (end == 4){
    MPI_Irecv(NULL,0,MPI_CHAR,iter,myRank,cdt.cm,&(cdt.req[myRank]));
  } else if (end == 5){
    __idleTimeDelta =TIME_SEC();
    MPI_Send(NULL,0,MPI_CHAR,iter,myRank,cdt.cm);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  } else if (end == 6){
    MPI_Status __stat;
    __idleTimeDelta =TIME_SEC();
    MPI_Wait(&(cdt.req[myRank]),&__stat);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  }
} 

#endif
/**
 * \brief computes the size of a tensor in NOT HOLLOW packed symmetric layout
 * \param[in] order tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
int64_t sy_packed_size(const int order, const int* len, const int* sym){
  int i, k, mp;
  int64_t size, tmp;

  if (order == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (order > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < order;i++){
    tmp = (tmp * mp) / k;
    k++;
    mp ++;
    
    if (sym[i] == 0){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < order - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
}




/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] order tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
int64_t packed_size(const int order, const int* len, const int* sym){

  int i, k, mp;
  int64_t size, tmp;

  if (order == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (order > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < order;i++){
    tmp = (tmp * mp) / k;
    k++;
    if (sym[i] != 1)
      mp--;
    else
      mp ++;
    
    if (sym[i] == 0){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < order - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
}

/*
 * \brief calculates dimensional indices corresponding to a symmetric-packed index
 *        For each symmetric (SH or AS) group of size sg we have
 *          idx = n*(n-1)*...*(n-sg) / d*(d-1)*...
 *        therefore (idx*sg!)^(1/sg) >= n-sg
 *        or similarly in the SY case ... >= n
 *
 * \param[in] order number of dimensions in the tensor 
 * \param[in] lens edge lengths 
 * \param[in] sym symmetry
 * \param[in] idx index in the global tensor, in packed format
 * \param[out] idx_arr preallocated to size order, computed to correspond to idx
 */
void calc_idx_arr(int         order,
                  int const * lens,
                  int const * sym,
                  int64_t    idx,
                  int *       idx_arr){
  int64_t idx_rem = idx;
  memset(idx_arr, 0, order*sizeof(int));
  for (int dim=order-1; dim>=0; dim--){
    if (idx_rem == 0) break;
    if (dim == 0 || sym[dim-1] == NS){
      int64_t lda = packed_size(dim, lens, sym);
      idx_arr[dim] = idx_rem/lda;
      idx_rem -= idx_arr[dim]*lda;
    } else {
      int plen[dim+1];
      memcpy(plen, lens, (dim+1)*sizeof(int));
      int sg = 2;
      int fsg = 2;
      while (dim >= sg && sym[dim-sg] != NS) { sg++; fsg*=sg; }
      int64_t lda = packed_size(dim-sg+1, lens, sym);
      double fsg_idx = (((double)idx_rem)*fsg)/lda;
      int kidx = (int)pow(fsg_idx,1./sg);
      //if (sym[dim-1] != SY) 
      kidx += sg+1;
      int mkidx = kidx;
#if DEBUG >= 1
      for (int idim=dim-sg+1; idim<=dim; idim++){
        plen[idim] = mkidx+1;
      }
      int64_t smidx = packed_size(dim+1, plen, sym);
      ASSERT(smidx > idx_rem);
#endif
      int64_t midx = 0;
      for (; mkidx >= 0; mkidx--){
        for (int idim=dim-sg+1; idim<=dim; idim++){
          plen[idim] = mkidx;
        }
        midx = packed_size(dim+1, plen, sym);
        if (midx <= idx_rem) break;
      }
      if (midx == 0) mkidx = 0;
      idx_arr[dim] = mkidx;
      idx_rem -= midx;
    }
  }
  ASSERT(idx_rem == 0);
}

/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] n a positive number
 * \param[out] nfactor number of factors in n
 * \param[out] factor array of length nfactor, corresponding to factorization of n
 */
void factorize(int n, int *nfactor, int **factor){
  int tmp, nf, i;
  int * ff;
  tmp = n;
  nf = 0;
  while (tmp > 1){
    for (i=2; i<=n; i++){
      if (tmp % i == 0){
        nf++;
        tmp = tmp/i;
        break;
      }
    }
  }
  if (nf == 0){
    *nfactor = nf;
  } else {
    ff  = (int*)CTF_alloc(sizeof(int)*nf);
    tmp = n;
    nf = 0;
    while (tmp > 1){
      for (i=2; i<=n; i++){
        if (tmp % i == 0){
          ff[nf] = i;
          nf++;
          tmp = tmp/i;
          break;
        }
      }
    }
    *factor = ff;
    *nfactor = nf;
  }
}

void cvrt_idx(int         order,
              int const * lens,
              int64_t     idx,
              int *       idx_arr){
  int i;
  int64_t cidx = idx;
  for (i=0; i<order; i++){
    idx_arr[i] = cidx%lens[i];
    cidx = cidx/lens[i];
  }
}

void cvrt_idx(int         order,
              int const * lens,
              int64_t     idx,
              int **      idx_arr){
  (*idx_arr) = (int*)CTF_alloc(order*sizeof(int));
  cvrt_idx(order, lens, idx, *idx_arr);
}

void cvrt_idx(int         order,
              int const * lens,
              int const * idx_arr,
              int64_t *   idx){
  int i;
  int64_t lda = 1;
  *idx = 0;
  for (i=0; i<order; i++){
    (*idx) += idx_arr[i]*lda;
    lda *= lens[i];
  }

}

