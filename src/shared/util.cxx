/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following 
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR 
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY 
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF 
 * SUCH DAMAGE. */

#include <stdio.h>
#include <stdint.h>
#include "string.h"
#include "assert.h"
#include "util.h"

#if (defined BGP || defined BGQ)
#define UTIL_ZGEMM      zgemm
#define UTIL_DGEMM      dgemm
#define UTIL_DAXPY      daxpy
#define UTIL_ZAXPY      zaxpy
#define UTIL_DSCAL      dscal
#define UTIL_DDOT       ddot
#else
#define UTIL_ZGEMM      zgemm_
#define UTIL_DGEMM      dgemm_
#define UTIL_DAXPY      daxpy_
#define UTIL_ZAXPY      zaxpy_
#define UTIL_DSCAL      dscal_
#define UTIL_DDOT       ddot_
#endif

#ifdef USE_JAG
extern "C"
void jag_zgemm( char *, char *,
                int *,  int *,
                int *,  double *,
                double *,       int *,
                double *,       int *,
                double *,       double *,
                                int *);

#endif

extern "C"
void UTIL_DGEMM(const char *,   const char *,
                const int *,    const int *,
                const int *,    const double *,
                const double *, const int *,
                const double *, const int *,
                const double *, double *,
                                const int *);

extern "C"
void UTIL_ZGEMM(const char *,   const char *,
                const int *,    const int *,
                const int *,    const std::complex<double> *,
                const std::complex<double> *,   const int *,
                const std::complex<double> *,   const int *,
                const std::complex<double> *,   std::complex<double> *,
                                const int *);


extern "C"
void UTIL_DAXPY(const int * n,          double * dA,
                const double * dX,      const int * incX,
                double * dY,            const int * incY);

extern "C"
void UTIL_ZAXPY(const int * n,                          std::complex<double> * dA,
                const std::complex<double> * dX,        const int * incX,
                std::complex<double> * dY,              const int * incY);


extern "C"
void UTIL_DSCAL(const int *n,           double *dA,
                const double * dX,      const int *incX);

extern "C"
double UTIL_DDOT(const int * n,         const double * dX,      
                 const int * incX,      const double * dY,      
                 const int * incY);

void cdgemm(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const double a,
            const double * A,   const int lda,
            const double * B,   const int ldb,
            const double b,     double * C,
                                const int ldc){
  UTIL_DGEMM(&transa, &transb, &m, &n, &k, &a, A,
             &lda, B, &ldb, &b, C, &ldc);
}

void czgemm(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const std::complex<double> a,
            const std::complex<double> * A,     const int lda,
            const std::complex<double> * B,     const int ldb,
            const std::complex<double> b,       std::complex<double> * C,
                                const int ldc){
#ifdef USE_JAG
  jag_zgemm((char*)&transa, (char*)&transb, (int*)&m, (int*)&n, (int*)&k, (double*)&a, (double*)A,
             (int*)&lda, (double*)B, (int*)&ldb, (double*)&b, (double*)C, (int*)&ldc);
#else
  UTIL_ZGEMM(&transa, &transb, &m, &n, &k, &a, A,
             &lda, B, &ldb, &b, C, &ldc);
#endif
}

void czaxpy(const int n,        
            std::complex<double> dA,
            const std::complex<double> * dX,
            const int incX,
            std::complex<double> * dY,
            const int incY){
  UTIL_ZAXPY(&n, &dA, dX, &incX, dY, &incY);
}


void cdaxpy(const int n,        double dA,
            const double * dX,  const int incX,
            double * dY,        const int incY){
  UTIL_DAXPY(&n, &dA, dX, &incX, dY, &incY);
}

void cdscal(const int n,        double dA,
            const double * dX,  const int incX){
  UTIL_DSCAL(&n, &dA, dX, &incX);
}

double cddot(const int n,       const double *dX,
             const int incX,    const double *dY,
             const int incY){
  return UTIL_DDOT(&n, dX, &incX, dY, &incY);
}


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
          const CommData *cdt, 
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
    MPI_Reduce((void*)&__commTime, (void*)&__commTimeDelta, 1, COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt->cm); 
    __commTime = __commTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent doing communication on average per iteration\n", __commTime/iter); 

    MPI_Reduce((void*)&__idleTime, (void*)&__idleTimeDelta, 1,
            COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt->cm);
    __idleTime = __idleTimeDelta/p;
    if (myRank == 0)
      printf("%lf seconds spent idle per iteration\n", __idleTime/iter); 
  } else if (end == 3){
    __commTime =0.0;
    __idleTime =0.0;
  } else if (end == 4){
    MPI_Irecv(NULL,0,MPI_CHAR,iter,myRank,cdt->cm,&(cdt->req[myRank]));
  } else if (end == 5){
    __idleTimeDelta =TIME_SEC();
    MPI_Send(NULL,0,MPI_CHAR,iter,myRank,cdt->cm);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  } else if (end == 6){
    MPI_Status __stat;
    __idleTimeDelta =TIME_SEC();
    MPI_Wait(&(cdt->req[myRank]),&__stat);
    __idleTime += TIME_SEC() - __idleTimeDelta;
    __commTimeDelta = TIME_SEC(); 
  }
} 

#endif
/**
 * \brief computes the size of a tensor in NOT HOLLOW packed symmetric layout
 * \param[in] ndim tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
int64_t sy_packed_size(const int ndim, const int* len, const int* sym){
  int i, k, mp;
  int64_t size, tmp;

  if (ndim == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (ndim > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < ndim;i++){
    tmp = (tmp * mp) / k;
    k++;
    mp ++;
    
    if (sym[i] == 0){
      size *= tmp;
      k = 1;
      tmp = 1;
      if (i < ndim - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
}




/**
 * \brief computes the size of a tensor in packed symmetric layout
 * \param[in] ndim tensor dimension
 * \param[in] len tensor edge _elngths
 * \param[in] sym tensor symmetries
 * \return size of tensor in packed layout
 */
int64_t packed_size(const int ndim, const int* len, const int* sym){

  int i, k, mp;
  int64_t size, tmp;

  if (ndim == 0) return 1;

  k = 1;
  tmp = 1;
  size = 1;
  if (ndim > 0)
    mp = len[0];
  else
    mp = 1;
  for (i = 0;i < ndim;i++){
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
      if (i < ndim - 1) mp = len[i + 1];
    }
  }
  size *= tmp;

  return size;
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
    *nfactor    = 0;
  } else {
    ff  = (int*)malloc(sizeof(int)*nf);
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
