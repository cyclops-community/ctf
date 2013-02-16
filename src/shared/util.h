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

#ifndef __UTIL_H__
#define __UTIL_H__

#include <string.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <complex>
#include <unistd.h>
#include "comm.h"


typedef int64_t long_int;
volatile static long_int long_int_max = INT64_MAX;
//doesn't work with OpenMPI
//volatile static long_int mpi_long_int = MPI_LONG_LONG_INT;

#ifndef ENABLE_ASSERT
#ifdef DEBUG
#define ENABLE_ASSERT 1
#else
#define ENABLE_ASSERT 0
#endif
#endif
#ifndef HOPPER
inline
uint64_t getTotalSystemMemory()
{
  uint64_t pages = (uint64_t)sysconf(_SC_PHYS_PAGES);
  uint64_t page_size = (uint64_t)sysconf(_SC_PAGE_SIZE);
  return pages * page_size;
}
#endif

#include <execinfo.h>
#include <signal.h>
inline void handler() {
#if (!BGP && !BGQ && !HOPPER)
  int i, size;
  void *array[10];

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  backtrace_symbols(array, size);
  char syscom[256*size];
  for (i=1; i<size; ++i)
  {
    char buf[256];
    char buf2[256];
    int bufsize = 256;
    int sz = readlink("/proc/self/exe", buf, bufsize);
    buf[sz] = '\0';
    sprintf(buf2,"addr2line %p -e %s", array[i], buf);
    if (i==1)
      strcpy(syscom,buf2);
    else
      strcat(syscom,buf2);

  }
  assert(system(syscom)==0);
#endif
}
#ifndef LIBT_ASSERT
#if ENABLE_ASSERT
#define LIBT_ASSERT(...)                \
do { if (!(__VA_ARGS__)) handler(); assert(__VA_ARGS__); } while (0)
#else
#define LIBT_ASSERT(...)
#endif
#endif

#define ABORT                                   \
  do{                                           \
  handler(); MPI_Abort(MPI_COMM_WORLD, -1); } while(0)

//proper modulus for 'a' in the range of [-b inf]
#ifndef WRAP
#define WRAP(a,b)       ((a + b)%b)
#endif

#ifndef ALIGN_BYTES
#define ALIGN_BYTES     16
#endif

#ifndef MIN
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#ifndef MAX
#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef LOC
#define LOC \
  do { printf("debug:%s:%d ",__FILE__,__LINE__); } while(0)
#endif

#ifndef ERROR
#define ERROR(...) \
do { printf("error:%s:%d ",__FILE__,__LINE__); printf(__VA_ARGS__); printf("\n"); quit(1); } while(0)
#endif

#ifndef WARN
#define WARN(...) \
do { printf("warning: "); printf(__VA_ARGS__); printf("\n"); } while(0)
#endif

#ifdef DEBUG
  #ifndef DPRINTF
  #define DPRINTF(i,...) \
    do { if (i<=DEBUG) { LOC; printf(__VA_ARGS__); } } while (0)
  #endif
  #ifndef DEBUG_PRINTF
  #define DEBUG_PRINTF(...) \
    do { DPRINTF(5,__VA_ARGS__); } while(0)
  #endif
  #ifndef RANK_PRINTF
  #define RANK_PRINTF(myRank,rank,...) \
    do { if (myRank == rank) { LOC; printf("P[%d]: ",rank); printf(__VA_ARGS__); } } while(0)
  #endif
        #ifndef PRINT_INT
        #define PRINT_INT(var) \
          do {  LOC; printf(#var); printf("=%d\n",var); } while(0)
        #endif
        #ifndef PRINT_DOUBLE
        #define PRINT_DOUBLE(var) \
          do {  LOC; printf(#var); printf("=%lf\n",var); } while(0)
        #endif
#else
  #ifndef DPRINTF
  #define DPRINTF(...)
  #endif
  #ifndef DEBUG_PRINTF
  #define DEBUG_PRINTF(...)
  #endif
  #ifndef RANK_PRINTF
  #define RANK_PRINTF(...)
  #endif
  #ifndef PRINT_INT
  #define PRINT_INT(var)
  #endif
#endif

#ifdef VERBOSE
  #ifndef VERBOSE_PRINTF
  #define VERBOSE_PRINTF(...) \
    do { LOC; printf(__VA_ARGS__); } while(0)
  #endif
#else
  #ifndef VERBOSE_PRINTF
  #define VERBOSE_PRINTF
  #endif
#endif

#ifdef DUMPDEBUG
  #ifndef DUMPDEBUG_PRINTF
  #define DUMPDEBUG_PRINTF(...) \
    do { LOC; printf(__VA_ARGS__); } while(0)
  #endif
#else
  #ifndef DUMPDEBUG_PRINTF
  #define DUMPDEBUG_PRINTF(...)
  #endif
#endif

#ifdef TAU
#include <stddef.h>
#include <Profile/Profiler.h>
#define TAU_FSTART(ARG)                                 \
    TAU_PROFILE_TIMER(timer##ARG, #ARG, "", TAU_USER);  \
    TAU_PROFILE_START(timer##ARG)

#define TAU_FSTOP(ARG)                                  \
    TAU_PROFILE_STOP(timer##ARG)

#else
#define TAU_PROFILE(NAME,ARG,USER)
#define TAU_PROFILE_TIMER(ARG1, ARG2, ARG3, ARG4)
#define TAU_PROFILE_STOP(ARG)
#define TAU_PROFILE_START(ARG)
#define TAU_FSTART(ARG)
#define TAU_FSTOP(ARG)
#endif

#if (defined(TAU) || defined(COMM_TIME))
#define INIT_IDLE_TIME                  \
  volatile double __idleTime=0.0;       \
  volatile double __idleTimeDelta=0.0;
#define INSTRUMENT_BARRIER(COMM)        do {    \
  __idleTimeDelta = TIME_SEC();                 \
  COMM_BARRIER(COMM);                           \
  __idleTime += TIME_SEC() - __idleTimeDelta;   \
  } while(0)
#define INSTRUMENT_GLOBAL_BARRIER(COMM) do {    \
  __idleTimeDelta = TIME_SEC();                 \
  GLOBAL_BARRIER(COMM);                         \
  __idleTime += TIME_SEC() - __idleTimeDelta;   \
  } while(0)
#define AVG_IDLE_TIME(cdt, p)                                   \
do{                                                             \
  REDUCE((void*)&__idleTime, (void*)&__idleTimeDelta, 1,        \
          COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt);                  \
  __idleTime = __idleTimeDelta/p;                               \
}while(0)
#define IDLE_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent idle per iteration\n",         \
      __idleTime/iter); } while(0)

#else
#define INSTRUMENT_BARRIER(COMM)
#define INSTRUMENT_GLOBAL_BARRIER(COMM)
#define INIT_IDLE_TIME
#define AVG_IDLE_TIME(cdt, p)
#define IDLE_TIME_PRINT_ITER(iter)
#endif

#define TIME(STRING) TAU_PROFILE(STRING, " ", TAU_DEFAULT)

#ifdef COMM_TIME
//ugly and scope limited, but whatever.
#define INIT_COMM_TIME                          \
  volatile double __commTime =0.0, __commTimeDelta;     \
  volatile double __critTime =0.0, __critTimeDelta;

#define COMM_TIME_START()                       \
  do { __commTimeDelta = TIME_SEC(); } while(0)
#define COMM_TIME_END()                         \
  do { __commTime += TIME_SEC() - __commTimeDelta; } while(0)
#define CRIT_TIME_START()                       \
  do {                                          \
    __commTimeDelta = TIME_SEC();               \
    __critTimeDelta = TIME_SEC();               \
  } while(0)
#define CRIT_TIME_END()                         \
  do {                                          \
    __commTime += TIME_SEC() - __commTimeDelta; \
    __critTime += TIME_SEC() - __critTimeDelta; \
  } while(0)
#define COMM_TIME_PRINT()                       \
  do { printf("%lf seconds spent doing communication\n", __commTime); } while(0)
#define COMM_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent doing communication per iteration\n", __commTime/iter); } while(0)
#define CRIT_TIME_PRINT_ITER(iter)                              \
  do { printf("%lf seconds spent doing communication along critical path per iteration\n", __critTime/iter); \
  } while(0)
#define AVG_COMM_TIME(cdt, p)                                                           \
do{                                                                                     \
  REDUCE((void*)&__commTime, (void*)&__commTimeDelta, 1, COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt);           \
  __commTime = __commTimeDelta/p;                                                       \
}while(0)
#define SUM_CRIT_TIME(cdt, p)                                                           \
do{                                                                                     \
  REDUCE((void*)&__critTime, (void*)&__critTimeDelta, 1, COMM_DOUBLE_T, COMM_OP_SUM, 0, cdt);           \
  __critTime = __critTimeDelta;                                                 \
}while(0)


void __CM(const int     end,
          const CommData *cdt,
          const int     p,
          const int     iter,
          const int     myRank);
#else
#define __CM(...)
#define INIT_COMM_TIME
#define COMM_TIME_START()
#define COMM_TIME_END()
#define COMM_TIME_PRINT()
#define COMM_TIME_PRINT_ITER(iter)
#define AVG_COMM_TIME(cdt, p)
#define CRIT_TIME_START()
#define CRIT_TIME_END()
#define CRIT_TIME_PRINT_ITER(iter)
#define SUM_CRIT_TIME(cdt, p)
#endif


void cdgemm(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const double a,
            const double * A,   const int lda,
            const double * B,   const int ldb,
            const double b,     double * C,
                                const int ldc);

void czgemm(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const std::complex<double> a,
            const std::complex<double> * A,     const int lda,
            const std::complex<double> * B,     const int ldb,
            const std::complex<double> b,       std::complex<double> * C,
                                const int ldc);

template <typename dtype>
void cxgemm(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const dtype a,
            const dtype * A,    const int lda,
            const dtype * B,    const int ldb,
            const dtype b,      dtype * C,
                                const int ldc);

void cdaxpy(const int n,        double dA,
            const double * dX,  const int incX,
            double * dY,        const int incY);

void czaxpy(const int n,                        std::complex<double> dA,
            const std::complex<double> * dX,    const int incX,
            std::complex<double> * dY,          const int incY);

template <typename dtype>
void cxaxpy(const int n,        dtype dA,
            const dtype * dX,   const int incX,
            dtype * dY, const int incY);

void cdscal(const int n,        double dA,
            const double * dX,  const int incX);

double cddot(const int n,       const double *dX,
             const int incX,    const double *dY,
             const int incY);


template<typename dtype>
void transp(const int size,  const int lda_i, const int lda_o,
            const dtype *A, dtype *B);

template<typename dtype>
void coalesce_bwd(dtype         *B,
                  dtype const   *B_aux,
                  int const     k,
                  int const     n,
                  int const     kb);

/* Copies submatrix to submatrix */
template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B);

template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B,
             const dtype a,  const dtype b);

void print_matrix(double *M, int n, int m);

//double util_dabs(double x);

int64_t sy_packed_size(const int ndim, const int* len, const int* sym);

int64_t packed_size(const int ndim, const int* len, const int* sym);

void factorize(int n, int *nfactor, int **factor);

inline
int gcd(int a, int b){
  if (b==0) return a;
  return gcd(b, a%b);
}

inline
int lcm(int a, int b){
  return a*b/gcd(a,b);
}

/**
 * \brief Copies submatrix to submatrix (column-major)
 * \param[in] nrow number of rows
 * \param[in] ncol number of columns
 * \param[in] lda_A lda along rows for A
 * \param[in] lda_B lda along rows for B
 * \param[in] A matrix to read from
 * \param[in,out] B matrix to write to
 */
template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B){
  if (lda_A == nrow && lda_B == nrow){
    memcpy(B,A,nrow*ncol*sizeof(dtype));
  } else {
    int i;
    for (i=0; i<ncol; i++){
      memcpy(B+lda_B*i,A+lda_A*i,nrow*sizeof(dtype));
    }
  }
}

/**
 * \brief Copies submatrix to submatrix with scaling (column-major)
 * \param[in] nrow number of rows
 * \param[in] ncol number of columns
 * \param[in] lda_A lda along rows for A
 * \param[in] lda_B lda along rows for B
 * \param[in] A matrix to read from
 * \param[in,out] B matrix to write to
 * \param[in] a factor to scale A
 * \param[in] b factor to scale B
 */
template<typename dtype>
void lda_cpy(const int nrow,  const int ncol,
             const int lda_A, const int lda_B,
             const dtype *A,        dtype *B,
             const dtype a,  const dtype b){
  int i,j;
  if (lda_A == nrow && lda_B == nrow){
    for (j=0; j<nrow*ncol; j++){
      B[j] = B[j]*b + A[j]*a;
    }
  } else {
    for (i=0; i<ncol; i++){
      for (j=0; j<nrow; j++){
        B[lda_B*i + j] = B[lda_B*i + j]*b + A[lda_A*i + j]*a;
      }
    }
  }
}

/**
 * \brief we receive a contiguous buffer kb-by-n B and (k-kb)-by-n B_aux
 * which is the block below.
 * To get a k-by-n buffer, we need to combine this buffer with our original
 * block. Since we are working with column-major ordering we need to interleave
 * the blocks. Thats what this function does.
 * \param[in,out] B the buffer to coalesce into
 * \param[in] B_aux the second buffer to coalesce from
 * \param[in] k the total number of rows
 * \param[in] n the number of columns
 * \param[in] kb the number of rows in a B originally
 */
template<typename dtype>
void coalesce_bwd(dtype         *B,
                  dtype const   *B_aux,
                  int const     k,
                  int const     n,
                  int const     kb){
  int i;
  for (i=n-1; i>=0; i--){
    memcpy(B+i*k+kb, B_aux+i*(k-kb), (k-kb)*sizeof(dtype));
    if (i>0) memcpy(B+i*k, B+i*kb, kb*sizeof(dtype));
  }
}


/* Copies submatrix to submatrix */
template<typename dtype>
void transp(const int size,  const int lda_i, const int lda_o,
            const dtype *A, dtype *B){
  if (lda_i == 1){
    memcpy(B,A,size*sizeof(dtype));
  }
  int i,j,o;
  LIBT_ASSERT(size%lda_o == 0);
  LIBT_ASSERT(lda_o%lda_i == 0);
  for (o=0; o<size/lda_o; o++){
    for (j=0; j<lda_i; j++){
      for (i=0; i<lda_o/lda_i; i++){
        B[o*lda_o + j*lda_o/lda_i + i] = A[o*lda_o+j+i*lda_i];
      }
    }
  }
}

template<typename dtype>
dtype get_zero(){
  ABORT;
}
template<typename dtype>
dtype get_one(){
  ABORT;
}


template<> inline
double get_zero<double>() { return 0.0; }

template<> inline
std::complex<double> get_zero< std::complex<double> >() { return std::complex<double>(0.0,0.0); }

template<> inline
double get_one<double>() { return 1.0; }

template<> inline
std::complex<double> get_one< std::complex<double> >() { return std::complex<double>(1.0,0.0); }


template <> inline
void cxgemm<double>(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const double a,
            const double * A,   const int lda,
            const double * B,   const int ldb,
            const double b,     double * C,
                                const int ldc){
  cdgemm(transa, transb, m, n, k, a, A, lda, B, ldb, b, C, ldc);
}

template <> inline
void cxgemm< std::complex<double> >(const char transa,  const char transb,
            const int m,        const int n,
            const int k,        const std::complex<double> a,
            const std::complex<double> * A,     const int lda,
            const std::complex<double> * B,     const int ldb,
            const std::complex<double> b,       std::complex<double> * C,
                                const int ldc){
  czgemm(transa, transb, m, n, k, a, A, lda, B, ldb, b, C, ldc);
}

template <> inline
void cxaxpy<double>(const int n,        double dA,
                    const double * dX,  const int incX,
                    double * dY,        const int incY){
  cdaxpy(n, dA, dX, incX, dY, incY);
}

template <> inline
void cxaxpy< std::complex<double> >
                    (const int n,
                     std::complex<double> dA,
                     const std::complex<double> * dX,
                     const int incX,
                     std::complex<double> * dY,
                     const int incY){
  czaxpy(n, dA, dX, incX, dY, incY);
}



#endif
