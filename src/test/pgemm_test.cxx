
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



#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
//#include <assert.h>
#include "../dist_tensor/cyclopstf.hpp"
#include "../shared/util.h"

#define ZGEMM_TEST
#ifdef ZGEMM_TEST
typedef std::complex<double> VAL_TYPE;
#else
typedef double VAL_TYPE;
#endif

#define NUM_ITER 5

//proper modulus for 'a' in the range of [-b inf]
#define WRAP(a,b)       ((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#if (defined BGP || defined BGQ)
#define DESCINIT        descinit
#define PDGEMM          pdgemm
#define PZGEMM          pzgemm
#else   
#define DESCINIT        descinit_
#define PDGEMM          pdgemm_
#define PZGEMM          pzgemm_
#endif

extern "C" {
void Cblacs_pinfo(int*,int*);

void Cblacs_get(int,int,int*);

int Cblacs_gridinit(int*,char*,int,int);

void DESCINIT(int *, const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, const int *,
                const int *, int *);

void PDGEMM( char *,     char *,
             int *,      int *,
             int *,      double *,
             double *,   int *,
             int *,      int *,
             double *,   int *,
             int *,      int *,
             double *,  double *,
             int *,      int *,
                                 int *);
void PZGEMM( char *,     char *,
             int *,      int *,
             int *,      std::complex<double> *,
             std::complex<double> *,     int *,
             int *,      int *,
             std::complex<double> *,     int *,
             int *,      int *,
             std::complex<double> *,    std::complex<double> *,
             int *,      int *,
                                 int *);
}

static void cdesc_init(int * desc, 
                      const int m,      const int n,
                      const int mb,     const int nb,
                      const int irsrc,  const int icsrc,
                      const int ictxt,  const int LLD,
                                        int * info){
  DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,
             &ictxt, &LLD, info);
}


static void cpdgemm( char n1,    char n2,
                    int sz1,     int sz2,
                    int sz3,     double ALPHA,
                    double * A,  int ia,
                    int ja,      int * desca,
                    double * B,  int ib,
                    int jb,      int * descb,
                    double BETA,        double * C,
                    int ic,      int jc,
                                         int * descc){
  PDGEMM(&n1, &n2, &sz1, 
         &sz2, &sz3, &ALPHA, 
         A, &ia, &ja, desca, 
         B, &ib, &jb, descb, &BETA,
         C, &ic, &jc, descc);
}
static void cpzgemm(char n1,     char n2,
                    int sz1,     int sz2,
                    int sz3,     std::complex<double> ALPHA,
                    std::complex<double> * A,    int ia,
                    int ja,      int * desca,
                    std::complex<double> * B,    int ib,
                    int jb,      int * descb,
                    std::complex<double> BETA,  std::complex<double> * C,
                    int ic,      int jc,
                                         int * descc){
  PZGEMM(&n1, &n2, &sz1, 
         &sz2, &sz3, &ALPHA, 
         A, &ia, &ja, desca, 
         B, &ib, &jb, descb, &BETA,
         C, &ic, &jc, descc);
}
  
int uint_log2( int n){
   int j;
  for (j=0; n!=1; j++) n>>=1;
  return j;
}

//extern "C"  void pbm();

int main(int argc, char **argv) {
/*void pbm() {

  int argc;
  char **argv;*/
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (argc < 3 || argc > 4) {
    if (myRank == 0) printf("%s [log2_matrix_dimension] [log2_block_dimension] [number of iterations]\n",
                               argv[0]);
    //printf("%s [matrix_dimension_X] [matrix_dimension_Y] [block_dimension_X] [block_dimension_Y]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  const  int log_matrixDim = atoi(argv[1]);
  const  int log_blockDim = atoi(argv[2]);
  const  int matrixDim = 1<<log_matrixDim;
  const  int blockDim = 1<<log_blockDim;

  int num_iter;
  if (argc > 3) num_iter = atoi(argv[3]);
  else num_iter = NUM_ITER;

  double ALPHA = 2.3;
  double BETA = 0.7;



  if (myRank == 0){ 
    printf("MATRIX MULTIPLICATION OF SQUARE MATRICES\n");
    printf("MATRIX DIMENSION IS %d\n", matrixDim);
    printf("BLOCK DIMENSION IS %d\n", blockDim);
    printf("PERFORMING %d ITERATIONS\n", num_iter);
    printf("WITH RANDOM DATA\n");
    printf("ALPHA = %lf, BETA = %lf\n",ALPHA,BETA);
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_X mod block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_Y mod block_size_Y != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  const  int log_num_blocks_dim = log_matrixDim - log_blockDim;
  const  int num_blocks_dim = 1<<log_num_blocks_dim;

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
  }

  if (num_blocks_dim*num_blocks_dim != numPes){
    if (myRank == 0) printf("NUMBER OF BLOCKS MUST BE EQUAL TO NUMBER OF PROCESSORS\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int const myRow = myRank / num_blocks_dim;
  int const myCol = myRank % num_blocks_dim;
  int iter, i, j;

  VAL_TYPE * mat_A = (VAL_TYPE*)malloc(blockDim*blockDim*sizeof(VAL_TYPE));
  VAL_TYPE * mat_B = (VAL_TYPE*)malloc(blockDim*blockDim*sizeof(VAL_TYPE));
  VAL_TYPE * mat_C = (VAL_TYPE*)malloc(blockDim*blockDim*sizeof(VAL_TYPE));
  VAL_TYPE * mat_C_CTF = (VAL_TYPE*)malloc(blockDim*blockDim*sizeof(VAL_TYPE));
  

  VAL_TYPE ans_verify;

//  srand48(13);
  for (i=0; i < blockDim; i++){
    for (j=0; j < blockDim; j++){
  /* myCol and myRow swapped in the below two lines since
     Scalapack wants column-major */
      srand(myCol*matrixDim*blockDim + myRow*blockDim + i*matrixDim+j);
#ifdef ZGEMM_TEST
      mat_A[i*blockDim+j].real() = drand48();
      mat_A[i*blockDim+j].imag() = drand48();
      mat_B[i*blockDim+j].real()= drand48();
      mat_B[i*blockDim+j].imag()= drand48();
      mat_C[i*blockDim+j].real()= drand48();
      mat_C[i*blockDim+j].imag()= drand48();
      mat_C_CTF[i*blockDim+j] = mat_C[i*blockDim+j];
#else
      mat_A[i*blockDim+j] = drand48();
      mat_B[i*blockDim+j] = drand48();
      mat_C[i*blockDim+j] = drand48();
      mat_C_CTF[i*blockDim+j] = mat_C[i*blockDim+j];
#endif
    }
  }

  int icontxt, info;
  int iam, inprocs;
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  char cC = 'C';
  Cblacs_gridinit(&icontxt, &cC, num_blocks_dim, num_blocks_dim);

  int desc_a[9];
  int desc_b[9];
  int desc_c[9];
  cdesc_init(desc_a, matrixDim, matrixDim,
                    blockDim, blockDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
  assert(info==0);
  cdesc_init(desc_b, matrixDim, matrixDim,
                    blockDim, blockDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
  assert(info==0);
  cdesc_init(desc_c, matrixDim, matrixDim,
                    blockDim, blockDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
  assert(info==0);

#ifdef ZGEMM_TEST
  tCTF< std::complex<double> > * myctf = new tCTF< std::complex<double> >;
  myctf->init(MPI_COMM_WORLD, MACHINE_BGQ, myRank,numPes);
  cpzgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
          mat_A, 1, 1, desc_a,
          mat_B, 1, 1, desc_b, BETA,
          mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pzgemm, starting CTF pzgemm\n");

  myctf->pgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
              mat_A, 1, 1, desc_a,
              mat_B, 1, 1, desc_b, BETA,
              mat_C_CTF, 1, 1, desc_c); 

  for (i=0; i<blockDim; i++){
    for (j=0; j<blockDim; j++){
      if (fabs(mat_C[i*blockDim+j].real()-mat_C_CTF[i*blockDim+j].real()) > 1.E-6 && 
          fabs(mat_C[i*blockDim+j].imag()-mat_C_CTF[i*blockDim+j].imag()) > 1.E-6){
        printf("[%d] incorrect answer at C[%d,%d]\n",myRank,i,j);
      }
    }
  }
  if (myRank == 0)
    printf("PZGEMM Verification complete, correct if no errors.\n");
                                    
  double startTime, endTime;
  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    cpzgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
            mat_A, 1, 1, desc_a,
            mat_B, 1, 1, desc_b, BETA,
            mat_C, 1, 1, desc_c); 
    if (iter == 0)
      ans_verify = mat_C[2];
  }


#else
  CTF * myctf = new CTF;
  myctf->init(MPI_COMM_WORLD, MACHINE_BGQ, myRank,numPes);
  cpdgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
          mat_A, 1, 1, desc_a,
          mat_B, 1, 1, desc_b, BETA,
          mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pdgemm, starting CTF pdgemm\n");

  myctf->pgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
              mat_A, 1, 1, desc_a,
              mat_B, 1, 1, desc_b, BETA,
              mat_C_CTF, 1, 1, desc_c); 

  for (i=0; i<blockDim; i++){
    for (j=0; j<blockDim; j++){
      if (fabs(mat_C[i*blockDim+j]-mat_C_CTF[i*blockDim+j]) > 1.E-6){
        printf("[%d] incorrect answer %lf at C[%d,%d] = %lf\n",myRank,mat_C_CTF[i*blockDim+j],i,j,mat_C[i*blockDim+j]);
      }
    }
  }
  if (myRank == 0)
    printf("PDGEMM Verification complete, correct if no errors.\n");
                                    
  double startTime, endTime;
  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    cpdgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
            mat_A, 1, 1, desc_a,
            mat_B, 1, 1, desc_b, BETA,
            mat_C, 1, 1, desc_c); 
    if (iter == 0)
      ans_verify = mat_C[2];
  }
#endif

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %u ScaLAPACK iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/num_iter);
    printf("Gigaflops: %f\n", 2.*matrixDim*matrixDim*matrixDim/
                                ((endTime - startTime)/num_iter)*1E-9);
    //printf("Ans=%lf\n",ans_verify);
  }
  
#ifdef TAU
  TAU_PROFILE_TIMER(timer, "main", "int (int, char**)", TAU_USER);
  TAU_PROFILE_START(timer);
  TAU_PROFILE_INIT(argc, argv);
  TAU_PROFILE_SET_NODE(myRank);
  TAU_PROFILE_SET_CONTEXT(0);
#endif

  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    myctf->pgemm('N','N', matrixDim, matrixDim, matrixDim, ALPHA, 
            mat_A, 1, 1, desc_a,
            mat_B, 1, 1, desc_b, BETA,
            mat_C, 1, 1, desc_c); 
    if (iter == 0)
      ans_verify = mat_C[2];
  }

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %u CTF PGEMM iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/num_iter);
    printf("Gigaflops: %f\n", 2.*matrixDim*matrixDim*matrixDim/
                                ((endTime - startTime)/num_iter)*1E-9);
    //printf("Ans=%lf\n",ans_verify);
  }

  TAU_PROFILE_STOP(timer);


  MPI_Finalize();
  return 0;
} /* end function main */


