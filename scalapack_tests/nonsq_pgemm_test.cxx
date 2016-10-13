/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/



#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <math.h>
#include "ctf.hpp"

using namespace CTF;

#define ZGEMM_TEST
#ifdef ZGEMM_TEST
typedef std::complex<double> VAL_TYPE;
#else
typedef double VAL_TYPE;
#endif
//#define TESTPAD

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
  int64_t m, k, n;
  int nprow, npcol, alloc_host_buf;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

/*  if (argc < 3 || argc > 4) {
    if (myRank == 0) printf("%s [log2_matrix_dimension] [log2_block_dimension] [number of iterations]\n",
                               argv[0]);
    //printf("%s [matrix_dimension_X] [matrix_dimension_Y] [block_dimension_X] [block_dimension_Y]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }*/

  if (argc != 1 && argc >= 8) {
    if (myRank == 0) printf("%s [m] [n] [k] [nprow] [niter] [alloc_host_buf]\n",
                               argv[0]);
    //printf("%s [matrix_dimension_X] [matrix_dimension_Y] [block_dimension_X] [block_dimension_Y]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  if (argc <= 4){
    m = 400;
    n = 400;
    k = 800;
    if (sqrt(numPes)*sqrt(numPes) == numPes){
      nprow = sqrt(numPes);
      npcol = sqrt(numPes);
    } else {
      nprow = numPes;
      npcol = 1;
    }
  }
  if (argc > 4){
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
    nprow = atoi(argv[4]);
    npcol = numPes/nprow;
    assert(numPes == nprow*npcol);
  }

  int num_iter;
  if (argc > 5) num_iter = atoi(argv[5]);
  else num_iter = NUM_ITER;
  
  if (argc > 6) alloc_host_buf = atoi(argv[6]);
  else alloc_host_buf = 1;

#ifdef ZGEMM_TEST
  std::complex<double> ALPHA = std::complex<double>(.3,0.);
  std::complex<double> BETA =std::complex<double>(.7,0.); 
#else
  double ALPHA = 0.3;
  double BETA = .7;
#endif
  double DALPHA = 0.3;
  double DBETA = .7;



  if (myRank == 0){ 
    printf("matrix multiplication of matrices\n");
    printf("matrix dimensions are %ld by %ld by %ld\n", m, n, k);
    printf("processor grid is %d by %d\n", nprow, npcol);
    printf("performing %d iterations\n", num_iter);
    printf("with random data\n");
    printf("alloc_host_buf = %d\n",alloc_host_buf);
//    printf("ALPHA = %lf, BETA = %lf\n",ALPHA,BETA);
  }

  int iter, i, j;
  int64_t nblk, mblk;
  int64_t fnblk, fmblk, fkblk;

//#define TESTPAD
#ifdef TESTPAD
  nblk = n/npcol+4;
  mblk = m/std::min(nprow,npcol);
  fnblk = n/npcol+4;
  fmblk = m/std::min(nprow,npcol);
  fkblk = k/nprow;
#else
  nblk = n/npcol;
  mblk = m/std::min(nprow,npcol);
  fnblk = n/npcol;
  fmblk = m/std::min(nprow,npcol);
  fkblk = k/nprow;
#endif

#ifdef OFFLOAD
  VAL_TYPE * mat_A;
  VAL_TYPE * mat_B; 
  VAL_TYPE * mat_C;
  VAL_TYPE * mat_C_CTF;
 
  if (alloc_host_buf){
    host_pinned_alloc((void**)&mat_A,fkblk*fmblk*sizeof(VAL_TYPE));
    host_pinned_alloc((void**)&mat_B,fnblk*fkblk*sizeof(VAL_TYPE));
    host_pinned_alloc((void**)&mat_C,nblk*mblk*sizeof(VAL_TYPE));
    host_pinned_alloc((void**)&mat_C_CTF,nblk*mblk*sizeof(VAL_TYPE));
  } else {
    mat_A = (VAL_TYPE*)malloc((fkblk*fmblk*sizeof(VAL_TYPE)));
    mat_B = (VAL_TYPE*)malloc((fnblk*fkblk*sizeof(VAL_TYPE)));
    mat_C = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));
    mat_C_CTF = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));
  }
#else
  VAL_TYPE * mat_A = (VAL_TYPE*)malloc((fkblk*fmblk*sizeof(VAL_TYPE)));
  VAL_TYPE * mat_B = (VAL_TYPE*)malloc((fnblk*fkblk*sizeof(VAL_TYPE)));
  VAL_TYPE * mat_C = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));
  VAL_TYPE * mat_C_CTF = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));
#endif

  VAL_TYPE ans_verify;

  srand48(13*myRank);
  for (i=0; i<fkblk; i++){
    for (j=0; j<fmblk; j++){
#ifdef ZGEMM_TEST
      mat_A[i*(fmblk)+j].real(drand48());
      mat_A[i*(fmblk)+j].imag(drand48());
#else
      mat_A[i*(fmblk)+j] = drand48();
#endif
    }
  }
  
  for (i=0; i<fnblk; i++){
    for (j=0; j<fkblk; j++){
#ifdef ZGEMM_TEST
      if (i>=n){
        mat_B[i*(fkblk)+j].real(0.0);
        mat_B[i*(fkblk)+j].imag(0.0);
      } else {                            
        mat_B[i*(fkblk)+j].real(drand48());
        mat_B[i*(fkblk)+j].imag(drand48());
      }
#else
      mat_B[i*(fkblk)+j] = drand48();
#endif
    }
  }
  
  for (i=0; i<nblk; i++){
    for (j=0; j<mblk; j++){
#ifdef ZGEMM_TEST
      if (i>=n){
        mat_C[i*(mblk)+j].real(0.0);
        mat_C[i*(mblk)+j].imag(0.0);
      } else {
        mat_C[i*(mblk)+j].real(drand48());
        mat_C[i*(mblk)+j].imag(drand48());
      }
#else
      mat_C[i*(mblk)+j] = drand48();
#endif
      mat_C_CTF[i*(mblk)+j] = mat_C[i*(mblk)+j];
    }
  }

  /* myCol and myRow swapped in the below two lines since
     Scalapack wants column-major */
/*  for (i=0; i < blockDim; i++){
    for (j=0; j < blockDim; j++){
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
  }*/


  int icontxt, info;
  int iam, inprocs;
  Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(-1, 0, &icontxt);
  char cC = 'C';
  Cblacs_gridinit(&icontxt, &cC, nprow, npcol);

  int desc_a[9];
  int desc_b[9];
  int desc_c[9];
  cdesc_init(desc_a, k, m,
             fkblk, fmblk,
             0,  0,
             icontxt, fkblk, 
             &info);
  assert(info==0);
  cdesc_init(desc_b, k, n,
             fkblk, fnblk,
             0,  0,
             icontxt, fkblk, 
             &info);
  assert(info==0);
  cdesc_init(desc_c, m, n,
             mblk, nblk,
             0,  0,
             icontxt, mblk, 
             &info);
  assert(info==0);
  
  World dw(MPI_COMM_WORLD);

#ifdef ZGEMM_TEST
  double scalaTime = MPI_Wtime();
  cpzgemm('T','N', m, n, k, ALPHA, 
          mat_A, 1, 1, desc_a,
          mat_B, 1, 1, desc_b, BETA,
          mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pzgemm in %lf sec, starting CTF pzgemm\n", MPI_Wtime()-scalaTime);

  Matrix< std::complex<double> > A(desc_a, mat_A, dw);
  Matrix< std::complex<double> > B(desc_b, mat_B, dw);
  Matrix< std::complex<double> > C(desc_c, mat_C_CTF, dw);

  double ctfTime = MPI_Wtime();
  (DBETA*C["ij"])+=DALPHA*A["ki"]*B["kj"];
  if (myRank == 0)
    printf("Performed CTF pzgemm in %lf sec\n", MPI_Wtime()-ctfTime);

  C.read_mat(desc_c, mat_C_CTF);

/*  myctf->pgemm('T','N', m, n, k, ALPHA, 
              mat_A, 1, 1, desc_a,
              mat_B, 1, 1, desc_b, BETA,
              mat_C_CTF, 1, 1, desc_c); */

  if (myRank < npcol*npcol){
    for (i=0; i<mblk; i++){
      for (j=0; j<nblk; j++){
        if (fabs(mat_C[i*(nblk)+j].real()-mat_C_CTF[i*(nblk)+j].real()) > 1.E-6 && 
            fabs(mat_C[i*(nblk)+j].imag()-mat_C_CTF[i*(nblk)+j].imag()) > 1.E-6){
          printf("[%d] incorrect answer %lf %lf at C[%d,%d]=%lf,%lf\n",
                  myRank, mat_C_CTF[i*(nblk)+j].real(),
                  mat_C_CTF[i*(nblk)+j].imag(), i,j,
                  mat_C[i*(nblk)+j].real(), mat_C[i*(nblk)+j].imag());
        }
      }
    }
  }
  if (myRank == 0)
    printf("PZGEMM Verification complete, correct if no errors.\n");
                                    
  double startTime, endTime;
  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    cpzgemm('T','N', m, n, k, ALPHA, 
            mat_A, 1, 1, desc_a,
            mat_B, 1, 1, desc_b, BETA,
            mat_C, 1, 1, desc_c); 
    if (iter == 0)
      ans_verify = mat_C[2];
  }


#else
  CTF * myctf = new CTF;
  myctf->init(MPI_COMM_WORLD,  myRank,numPes);
  cpdgemm('T','N', m, n, k, ALPHA, 
          mat_A, 1, 1, desc_a,
          mat_B, 1, 1, desc_b, BETA,
          mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pdgemm, starting CTF pdgemm\n");

  myctf->pgemm('T','N', m, n, k, ALPHA, 
              mat_A, 1, 1, desc_a,
              mat_B, 1, 1, desc_b, BETA,
              mat_C_CTF, 1, 1, desc_c); 

  for (i=0; i<mblk; i++){
    for (j=0; j<nblk; j++){
      if (fabs(mat_C[i*(nblk)+j]-mat_C_CTF[i*(nblk)+j]) > 1.E-6){
        printf("[%d] incorrect answer %lf at C[%d,%d]=%lf\n",
                myRank, mat_C_CTF[i*(nblk)+j], i,j,
                mat_C[i*(nblk)+j]);
      }
    }
  }
  if (myRank == 0)
    printf("PDGEMM Verification complete, correct if no errors.\n");
                                    
  double startTime, endTime;
  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    cpdgemm('T','N', m, n, k, ALPHA, 
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
#ifdef ZGEMM_TEST
    printf("Gigaflops: %f\n", 8.*m*n*k/
                                ((endTime - startTime)/num_iter)*1E-9);
#else
    printf("Gigaflops: %f\n", 2.*m*n*k/
                                ((endTime - startTime)/num_iter)*1E-9);
#endif
    //printf("Ans=%lf\n",ans_verify);
  }
  startTime = MPI_Wtime();
  for (iter=0; iter < num_iter; iter++){
    (DBETA*C["ij"])+=DALPHA*A["ki"]*B["kj"];
    //seq_square_matmul(mat_A, mat_B, mat_C, blockDim, 0);
    /*myctf->pgemm( 'T','N', m, n, k, ALPHA, 
                  mat_A, 1, 1, desc_a,
                  mat_B, 1, 1, desc_b, BETA,
                  mat_C, 1, 1, desc_c); 
    if (iter == 0)
      ans_verify = mat_C[2];*/
  }

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %u CTF PGEMM iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/num_iter);
#ifdef ZGEMM_TEST
    printf("Gigaflops: %f\n", 8.*m*n*k/
                                ((endTime - startTime)/num_iter)*1E-9);
#else
    printf("Gigaflops: %f\n", 2.*m*n*k/
                                ((endTime - startTime)/num_iter)*1E-9);
#endif
  }

  MPI_Finalize();
  return 0;
} /* end function main */


