/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/



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
#define WRAP(a,b)	((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
#if (defined BGP || defined BGQ)
#define DESCINIT 	descinit
#define PDGEMM		pdgemm
#define PZGEMM		pzgemm
#else	
#define DESCINIT 	descinit_
#define PDGEMM		pdgemm_
#define PZGEMM		pzgemm_
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

void PDGEMM( char *,	 char *,
	     int *,	 int *,
	     int *,	 double *,
	     double *,	 int *,
	     int *,	 int *,
	     double *,	 int *,
	     int *,	 int *,
	     double *,	double *,
	     int *,	 int *,
				 int *);
void PZGEMM( char *,	 char *,
	     int *,	 int *,
	     int *,	 std::complex<double> *,
	     std::complex<double> *,	 int *,
	     int *,	 int *,
	     std::complex<double> *,	 int *,
	     int *,	 int *,
	     std::complex<double> *,	std::complex<double> *,
	     int *,	 int *,
				 int *);
}

static void cdesc_init(int * desc, 
		      const int m,	const int n,
		      const int mb, 	const int nb,
		      const int irsrc,	const int icsrc,
		      const int ictxt,	const int LLD,
					int * info){
  DESCINIT(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,
	     &ictxt, &LLD, info);
}


static void cpdgemm( char n1,	 char n2,
		    int sz1,	 int sz2,
		    int sz3,	 double ALPHA,
		    double * A,	 int ia,
		    int ja,	 int * desca,
		    double * B,	 int ib,
		    int jb,	 int * descb,
		    double BETA,	double * C,
		    int ic,	 int jc,
					 int * descc){
  PDGEMM(&n1, &n2, &sz1, 
	 &sz2, &sz3, &ALPHA, 
	 A, &ia, &ja, desca, 
	 B, &ib, &jb, descb, &BETA,
	 C, &ic, &jc, descc);
}
static void cpzgemm(char n1,	 char n2,
		    int sz1,	 int sz2,
		    int sz3,	 std::complex<double> ALPHA,
		    std::complex<double> * A,	 int ia,
		    int ja,	 int * desca,
		    std::complex<double> * B,	 int ib,
		    int jb,	 int * descb,
		    std::complex<double> BETA,	std::complex<double> * C,
		    int ic,	 int jc,
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
  int m, k, n;
  int nprow, npcol;
  int tid_A, tid_B, tid_C;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);


  if (argc != 1 && argc >= 7) {
    if (myRank == 0) printf("%s [m] [n] [k] [nprow] [niter]\n",
			       argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  if (argc <= 4){
    m = 400;
    n = 300;
    k = 500;
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

#ifdef ZGEMM_TEST
  std::complex<double> ALPHA = std::complex<double>(2.0,-.7);
  std::complex<double> BETA =std::complex<double>(1.3,.4); 
#else
  double ALPHA = 0.3;
  double BETA = .7;
#endif



  if (myRank == 0){ 
    printf("MATRIX MULTIPLICATION OF MATRICES\n");
    printf("MATRIX DIMENSIONS ARE %d by %d by %d\n", m, n, k);
    printf("PROCESSOR GRID IS %d by %d\n", nprow, npcol);
    printf("PERFORMING %d ITERATIONS\n", num_iter);
    printf("WITH RANDOM DATA\n");
//    printf("ALPHA = %lf, BETA = %lf\n",ALPHA,BETA);
  }

  int iter, i, j, nblk, mblk;

  nblk = n/npcol;
  mblk = m/nprow;

  VAL_TYPE * mat_A = (VAL_TYPE*)malloc(k*m*sizeof(VAL_TYPE)/numPes);
  VAL_TYPE * mat_B = (VAL_TYPE*)malloc(n*k*sizeof(VAL_TYPE)/numPes);
  VAL_TYPE * mat_C = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));
  VAL_TYPE * mat_C_CTF = (VAL_TYPE*)malloc(nblk*mblk*sizeof(VAL_TYPE));

  VAL_TYPE ans_verify;

  srand48(13*myRank);
  for (i=0; i<k/npcol; i++){
    for (j=0; j<m/nprow; j++){
#ifdef ZGEMM_TEST
      mat_A[i*(m/nprow)+j].real() = drand48();
      mat_A[i*(m/nprow)+j].imag() = drand48();
#else
      mat_A[i*(m/nprow)+j] = drand48();
#endif
    }
  }
  
  for (i=0; i<n/npcol; i++){
    for (j=0; j<k/nprow; j++){
#ifdef ZGEMM_TEST
      mat_B[i*(k/nprow)+j].real() = drand48();
      mat_B[i*(k/nprow)+j].imag() = drand48();
#else
      mat_B[i*(k/nprow)+j] = drand48();
#endif
    }
  }
  
  for (i=0; i<nblk; i++){
    for (j=0; j<mblk; j++){
#ifdef ZGEMM_TEST
      mat_C[i*(mblk)+j].real() = drand48();
      mat_C[i*(mblk)+j].imag() = drand48();
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
		    k/nprow, m/npcol,
		    0,	0,
		    icontxt, k/nprow, 
				 &info);
  assert(info==0);
  cdesc_init(desc_b, k, n,
		    k/nprow, n/npcol,
		    0,	0,
		    icontxt, k/nprow, 
				 &info);
  assert(info==0);
  cdesc_init(desc_c, m, n,
		    mblk, nblk,
		    0,	0,
		    icontxt, mblk, 
				 &info);
  assert(info==0);

#ifdef ZGEMM_TEST
  tCTF< std::complex<double> > * myctf = new tCTF< std::complex<double> >;
  myctf->init(MPI_COMM_WORLD,myRank,numPes,MACHINE_BGQ);

  cpzgemm('T','N', m, n, k, ALPHA, 
	  mat_A, 1, 1, desc_a,
	  mat_B, 1, 1, desc_b, BETA,
	  mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pzgemm, starting CTF pzgemm\n");

/*  myctf->pgemm('T','N', m, n, k, ALPHA, 
	      mat_A, 1, 1, desc_a,
	      mat_B, 1, 1, desc_b, BETA,
	      mat_C_CTF, 1, 1, desc_c); */
  myctf->def_scala_mat(desc_a, mat_A, &tid_A);
  myctf->def_scala_mat(desc_b, mat_B, &tid_B);
  myctf->def_scala_mat(desc_c, mat_C_CTF, &tid_C);
  myctf->pgemm('T', 'N', m, n, k, ALPHA, tid_A, tid_B, BETA, tid_C);
  myctf->read_scala_mat(tid_C, mat_C_CTF);

#if 0
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
#endif
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
  myctf->init(MPI_COMM_WORLD,MACHINE_BGQ,myRank,numPes);
  cpdgemm('N','N', m, n, k, ALPHA, 
	  mat_A, 1, 1, desc_a,
	  mat_B, 1, 1, desc_b, BETA,
	  mat_C, 1, 1, desc_c); 

  if (myRank == 0)
    printf("Performed ScaLAPACK pdgemm, starting CTF pdgemm\n");

  /*myctf->pgemm('N','N', m, n, k, ALPHA, 
	      mat_A, 1, 1, desc_a,
	      mat_B, 1, 1, desc_b, BETA,
	      mat_C_CTF, 1, 1, desc_c); */
  myctf->def_scala_mat(desc_a, mat_A, &tid_A);
  myctf->def_scala_mat(desc_b, mat_B, &tid_B);
  myctf->def_scala_mat(desc_c, mat_C_CTF, &tid_C);
  myctf->pgemm('T', 'N', m, n, k, ALPHA, tid_A, tid_B, BETA, tid_C);
  myctf->read_scala_mat(tid_C, mat_C_CTF);


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
    cpdgemm('N','N', m, n, k, ALPHA, 
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
    printf("Gigaflops: %f\n", 2.*m*n*k/
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
    /*myctf->pgemm('T','N', m, n, k, ALPHA, 
	    mat_A, 1, 1, desc_a,
	    mat_B, 1, 1, desc_b, BETA,
	    mat_C, 1, 1, desc_c); */
    myctf->pgemm('T', 'N', m, n, k, ALPHA, tid_A, tid_B, BETA, tid_C);
    if (iter == 0)
      ans_verify = mat_C[2];
  }

  if(myRank == 0) {
    endTime = MPI_Wtime();
    printf("Completed %u CTF PGEMM iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", (endTime - startTime)/num_iter);
    printf("Gigaflops: %f\n", 2.*m*n*k/
				((endTime - startTime)/num_iter)*1E-9);
    //printf("Ans=%lf\n",ans_verify);
  }

  TAU_PROFILE_STOP(timer);

  myctf->exit();

  MPI_Finalize();
  return 0;
} /* end function main */


