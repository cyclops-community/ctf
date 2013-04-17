/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "ctr_comm.h"

#define COMPLEX

#ifdef COMPLEX
#include <complex>
using namespace std;
typedef complex<double> test_dtype;
#else
typedef double test_dtype;
#endif

/**
 * \brief a unit test for multiple recursive algs
 * \param[in] usr_sqr defines alg to test (0 is rect, 1 is square, and 2 is 
                1D virtualized)
 * \param[in] m A is m by k
 * \param[in] n B is k by n
 * \param[in] k A is m by k and B is k by m
 * \param[in] seed runtime set seed
 * \param[in] np_x number of processor in x dimensions
 * \param[in] nlyr number of 2.5D layers
 * \param[in] cdt_glb global communicator
 */
static 
void unit_2d_bcast(int const    use_sqr,
                   int const    m, 
                   int const    n, 
                   int const    k, 
                   int const    seed, 
                   int const    np_x, 
                   int const    nlyr, 
                   CommData     *cdt_glb){
  int myRank, numPes;
  int i,j,ib,jb,virt_x,virt_y,iv;
  int nb,mb,kb_A,kb_B,kX;

  test_dtype * mat_A, * mat_B, * mat_C, * buffer;


  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;

  if (myRank == 0){ 
    printf("TESTING 4D MM CANNON / TOPO BCAST ALG\n");
  }

  const int np_y = (numPes/nlyr)/np_x;

  if (myRank == 0){
    printf("NUM X PROCS IS %d\n", np_x);
    printf("NUM Y PROCS IS %d\n", np_y);
    printf("NUM Z PROCS IS %d\n", nlyr);
  }

  if (np_x*np_y*nlyr != numPes){
    if (myRank == 0) 
      printf("ERROR: PROCESSOR GRID MISMATCH\n");
    ABORT;
  }

  if (n%np_x != 0 || m%np_y != 0 || k%np_x != 0 || k%np_y != 0){
    if (myRank == 0) 
      printf("MATRIX DIMENSIONS MUST BE DIVISBLE BY PROCESSOR GRIDS DIMENSIONS\n");
    ABORT;
  }

  virt_x=1;
  virt_y=1;
  if (use_sqr == 2){
    if (np_x == 1) virt_x = np_y;
    if (np_y == 1) virt_y = np_x;
  }
  
  mb = m/np_y;
  nb = n/np_x;
  kb_A = k/(np_x*virt_x);
  kb_B = k/(np_y*virt_y);

 
  const int nreq = 4;
  const int nbcast = 4;

  CommData_t * cdt_lyr, * cdt_x, * cdt_y, * cdt_z;
  cdt_lyr = (CommData_t*)malloc(sizeof(CommData_t));
  cdt_x = (CommData_t*)malloc(sizeof(CommData_t));
  cdt_y = (CommData_t*)malloc(sizeof(CommData_t));
  cdt_z = (CommData_t*)malloc(sizeof(CommData_t));

  SETUP_SUB_COMM(cdt_glb, cdt_lyr, 
                 myRank%(numPes/nlyr), 
                 myRank/(numPes/nlyr), 
                 numPes/nlyr, nreq, nbcast);
  
  SETUP_SUB_COMM(cdt_glb, cdt_z, 
                 myRank/(numPes/nlyr), 
                 myRank%(numPes/nlyr), 
                 nlyr, nreq, nbcast);

  const int lyr_rank = cdt_lyr->rank;

  SETUP_SUB_COMM(cdt_lyr, cdt_y, 
                 lyr_rank%np_y, 
                 lyr_rank/np_y, 
                 np_y, nreq, nbcast);
  SETUP_SUB_COMM(cdt_lyr, cdt_x, 
                 lyr_rank/np_y, 
                 lyr_rank%np_y, 
                 np_x, nreq, nbcast);

  assert(posix_memalign((void**)&mat_A, 
                        ALIGN_BYTES, 
                        mb*kb_A*virt_x*sizeof(test_dtype)) == 0);
  assert(posix_memalign((void**)&mat_B, 
                        ALIGN_BYTES, 
                        kb_B*nb*virt_y*sizeof(test_dtype)) == 0);
  assert(posix_memalign((void**)&mat_C, 
                        ALIGN_BYTES, 
                        mb*nb*sizeof(test_dtype)) == 0);
  assert(posix_memalign((void**)&buffer, 
                        ALIGN_BYTES, 
                        3*(mb*nb+nb*kb_B+kb_A*mb)*sizeof(test_dtype)) == 0);

  for (iv = 0; iv <virt_x; iv++){
    for (i=0; i<kb_A; i++){
      for (j=0; j<mb; j++){
        srand48(seed + m*((iv+cdt_x->rank)*kb_A+i) + mb*cdt_y->rank+j);
#ifdef COMPLEX
        mat_A[iv*mb*kb_A+i*mb+j].real() = drand48();
        mat_A[iv*mb*kb_A+i*mb+j].imag() = drand48();
#else
        mat_A[iv*mb*kb_A+i*mb+j] = drand48();
#endif
      }
    }
  }

  for (iv = 0; iv <virt_y; iv++){
    for (i=0; i<nb; i++){
      for (j=0; j<kb_B; j++){
        srand48(seed*seed + k*(cdt_x->rank*nb+i) + kb_B*(iv+cdt_y->rank)+j);
#ifdef COMPLEX
        mat_B[iv*nb*kb_B+i*kb_B+j].real() = drand48();
        mat_B[iv*nb*kb_B+i*kb_B+j].imag() = drand48();
#else
        mat_B[iv*nb*kb_B+i*kb_B+j] = drand48();
#endif
      }
    }
  }

  kX = k;

  ctr_1d_sqr_bcast<test_dtype> * c1d = new ctr_1d_sqr_bcast<test_dtype>;
  ctr_2d_sqr_bcast<test_dtype> * c2ds = new ctr_2d_sqr_bcast<test_dtype>;
  ctr_2d_rect_bcast<test_dtype> * c2dr = new ctr_2d_rect_bcast<test_dtype>;
  ctr_2d_general<test_dtype> * c2dgen = new ctr_2d_general<test_dtype>;
  ctr_lyr<test_dtype> * clyr = new ctr_lyr<test_dtype>;
  ctr_dgemm<test_dtype> * cdg = new ctr_dgemm<test_dtype>;

  clyr->A       = mat_A;
  clyr->B       = mat_B;
  clyr->C       = mat_C;
  clyr->idx_lyr = 0;
  clyr->num_lyr = 1;
  clyr->beta    = 0.0;
  
  if (use_sqr== 3){
    c2dgen->cdt_A       = cdt_x;
    c2dgen->cdt_B       = cdt_y;
    c2dgen->cdt_C       = NULL;
    c2dgen->rec_ctr     = cdg;
    c2dgen->buffer      = NULL;
    c2dgen->ctr_lda_A   = 1;
    c2dgen->ctr_lda_B   = nb;
    c2dgen->ctr_lda_C   = 1;
    c2dgen->ctr_sub_lda_A = mb;
    c2dgen->ctr_sub_lda_B = 1;
    c2dgen->ctr_sub_lda_C = 0;
    c2dgen->edge_len    = kX;
    clyr->rec_ctr       = c2dgen;
    kX                  = kX / (MAX(cdt_x->np,cdt_y->np));
//    ctr_2d_rect_bcast((void*)&rect_args, &gen_args);
  } else if (use_sqr== 2){
    if (cdt_x->np == 1){
      c1d->cdt          = cdt_y;
      c1d->cdt_dir      = 1;
      c1d->sz           = kb_B*nb;
      c1d->ctr_sub_lda  = kb_A*mb;
      c1d->ctr_lda      = 1;
    } else {
      c1d->cdt          = cdt_x;
      c1d->cdt_dir      = 0;
      c1d->sz           = mb*kb_A;
      c1d->ctr_sub_lda  = kb_B*nb;
      c1d->ctr_lda      = 1;
    }
    c1d->rec_ctr        = cdg;
    c1d->buffer         = buffer+nb*mb;
    c1d->k              = kX;
    clyr->rec_ctr       = c1d;
    kX                  = kX / (MAX(cdt_x->np,cdt_y->np));
//    ctr_2d_sqr_bcast((void*)&sqr_args, &gen_args);
  } else if (use_sqr){
    c2ds->cdt_x                 = cdt_x;
    c2ds->cdt_y                 = cdt_y;
    c2ds->buffer        = buffer+nb*mb;
    c2ds->rec_ctr       = cdg;
    c2ds->sz_A          = mb*kb_A;
    c2ds->sz_B          = kb_B*nb;
    c2ds->k             = kX;
    clyr->rec_ctr       = c2ds;
    kX                  = kX/cdt_x->np;
//    ctr_2d_sqr_bcast((void*)&sqr_args, &gen_args);
  } else {
    c2dr->cdt_x         = cdt_x;
    c2dr->cdt_y         = cdt_y;
    c2dr->rec_ctr       = cdg;
    c2dr->buffer        = buffer+nb*mb;
    c2dr->ctr_lda_A     = 1;
    c2dr->ctr_lda_B     = nb;
    c2dr->ctr_sub_lda_A = mb;
    c2dr->ctr_sub_lda_B = 1;
    c2dr->k             = kX;
    clyr->rec_ctr       = c2dr;
    kX                  = kX / (MAX(cdt_x->np,cdt_y->np));
//    ctr_2d_rect_bcast((void*)&rect_args, &gen_args);
  }
  cdg->transp_A = 'n';
  cdg->transp_B = 'n';
  cdg->alpha    = 1.0;
  cdg->n        = nb;
  cdg->m        = mb;
  cdg->k        = kX;

  clyr->buffer  = buffer;
  clyr->sz_C    = nb*mb;
  clyr->cdt     = cdt_z;
      
  clyr->run();

  free(mat_A); 
  free(mat_B); 
  free(buffer); 
  
  assert(posix_memalign((void**)&mat_A, 
                        ALIGN_BYTES, 
                        m*k*sizeof(test_dtype)) == 0);
  assert(posix_memalign((void**)&mat_B, 
                        ALIGN_BYTES, 
                        k*n*sizeof(test_dtype)) == 0);
  assert(posix_memalign((void**)&buffer, 
                        ALIGN_BYTES, 
                        m*n*sizeof(test_dtype)) == 0);
 
 

  for (i=0; i<k; i++){
    for (j=0; j<m; j++){
      srand48(seed + m*i + j);
#ifdef COMPLEX
      mat_A[i*m+j].real() = drand48();
      mat_A[i*m+j].imag() = drand48();
#else
      mat_A[i*m+j] = drand48();
#endif
    }
  }

  for (i=0; i<n; i++){
    for (j=0; j<k; j++){
      srand48(seed*seed + k*i + j);
#ifdef COMPLEX
      mat_B[i*k+j].real() = drand48();
      mat_B[i*k+j].imag() = drand48();
#else
      mat_B[i*k+j] = drand48();
#endif
    }
  }

#ifdef COMPLEX
  czgemm('n','n',m,n,k,1.0,mat_A,m,mat_B,k,0.0,buffer,m);
#else
  cdgemm('n','n',m,n,k,1.0,mat_A,m,mat_B,k,0.0,buffer,m);
#endif
#ifdef VERBOSE
  GLOBAL_BARRIER(cdt_glb);
  if (cdt_glb->rank == 0){
    printf("\n");
    print_matrix(mat_A,m,k);
    printf("\n");
    print_matrix(mat_B,k,n);
    printf("\n");
    print_matrix(buffer,m,n);
    printf("\n");
    print_matrix(mat_C,mb,nb);
    printf("\n");
  }
  GLOBAL_BARRIER(cdt_glb);
#endif
  
  bool pass = true;
  bool global_pass;
  for (i=0; i<nb; i++){
    for (j=0; j<mb; j++){
      ib = cdt_x->rank*nb+i;
      jb = cdt_y->rank*mb+j;
#ifdef COMPLEX
      if (fabs(buffer[ib*m+jb].real()-mat_C[i*mb+j].real()) > 1.E-6
          || fabs(buffer[ib*m+jb].imag()-mat_C[i*mb+j].imag()) > 1.E-6){
#else
      if (fabs(buffer[ib*m+jb]-mat_C[i*mb+j]) > 1.E-6){
#endif
        pass = false;
        DEBUG_PRINTF("[%d] mat_C[%d,%d]=%lf should have been %lf\n", 
                      lyr_rank,jb,ib,mat_C[i*mb+j],buffer[ib*m+jb]);
      }
    }
  }
  REDUCE(&pass, &global_pass, 1, COMM_CHAR_T, COMM_OP_BAND, 0, cdt_glb);
  if (cdt_glb->rank == 0) { 
    if (global_pass){
      printf("[%d] UNIT 2D BCAST TEST PASSED\n",cdt_glb->rank);
    } else {     
      printf("[%d] !!! UNIT 2D BCAST TEST FAILED!!!\n",cdt_glb->rank);
    }
  }

  FREE_CDT((cdt_x));
  FREE_CDT((cdt_y));
  FREE_CDT((cdt_lyr));
} 

/* Defines elsewhere deprecate */
static
char* getCmdOption(char ** begin, char ** end, const std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

/**
 * \brief main for a unit test of recurive contraction functions
 */
int main(int argc, char **argv) {
  int myRank, numPes, m, n, k, seed, npx, nlyr;

  CommData_t *cdt_glb = (CommData_t*)malloc(sizeof(CommData_t));
  RINIT_COMM(numPes, myRank, 4, 4, cdt_glb);

  if (getCmdOption(argv, argv+argc, "-n")){
    n = atoi(getCmdOption(argv, argv+argc, "-n"));
    if (n <= 0) n = 128;
  } else n = 128;
  if (getCmdOption(argv, argv+argc, "-k")){
    k = atoi(getCmdOption(argv, argv+argc, "-k"));
    if (k <= 0) k = 64;
  } else k = 64;
  if (getCmdOption(argv, argv+argc, "-m")){
    m = atoi(getCmdOption(argv, argv+argc, "-m"));
    if (m <= 0) m = 32;
  } else m = 32;
  if (getCmdOption(argv, argv+argc, "-seed")){
    seed = atoi(getCmdOption(argv, argv+argc, "-seed"));
    if (seed <= 0) seed = 1000;
  } else seed = 1000;
  if (getCmdOption(argv, argv+argc, "-npx")){
    npx = atoi(getCmdOption(argv, argv+argc, "-npx"));
    if (npx <= 0) npx = 1;
  } else npx = 1;
  assert(numPes%npx == 0);
  if (getCmdOption(argv, argv+argc, "-nlyr")){
    nlyr = atoi(getCmdOption(argv, argv+argc, "-nlyr"));
    if (nlyr <= 0) nlyr = 1;
  } else nlyr = 1;

  if (myRank == 0) {
    printf("-m=%d -n=%d -k=%d -seed = %d -npx = %d -nlyr = %d\n",m,n,k,seed,npx,nlyr);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif
  }

  GLOBAL_BARRIER(cdt_glb);
  if (myRank == 0) printf("TESTING RECTANGULAR ALGORITHM\n");
  unit_2d_bcast(0, n, m, k, seed, npx, nlyr, cdt_glb);
  if (myRank == 0) printf("TESTING GENERAL ALGORITHM\n");
  unit_2d_bcast(3, n, m, k, seed, npx, nlyr, cdt_glb);
  GLOBAL_BARRIER(cdt_glb);
  if ((numPes/nlyr)/npx == npx){
    if (myRank == 0) printf("TESTING SQUARE ALGORITHM\n");
    GLOBAL_BARRIER(cdt_glb);
    unit_2d_bcast(1, n, m, k, seed, npx, nlyr, cdt_glb);
  }
  GLOBAL_BARRIER(cdt_glb);
  if (npx == 1 || (numPes/nlyr)/npx == 1){
    if (myRank == 0) printf("TESTING SQUARE VIRT ALGORITHM\n");
    GLOBAL_BARRIER(cdt_glb);
    unit_2d_bcast(2, n, m, k, seed, npx, nlyr, cdt_glb);
  }
  GLOBAL_BARRIER(cdt_glb);
  COMM_EXIT;
  return 0;
}

