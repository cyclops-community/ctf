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

#include "../shared/util.h"
#include "ctr_comm.h"

/**
 * \brief a bench test for multiple recursive algs
 * \param[in] usr_sqr defines alg to test (0 is rect, 1 is square, and 2 is 
                1D virtualized)
 * \param[in] m A is m by k
 * \param[in] n B is k by n
 * \param[in] k A is m by k and B is k by m
 * \param[in] seed runtime set seed
 * \param[in] niter is then number of iterations to test
 * \param[in] np_x number of processor in x dimensions
 * \param[in] nlyr number of 2.5D layers
 * \param[in] cdt_glb global communicator
 */
static 
void bench_2d_bcast(int const   use_sqr,
                   int const    m, 
                   int const    n, 
                   int const    k, 
                   int const    seed, 
                   int const    niter, 
                   int const    np_x, 
                   int const    nlyr, 
                   CommData     *cdt_glb){
  int myRank, numPes;
  int i,j,ib,jb,virt_x,virt_y,iv;
  int nb,mb,kb_A,kb_B,kX;

  double * mat_A, * mat_B, * mat_C, * buffer;


  myRank = cdt_glb->rank;
  numPes = cdt_glb->np;

  if (myRank == 0){ 
    printf("BENCHMARKING 4D MM CANNON / TOPO BCAST ALG\n");
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
                        mb*kb_A*virt_x*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_B, 
                        ALIGN_BYTES, 
                        kb_B*nb*virt_y*sizeof(double)) == 0);
  assert(posix_memalign((void**)&mat_C, 
                        ALIGN_BYTES, 
                        mb*nb*sizeof(double)) == 0);
  assert(posix_memalign((void**)&buffer, 
                        ALIGN_BYTES, 
                        3*(mb*nb+nb*kb_B+kb_A*mb)*sizeof(double)) == 0);

  srand48(seed);
  for (iv = 0; iv <virt_x; iv++){
    for (i=0; i<kb_A; i++){
      for (j=0; j<mb; j++){
        mat_A[iv*mb*kb_A+i*mb+j] = drand48();
      }
    }
  }

  for (iv = 0; iv <virt_y; iv++){
    for (i=0; i<nb; i++){
      for (j=0; j<kb_B; j++){
        mat_B[iv*nb*kb_B+i*kb_B+j] = drand48();
      }
    }
  }

  kX = k;

  ctr_1d_sqr_bcast * c1d = new ctr_1d_sqr_bcast;
  ctr_2d_sqr_bcast * c2ds = new ctr_2d_sqr_bcast;
  ctr_2d_rect_bcast * c2dr = new ctr_2d_rect_bcast;
  ctr_lyr * clyr = new ctr_lyr;
  ctr_dgemm * cdg = new ctr_dgemm;

  clyr->A       = mat_A;
  clyr->B       = mat_B;
  clyr->C       = mat_C;
  clyr->idx_lyr = 0;
  clyr->num_lyr = 1;
  clyr->beta    = 0.0;
  
  if (use_sqr== 2){
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
    c1d->buffer                 = buffer+nb*mb;
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
  
      
  GLOBAL_BARRIER(cdt_glb);
  double str_time, end_time;
  str_time = TIME_SEC(); 
  clyr->run();
  if (myRank == 0) printf("first iteration took %lf secs\n", (TIME_SEC()-str_time));
  for (i=1; i<niter; i++){
    clyr->run();
  }
  GLOBAL_BARRIER(cdt_glb);
  end_time = TIME_SEC();
  if (myRank == 0){
    printf("benchmark completed\n");
    printf("performed %d iterations in %lf sec/iteration\n", niter, 
            (end_time-str_time)/niter);
    printf("achieved %lf Gigaflops\n", 
            ((double)2*n*m*k)*1.E-9/((end_time-str_time)/niter));
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
 * \brief main for a bench test of recurive contraction functions
 */
int main(int argc, char **argv) {
  int myRank, numPes, m, n, k, seed, npx, nlyr, niter;

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
  if (getCmdOption(argv, argv+argc, "-niter")){
    niter = atoi(getCmdOption(argv, argv+argc, "-niter"));
    if (niter <= 0) niter = 10;
  } else niter = 10;
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
    printf("-m=%d -n=%d -k=%d -seed = %d -niter = %d -npx = %d -nlyr = %d\n",m,n,k,seed,niter,npx,nlyr);
#ifdef USE_DCMF
    printf("USING DCMF FOR COMMUNICATION\n");
#else
    printf("USING MPI FOR COMMUNICATION\n");
#endif
  }

  GLOBAL_BARRIER(cdt_glb);
  if (myRank == 0) printf("BENCHMARKING RECTANGULAR ALGORITHM\n");
  bench_2d_bcast(0, n, m, k, seed, niter, npx, nlyr, cdt_glb);
  GLOBAL_BARRIER(cdt_glb);
  if ((numPes/nlyr)/npx == npx){
    if (myRank == 0) printf("BENCHMARKING SQUARE ALGORITHM\n");
    GLOBAL_BARRIER(cdt_glb);
    bench_2d_bcast(1, n, m, k, seed, niter, npx, nlyr, cdt_glb);
  }
  GLOBAL_BARRIER(cdt_glb);
  if (npx == 1 || (numPes/nlyr)/npx == 1){
    if (myRank == 0) printf("BENCHMARKING SQUARE VIRT ALGORITHM\n");
    GLOBAL_BARRIER(cdt_glb);
    bench_2d_bcast(2, n, m, k, seed, niter, npx, nlyr, cdt_glb);
  }
  GLOBAL_BARRIER(cdt_glb);
  COMM_EXIT;
  return 0;
}

