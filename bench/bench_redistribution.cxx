/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{ 
  * \addtogroup bench_redistributions
  * @{ 
  * \brief Benchmarks arbitrary NS redistribution
  */

//#include <boost/math/distributions/normal.hpp>
//
//boost::math::normal dist(0.0, 1.0);
//
//// 95% of distribution is below q:
//double q = quantile(dist, 0.95);

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include <iostream>
#include <fstream>
#include "../src/shared/util.h"

using namespace CTF;

void bench_redistribution(int          niter,
                          World &      dw,
                          int          order,
                          int const *  lens,
                          char const * idx,
                          int          prl1_ord,
                          int const *  prl1_lens,
                          char const * prl1_idx,
                          int          prl2_ord,
                          int const *  prl2_lens,
                          char const * prl2_idx,
                          int          blk1_ord,
                          int const *  blk1_lens,
                          char const * blk1_idx,
                          int          blk2_ord,
                          int const *  blk2_lens,
                          char const * blk2_idx){

  int rank, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int sym[order];
  int64_t N = 1;
  for (int i=0; i<order; i++){
    N*=lens[i];
    sym[i] = NS;
  }

  Partition prl1(prl1_ord, prl1_lens);
  Partition prl2(prl2_ord, prl2_lens);
  Partition blk1(blk1_ord, blk1_lens);
  Partition blk2(blk2_ord, blk2_lens);
  
  Tensor<> A(order, lens, sym, dw, idx, prl1[prl1_idx], blk1[blk1_idx], "A", 1);

  A.fill_random(-.5, .5);

  double t = 0.0;
  double t_min;
  double t_max;

  double btime;
  MPI_Barrier(MPI_COMM_WORLD);
  btime = MPI_Wtime();
  MPI_Barrier(MPI_COMM_WORLD);
  btime -= MPI_Wtime();
    
  double * data_ref = A.read(idx, prl2[prl2_idx], blk2[blk2_idx]);

#ifdef USE_FOMPI
  int N_DGTOG = 6;
#else
  int N_DGTOG = 5;
#endif

  std::ofstream f;
  if (rank == 0){
    char fname[1000];
    sprintf(fname, "bench_redist.p%d.o%d.N%d.pst-%s.vst-%s.ped-%s.ved-%s.dat", num_pes, order, lens[0], prl1_idx, blk1_idx, prl2_idx, blk2_idx);
    f.open(fname);
  }
 
  std::vector<double> times[N_DGTOG];
  for (int D=0; D<N_DGTOG; D++){
    DGTOG_SWITCH = D;
    char const * str_name;
    switch (D){
      case 0:
        str_name = "NAIVE";
        break;
      case 1:
        str_name = "ROR";
        break;
      case 2:
        str_name = "ROR_ISR";
        break;
      case 3:
        str_name = "ROR_PUT";
        break;
      case 4:
        str_name = "ROR_ISR_ANY";
        break;
      case 5:
        str_name = "ROR_PUT_ANY";
        break;
    }
    if (rank == 0) printf("Testing redistribution via kernel %s\n", str_name);

    double * data = A.read(idx, prl2[prl2_idx], blk2[blk2_idx]);
    int pass = 1;
    for (int64_t j=0; j<N/num_pes; j++){
      if (data[j] != data_ref[j]){ 
        pass = 0;
        printf("[%d] Incorrect! data[%ld] = %lf instead of %lf\n",rank,j, data[j],data_ref[j]);
      }
    }
    free(data);
    MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (pass){
      if (rank == 0) printf("Correctness test passed.\n");
      MPI_Barrier(MPI_COMM_WORLD);
      Timer_epoch te(str_name);
      te.begin();
      for (int i=0; i<niter; i++){
        double t_st = MPI_Wtime();
        double * data = A.read(idx, prl2[prl2_idx], blk2[blk2_idx]);
        MPI_Barrier(MPI_COMM_WORLD);
        times[D].push_back(MPI_Wtime() - t_st - btime);
        free(data);
      }
      te.end();
      std::sort(&times[D][0], &times[D][0]+niter);
      if (rank == 0){
        printf("Performed %d redistributions via kernel %s sec/iter: median = %lf (median effective end-to-end bandwidth, N/(t*p) = %lf GB/s), range = [%lf, %lf]\n",
                niter, str_name, times[D][niter/2], 1.E-9*N*sizeof(double)/(num_pes*times[D][niter/2]), times[D][0], times[D][niter-1]);
        f << str_name << " ";
        for (int i=0; i<niter; i++){
          f << times[D][i] << " ";
        }
        f << "\n";
      }
    }
  }
  if (rank == 0){
    printf("Data line kernel * [min, median max]:\n");
    for (int D=0; D<N_DGTOG; D++){
      printf("%lf %lf %lf ", times[D][0], times[D][niter/2], times[D][niter-1]);
    }
    printf("\n");
  }
  if (rank == 0){
    f.close();
  }

  free(data_ref); 
/*  if (rank == 0)
    printf("Performed %d redistributions in %lf time/iter %lf mem GB/sec\n",
            niter, (end_time-st_time)/niter, (2*N*1.E-9/((end_time-st_time)/niter))/num_pes);*/
} 

char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}


int main(int argc, char ** argv){
  int rank, np, niter, n, phase;
  int const in_num = argc;
  char ** input_str = argv;
  char const * idx;
  char const * prl1_idx;
  char const * prl2_idx;
  char const * blk1_idx;
  char const * blk2_idx;
  int64_t prl1, prl2, blk1, blk2;
  int order;
  int prl1_ord;
  int prl2_ord;
  int blk1_ord;
  int blk2_ord;
  int * lens;
  int * prl1_lens;
  int * prl2_lens;
  int * blk1_lens;
  int * blk2_lens;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  if (getCmdOption(input_str, input_str+in_num, "-phase")){
    phase = atoi(getCmdOption(input_str, input_str+in_num, "-phase"));
    if (phase < 0) phase = 10;
  } else phase = 10;

  if (getCmdOption(input_str, input_str+in_num, "-prl1")){
    prl1 = atoi(getCmdOption(input_str, input_str+in_num, "-prl1"));
    if (prl1 < 0) prl1 = np;
  } else prl1 = np;

  if (getCmdOption(input_str, input_str+in_num, "-prl2")){
    prl2 = atoi(getCmdOption(input_str, input_str+in_num, "-prl2"));
    if (prl2 < 0) prl2 = np;
  } else prl2 = np;

  if (getCmdOption(input_str, input_str+in_num, "-blk1")){
    blk1 = atoi(getCmdOption(input_str, input_str+in_num, "-blk1"));
    if (blk1 < 0) blk1 = np;
  } else blk1 = np;

  if (getCmdOption(input_str, input_str+in_num, "-blk2")){
    blk2 = atoi(getCmdOption(input_str, input_str+in_num, "-blk2"));
    if (blk2 < 0) blk2 = np;
  } else blk2 = np;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;

  if (getCmdOption(input_str, input_str+in_num, "-idx")){
    idx = getCmdOption(input_str, input_str+in_num, "-idx");
  } else idx = "ij";
  if (getCmdOption(input_str, input_str+in_num, "-prl1_idx")){
    prl1_idx = getCmdOption(input_str, input_str+in_num, "-prl1_idx");
  } else prl1_idx = "i";
  if (getCmdOption(input_str, input_str+in_num, "-prl2_idx")){
    prl2_idx = getCmdOption(input_str, input_str+in_num, "-prl2_idx");
  } else prl2_idx = "j";
  if (getCmdOption(input_str, input_str+in_num, "-blk1_idx")){
    blk1_idx = getCmdOption(input_str, input_str+in_num, "-blk1_idx");
  } else blk1_idx = "";
  if (getCmdOption(input_str, input_str+in_num, "-blk2_idx")){
    blk2_idx = getCmdOption(input_str, input_str+in_num, "-blk2_idx");
  } else blk2_idx = "";
  
  order = strlen(idx);
  prl1_ord = strlen(prl1_idx);
  prl2_ord = strlen(prl2_idx);
  blk1_ord = strlen(blk1_idx);
  blk2_ord = strlen(blk2_idx);

  if (rank==0){
    printf("Redistributing order %d tensor with all dims %d and idx %s from order %d proc grid with dims %ld and idx %s, to order %d proc grid with dims %ld and idx %s\n", order, n, idx, prl1_ord, prl1, prl1_idx, prl2_ord, prl2, prl2_idx);
    printf("Initial blocking order %d dims %ld and idx %s, to final blocking order %d dims %ld and idx %s\n", blk1_ord, blk1, blk1_idx, blk2_ord, blk2, blk2_idx);
  }


  lens = (int*)malloc(order*sizeof(int));
  for (int i=0; i<order; i++){
    lens[i] = n;
  }
  prl1_lens = (int*)malloc(prl1_ord*sizeof(int));
  for (int i=0; i<prl1_ord; i++){
    prl1_lens[prl1_ord-i-1] = prl1%phase;
    prl1 = prl1/phase;
  }
  if (rank == 0){ 
    printf("start topology:");
    for (int i=0; i<prl1_ord; i++){
      printf(" %d", prl1_lens[i]);
    }
    printf("\n");
  }
  prl2_lens = (int*)malloc(prl2_ord*sizeof(int));
  for (int i=0; i<prl2_ord; i++){
    prl2_lens[prl2_ord-i-1] = prl2%phase;
    prl2 = prl2/phase;
  }
  if (rank == 0){ 
    printf("end topology:");
    for (int i=0; i<prl2_ord; i++){
      printf(" %d", prl2_lens[i]);
    }
    printf("\n");
  }

  blk1_lens = (int*)malloc(blk1_ord*sizeof(int));
  for (int i=0; i<blk1_ord; i++){
    blk1_lens[blk1_ord-i-1] = blk1%phase;
    blk1 = blk1/phase;
  }
  if (rank == 0){ 
    printf("start blocking:");
    for (int i=0; i<blk1_ord; i++){
      printf(" %d", blk1_lens[i]);
    }
    printf("\n");
  }

  blk2_lens = (int*)malloc(blk2_ord*sizeof(int));
  for (int i=0; i<blk2_ord; i++){
    blk2_lens[blk2_ord-i-1] = blk2%phase;
    blk2 = blk2/phase;
  }
  if (rank == 0){ 
    printf("end blocking:");
    for (int i=0; i<blk2_ord; i++){
      printf(" %d", blk2_lens[i]);
    }
    printf("\n");
  }


  {
    CTF_World dw(argc, argv);
    bench_redistribution(niter, dw, order, lens, idx,
                         prl1_ord, prl1_lens, prl1_idx,
                         prl2_ord, prl2_lens, prl2_idx,
                         blk1_ord, blk1_lens, blk1_idx,
                         blk2_ord, blk2_lens, blk2_idx);
  }


  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */


