/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{ 
  * \addtogroup bench_contractions
  * @{ 
  * \brief Benchmarks arbitrary NS contraction
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

int bench_contraction(int          n,
                      int          niter,
                      char const * iA,
                      char const * iB,
                      char const * iC,
                      CTF_World   &dw){

  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int order_A, order_B, order_C;
  order_A = strlen(iA);
  order_B = strlen(iB);
  order_C = strlen(iC);

  int NS_A[order_A];
  int NS_B[order_B];
  int NS_C[order_C];
  int n_A[order_A];
  int n_B[order_B];
  int n_C[order_C];

  for (i=0; i<order_A; i++){
    n_A[i] = n;
    NS_A[i] = NS;
  }
  for (i=0; i<order_B; i++){
    n_B[i] = n;
    NS_B[i] = NS;
  }
  for (i=0; i<order_C; i++){
    n_C[i] = n;
    NS_C[i] = NS;
  }


  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(order_A, n_A, NS_A, dw, "A", 1);
  CTF_Tensor B(order_B, n_B, NS_B, dw, "B", 1);
  CTF_Tensor C(order_C, n_C, NS_C, dw, "C", 1);

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    C[iC] += A[iA]*B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf sec/iter\n", 
           niter, iC, iA, iB, (end_time-st_time)/niter);

  return 1;
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
  int rank, np, niter, n;
  int const in_num = argc;
  char ** input_str = argv;
  char const * A;
  char const * B;
  char const * C;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "ik";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "kj";
  if (getCmdOption(input_str, input_str+in_num, "-C")){
    C = getCmdOption(input_str, input_str+in_num, "-C");
  } else C = "ij";



  {
    CTF_World dw(argc, argv);
    int pass = bench_contraction(n, niter, A, B, C, dw);
    assert(pass);
  }


  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */


