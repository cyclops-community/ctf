/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup examples 
  * @{ 
  * \addtogroup CCSDT_T3_to_T2
  * @{ 
  * \brief A symmetric contraction from CCSDT compared with the explicitly permuted nonsymmetric form
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

  int ndim_A, ndim_B, ndim_C;
  ndim_A = strlen(iA);
  ndim_B = strlen(iB);
  ndim_C = strlen(iC);

  int NS_A[ndim_A];
  int NS_B[ndim_B];
  int NS_C[ndim_C];
  int n_A[ndim_A];
  int n_B[ndim_B];
  int n_C[ndim_C];

  for (i=0; i<ndim_A; i++){
    n_A[i] = n;
    NS_A[i] = NS;
  }
  for (i=0; i<ndim_B; i++){
    n_B[i] = n;
    NS_B[i] = NS;
  }
  for (i=0; i<ndim_C; i++){
    n_C[i] = n;
    NS_C[i] = NS;
  }


  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(ndim_A, n_A, NS_A, dw, "A", 1);
  CTF_Tensor B(ndim_B, n_B, NS_B, dw, "B", 1);
  CTF_Tensor C(ndim_C, n_C, NS_C, dw, "C", 1);

  double st_time = MPI_Wtime();

  for (i=0; i<niter; i++){
    C[iC] += A[iA]*B[iB];
  }

  double end_time = MPI_Wtime();

  if (rank == 0)
    printf("Performed %d iterations of C[\"%s\"] += A[\"%s\"]*B[\"%s\"] in %lf sec/iter\n", 
           niter, iA, iB, iC, (end_time-st_time)/niter);

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
  int rank, np, niter, n, m;
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


