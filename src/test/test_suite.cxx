/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <numeric>
#include <ctf.hpp>

#define TEST_SUITE
#include "../../examples/weight_4D.cxx"
#include "../../examples/sym3.cxx"
#include "../../examples/gemm.cxx"
#include "../../examples/gemm_4D.cxx"
#include "../../examples/trace.cxx"
#include "../../examples/dft.cxx"
#include "../../examples/dft_3D.cxx"
#include "../../examples/fast_sym.cxx"
#include "../../examples/fast_sym_4D.cxx"
#include "../../examples/ccsdt_t3_to_t2.cxx"
#include "../../examples/strassen.cxx"


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
  int rank, np, n;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 2) n = 6;
  } else n = 6;

  if (rank == 0){
    printf("Testing Cyclops Tensor Framework using %d processors\n",np);
  }

  std::vector<int> pass;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0)
      printf("Testing non-symmetric: NS = NS*NS weight with n = %d:\n",n);
    pass.push_back(weight_4D(n, NS, dw));

    if (rank == 0)
      printf("Testing symmetric: SY = SY*SY weight with n = %d:\n",n);
    pass.push_back(weight_4D(n, SY, dw));

    if (rank == 0)
      printf("Testing (anti-)skew-symmetric: AS = AS*AS weight with n = %d:\n",n);
    pass.push_back(weight_4D(n, AS, dw));

    if (rank == 0)
      printf("Testing symmetric-hollow: SH = SH*SH weight with n = %d:\n",n);
    pass.push_back(weight_4D(n, SH, dw));

    if (rank == 0)
      printf("Testing CCSDT T3->T2 with n= %d, m = %d:\n",n,n+1);
    pass.push_back(ccsdt_t3_to_t2(n, n+1, dw));
    
    if (rank == 0)
      printf("Testing non-symmetric: NS = NS*NS gemm with n = %d:\n",n*n);
    pass.push_back(gemm(n*n, n*n, n*n, NS, 1, dw));

    if (rank == 0)
      printf("Testing symmetric: SY = SY*SY gemm with n = %d:\n",n*n);
    pass.push_back(gemm(n*n, n*n, n*n, SY, 1, dw));

    if (rank == 0)
      printf("Testing (anti-)skew-symmetric: AS = AS*AS gemm with n = %d:\n",n*n);
    pass.push_back(gemm(n*n, n*n, n*n, AS, 1, dw));
    
    if (rank == 0)
      printf("Testing symmetric-hollow: SH = SH*SH gemm with n = %d:\n",n*n);
    pass.push_back(gemm(n*n, n*n, n*n, SH, 1, dw));

    if (rank == 0)
      printf("Testing non-symmetric: NS = NS*NS 4D gemm with n = %d:\n",n);
    pass.push_back(gemm_4D(n, NS, 1, dw));

    if (rank == 0)
      printf("Testing symmetric: SY = SY*SY 4D gemm with n = %d:\n",n);
    pass.push_back(gemm_4D(n, SY, 1, dw));

    if (rank == 0)
      printf("Testing (anti-)skew-symmetric: AS = AS*AS 4D gemm with n = %d:\n",n);
    pass.push_back(gemm_4D(n, AS, 1, dw));
    
    if (rank == 0)
      printf("Testing symmetric-hollow: SH = SH*SH 4D gemm with n = %d:\n",n);
    pass.push_back(gemm_4D(n, SH, 1, dw));

#ifndef USE_SYM_SUM
    if (rank == 0)
      printf("Testing a CCSDT 6D=4D*4D contraction with n = %d:\n",n);
    pass.push_back(sym3(n, dw));
#endif

    if (rank == 0)
      printf("Testing CCSDT T3->T2 with n= %d m = %d\n",n,n+1);
    pass.push_back(ccsdt_t3_to_t2(n, n+1, dw));
    
    if (rank == 0)
      printf("Testing a 2D trace operation with n = %d:\n",n*n);
    pass.push_back(trace(n*n, dw));
    
#ifndef USE_SYM_SUM
    if (rank == 0)
      printf("Testing fast symmetric multiplication operation with n = %d:\n",n*n);
    pass.push_back(fast_sym(n*n, dw));
    
    if (rank == 0)
      printf("Testing 4D fast symmetric contraction operation with n = %d:\n",n);
    pass.push_back(fast_sym_4D(n, dw));
#else
    if (rank == 0)
      printf("Currently unable to do fast symmetric multiplication operation with SYM SUM:\n");
    pass.push_back(0);
    pass.push_back(0);
#endif
    
    if (rank == 0)
      printf("Testing non-symmetric Strassen's algorithm with n = %d:\n",n*n);
    pass.push_back(strassen(n*n, NS, dw));
#ifdef USE_SYM
    if (rank == 0)
      printf("Testing skew-symmetric Strassen's algorithm with n = %d:\n",n*n);
    pass.push_back(strassen(n*n, AS, dw));
      if (rank == 0)
    printf("Currently cannot do asymmetric Strassen's algorithm with n = %d:\n",n*n);
    pass.push_back(0);
#endif

  }
  {
    cCTF_World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0)
      printf("Testing 1D DFT with n = %d:\n",n*n);
    pass.push_back(test_dft(n*n, dw));
    if (rank == 0)
      printf("Testing 3D DFT with n = %d:\n",n);
    pass.push_back(test_dft_3D(n, dw));

  }
  int num_pass = std::accumulate(pass.begin(), pass.end(), 0);
  if (rank == 0)
    printf("Testing completed, %d/%zu tests passed\n", num_pass, pass.size());


  MPI_Finalize();
  return 0;
}

