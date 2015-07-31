/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include <vector>
#include <numeric>
#include <ctf.hpp>

#define TEST_SUITE
#include "../examples/weigh_4D.cxx"
#include "../examples/gemm.cxx"
#include "../examples/gemm_4D.cxx"
#include "../examples/scalar.cxx"
#include "../examples/trace.cxx"
#include "diag_sym.cxx"
#include "diag_ctr.cxx"
#include "../examples/dft.cxx"
#include "../examples/dft_3D.cxx"
#include "../studies/fast_sym.cxx"
#include "../studies/fast_sym_4D.cxx"
#include "ccsdt_t3_to_t2.cxx"
#include "../examples/strassen.cxx"
#include "../examples/slice_gemm.cxx"
#include "readwrite_test.cxx"
#include "readall_test.cxx"
#include "../examples/subworld_gemm.cxx"
#include "multi_tsr_sym.cxx"
#include "repack.cxx"
#include "sy_times_ns.cxx"
#include "speye.cxx"
#include "sptensor_sum.cxx"
#include "../examples/endomorphism.cxx"
#include "../examples/endomorphism_cust.cxx"
#include "../examples/endomorphism_cust_sp.cxx"
#include "../examples/univar_function.cxx"
#include "../examples/univar_accumulator_cust.cxx"

using namespace CTF;

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
  int in_num = argc;
  char ** input_str = argv;

  //int nt;
  //MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &nt);
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
    World dw(MPI_COMM_WORLD, argc, argv);

    if (rank == 0)
      printf("Testing non-symmetric: NS = NS*NS weigh with n = %d:\n",n);
    pass.push_back(weigh_4D(n, NS, dw));

    if (rank == 0)
      printf("Testing symmetric: SY = SY*SY weigh with n = %d:\n",n);
    pass.push_back(weigh_4D(n, SY, dw));

    if (rank == 0)
      printf("Testing (anti-)skew-symmetric: AS = AS*AS weigh with n = %d:\n",n);
    pass.push_back(weigh_4D(n, AS, dw));

    if (rank == 0)
      printf("Testing symmetric-hollow: SH = SH*SH weigh with n = %d:\n",n);
    pass.push_back(weigh_4D(n, SH, dw));

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

    if (rank == 0)
      printf("Testing scalar operations\n");
    pass.push_back(scalar(dw));

    if (rank == 0)
      printf("Testing a 2D trace operation with n = %d:\n",n);
    pass.push_back(trace(n, dw));
    
    if (rank == 0)
      printf("Testing a diag sym operation with n = %d:\n",n);
    pass.push_back(diag_sym(n, dw));
    
    if (rank == 0)
      printf("Testing a diag ctr operation with n = %d m = %d:\n",n,n*n);
    pass.push_back(diag_ctr(n, n*n, dw));
    
    if (rank == 0)
      printf("Testing fast symmetric multiplication operation with n = %d:\n",n*n);
    pass.push_back(fast_sym(n*n, dw));

    if (rank == 0)
      printf("Testing 4D fast symmetric contraction operation with n = %d:\n",n);
    pass.push_back(fast_sym_4D(n, dw));
    
    if (rank == 0)
      printf("Testing multi-tensor symmetric contraction with m = %d n = %d:\n",n*n,n);
    pass.push_back(multi_tsr_sym(n^2,n, dw));
   
#ifndef PROFILE 
#ifndef BGQ
    if (rank == 0)
      printf("Testing gemm on subworld algorithm with n,m,k = %d div = 3:\n",n*n);
    pass.push_back(test_subworld_gemm(n*n, n*n, n*n, 3, dw));
#endif    

    if (rank == 0)
      printf("Testing non-symmetric Strassen's algorithm with n = %d:\n", n*n);
    pass.push_back(strassen(n*n, NS, dw));
    
    if (rank == 0)
      printf("Testing diagonal write with n = %d:\n",n);
    pass.push_back(readwrite_test(n, dw));
    
    if (rank == 0)
      printf("Testing readall test with n = %d m = %d:\n",n,n*n);
    pass.push_back(readall_test(n, n*n, dw));
    
    if (rank == 0)
      printf("Testing repack with n = %d:\n",n);
    pass.push_back(repack(n,dw));
    
    if (rank == 0)
      printf("Testing SY times NS with n = %d:\n",n);
    pass.push_back(sy_times_ns(n,dw));
    
    if (rank == 0)
      printf("Testing sparse summation with n = %d:\n",n);
    pass.push_back(sptensor_sum(n,dw));
    
    if (rank == 0)
      printf("Testing sparse identity with n = %d order = %d:\n",n,11);
    pass.push_back(speye(n,11,dw));
    
    if (rank == 0)
      printf("Testing endomorphism A_ijkl = A_ijkl^3 with n = %d:\n",n);
    pass.push_back(endomorphism(n,dw));

    if (rank == 0)
      printf("Testing endomorphism with custom function on a monoid A_ijkl = f(A_ijkl) with n = %d:\n",n);
    pass.push_back(endomorphism_cust(n,dw));

    if (rank == 0)
      printf("Testing endomorphism with custom function on a sparse set A_ijkl = f(A_ijkl) with n = %d:\n",n);
    pass.push_back(endomorphism_cust_sp(n,dw));

    if (rank == 0)
      printf("Testing univar_function .5*A_ijkl = .5*A_ijkl^4 with n = %d:\n",n);
    pass.push_back(univar_function(n,dw));

    if (rank == 0)
      printf("Testing univar_accumulator_cust integrates forces to particles with n = %d:\n",n);
    pass.push_back(univar_accumulator_cust(n,dw));

#if 0
    if (rank == 0)
      printf("Testing skew-symmetric Strassen's algorithm with n = %d:\n",n*n);
    pass.push_back(strassen(n*n, AS, dw));
      if (rank == 0)
    printf("Currently cannot do asymmetric Strassen's algorithm with n = %d:\n",n*n);
    pass.push_back(0);
#endif
#ifndef BGQ
    if (np == 1<<(int)log2(np)){
      if (rank == 0)
        printf("Testing non-symmetric sliced GEMM algorithm with (%d %d %d):\n",16,32,8);
      pass.push_back(test_slice_gemm(16, 32, 8, dw));
    }
#endif    
#endif
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

