/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
#include <vector>
#include <numeric>
#include <ctf.hpp>

#define TEST_SUITE
#include "weigh_4D.cxx"
#include "gemm_4D.cxx"
#include "scalar.cxx"
#include "diag_sym.cxx"
#include "diag_ctr.cxx"
#include "dft.cxx"
#include "ccsdt_t3_to_t2.cxx"
#include "readwrite_test.cxx"
#include "readall_test.cxx"
#include "subworld_gemm.cxx"
#include "multi_tsr_sym.cxx"
#include "repack.cxx"
#include "sy_times_ns.cxx"
#include "speye.cxx"
#include "sptensor_sum.cxx"
#include "endomorphism.cxx"
#include "endomorphism_cust.cxx"
#include "endomorphism_cust_sp.cxx"
#include "univar_function.cxx"
#include "bivar_function.cxx"
#include "bivar_transform.cxx"

#include "../examples/trace.cxx"
#include "../examples/fft_with_idx_partition.cxx"
#include "../examples/dft_3D.cxx"
#include "../examples/strassen.cxx"
#include "../examples/recursive_matmul.cxx"
#include "../examples/force_integration.cxx"
#include "../examples/force_integration_sparse.cxx"
#include "../examples/particle_interaction.cxx"
#include "../examples/spmv.cxx"
#include "../examples/jacobi.cxx"
#include "../examples/sssp.cxx"
#include "../examples/apsp.cxx"
#include "../examples/btwn_central.cxx"
#include "../examples/sparse_mp3.cxx"
#include "../examples/bitonic_sort.cxx"
#include "../examples/matmul.cxx"

#include "../studies/fast_sym.cxx"
#include "../studies/fast_sym_4D.cxx"

#ifdef USE_SCALAPACK
#include "../scalapack_tests/qr.cxx"
#include "../scalapack_tests/svd.cxx"
#include "../scalapack_tests/eigh.cxx"
#endif


using namespace CTF;

namespace CTF_int{
  int64_t get_tensor_data_bytes_allocated();
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
    Timer_epoch te("TEST_SUITE");
    te.begin();
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
      printf("Testing non-symmetric: NS = NS*NS matmul with n = %d:\n",n*n);
    pass.push_back(matmul(n*n, n*n, n*n, dw));

    if (rank == 0)
      printf("Testing symmetric: SY = SY*SY matmul with n = %d:\n",n*n);
    pass.push_back(matmul(n*n, n*n, n*n, dw, SY, SY, SY));

    if (rank == 0)
      printf("Testing (anti-)skew-symmetric: AS = AS*AS matmul with n = %d:\n",n*n);
    pass.push_back(matmul(n*n, n*n, n*n, dw, AS, AS, AS));
    
    if (rank == 0)
      printf("Testing symmetric-hollow: SH = SH*SH matmul with n = %d:\n",n*n);
    pass.push_back(matmul(n*n, n*n, n*n, dw, SH, SH, SH));

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
   
    if (rank == 0)
      printf("Testing gemm on subworld algorithm with n,m,k = %d div = 3:\n",n*n);
    pass.push_back(test_subworld_gemm(n*n, n*n, n*n, 3, dw));

    if (rank == 0)
      printf("Testing non-symmetric Strassen's algorithm with n = %d:\n", 2*n*n);
    pass.push_back(strassen(2*n*n, NS, dw));
    
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

#ifdef USE_SCALAPACK
    if (rank == 0)
      printf("Testing QR with m = %d n = %d:\n",n*n,n);
    pass.push_back(test_qr(n*n,n,dw));
 
    if (rank == 0)
      printf("Testing SVD with m = %d n = %d k=%d:\n",n*n,n+1,n+1);
    pass.push_back(test_svd(n*n,n+1,n+1,dw));
  
    if (rank == 0)
      printf("Testing symmetric eigensolve n = %d:\n",n*n+1);
    pass.push_back(test_eigh(n*n+1,dw));
 

#endif
    if (np == 1<<(int)log2(np)){
      if (rank == 0)
        printf("Testing non-symmetric sliced GEMM algorithm with (%d %d %d):\n",16,32,8);
      pass.push_back(test_recursive_matmul(16, 32, 8, dw));
    }
    if (rank == 0)
      printf("Testing 1D DFT with n = %d and int index type:\n",n*n);
    pass.push_back(test_dft<int>(n*n, dw));

    if (rank == 0)
      printf("Testing 1D DFT with n = %d and int64_t index type:\n",n*n);
    pass.push_back(test_dft<int64_t>(n*n, dw));

    if (rank == 0)
      printf("Testing FFT with Idx_partition with n = %d: m = %d\n",n,1<<(n/2));
    pass.push_back(fft_with_idx_partition(n, 1<<(n/2), dw));

//#ifdef __clang__
    //if (rank == 0)
    //  printf("WARNING: Skipping dft_3D test, due to known issue with Clang and optimizations -Ox for x>0\n");
//#else
    if (rank == 0)
      printf("Testing 3D DFT with n = %d:\n",n);
    pass.push_back(test_dft_3D(n, dw));
//#endif
    
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
      printf("Testing force_integration integrates forces to particles with n = %d:\n",n);
    pass.push_back(force_integration(n,dw));

    if (rank == 0)
      printf("Testing force_integration_sparse integrates sparse forces to particles with n = %d:\n",n);
    pass.push_back(force_integration_sparse(n,dw));
    
    if (rank == 0)
      printf("Testing bivar_function A_ijkl = f2(A_ijkl, B_ijkl) with n = %d:\n",n);
    pass.push_back(bivar_function(n,dw));
    
    if (rank == 0)
      printf("Testing custom bivar_function F[\"i\"] += f(P[\"i\"],P[\"j\"] with n = %d:\n",n);
    pass.push_back(particle_interaction(n,dw));
    
    if (rank == 0)
      printf("Testing bivar_transform 3(A_ijkl, B_ijkl, C_ijkl) with n = %d:\n",n);
    pass.push_back(bivar_transform(n,dw));
    
    if (rank == 0)
      printf("Testing sparse-matrix times vector with n=%d:\n",n);
    pass.push_back(spmv(n,false,dw));
 
    if (rank == 0)
      printf("Testing doubly-compressed sparse-matrix times vector with n=%d:\n",n);
    pass.push_back(spmv(n,true,dw));
    
    if (rank == 0)
      printf("Testing sparse-matrix times matrix with n=%d k=%d:\n",n*n,n);
    pass.push_back(matmul(n*n,n,n*n,dw,NS,NS,NS,.3));
    
    if (rank == 0)
      printf("Testing sparse-matrix times sparse-matrix with m=%d n=%d k=%d:\n",n,n*n,n+1);
    pass.push_back(matmul(n,n*n,n+1,dw,NS,NS,NS,.3,.3));

    if (rank == 0)
      printf("Testing sparse=sparse*sparse (spgemm) with m=%d n=%d k=%d:\n",n,n*n,n+1);
    pass.push_back(matmul(n,n*n,n+1,dw,NS,NS,NS,.3,.3,.3));

    if (rank == 0)
      printf("Testing Jacobi iteration with n=%d:\n",n);
    pass.push_back(jacobi(n,dw));
     
    if (rank == 0)
      printf("Testing SSSP via the Bellman-Ford algorithm n=%d:\n",n*n);
    pass.push_back(sssp(n*n,dw));
    
    if (rank == 0)
      printf("Testing APSP via path doubling with n=%d:\n",n*n);
    pass.push_back(apsp(n*n,dw));

    if (rank == 0)
      printf("Testing betweenness centrality with n=%d:\n",n);
    pass.push_back(btwn_cnt(n,dw,.2,2,1,1,1,1));
    
    if (rank == 0)
      printf("Testing MP3 calculation using sparse*dense with %d occupied and %d virtual orbitals:\n",n,2*n);
    pass.push_back(sparse_mp3(2*n,n,dw,.8,1,0,0,0,0));
    
    if (rank == 0)
      printf("Testing MP3 calculation using sparse*sparse with %d occupied and %d virtual orbitals:\n",n,2*n);
    pass.push_back(sparse_mp3(2*n,n,dw,.8,1,0,0,0,1));
    
    te.end();
    /*int logn = log2(n)+1;
    if (rank == 0)
      printf("Testing bitonic sorting with %d elements:\n",1<<logn);
    pass.push_back(bitonic(logn,dw));*/
  }
  int num_pass = std::accumulate(pass.begin(), pass.end(), 0);
  if (rank == 0)
    printf("Testing completed, %d/%zu tests passed\n", num_pass, pass.size());
  int64_t tot_mem_used = std::fabs(CTF_int::get_tensor_data_bytes_allocated());
  int64_t max_tot_mem_used;
  MPI_Reduce(&tot_mem_used, &max_tot_mem_used, 1, MPI_INT64_T, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0 && max_tot_mem_used > 0)
    printf("Warning: CTF memory accounting thinks %1.2E bytes have been left allocated, memory leak or bug possible\n", (double)max_tot_mem_used);

  MPI_Finalize();
  return 0;
}

