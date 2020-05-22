/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup benchmarks
  * @{ 
  * \addtogroup bench_nosym_transp
  * @{ 
  * \brief Benchmarks the nonsymemtric transpose kernel
  */

#include <ctf.hpp>
#include "../src/redistribution/nosym_transp.h"
#ifdef USE_OMP
#include "omp.h"
#endif
#include <assert.h>

using namespace CTF;

void bench_nosym_transp(int          n,
                        int          order,
                        int          niter,
                        char const * iA,
                        char const * iB){

  printf("Performing transposes n=%d, order=%d, %s<->%s:\n",n,order,iA,iB);
  Ring<> r;

  int64_t edge_len[order];
  int new_order[order];

  int64_t N=1;
  for (int i=0; i<order; i++){
    N*=n;
    edge_len[i] = n;
    new_order[i] = -1;
    for (int j=0; j<order; j++){
      if (iA[i] == iB[j]){
        assert(new_order[i] == -1);
        new_order[i] = j;
      }
    }
    assert(new_order[i] != -1);
  }

  double * data;
  int pm = posix_memalign((void**)&data, 16, N*sizeof(double));
  assert(pm==0);

  srand48(7);
  for (int64_t i=0; i<N; i++){
    data[i] = drand48()-.5;
  }

  //check correctness of transpose
  CTF_int::nosym_transpose(order, new_order, edge_len, (char*)data, 1, &r);
  CTF_int::nosym_transpose(order, new_order, edge_len, (char*)data, 0, &r);

  srand48(7);
  for (int64_t i=0; i<N; i++){
    assert(data[i] == drand48()-.5);
  }
  printf("Passed correctness test\n");

  double * data2;
  pm = posix_memalign((void**)&data2, 16, N*sizeof(double));
  assert(pm==0);

  double t_cpy_st = MPI_Wtime();
  memcpy(data2, data, N*sizeof(double));
  double t_cpy = MPI_Wtime()-t_cpy_st;
  printf("single-threaded memcpy %ld bandwidth is %lf sec %lf GB/sec\n",
          N, t_cpy, 1.E-9*N*sizeof(double)/t_cpy);

#ifdef USE_OMP
  t_cpy_st = MPI_Wtime();
  #pragma omp parallel
  {
    int ti = omp_get_thread_num();
    int nt = omp_get_num_threads();
    int64_t Nt = N/nt;
    memcpy(data2+Nt*ti, data+Nt*ti, Nt*sizeof(double));
  }
  t_cpy = MPI_Wtime()-t_cpy_st;
  printf("multi-threaded memcpy %ld bandwidth is %lf sec %lf GB/sec\n",
          N, t_cpy, 1.E-9*N*sizeof(double)/t_cpy);
#endif
  free(data2);

  double t_fwd = 0.0;
  double t_min_fwd;
  double t_max_fwd;
  double t_bwd = 0.0;
  double t_min_bwd;
  double t_max_bwd;


  for (int i=0; i<niter; i++){
    double t_st_fwd = MPI_Wtime();

    CTF_int::nosym_transpose(order, new_order, edge_len, (char*)data, 1, &r);

    t_fwd += MPI_Wtime() - t_st_fwd;
    if (i==0){
      t_min_fwd = t_fwd;
      t_max_fwd = t_fwd;
    } else {
      t_min_fwd = std::min(MPI_Wtime() - t_st_fwd, t_min_fwd);
      t_max_fwd = std::max(MPI_Wtime() - t_st_fwd, t_max_fwd);
    }
 
    double t_st_bwd = MPI_Wtime();

    CTF_int::nosym_transpose(order, new_order, edge_len, (char*)data, 0, &r);

    t_bwd += MPI_Wtime() - t_st_bwd;
    if (i==0){
      t_min_bwd = t_bwd;
      t_max_bwd = t_bwd;
    } else {
      t_min_bwd = std::min(MPI_Wtime() - t_st_bwd, t_min_bwd);
      t_max_bwd = std::max(MPI_Wtime() - t_st_bwd, t_max_bwd);
    }
 
  }
 
  printf("Performed %d iteartions\n",niter);
  printf("Forward sec/iter: average = %lf (GB/s = %lf), range = [%lf, %lf]\n",
          t_fwd/niter, 1.E-9*N*sizeof(double)/(t_fwd/niter), t_min_fwd, t_max_fwd);
  printf("Backward sec/iter: average = %lf (GB/s = %lf), range = [%lf, %lf]\n",
          t_bwd/niter, 1.E-9*N*sizeof(double)/(t_bwd/niter), t_min_bwd, t_max_bwd);

  free(data); 
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
  int niter, n;
  int const in_num = argc;
  char ** input_str = argv;
  char const * A;
  char const * B;
  MPI_Init(NULL, NULL);
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 10;
  } else n = 10;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 8;
  } else niter = 8;

  if (getCmdOption(input_str, input_str+in_num, "-A")){
    A = getCmdOption(input_str, input_str+in_num, "-A");
  } else A = "ij";
  if (getCmdOption(input_str, input_str+in_num, "-B")){
    B = getCmdOption(input_str, input_str+in_num, "-B");
  } else B = "ji";


  bench_nosym_transp(n, strlen(A), niter, A, B);

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */


