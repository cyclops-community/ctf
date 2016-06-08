/** \addtogroup examples 
  * @{ 
  * \defgroup spmm spmm
  * @{ 
  * \brief Multiplication of a random square sparse matrix by a dense matrix
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

int spmm(int     n,
         int     k,
         World & dw,
         double  sp=.50,
         int     niter=0,
         bool    bd=1){

  Matrix<> spA(n, n, SP, dw);
  Matrix<> dnA(n, n, dw);
  Matrix<> b(n, k, dw);
  Matrix<> c1(n, k, dw);
  Matrix<> c2(n, k, dw);

  srand48(dw.rank);

  b.fill_random(0.0,1.0);
  c1.fill_random(0.0,1.0);
  dnA.fill_random(0.0,1.0);

  spA["ij"] += dnA["ij"];
  spA.sparsify(sp);
  dnA["ij"] = 0.0;
  dnA["ij"] += spA["ij"];
  
  c2["ik"] = c1["ik"];
  
  c1["ik"] += dnA["ij"]*b["jk"];
  
  c2["ik"] += 0.5*spA["ij"]*b["jk"];
  c2["ik"] += 0.5*b["jk"]*spA["ij"];

  bool pass = c1.norm2() >= 1E-6;

  c1["ik"] -= c2["ik"];

  if (pass) pass = c1.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ c[\"ik\"] += A[\"ij\"]*b[\"jk\"] with sparse, A } passed \n");
    else
      printf("{ c[\"ik\"] += A[\"ij\"]*b[\"jk\"] with sparse, A } failed \n");
  }
#ifndef TEST_SUITE
  double min_time = DBL_MAX;
  double max_time = 0.0;
  double tot_time = 0.0;
  double times[niter];
  if (bd){
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of dense SPMM...\n", niter);
    }
    Timer_epoch dspmm("dense SPMM");
    dspmm.begin();
    for (int i=0; i<niter; i++){
      double start_time = MPI_Wtime();
      c1["ik"] += dnA["ij"]*b["jk"];
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    dspmm.end();
    
    if (dw.rank == 0){
      printf("Completed %d benchmarking iterations of dense SPMM (n=%d k=%d sp=%lf).\n", niter, n, k, sp);
      printf("All iterations times: ");
      for (int i=0; i<niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times,times+niter);
      printf("Dense MM (n=%d k=%d sp=%lf) Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",n,k,sp,min_time,tot_time/niter, times[niter/2], max_time);
    }
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of sparse SPMM...\n", niter);
    }
  }
  min_time = DBL_MAX;
  max_time = 0.0;
  tot_time = 0.0;
  Timer_epoch sspmm("sparse SPMM");
  sspmm.begin();
  for (int i=0; i<niter; i++){
    double start_time = MPI_Wtime();
    c1["ik"] += spA["ij"]*b["jk"];
    double end_time = MPI_Wtime();
    double iter_time = end_time-start_time;
    times[i] = iter_time;
    tot_time += iter_time;
    if (iter_time < min_time) min_time = iter_time;
    if (iter_time > max_time) max_time = iter_time;
  }
  sspmm.end();
  
  if (dw.rank == 0){
    printf("Completed %d benchmarking iterations of sparse SPMM (n=%d k=%d sp=%lf).\n", niter, n, k, sp);
    printf("All iterations times: ");
    for (int i=0; i<niter; i++){
      printf("%lf ", times[i]);
    }
    printf("\n");
    std::sort(times,times+niter);
    printf("Sparse MM (n=%d k=%d sp=%lf): Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",n,k,sp,min_time,tot_time/niter, times[niter/2], max_time);
  }

#endif
  return pass;
} 


#ifndef TEST_SUITE
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
  int rank, np, n, k, pass, niter, bd;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 7;
  } else k = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.0 || sp > 1.0) sp = .8;
  } else sp = .8;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;

  if (getCmdOption(input_str, input_str+in_num, "-bd")){
    bd = atoi(getCmdOption(input_str, input_str+in_num, "-bd"));
    if (bd != 0 && bd != 1) bd = 1;
  } else bd = 1;

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying %d-by-%d sparse (%lf zeros) matrix by %d-by-%d dense matrix\n",n,n,sp,n,k);
    }
    pass = spmm(n, k, dw, sp, niter, bd);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
