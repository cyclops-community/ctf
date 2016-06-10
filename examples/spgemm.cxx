/** \addtogroup examples 
  * @{ 
  * \defgroup spgemm spmm
  * @{ 
  * \brief Multiplication of a random square sparse matrix by another into a sparse output
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

int spgemm(int     m,
           int     n,
           int     k,
           World & dw,
           double  sp=.50,
           int     niter=0,
           bool    bd=1){

  Matrix<> spA(m, k, SP, dw);
  Matrix<> dnA(m, k, dw);
  Matrix<> spB(k, n, SP, dw);
  Matrix<> dnB(k, n, dw);
  Matrix<> spC(m, n, SP, dw);
  Matrix<> dnC(m, n, dw);

  srand48(dw.rank);

  dnB.fill_random(0.0,1.0);
  dnA.fill_random(0.0,1.0);

  spA["ij"] += dnA["ij"];
  spA.sparsify(sp);
  spB["ij"] += dnB["ij"];
  spB.sparsify(sp);

  dnA["ij"] = 0.0;
  dnA["ij"] += spA["ij"];
  dnB["ij"] = 0.0;
  dnB["ij"] += spB["ij"];
  
  dnC["ik"] += dnA["ij"]*dnB["jk"];
  
  spC["ik"] += .5*spA["ij"]*spB["jk"];
  spC["ik"] += .5*spB["jk"]*spA["ij"];
/*  spA.print();
  spB.print();
  dnC.print();
  spC.print();*/

  bool pass = dnC.norm2() >= 1E-6;

  dnC["ik"] -= spC["ik"];

  if (pass) pass = dnC.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ C[\"ik\"] += A[\"ij\"]*B[\"jk\"] with sparse A, B, C } passed \n");
    else
      printf("{ C[\"ik\"] += A[\"ij\"]*B[\"jk\"] with sparse A, B, C } failed \n");
  }
#ifndef TEST_SUITE
  double min_time = DBL_MAX;
  double max_time = 0.0;
  double tot_time = 0.0;
  double times[niter];
  if (bd){
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of dense SPGEMM...\n", niter);
    }
    Timer_epoch dspgemm("dense SPGEMM");
    dspgemm.begin();
    for (int i=0; i<niter; i++){
      double start_time = MPI_Wtime();
      dnC["ik"] += dnA["ij"]*dnB["jk"];
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    dspgemm.end();
    
    if (dw.rank == 0){
      printf("Completed %d benchmarking iterations of dense SPGEMM (m= %d n=%d k=%d sp=%lf).\n", niter, m, n, k, sp);
      printf("All iterations times: ");
      for (int i=0; i<niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times,times+niter);
      printf("Dense MM (m=%d n=%d k=%d sp=%lf) Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",m,n,k,sp,min_time,tot_time/niter, times[niter/2], max_time);
    }
    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of sparse SPGEMM...\n", niter);
    }
    min_time = DBL_MAX;
    max_time = 0.0;
    tot_time = 0.0;
    Timer_epoch sspgemm("sparse SPGEMM");
    sspgemm.begin();
    for (int i=0; i<niter; i++){
      double start_time = MPI_Wtime();
      spC["ik"] += spA["ij"]*spB["jk"];
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    sspgemm.end();
    
    if (dw.rank == 0){
      printf("Completed %d benchmarking iterations of sparse SPGEMM (m=%d n=%d k=%d sp=%lf).\n", niter, m, n, k, sp);
      printf("All iterations times: ");
      for (int i=0; i<niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times,times+niter);
      printf("Sparse MM (m=%d n=%d k=%d sp=%lf): Min time=%lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",m,n,k,sp,min_time,tot_time/niter, times[niter/2], max_time);
    }
  
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
  int rank, np, m, n, k, pass, niter, bd;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 7;
  } else m = 7;

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
      printf("Multiplying %d-by-%d sparse (%lf pct zeros) matrix by %d-by-%d sparse matrix (%lf pct zeros)\n",m,k,100.*sp,k,n,100.*sp);
    }
    pass = spgemm(m, n, k, dw, sp, niter, bd);
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
