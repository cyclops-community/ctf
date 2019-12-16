/** \addtogroup examples 
  * @{ 
  * \defgroup matmul Matrix multiplication
  * @{ 
  * \brief Multiplication of two matrices with user-defined attributes of symmetry and sparsity
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;


/**
 * \brief (if test) tests and (if bench) benchmarks m*n*k matrix multiplication with matrices of specified symmetry and sparsity fraction
 * \param[in] m number of rows in C, A
 * \param[in] n number of cols in C, B
 * \param[in] k number of rows in A, cols in B
 * \param[in] dw set of processors on which to execute matmul
 * \param[in] sym_A in {NS, SY, AS, SH} symmetry attributes of A 
 * \param[in] sym_B in {NS, SY, AS, SH} symmetry attributes of B 
 * \param[in] sym_C in {NS, SY, AS, SH} symmetry attributes of C 
 * \param[in] sp_A fraction of nonzeros in A (if 1. A stored as dense)
 * \param[in] sp_B fraction of nonzeros in B (if 1. B stored as dense)
 * \param[in] sp_C fraction of nonzeros in C (if 1. C stored as dense)
 * \param[in] test whether to test
 * \param[in] bench whether to benchmark
 * \param[in] niter how many iterations to compute
 */
int matmul(int     m,
           int     n,
           int     k,
           World & dw,
           int     sym_A=NS, 
           int     sym_B=NS, 
           int     sym_C=NS, 
           double  sp_A=1.,
           double  sp_B=1.,
           double  sp_C=1.,
           bool    test=true,
           bool    bench=false,
           int     niter=10){
  assert(test || bench);

  int sA = sp_A < 1. ? SP : 0;
  int sB = sp_B < 1. ? SP : 0;
  int sC = sp_C < 1. ? SP : 0;

  /* initialize matrices with attributes */
  Matrix<> A(m, k, sym_A|sA, dw);
  Matrix<> B(k, n, sym_B|sB, dw);
  Matrix<> C(m, n, sym_C|sC, dw, "C");


  /* fill with random data */
  srand48(dw.rank);
  if (sp_A < 1.)
    A.fill_sp_random(0.0,1.0,sp_A);
  else
    A.fill_random(0.0,1.0);
  if (sp_B < 1.)
    B.fill_sp_random(0.0,1.0,sp_B);
  else
    B.fill_random(0.0,1.0);
  if (sp_C < 1.)
    C.fill_sp_random(0.0,1.0,sp_C);
  else
    C.fill_random(0.0,1.0);

  bool pass = true;

  if (test){ 
    /* initialize matrices with default attributes (nonsymmetric, dense) */
    Matrix<> ref_A(m, k, dw);
    Matrix<> ref_B(k, n, dw);
    Matrix<> ref_C(m, n, dw, "ref_C");

    /* copy (sparse) initial data to a set of reference matrices */
    ref_A["ij"] = A["ij"];
    ref_B["ij"] = B["ij"];
    ref_C["ij"] = C["ij"];

    /* compute reference answer */
    switch (sym_C){
      case NS:
        ref_C["ik"] += ref_A["ij"]*ref_B["jk"];
        break;
      case SY:
      case SH:
        ref_C["ik"] += ref_A["ij"]*ref_B["jk"];
        ref_C["ik"] += ref_A["kj"]*ref_B["ji"];
        if (sym_C == SH) ref_C["ii"] = 0.0;
        break;
      case AS:
        ref_C["ik"] += ref_A["ij"]*ref_B["jk"];
        ref_C["ik"] -= ref_A["kj"]*ref_B["ji"];
        break;
    }
    /* compute answer for matrices with attributes as specified */
    C["ik"] += .5*A["ij"]*B["jk"];
    C["ik"] += .5*B["jk"]*A["ij"];

           
    /* compute difference in answer */
    ref_C["ik"] -= C["ik"];

    pass = ref_C.norm2() <= 1.E-6;

    if (dw.rank == 0){
      if (pass) 
        printf("{ C[\"ik\"] += A[\"ij\"]*B[\"jk\"] with A (%d*%d sym %d sp %lf), B (%d*%d sym %d sp %lf), C (%d*%d sym %d sp %lf) } passed \n",m,k,sym_A,sp_A,k,n,sym_B,sp_B,m,n,sym_C,sp_C);
      else
        printf("{ C[\"ik\"] += A[\"ij\"]*B[\"jk\"] with A (%d*%d sym %d sp %lf), B (%d*%d sym %d sp %lf), C (%d*%d sym %d sp %lf) } failed \n",m,k,sym_A,sp_A,k,n,sym_B,sp_B,m,n,sym_C,sp_C);
    }
  }
  if (bench){
    double min_time = DBL_MAX;
    double max_time = 0.0;
    double tot_time = 0.0;
    double times[niter];

    if (dw.rank == 0){
      printf("Starting %d benchmarking iterations of matrix multiplication with specified attributes...\n", niter);
      initialize_flops_counter();
    }
    min_time = DBL_MAX;
    max_time = 0.0;
    tot_time = 0.0;
    Timer_epoch smatmul("specified matmul");
    smatmul.begin();
    for (int i=0; i<niter; i++){
      //A.print(); 
      //B.print();

      double start_time = MPI_Wtime();
      C["ik"] = A["ij"]*B["jk"];
      //C.print();
      double end_time = MPI_Wtime();
      double iter_time = end_time-start_time;
      times[i] = iter_time;
      tot_time += iter_time;
      if (iter_time < min_time) min_time = iter_time;
      if (iter_time > max_time) max_time = iter_time;
    }
    smatmul.end();
    
    if (dw.rank == 0){
      printf("iterations completed, did %ld flops.\n",CTF::get_estimated_flops());
      printf("All iterations times: ");
      for (int i=0; i<niter; i++){
        printf("%lf ", times[i]);
      }
      printf("\n");
      std::sort(times,times+niter);
      printf("A (%d*%d sym %d sp %lf), B (%d*%d sym %d sp %lf), C (%d*%d sym %d sp %lf) Min time = %lf, Avg time = %lf, Med time = %lf, Max time = %lf\n",m,k,sym_A,sp_A,k,n,sym_B,sp_B,m,n,sym_C,sp_C,min_time,tot_time/niter, times[niter/2], max_time);
    }
  
  }
    
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
  int rank, np, m, n, k, pass, niter, bench, sym_A, sym_B, sym_C, test;
  double sp_A, sp_B, sp_C;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 17;
  } else m = 17;

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 32;
  } else n = 32;

  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 9;
  } else k = 9;

  if (getCmdOption(input_str, input_str+in_num, "-sym_A")){
    sym_A = atoi(getCmdOption(input_str, input_str+in_num, "-sym_A"));
    if (sym_A != AS && sym_A != SY && sym_A != SH) sym_A = NS;
  } else sym_A = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sym_B")){
    sym_B = atoi(getCmdOption(input_str, input_str+in_num, "-sym_B"));
    if (sym_B != AS && sym_B != SY && sym_B != SH) sym_B = NS;
  } else sym_B = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sym_C")){
    sym_C = atoi(getCmdOption(input_str, input_str+in_num, "-sym_C"));
    if (sym_C != AS && sym_C != SY && sym_C != SH) sym_C = NS;
  } else sym_C = NS;

  if (getCmdOption(input_str, input_str+in_num, "-sp_A")){
    sp_A = atof(getCmdOption(input_str, input_str+in_num, "-sp_A"));
    if (sp_A < 0.0 || sp_A > 1.0) sp_A = .2;
  } else sp_A = .2;

  if (getCmdOption(input_str, input_str+in_num, "-sp_B")){
    sp_B = atof(getCmdOption(input_str, input_str+in_num, "-sp_B"));
    if (sp_B < 0.0 || sp_B > 1.0) sp_B = .2;
  } else sp_B = .2;

  if (getCmdOption(input_str, input_str+in_num, "-sp_C")){
    sp_C = atof(getCmdOption(input_str, input_str+in_num, "-sp_C"));
    if (sp_C < 0.0 || sp_C > 1.0) sp_C = .2;
  } else sp_C = .2;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 10;
  } else niter = 10;

  if (getCmdOption(input_str, input_str+in_num, "-bench")){
    bench = atoi(getCmdOption(input_str, input_str+in_num, "-bench"));
    if (bench != 0 && bench != 1) bench = 1;
  } else bench = 1;
  
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test != 0 && test != 1) test = 1;
  } else test = 1;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying A (%d*%d sym %d sp %lf) and B (%d*%d sym %d sp %lf) into C (%d*%d sym %d sp %lf) \n",m,k,sym_A,sp_A,k,n,sym_B,sp_B,m,n,sym_C,sp_C);
    }
    pass = matmul(m, n, k, dw, sym_A, sym_B, sym_C, sp_A, sp_B, sp_C, test, bench, niter);
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
