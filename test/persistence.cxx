/** \addtogroup examples 
  * @{ 
  * \defgroup persistence Tensor contraction 
  * @{ 
  * \brief Multiplication of two matrices
  */

/**
 * Non symmetric
 * TODO: Test for rank>2 tensor
 *       Sparse
 */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

/**
 * \brief tests m*n*k matrix multiplication
 * \param[in] m number of rows in C, A
 * \param[in] n number of cols in C, B
 * \param[in] k number of rows in A, cols in B
 * \param[in] dw set of processors on which to execute matmul
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
           double  sp_A=1.,
           double  sp_B=1.,
           double  sp_C=1.,
           bool    test=true,
           int     bench=false,
           int     niter=3){
  assert(test || bench);

  /*
  int sA = sp_A < 1. ? SP : 0;
  int sB = sp_B < 1. ? SP : 0;
  int sC = sp_C < 1. ? SP : 0;
  */

  Matrix<> ref_A(m, k, dw);
  Matrix<> ref_B(k, n, dw);
  Matrix<> ref_C(m, n, dw, "ref_C");

  srand48(dw.rank);
  if (sp_A < 1.)
    ref_A.fill_sp_random(0.0,1.0,sp_A);
  else
    ref_A.fill_random(0.0,1.0);
  if (sp_B < 1.)
    ref_B.fill_sp_random(0.0,1.0,sp_B);
  else
    ref_B.fill_random(0.0,1.0);
  if (sp_C < 1.)
    ref_C.fill_sp_random(0.0,1.0,sp_C);
  else
    ref_C.fill_random(0.0,1.0);

  bool pass = false;
  if (bench == 1) {
    printf("Benchmarking is not yet implemented\n");
    return 0;
  }
  if (test == 1) {
    double err;
    /* copy initial data to a set of reference matrices */
    Matrix<> A0(m, k, dw);
    Matrix<> B0(k, n, dw);
    Matrix<> C0(m, n, dw, "C");
    A0["ij"] = ref_A["ij"];
    B0["ij"] = ref_B["ij"];
    C0["ij"] = ref_C["ij"];

    for (int i = 0; i < niter; i++) {
      C0.contract(1.0, A0, "ij", B0, "jk", 0.0, "ik");
      A0["ij"] = C0["ij"];
    }

    /* ----------------------------------------------------------------------------------------- */

    /* contract using persistence call */
    Matrix<> A1(m, k, dw, "A1");
    Matrix<> B1(k, n, dw);
    Matrix<> C1(m, n, dw, "C1");
    A1["ij"] = ref_A["ij"];
    B1["ij"] = ref_B["ij"];
    C1["ij"] = ref_C["ij"];

    Contract<> *x;
    x = new Contract<>(1.0, A1, "ij", B1, "jk", 0.0, C1, "ik");
    for (int i = 0; i < niter; i++) {
      x->execute();
      A1["ij"] = C1["ij"];
    }
    delete x; // Release A1, B1, C1

    /* compute difference in answer */
    C1["ij"] -= C0["ij"];
    err = C1.norm2();
    pass = err <= 1.E-6;
    assert(pass);
    /* check if tensors A and B have corrupted data after releaseA() and releaseB() */
    A1["ij"] -= A0["ij"];
    err = A1.norm2();
    pass = (pass == true) ? (err <= 1.E-6) : false;
    assert(pass);
    B1["ij"] -= B0["ij"];
    err = B1.norm2();
    pass = (pass == true) ? (err <= 1.E-6) : false;
    assert(pass);
    if (dw.rank == 0) {
      printf("Error when execute() is called for %d iterations is: %lf\n", niter, err);
    }

    /* ----------------------------------------------------------------------------------------- */
   
    /* prepareA() call; upload AA in place of A */ 
    Matrix<> A2(m, k, dw);
    Matrix<> B2(k, n, dw);
    Matrix<> C2(m, n, dw, "C2");
    A2["ij"] = ref_A["ij"];
    B2["ij"] = ref_B["ij"];
    C2["ij"] = ref_C["ij"];

    Contract<> *y;
    y = new Contract<>(1.0, A2, "ij", B2, "jk", 0.0, C2, "ik");
    
    Matrix<> *AA = new Matrix<>(m, k, dw);
    (*AA)["ij"] = ref_A["ij"];
    Matrix<> *BB = new Matrix<>(k, n, dw);
    (*BB)["ij"] = ref_B["ij"];
    
    Matrix<> *tempA = nullptr;
    Matrix<> *tempB = nullptr;
    
    for (int i = 0; i < niter; i++) {      
      y->prepareA(*AA, "ij"); // upload AA in place of A2, and then release A2
      y->prepareB(*BB, "ij"); // upload BB in place of B2, and then release B2
      
      y->execute();
     
      if (tempA != nullptr) delete tempA;
      tempA = AA;
      AA = new Matrix<>(m, k, dw); // create a new Matrix to be uploaded in the next iteration
      (*AA)["ij"] = C2["ij"];
      
      if (tempB != nullptr) delete tempB;
      tempB = BB;
      BB = new Matrix<>(k, n, dw); // create a new Matrix to be uploaded in the next iteration
      (*BB)["ij"] = ref_B["ij"];
    }
    delete y; // Release A, B, C
    A2["ij"] = (*AA)["ij"];
    delete AA;
    delete BB;

    /* compute difference in answer */
    C2["ij"] -= C0["ij"];
    err = C2.norm2();
    pass = err <= 1.E-6;
    assert(pass);
    /* check if tensors A and B have corrupted data after releaseA() and releaseB() */
    A2["ij"] -= A0["ij"];
    err = A2.norm2();
    pass = (pass == true) ? (err <= 1.E-6) : false;
    assert(pass);
    B2["ij"] -= B0["ij"];
    err = B2.norm2();
    assert(pass);
    pass = (pass == true) ? (err <= 1.E-6) : false;
    if (dw.rank == 0) {
      printf("Error when execute() followed by calls prepareA() and prepareB() for %d iterations is: %lf\n", niter, err);
    }
    /* ----------------------------------------------------------------------------------------- */
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
  int rank, np, m, n, k, pass, niter, bench, test;
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
    if (n < 0) n = 9;
  } else n = 9;

  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 9;
  } else k = 9;

  if (getCmdOption(input_str, input_str+in_num, "-sp_A")){
    sp_A = atof(getCmdOption(input_str, input_str+in_num, "-sp_A"));
    if (sp_A < 0.0 || sp_A > 1.0) sp_A = 1.;
  } else sp_A = 1.;

  if (getCmdOption(input_str, input_str+in_num, "-sp_B")){
    sp_B = atof(getCmdOption(input_str, input_str+in_num, "-sp_B"));
    if (sp_B < 0.0 || sp_B > 1.0) sp_B = 1.;
  } else sp_B = 1.;

  if (getCmdOption(input_str, input_str+in_num, "-sp_C")){
    sp_C = atof(getCmdOption(input_str, input_str+in_num, "-sp_C"));
    if (sp_C < 0.0 || sp_C > 1.0) sp_C = 1.;
  } else sp_C = 1.;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 1;
  } else niter = 4;

  if (getCmdOption(input_str, input_str+in_num, "-bench")){
    bench = atoi(getCmdOption(input_str, input_str+in_num, "-bench"));
    if (bench < 0) bench = 0;
  } else bench = 0;
  
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test != 0 && test != 1) test = 1;
  } else test = 1;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying A (%d*%d sp %lf) and B (%d*%d sp %lf) into C (%d*%d sp %lf) \n",m,k,sp_A,k,n,sp_B,m,n,sp_C);
    }
    pass = matmul(m, n, k, dw, sp_A, sp_B, sp_C, test, bench, niter);
    assert(pass);
    if (rank == 0) printf("Test passed for %d iterations\n", niter);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
