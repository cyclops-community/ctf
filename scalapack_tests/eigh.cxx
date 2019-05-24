/** \addtogroup tests 
  * @{ 
  * \defgroup eigh eigh
  * @{ 
  * \brief symmetric eigensolve factorization of CTF matrices
  */

#include <ctf.hpp>
#include "conj.h"
using namespace CTF;


template <typename dtype>
bool eigh(Matrix<dtype> A){
  // Perform symmetric eigensolve
  Matrix<dtype> U;
  Vector<dtype> D;
  A.eigh(U,D);
  int n = A.nrow;
  // Test orthogonality
  Matrix<dtype> E(n,n,*A.wrld);

  E["ii"] = 1.;

  E["ij"] -= U["ki"]*conj<dtype>(U)["kj"];

  bool pass_orthogonality = true;

  double nrm1, nrm2;
  E.norm2(nrm1);
  if (nrm1 > n*n*1.E-6){
    pass_orthogonality = false;
  }

  E.set_zero();
  E["ij"] = A["ik"]*U["kj"] - U["ij"]*D["j"];

  bool pass_residual = true;
  E.norm2(nrm2);
  if (nrm2 > n*n*1.E-6){
    pass_residual = false;
  }

#ifndef TEST_SUITE
  if (A.wrld->rank == 0){
    printf("symmetric eigensolve orthogonality check returned %d, residual check %d\n", pass_orthogonality, pass_residual);
  }
#else
  if (!pass_residual || ! pass_orthogonality){
    if (A.wrld->rank == 0){
      printf("symmetric eigensolve orthogonality check returned %d (%lf), residual check %d (%lf)\n", pass_orthogonality, nrm1, pass_residual, nrm2);
    }
  }
#endif
  return pass_residual & pass_orthogonality;
} 

template <typename dtype>
bool test_eigh_dt(int n, World dw){
  bool pass = true;
  Matrix<dtype> A_ss(n,n,SP|SY,dw);
  A_ss.fill_sp_random(-1.,1.,.8);
  Matrix<dtype> A_s(n,n,SY,dw);
  Matrix<dtype> A(n,n,dw);
  A_s["ij"] = A_ss["ij"];
  A["ij"] = A_ss["ij"];
  pass = pass & eigh<dtype>(A);
  pass = pass & eigh<dtype>(A_s);
  pass = pass & eigh<dtype>(A_ss);

  return pass;
}

bool test_eigh(int n, World dw){
  bool pass = true;
  pass = pass & test_eigh_dt<float>(n, dw);
  pass = pass & test_eigh_dt<double>(n, dw);
  pass = pass & test_eigh_dt<std::complex<float>>(n, dw);
  pass = pass & test_eigh_dt<std::complex<double>>(n, dw);

  if (dw.rank == 0){
    if (pass){
      printf("{ AX = XD and X^HX = I } passed\n");
    } else {
      printf("{ AX = XD and X^HX = I } failed\n");
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
  int rank, np, n, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 5;
  } else n = 5;
 
  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Testing %d-by-%d symmetric eigensolve\n", n, n);
    }
    pass = test_eigh(n, dw);
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
