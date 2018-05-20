/** \addtogroup tests 
  * @{ 
  * \defgroup qr qr
  * @{ 
  * \brief QR factorization of CTF matrices
  */

#include <ctf.hpp>
using namespace CTF;

template <typename dtype>
bool qr(int     m,
        int     n,
        World & dw){

  Matrix<dtype> A(m,n,dw);

  // Perform QR
  Matrix<dtype> Q,R;
  A.qr(Q,R);

  // Test orthogonality
  Matrix<dtype> E(n,n,dw);

  E["ij"] = Q["ik"]*Q["jk"];

  bool pass_orthogonality = true;

  double nrm;
  E.norm2(nrm);
  if (nrm > 1.E-6){
    pass_orthogonality = false;
  }

  A["ij"] = Q["ik"]*R["kj"];

  bool pass_residual = true;
  A.norm2(nrm);
  if (nrm > 1.E-6){
    pass_residual = false;
  }

#ifndef TEST_SUITE
  if (dw.rank == 0){
    printf("QR orthogonality check returned %d, residual check %d\n", pass_orthogonality, pass_residual);
  }
#endif
  return pass_residual & pass_orthogonality;
} 

bool test_qr(int m, int n, World dw){
  bool pass = true;
  pass = pass & qr<float>(m,n,dw);
  pass = pass & qr<double>(m,n,dw);
  pass = pass & qr<std::complex<float>>(m,n,dw);
  pass = pass & qr<std::complex<double>>(m,n,dw);
  if (dw.rank == 0){
    if (pass){
      printf("{ A = QR and Q^TQ = I } passed\n");
    } else {
      printf("{ A = QR and Q^TQ = I } failed\n");
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
  int rank, np, m, n, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 13;
  } else m = 13;


  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Testing %d-by-%d QR factorization\n", m, n);
    }
    pass = test_qr(m, n, dw);
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
