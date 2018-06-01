/** \addtogroup tests 
  * @{ 
  * \defgroup svd svd
  * @{ 
  * \brief SVD factorization of CTF matrices
  */

#include <ctf.hpp>
using namespace CTF;

template <typename dtype>
Matrix<dtype> conj(Matrix<dtype> & A){
  return A;
}
template <>
Matrix< std::complex<float> > conj(Matrix< std::complex<float> > & A){
  Matrix< std::complex<float> > B(A);
  B["ij"] = Function< std::complex<float>>([](std::complex<float> a){ return std::conj(a); })(A["ij"]);
  return B;
}
template <>
Matrix<std::complex<double>> conj(Matrix<std::complex<double>> & A){
  Matrix<std::complex<double>> B(A);
  B["ij"] = Function<std::complex<double>>([](std::complex<double> a){ return std::conj(a); })(A["ij"]);
  return B;
}

template <typename dtype>
bool svd(Matrix<dtype> A,
        int     m,
        int     n,
        int     k,
        World & dw){

  // Perform SVD
  Matrix<dtype> U,VT;
  Vector<dtype> S;
  A.svd(U,S,VT,k);
  // Test orthogonality
  Matrix<dtype> E(k,k,dw);

  E["ii"] = 1.;

  E["ij"] -= U["ki"]*conj<dtype>(U)["kj"];

  bool pass_orthogonality = true;

  double nrm;
  E.norm2(nrm);
  if (nrm > m*n*1.E-6){
    pass_orthogonality = false;
  }

  E["ii"] = 1.;

  E["ij"] -= VT["ik"]*conj<dtype>(VT)["jk"];

  E.norm2(nrm);
  if (nrm > m*n*1.E-6){
    pass_orthogonality = false;
  }

  A["ij"] -= U["ik"]*S["k"]*VT["kj"];

  bool pass_residual = true;
  A.norm2(nrm);
  if (nrm > m*n*n*1.E-6){
    pass_residual = false;
  }

#ifndef TEST_SUITE
  if (dw.rank == 0){
    printf("SVD orthogonality check returned %d, residual check %d\n", pass_orthogonality, pass_residual);
  }
#endif
  return pass_residual & pass_orthogonality;
} 

bool test_svd(int m, int n, int k, World dw){
  bool pass = true;
  Matrix<float> A(m,n,dw);
  Matrix<float> AA(m,n,dw);
  A.fill_random(0.,1.);
  AA.fill_random(0.,1.);
  pass = pass & svd<float>(A,m,n,k,dw);

  Matrix<double> B(m,n,dw);
  Matrix<double> BB(m,n,dw);
  B.fill_random(0.,1.);
  BB.fill_random(0.,1.);
  pass = pass & svd<double>(B,m,n,k,dw);

  Matrix<std::complex<float>> cA(m,n,dw);
  cA["ij"] = Function<float,float,std::complex<float>>([](float a, float b){ return std::complex<float>(a,b); })(A["ij"],AA["ij"]);
  pass = pass & svd<std::complex<float>>(cA,m,n,k,dw);


  Matrix<std::complex<double>> cB(m,n,dw);
  cB["ij"] = Function<double,double,std::complex<double>>([](double a, double b){ return std::complex<double>(a,b); })(B["ij"],BB["ij"]);
  pass = pass & svd<std::complex<double>>(cB,m,n,k,dw);
  if (dw.rank == 0){
    if (pass){
      printf("{ A = USVT and U^TU = I } passed\n");
    } else {
      printf("{ A = USVT and U^TU = I } failed\n");
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
  int rank, np, m, n, k, pass;
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
    if (n < 0) n = 5;
  } else n = 5;


  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = std::min(m,n);
  } else k = std::min(m,n);

  assert(k<=std::min(m,n));

 
  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Testing rank %d %d-by-%d SVD factorization\n", k, m, n);
    }
    pass = test_svd(m, n, k, dw);
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
