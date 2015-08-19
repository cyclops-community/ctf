/** \addtogroup examples 
  * @{ 
  * \defgroup spmv spmv
  * @{ 
  * \brief Multiplication of a random square sparse matrix by a vector
  */

#include <ctf.hpp>
using namespace CTF;

int spmv(int     n,
         World & dw){

  Matrix<> spA(true,  n, n, NS, dw);
  Matrix<> dnA(false, n, n, NS, dw);
  Vector<> b(n, dw);
  Vector<> c1(n, dw);
  Vector<> c2(n, dw);

  b.fill_random(0.0,1.0);
  c1.fill_random(0.0,1.0);
  dnA.fill_random(0.0,1.0);

  spA["ij"] += dnA["ij"];
  spA.sparsify(.5);
  dnA["ij"] = 0.0;
  dnA["ij"] += spA["ij"];
  
  printf("dense A\n");
  dnA.print();
  printf("sparse A\n");
  spA.print();

  c2["i"] = c1["i"];
  
  c1["i"] += dnA["ij"]*b["j"];

  
  c2["i"] += spA["ij"]*b["j"];

  printf("b\n");
  b.print();
  printf("dense c + A * b\n");
  c1.print();
  printf("sparse c + A * b\n");
  c2.print();


  assert(c2.norm2() >= 1E-6);

  c2["i"] -= c1["i"];

  bool pass = c2.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse, A } passed \n");
    else
      printf("{ c[\"i\"] += A[\"ij\"]*b[\"j\"] with sparse, A } failed \n");
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
    if (n < 0) n = 7;
  } else n = 7;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying sparse matrix by vector with n=%d\n",n);
    }
    pass = spmv(n, dw);
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
