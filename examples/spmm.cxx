/** \addtogroup examples 
  * @{ 
  * \defgroup spmm spmm
  * @{ 
  * \brief Multiplication of a random square sparse matrix by a vector
  */

#include <ctf.hpp>
using namespace CTF;

int spmm(int     n,
         int     k,
         World & dw){

  Matrix<> spA(true,  n, n, NS, dw);
  Matrix<> dnA(false, n, n, NS, dw);
  Matrix<> b(n, k, NS, dw);
  Matrix<> c1(n, k, NS, dw);
  Matrix<> c2(n, k, NS, dw);

  srand48(dw.rank);

  b.fill_random(0.0,1.0);
  c1.fill_random(0.0,1.0);
  dnA.fill_random(0.0,1.0);

  spA["ij"] += dnA["ij"];
  spA.sparsify(.5);
  dnA["ij"] = 0.0;
  dnA["ij"] += spA["ij"];
  
  /*printf("dense A\n");
  dnA.print();
  printf("sparse A\n");
  spA.print();*/

  c2["ik"] = c1["ik"];
  
  c1["ik"] += dnA["ij"]*b["jk"];
  
  c2["ik"] += spA["ij"]*b["jk"];
  //c2["ik"] += 0.5*b["jk"]*spA["ij"];

  /*if (dw.rank == 0) printf("b\n");
  b.print();
  if (dw.rank == 0) printf("dense c + A * b\n");
  c1.print();
  if (dw.rank == 0) printf("sparse c + A * b\n");
  c2.print();*/

  assert(c2.norm2() >= 1E-6);

  c2["ik"] -= c1["ik"];

  bool pass = c2.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ c[\"ik\"] += A[\"ij\"]*b[\"jk\"] with sparse, A } passed \n");
    else
      printf("{ c[\"ik\"] += A[\"ij\"]*b[\"jk\"] with sparse, A } failed \n");
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
  int rank, np, n, k, pass;
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

  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Multiplying %d-by-%d sparse matrix by %d-by-%d dense matrix\n",n,n,n,k);
    }
    pass = spmm(n, k, dw);
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
