/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup reduce_bcast reduce_bcast 
  * @{ 
  * \brief Summation along tensor diagonals
  */

#include <ctf.hpp>

using namespace CTF;

int reduce_bcast(int     n,
                 World & dw){
  int pass;

  Matrix<> A(n,n,dw);
  Matrix<> B(n,1,dw);
  Matrix<> C(n,n,dw);
  Matrix<> C2(n,n,dw);
  Vector<> d(n,dw);

  srand48(13*dw.rank);

  A.fill_random(0.,1.);
  B.fill_random(0.,1.);
  C.fill_random(0.,1.);
  C2["ij"] = C["ij"];
  d.fill_random(0.,1.);

  C["ij"] += B["ik"];

  d["i"] = B["ij"];

  C2["ij"] += d["i"];

  C["ij"] -= C2["ij"];

  pass = true;
  if (C.norm2() > 1.E-6){
    pass = false;
    if (dw.rank == 0)
      printf("{ (A[\"ij\"]+=B[\"ik\"] with square B } failed \n");
    return pass;
  }

  C["ij"] = C2["ij"];

  C["ij"] += B["ik"];

  d["i"] = B["ik"];
  
  C2["ij"] += d["i"];

  C["ij"] -= C2["ij"];

  if (C.norm2() > 1.E-6)
    pass = false;

  if (pass){
    if (dw.rank == 0)
      printf("{ (A[\"ij\"]+=B[\"ik\"] } passed \n");
  } else {
    if (dw.rank == 0)
      printf("{ (A[\"ij\"]+=B[\"ik\"] with column vector B } failed \n");
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
  int rank, np, n;
  int in_num = argc;
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
    reduce_bcast(n, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
