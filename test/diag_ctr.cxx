/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup diag_ctr diag_ctr
  * @{ 
  * \brief Summation along tensor diagonals
  */
#include <ctf.hpp>

using namespace CTF;

int diag_ctr(int     n,
             int     m,
             World & dw){
  int rank, i, num_pes, pass;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,m,n,m};

  //* Creates distributed tensors initialized with zeros
  Tensor<> A(4, sizeN4, shapeN4, dw);

  srand48(13*rank);

  Matrix<> mA(n,m,NS,dw);
  Matrix<> mB(n,m,NS,dw);
  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  A.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  pass = 1;
  double tr = 0.0;
  tr += A["aiai"];
  if (fabs(tr) < 1.E-10){
    pass = 0;
  }
  mA["ai"] = A["aiai"];
  tr -= mA["ai"];

  if (fabs(tr) > 1.E-10)
    pass = 0;
  if (pass){
    if (rank == 0)
      printf("{ sum(ai)A[\"aiai\"]=sum(ai)mA[\"ai\"] } passed \n");
  } else {
    if (rank == 0)
      printf("{ sum(ai)A[\"aiai\"]=sum(ai)mA[\"ai\"] } failed \n");
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
  int rank, np, n, m;
  int in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 7;
  } else m = 7;

  {
    World dw(argc, argv);
    diag_ctr(n, m, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
