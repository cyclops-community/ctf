/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup diag_sym diag_sym 
  * @{ 
  * \brief Summation along tensor diagonals
  */

#include <ctf.hpp>

using namespace CTF;

int diag_sym(int     n,
             World & dw){
  int rank, i, num_pes, pass;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {SY,NS,SY,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  Tensor<> A(4, sizeN4, shapeN4, dw);
  Tensor<> B(4, sizeN4, shapeN4, dw);
  Tensor<> C(4, sizeN4, shapeN4, dw);

  srand48(13*rank);

  Matrix<> mA(n,n,NS,dw);
  Matrix<> mB(n,n,NS,dw);
  mA.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  mA.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  mB.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  mB.write(np, indices, pairs);
  delete [] pairs;
  free(indices);

  A["abij"] = mA["ii"];
  B["abij"] = mA["jj"];
  A["abij"] -= mB["aa"];
  B["abij"] -= mB["bb"];
  C["abij"] = A["abij"]-B["abij"];

  double norm = C.norm2();
  
  if (norm < 1.E-10){
    pass = 1;
    if (rank == 0)
      printf("{ (A[\"(ab)(ij)\"]=mA[\"ii\"]-mB[\"aa\"]=mA[\"jj\"]-mB[\"bb\"] } passed \n");
  } else {
    pass = 0;
    if (rank == 0)
      printf("{ (A[\"(ab)(ij)\"]=mA[\"ii\"]-mB[\"aa\"]=mA[\"jj\"]-mB[\"bb\"] } failed \n");
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
    diag_sym(n, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
