/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup sy_times_ns sy_times_ns 
  * @{ 
  * \brief Tests contraction of a symmetric index group with a nonsymmetric one
  */

#include <ctf.hpp>

using namespace CTF;

int sy_times_ns(int     n,
                World & dw){
  int rank, i, num_pes, pass;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  Tensor<> B(4, sizeN4, shapeN4, dw);

  Matrix<> A(n, n, SY, dw);
  Matrix<> An(n, n, NS, dw);
  Matrix<> C(n, n, SY, dw, "C");
  Matrix<> Cn(n, n, NS, dw, "Cn");

  srand48(13*rank);


  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
//  A.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
//  B.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  C.write(np, indices, pairs);
  delete [] pairs;
  free(indices);

  An["ij"] = A["ij"];
  Cn["ij"] = C["ij"];

  C["ij"] += A["ij"]*B["ijkl"];
  Cn["ij"] += An["ij"]*B["ijkl"];
  Cn["ji"] += An["ij"]*B["ijkl"];


  Cn["ij"] -= C["ij"];

  double norm = Cn.norm2();
  
  if (norm < 1.E-10){
    pass = 1;
    if (rank == 0)
      printf("{ C[\"(ij)\"]=A[\"(ij)\"]*B[\"ijkl\"] } passed \n");
  } else {
    pass = 0;
    if (rank == 0)
      printf("{ C[\"(ij)\"]=A[\"(ij)\"]*B[\"ijkl\"] } failed \n");
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
    sy_times_ns(n, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
