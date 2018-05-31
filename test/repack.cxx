/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup repack repack 
  * @{ 
  * \brief Tests contraction of a symmetric index group with a nonsymmetric one
  */

#include <ctf.hpp>

using namespace CTF;

int repack(int     n,
           World & dw){
  int rank, i, num_pes, pass;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {NS,NS,NS,NS};
  int shapeS4[] = {NS,NS,SY,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  Tensor<> An(4, sizeN4, shapeN4, dw);
  Tensor<> As(4, sizeN4, shapeS4, dw);

  As.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  As.write(np, indices, pairs);
  An.write(np, indices, pairs);

  Tensor<> Anr(An, shapeS4);
 
  Anr["ijkl"] -= As["ijkl"];

  double norm = Anr.norm2();

  if (norm < 1.E-6)
    pass = 1;
  else
    pass = 0;
  
  if (!pass)
    printf("{ NS -> SY repack } failed \n");
  else {
    Tensor<> Anur(As, shapeN4);
    Tensor<> Asur(As, shapeN4);
    Asur["ijkl"] = 0.0;
    Asur.write(np, indices, pairs);
    Anur["ijkl"] -= Asur["ijkl"];

    norm = Anur.norm2();

    if (norm < 1.E-6){
      pass = 1;
      if (rank == 0)
        printf("{ NS -> SY -> NS repack } passed \n");
    } else {
      pass = 0;
      if (rank == 0)
        printf("{ SY -> NS repack } failed \n");
    }

  }
  delete [] pairs;
  free(indices);
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
    repack(n, dw);
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
