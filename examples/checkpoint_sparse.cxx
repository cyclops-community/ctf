/*Copyright (c) 2016, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup sparse_checkpoint sparse_checkpoint 
  * @{ 
  * \brief tests read and write dense data to file functionality
  */

#include <ctf.hpp>
using namespace CTF;

int sparse_checkpoint(int     n,
               World & dw,
               int     qtf=NS){

  Matrix<> A(n, n, qtf, dw);
  Matrix<> A2(n, n, qtf, dw);
  Matrix<> A3(n, n, qtf, dw);
  Matrix<> A4(n, n, qtf, dw);
  Matrix<> A5(n, n, qtf, dw);

  int lens_u[] = {5,5,5};

  Tensor<> u(3, lens_u);
  u.print();
  u.read_sparse_from_file("tensor.txt");
  u.print();
  srand48(13*dw.rank);

  bool pass =true;
   
  if (dw.rank == 0){
    if (!pass){
      printf("{ sparse_checkpointing using dense data representation with qtf=%d } failed\n",qtf);
    } else {
      printf("{ sparse_checkpointing using dense data representation with qtf=%d } passed\n",qtf);
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
  int rank, np, n, qtf;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-qtf")){
    qtf = atoi(getCmdOption(input_str, input_str+in_num, "-qtf"));
    if (qtf < 0) qtf = NS;
  } else qtf = NS;



  {
    World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Checking sparse_checkpoint calculation n = %d, p = %d, qtf = %d:\n",n,np,qtf);
    }
    int pass = sparse_checkpoint(n,dw,qtf);
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
