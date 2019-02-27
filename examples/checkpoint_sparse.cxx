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
  int lens_u[] = {n,n,n};

  Tensor<> u(3, true, lens_u);
  u.fill_sp_random(0., 1., .1);
  u.write_sparse_to_file("checkpoint_sparse_tensor.txt");
  Tensor<> v(3, true, lens_u);
  
  v.read_sparse_from_file("checkpoint_sparse_tensor.txt");
  srand48(13*dw.rank);
  v["ijk"] -= u["ijk"];

  bool pass = v.norm2() < 1.e-7*n*n*.1*n;

  MPI_Info info;
  MPI_File_delete("checkpoint_sparse_tensor.txt", info);
   
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
