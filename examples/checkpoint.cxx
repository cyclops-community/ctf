/*Copyright (c) 2016, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup checkpoint checkpoint 
  * @{ 
  * \brief tests read and write dense data to file functionality
  */

#include <ctf.hpp>
using namespace CTF;

int checkpoint(int     n,
               World & dw,
               int     qtf=NS){

  Matrix<> A(n, n, qtf, dw);
  Matrix<> A2(n, n, qtf, dw);
  Matrix<> A3(n, n, qtf, dw);
  Matrix<> A4(n, n, qtf, dw);
  Matrix<> A5(n, n, qtf, dw);

  srand48(13*dw.rank);
  A.fill_random(0.0,1.0);
  A.print();
  A["ii"] = 0.0;
  A2["ij"] = A["ij"];
  A3["ij"] = 2.*A["ij"];
 
  MPI_File file;
  MPI_File_open(dw.comm, "CTF_checkpoint_test_file.bin",  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &file);
  A2.write_dense_to_file(file);
  A3.write_dense_to_file(file,n*n*sizeof(double));
  MPI_File_close(&file);
  
  MPI_File_open(dw.comm, "CTF_checkpoint_test_file.bin",  MPI_MODE_RDONLY | MPI_MODE_DELETE_ON_CLOSE, MPI_INFO_NULL, &file);
  A4.read_dense_from_file(file);
 
  A4.print(); 
  A["ij"] -= A4["ij"];
  int pass = A.norm2() <= 1.e-9*n; 
  
  A5.read_dense_from_file(file,n*n*sizeof(double));
  MPI_File_close(&file);
  A5["ij"] -= 2.*A4["ij"];
  pass = pass & (A5.norm2() <= 1.e-9*n); 
    
  if (dw.rank == 0){
    if (!pass){
      printf("{ checkpointing using dense data representation with qtf=%d } failed\n",qtf);
    } else {
      printf("{ checkpointing using dense data representation with qtf=%d } passed\n",qtf);
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
      printf("Checking checkpoint calculation n = %d, p = %d, qtf = %d:\n",n,np,qtf);
    }
    int pass = checkpoint(n,dw,qtf);
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
