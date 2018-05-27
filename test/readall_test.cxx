/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup readall_test readall_test
  * @{ 
  * \brief Summation along tensor diagonals
  */

#include <ctf.hpp>

using namespace CTF;

int readall_test(int   n,
                 int   m,
                 World &dw){
  int rank, i, num_pes, pass;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,m,n,m};

  //* Creates distributed tensors initialized with zeros
  Tensor<> A(4, sizeN4, shapeN4, dw);

  std::vector<double> vals;
  std::vector<int64_t> inds;
  if (rank == 0){
    World sw(MPI_COMM_SELF);
    
    Tensor<> sA(4, sizeN4, shapeN4, sw);

    
    if (rank == 0){ 
      srand48(13*rank);
      for (i=0; i<n*m*n*m; i++){
        vals.push_back(drand48());
        inds.push_back(i);
      }
    }
    
    sA[inds] = vals;

    A.add_from_subworld(&sA);
  } else 
    A.add_from_subworld(NULL);

  double * vs;
  int64_t ns;

  A.read_all(&ns, &vs);

  assert(ns == n*n*m*m);
  

  pass = 1;
  if (rank == 0){
    for (i=0; i<ns; i++){
      if (fabs(vs[i]-vals[i])>1.E-10)
        pass = 0;
    }
  }
  delete [] vs;

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
    if (m < 0) m = 9;
  } else m = 9;



  {
    World dw(argc, argv);
    readall_test(n, m, dw);
  }

  MPI_Finalize();
  return 0;
}
#endif
/**
 * @} 
 * @}
 */

