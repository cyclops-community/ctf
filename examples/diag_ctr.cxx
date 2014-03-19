/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup diag_ctr 
  * @{ 
  * \brief Summation along tensor diagonals
  */

#include <stdio.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

int diag_ctr(int const    n,
             int const    m,
             CTF_World   &dw){
  int rank, i, num_pes, pass;
  int64_t np;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {NS,NS,NS,NS};
  int sizeN4[] = {n,m,n,m};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(4, sizeN4, shapeN4, dw);

  srand48(13*rank);

  CTF_Matrix mA(n,m,NS,dw);
  CTF_Matrix mB(n,m,NS,dw);
  A.fill_random(-.5,.5);
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
      printf("{sum(ai)A[\"aiai\"]=sum(ai)mA[\"ai\"]} passed \n");
  } else {
    if (rank == 0)
      printf("{sum(ai)A[\"aiai\"]=sum(ai)mA[\"ai\"]} failed \n");
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
  int const in_num = argc;
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
    CTF_World dw(argc, argv);
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
