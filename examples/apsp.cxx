/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup apsp
  * @{ 
  * \brief Matrix multiplication
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

int add(int a, int b) { return a+b; }
int mul(int a, int b) { return a*b; }

int apsp(int        n,
         int        sym,
         CTF_World &dw){
  int rank, num_pes;
 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  tCTF_Semiring<int, &add, &mul> ts(0, 1, MPI_SUM);

  CTF_Semiring s = ts;

  return 1;
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
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;
  
  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);

    int pass;    
    if (rank == 0){
      printf("Non-symmetric: NS = NS*NS apsp:\n");
    }
    pass = apsp(n, NS, dw);
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
