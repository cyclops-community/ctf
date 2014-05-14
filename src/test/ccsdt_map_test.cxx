/** Copyright (c) 2011, Edgar Solomonik, all rights reserved.
  * \addtogroup examples 
  * @{ 
  * \addtogroup CCSDT_T3_to_T2
  * @{ 
  * \brief A symmetric contraction from CCSDT compared with the explicitly permuted nonsymmetric form
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

int ccsdt_map_test(int const     n,
                   CTF_World    &dw){

  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  //int shapeAS6[] = {AS,AS,NS,AS,AS,NS};
  int shapeNS6[] = {NS,NS,NS,NS,NS,NS};
  int nnnnnn[] = {n,n,n,n,n,n};
  int shapeNS4[] = {NS,NS,NS,NS};
  int nnnn[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor W(4, nnnn, shapeNS4, dw, "W", 1);
  CTF_Tensor T(4, nnnn, shapeNS4, dw, "T", 1);
  CTF_Tensor Z(6, nnnnnn, shapeNS6, dw, "Z", 1);

  Z["hijmno"] += W["hijk"]*T["kmno"];

  return 1;
} 

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
  int rank, np, niter, n, m;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;



  {
    CTF_World dw(argc, argv);
    int pass = ccsdt_map_test(n, dw);
    assert(pass);
  }


  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */


