/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroupmulti_tsr_sym
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

int multi_tsr_sym(int         m,
                  int         n,
                  CTF_World  &dw){
  int rank, i, num_pes;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

#ifndef TEST_SUITE
  if (rank == 0)
    printf("m = %d, n = %d, p = %d\n", 
            m,n,num_pes);
#endif
  
  //* Creates distributed tensors initialized with zeros
  CTF_Matrix A(n, m, NS, dw);
  CTF_Matrix C_NS(n, n, NS, dw);
  CTF_Matrix C_SY(n, n, SY, dw);
  CTF_Matrix diff(n, n, NS, dw);

  srand48(13*rank);
  //* Writes noise to local data based on global index
  A.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  A.write(np, indices, pairs);
  free(pairs);
  free(indices);

  C_NS["ij"] = A["ik"]*A["jk"];
  C_SY["ij"] = A["ik"]*A["jk"];

  diff["ij"] = C_SY["ij"] - C_NS["ij"];

  double err = diff.norm2();
  if (rank == 0){
    if (err < 1.E-6)
      printf("{ A[\"ik\"]*A[\"jk\"] = C_NS[\"ij\"] = C_AS[\"ij\"] passed.\n");
    else 
      printf("{ A[\"ik\"]*A[\"jk\"] = C_NS[\"ij\"] = C_AS[\"ij\"] failed!\n");
  }

  return err < 1.E-6;
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
  int rank, np, niter, n, m, k, pass;
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
    CTF_World dw(MPI_COMM_WORLD, argc, argv);

    pass =multi_tsr_sym(m, n, dw);
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
