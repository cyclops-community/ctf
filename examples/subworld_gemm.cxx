/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup subworld_gemm
  * @{ 
  * \brief Performs recursive parallel matrix multiplication using the slice interface to extract blocks
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <ctf.hpp>


int test_subworld_gemm(int n,
                       int m,
                       int k,
                       int div_,
                       CTF_World &dw){
  int rank, num_pes;
  int64_t i, np;
  double * pairs, err;
  int64_t * indices;
  
  
  CTF_Matrix C(m, n, NS, dw);
  CTF_Matrix C_ans(m, n, NS, dw);
  CTF_Matrix A(m, k, NS, dw);
  CTF_Matrix B(k, n, NS, dw);
  
  MPI_Comm pcomm = dw.comm;
  MPI_Comm_rank(pcomm, &rank);
  MPI_Comm_size(pcomm, &num_pes);
  
  int div = div_;
  if (div > num_pes) div = num_pes;

  
  srand48(13*rank);
  A.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; 
  A.write(np, indices, pairs);
  free(pairs);
  free(indices);
  B.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; 
  B.write(np, indices, pairs);
  free(pairs);
  free(indices);

  
  int cnum_pes = num_pes / div;
  int color = rank/cnum_pes;
  int crank = rank%cnum_pes;
   
  MPI_Comm ccomm; 
  MPI_Comm_split(pcomm, color, crank, &ccomm);
  CTF_World sworld(ccomm);
  
  C_ans["ij"] = ((double)div)*A["ik"]*B["kj"];

  CTF_Matrix subA(m, k, NS, sworld);
  CTF_Matrix subB(k, n, NS, sworld);
  CTF_Matrix subC(m, n, NS, sworld);

  for (int c=0; c<num_pes/cnum_pes; c++){
    if (c==color){
      A.add_to_subworld(&subA,1.0,0.0);
      B.add_to_subworld(&subB,1.0,0.0);
    } else {
      A.add_to_subworld(NULL,1.0,0.0);
      B.add_to_subworld(NULL,1.0,0.0);
    }    
  }

  if (rank < cnum_pes*div)
    subC["ij"] = subA["ik"]*subB["kj"];

  for (int c=0; c<num_pes/cnum_pes; c++){
    if (c==color){
      C.add_from_subworld(&subC, 1.0, 1.0);
    } else {
      C.add_from_subworld(NULL, 1.0, 1.0);
    }    
  }
  

  C_ans["ij"] -= C["ij"];

  err = C_ans.norm2();

  if (rank == 0){
    if (err<1.E-9)
      printf("{ GEMM on subworlds } passed\n");
    else
      printf("{ GEMM on subworlds } FAILED, error norm = %E\n",err);
  }
  return err<1.E-9;
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
  int rank, np, niter, n, m, k, pass, div;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 23;
  } else n = 23;
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 17;
  } else m = 17;
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 31;
  } else k = 31;
  if (getCmdOption(input_str, input_str+in_num, "-div")){
    div = atoi(getCmdOption(input_str, input_str+in_num, "-div"));
    if (div < 0) div = 2;
  } else div = 2;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    int pass;    
    if (rank == 0){
      printf("Non-symmetric: NS = NS*NS test_subworld_gemm:\n");
    }
    pass = test_subworld_gemm(n, m, k, div, dw);
    assert(pass);
  }

  MPI_Finalize();
  return 0;
}
#endif

