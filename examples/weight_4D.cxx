/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>
#include "../src/shared/util.h"

void divide(double const alpha, double const a, double const b, double & c){
  c+=alpha*(a/b);
}

int weight_4D(int const    n,
              int const    sym,
              CTF_World   &dw){
  int rank, i, num_pes;
  long long np, np_A;
  double * pairs, * post_pairs_C, * pairs_A;
  long long * indices, * indices_A;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
  
  int shapeN4[] = {sym,NS,sym,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(4, sizeN4, shapeN4, dw);
  CTF_Tensor B(4, sizeN4, shapeN4, dw);
  CTF_Tensor C(4, sizeN4, shapeN4, dw);

  srand48(13*rank);
  A.get_local_data(&np_A, &indices_A, &pairs_A);
  for (i=0; i<np_A; i++ ) pairs_A[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  A.write_remote_data(np_A, indices_A, pairs_A);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  B.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  C.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);

  C["ijkl"] = A["ijkl"]*B["klij"];

  CTF_fctr fctr;
  fctr.func_ptr = &divide;

  C.contract(1.0, C, "ijkl", B, "klij", 0.0, "ijkl", fctr);

  post_pairs_C = (double*)malloc(np_A*sizeof(double));
  C.get_remote_data(np_A, indices_A, post_pairs_C);
  
  int pass = 1; 
  for (i=0; i<np_A; i++){
    if (fabs(pairs_A[i]) > 1.E-6 &&
           fabs((double)post_pairs_C[i]-(double)pairs_A[i])/(double)pairs_A[i]>1.E-6){
      pass = 0;
    }
  }
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass){
      printf("{ C[\"ijkl\"] = A[\"ijkl\"]*B[\"ijkl\"] } passed\n");
    } else {
      printf("{ C[\"ijkl\"] = A[\"ijkl\"]*B[\"ijkl\"] } failed\n");
    }
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  
  free(indices_A);
  free(pairs_A);
  free(post_pairs_C);
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

    if (rank == 0){
      printf("Computing C_ijkl = A_ijkl*B_kilj\n");
      printf("Non-symmetric: NS = NS*NS weight:\n");
    }
    weight_4D(n, NS, dw);
    if (rank == 0){
      printf("Symmetric: SY = SY*SY weight:\n");
    }
    weight_4D(n, SY, dw);
    if (rank == 0){
      printf("(Anti-)Skew-symmetric: AS = AS*AS weight:\n");
    }
    weight_4D(n, AS, dw);
  }


  MPI_Finalize();
  return 0;
}

#endif
