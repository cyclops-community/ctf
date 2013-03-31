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

void ccsdt_t3_to_t2(int const  n,
                    CTF_World  &dw,
                    char const *dir){
  int rank, i, num_pes;
  int64_t np;
  double * pairs;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  if (rank == 0)
    printf("n = %d\n", n);
  
  int shapeAS4[] = {AS,NS,AS,NS};
  int shapeAS6[] = {AS,AS,NS,AS,AS,NS};
  int shapeNS4[] = {NS,NS,NS,NS};
  int shapeNS6[] = {NS,NS,NS,NS,NS,NS};
  int sizeN4[] = {n,n,n,n};
  int sizeN6[] = {n,n,n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor AS_A(4, sizeN4, shapeAS4, dw);
  CTF_Tensor AS_B(6, sizeN6, shapeAS6, dw);
  CTF_Tensor AS_C(4, sizeN4, shapeAS4, dw);
  CTF_Tensor NS_A(4, sizeN4, shapeNS4, dw);
  CTF_Tensor NS_B(6, sizeN6, shapeNS6, dw);
  CTF_Tensor NS_C(4, sizeN4, shapeNS4, dw);

  if (rank == 0)
    printf("tensor creation succeed\n");

  //* Writes noise to local data based on global index
  AS_A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(indices[i]);
  AS_A.write_remote_data(np, indices, pairs);
  NS_A.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  AS_B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(.33+indices[i]);
  AS_B.write_remote_data(np, indices, pairs);
  AS_B.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  AS_C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(.66+indices[i]);
  AS_C.write_remote_data(np, indices, pairs);
  NS_C.write_remote_data(np, indices, pairs);

  AS_C["ijmn"] += .5*AS_A["mnje"]*AS_B["abeimn"];

  NS_C["ijmn"] += NS_A["mnje"]*NS_B["abeimn"];
  NS_C["ijmn"] -= NS_A["mnje"]*NS_B["abemin"];
  NS_C["ijmn"] -= NS_A["mnje"]*NS_B["abenmi"];
  NS_C["ijmn"] -= NS_A["mnje"]*NS_B["aebimn"];
  NS_C["ijmn"] += NS_A["mnje"]*NS_B["aebmin"];
  NS_C["ijmn"] += NS_A["mnje"]*NS_B["aebnmi"];
  NS_C["ijmn"] -= NS_A["mnje"]*NS_B["eabimn"];
  NS_C["ijmn"] += NS_A["mnje"]*NS_B["eabmin"];
  NS_C["ijmn"] += NS_A["mnje"]*NS_B["eabnmi"];
  NS_C["ijmn"] -= NS_A["mnej"]*NS_B["abeimn"];
  NS_C["ijmn"] += NS_A["mnej"]*NS_B["abemin"];
  NS_C["ijmn"] += NS_A["mnej"]*NS_B["abenmi"];
  NS_C["ijmn"] += NS_A["mnej"]*NS_B["aebimn"];
  NS_C["ijmn"] -= NS_A["mnej"]*NS_B["aebmin"];
  NS_C["ijmn"] -= NS_A["mnej"]*NS_B["aebnmi"];
  NS_C["ijmn"] += NS_A["mnej"]*NS_B["eabimn"];
  NS_C["ijmn"] -= NS_A["mnej"]*NS_B["eabmin"];
  NS_C["ijmn"] -= NS_A["mnej"]*NS_B["eabnmi"];
  double nrm_AS = AS_C.reduce(CTF_OP_SQNRM2);
  double nrm_NS = NS_C.reduce(CTF_OP_SQNRM2);
  if (rank == 0) printf("norm of AS_C = %lf NS_C = %lf\n", nrm_AS, nrm_NS);
  AS_C["ijmn"] -= NS_C["ijmn"];
  
  double nrm = AS_C.reduce(CTF_OP_SQNRM2);
  if (rank == 0) printf("norm of AS_C after contraction should be zero, is = %lf\n", nrm);

  free(pairs);
  free(indices);
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
  int rank, np, niter, n;
  int const in_num = argc;
  char dir[120];
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 3;
  } else niter = 3;



  CTF_World dw;

  ccsdt_t3_to_t2(n, dw, dir);


  MPI_Finalize();
  return 0;
}



