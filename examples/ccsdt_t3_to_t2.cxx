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
                    int const  sym,
                    int const  niter,
                    CTF_World  &dw,
                    char const *dir){
  int rank, i, num_pes;
  int64_t np;
  double * pairs;
  double t, time;
  int64_t * indices;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  if (rank == 0)
    printf("n = %d\n", n);
  
  int shapeN4[] = {sym,NS,sym,NS};
  int shapeN6[] = {sym,sym,NS,sym,sym,NS};
  int sizeN4[] = {n,n,n,n};
  int sizeN6[] = {n,n,n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(4, sizeN4, shapeN4, dw);
  CTF_Tensor B(6, sizeN6, shapeN6, dw);
  CTF_Tensor C(4, sizeN4, shapeN4, dw);

  if (rank == 0)
    printf("tensor creation succeed\n");

  //* Writes noise to local data based on global index
  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(indices[i]);
  A.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(.33+indices[i]);
  B.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);
  C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = (1.E-3)*sin(.66+indices[i]);
  C.write_remote_data(np, indices, pairs);
  free(pairs);
  free(indices);

  C["ijmn"] += (.5*i)*A["mnje"]*B["abeimn"];
  double nrm = C.reduce(CTF_OP_SQNRM2);
  if (rank == 0) printf("norm of C after contraction is = %lf\n", nrm);

  t = MPI_Wtime();
  for (i=0; i<niter; i++){
    C["ijmn"] += (.5*i)*A["mnje"]*B["abeimn"];
  }

  time = MPI_Wtime()- t;
  if (rank == 0){
    double nd = (double)n;
    double c = 2.E-9;
    if (sym == SY || sym == AS){
      c = c/2.;
    }
    printf("%lf seconds/contrction %lf GF\n",
            time/niter,niter*c*nd*nd*nd*nd*nd*nd*nd*nd/time);
  }
  
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

  if (rank == 0){
    printf("Computing C_ijmn = A_mnje*B_abeimn\n");
    printf("Non-symmetric: NS = NS*NS:\n");
  }
  ccsdt_t3_to_t2(n, NS, niter, dw, dir);
  if (rank == 0){
    printf("(Anti-)Skew-symmetric: AS = AS*AS:\n");
  }
  ccsdt_t3_to_t2(n, AS, niter, dw, dir);


  MPI_Finalize();
  return 0;
}



