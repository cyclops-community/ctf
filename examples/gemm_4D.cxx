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

void gemm_4D(int const  n,
             int const  sym,
             int const  niter,
             CTF_World  &dw,
             char const *dir){
  int rank, i, num_pes;
  int64_t np;
  double * pairs, * pairs_AB, * pairs_BC;
  double t, time;
  int64_t * indices, * indices_AB, * indices_BC;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  if (rank == 0)
    printf("n = %d\n", n);
  
  int shapeN4[] = {sym,NS,sym,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A(4, sizeN4, shapeN4, dw);
  CTF_Tensor B(4, sizeN4, shapeN4, dw);
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


  t = MPI_Wtime();
  for (i=0; i<niter; i++){
    C["ijkl"] += (.3*i)*A["ijmn"]*B["mnkl"];
  }
  time = MPI_Wtime()- t;
  if (rank == 0){
    double nd = (double)n;
    double c = 2.E-9;
    if (sym == SY || sym == AS){
      c = c/8.;
    }
    printf("%lf seconds/GEMM %lf GF\n",
            time/niter,niter*c*nd*nd*nd*nd*nd*nd/time);
    printf("Verifying associativity\n");
  }
  
  /* verify D=(A*B)*C = A*(B*C) */
  CTF_Tensor D(4, sizeN4, shapeN4, dw);
  
  D["ijkl"] = A["ijmn"]*B["mnkl"];
  D["ijkl"] = D["ijmn"]*C["mnkl"];
  C["ijkl"] = B["ijmn"]*C["mnkl"];
  C["ijkl"] = A["ijmn"]*C["mnkl"];
  
  if (rank == 0)
    printf("Completed (A*B)*C and A*(B*C) computations, verifying...\n");

  C.align(D);  
  C.get_local_data(&np, &indices_BC, &pairs_BC);
  D.get_local_data(&np, &indices_AB, &pairs_AB);
  for (i=0; i<np; i++){
    assert(fabs((double)pairs_BC[i]-(double)pairs_AB[i])<1.E-6);
  }
  free(pairs_AB);
  free(pairs_BC);
  if (rank == 0)
    printf("Verification completed successfully.\n");
  
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
    printf("Computing C_ijkl = A_ijmn*B_klmn\n");
    printf("Non-symmetric: NS = NS*NS gemm:\n");
  }
  gemm_4D(n, NS, niter, dw, dir);
  if (rank == 0){
    printf("Symmetric: SY = SY*SY gemm:\n");
  }
  gemm_4D(n, SY, niter, dw, dir);
  if (rank == 0){
    printf("(Anti-)Skew-symmetric: AS = AS*AS gemm:\n");
  }
  gemm_4D(n, AS, niter, dw, dir);


  MPI_Finalize();
  return 0;
 }

