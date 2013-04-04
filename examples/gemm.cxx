/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <algorithm>
#include <ctf.hpp>

void gemm(int const  m,
          int const  n,
          int const  k,
          int const  sym,
          int const  niter,
          char const *dir){
  int rank, i, num_pes;
  int64_t np;
  double * pairs, * pairs_AB, * pairs_BC;
  int64_t * indices, * indices_AB, * indices_BC;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
  CTF_World dw;

  if (rank == 0)
    printf("m = %d, n = %d, k = %d, p = %d, niter = %d\n", 
            m,n,k,num_pes,niter);
  
  //* Creates distributed tensors initialized with zeros
  CTF_Matrix A(m, k, sym, dw);
  CTF_Matrix B(k, n, sym, dw);
  CTF_Matrix C(m, n, NS,  dw);

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

  double t;

  t = MPI_Wtime();
  for (i=0; i<niter; i++){
    C["ij"] += (.3*i)*A["ik"]*B["kj"];
  }
  t = MPI_Wtime() - t;
  if (rank == 0){
    printf("%lf seconds/GEMM, %lf GF\n",t/niter, 
           niter*2.*((double)m)*((double)n)*((double)k)*1.E-9/t);
  }
 
  if (m==n && n==k){ 
    if (rank == 0)
      printf("Verifying associativity\n");
    /* verify D=(A*B)*C = A*(B*C) */
    CTF_Matrix D(m, n, NS, dw);
    CTF_Matrix E(m, n, NS, dw);
    
    D["ij"] = A["ik"]*B["kj"];
    D["ij"] = D["ik"]*C["kj"];
    E["ij"] = B["ik"]*C["kj"];
    E["ij"] = A["ik"]*E["kj"];
    
    if (rank == 0)
      printf("Completed (A*B)*C and A*(B*C) computations, verifying...\n");
    
    D.align(E);
    D.get_local_data(&np, &indices_BC, &pairs_BC);
    E.get_local_data(&np, &indices_AB, &pairs_AB);
    for (i=0; i<np; i++){
      if (fabs((pairs_BC[i]-pairs_AB[i])/pairs_AB[i])>1.E-6){
        printf("P[%d]: element %lld is of (A*B)*C is %lf,",
                rank,indices_AB[i],pairs_AB[i]);
        printf("but for A*(B*C) it is %lf\n", pairs_BC[i]);
        assert(0);
      }
    }
    free(pairs_AB);
    free(pairs_BC);
    free(indices_AB);
    free(indices_BC);
    if (rank == 0)
      printf("Verification completed successfully.\n");
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
  int rank, np, niter, n, m, k;
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
  if (getCmdOption(input_str, input_str+in_num, "-m")){
    m = atoi(getCmdOption(input_str, input_str+in_num, "-m"));
    if (m < 0) m = 7;
  } else m = 7;
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 7;
  } else k = 7;
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 5;
  } else niter = 5;

  
  if (rank == 0){
    printf("Non-symmetric: NS = NS*NS gemm:\n");
  }
  gemm(m, n, k, NS, niter, dir);
  if (m==n && n==k){ 
    if (rank == 0){
      printf("Symmetric: NS = SY*SY gemm:\n");
    }
    gemm(m, n, k, SY, niter, dir);
    if (rank == 0){
      printf("(Anti-)Skew-symmetric: NS = AS*AS gemm:\n");
    }
    gemm(m, n, k, AS, niter, dir);
  }

  MPI_Finalize();
  return 0;
 }

