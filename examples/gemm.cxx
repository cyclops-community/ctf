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
          CTF_World  *dw,
          char const *dir){
  int rank, i, num_pes;
  int64_t np;
  double * pairs, * pairs_AB, * pairs_BC;
  int64_t * indices, * indices_AB, * indices_BC;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  if (rank == 0)
    printf("m = %d, n = %d, k = %d, p = %d, niter = %d\n", 
            m,n,k,num_pes,niter);
  
  int shape_NS[] = {NS,NS};
  int shape[] = {sym,NS};
  int size_A[] = {m,k};
  int size_B[] = {k,n};
  int size_C[] = {m,n};

  //* Creates distributed tensors initialized with zeros
  CTF_Tensor A = CTF_Tensor(2, size_A, shape, dw);
  CTF_Tensor B = CTF_Tensor(2, size_B, shape, dw);
  CTF_Tensor C = CTF_Tensor(2, size_C, shape_NS, dw);

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
    CTF_Tensor D = CTF_Tensor(2, size_C, shape_NS, dw);
    CTF_Tensor E = CTF_Tensor(2, size_C, shape_NS, dw);
    
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

  CTF_World * dw;
  dw = new CTF_World();

  
  if (rank == 0){
    printf("Non-symmetric: NS = NS*NS gemm:\n");
  }
  gemm(m, n, k, NS, niter, dw, dir);
  if (m==n && n==k){ 
    if (rank == 0){
      printf("Symmetric: NS = SY*SY gemm:\n");
    }
    gemm(m, n, k, SY, niter, dw, dir);
    if (rank == 0){
      printf("(Anti-)Skew-symmetric: NS = AS*AS gemm:\n");
    }
    gemm(m, n, k, AS, niter, dw, dir);
  }

  delete dw;

  MPI_Finalize();
  return 0;
 }

