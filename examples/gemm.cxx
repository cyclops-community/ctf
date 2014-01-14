/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/
/** \addtogroup examples 
  * @{ 
  * \defgroup gemm
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

int  gemm(int const     m,
          int const     n,
          int const     k,
          int const     sym,
          int const     niter,
          CTF_World    &dw){
  int rank, i, num_pes;
  int64_t np;
  double * pairs, * pairs_AB, * pairs_BC;
  int64_t * indices, * indices_AB, * indices_BC;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

#ifndef TEST_SUITE
  if (rank == 0)
    printf("m = %d, n = %d, k = %d, p = %d, niter = %d\n", 
            m,n,k,num_pes,niter);
#endif
  
  //* Creates distributed tensors initialized with zeros
  CTF_Matrix A(m, k, sym, dw);
  CTF_Matrix B(k, n, sym, dw);
  CTF_Matrix C(m, n, NS,  dw);

  srand48(13*rank);
  //* Writes noise to local data based on global index
  A.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  A.write(np, indices, pairs);
  free(pairs);
  free(indices);
  B.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  B.write(np, indices, pairs);
  free(pairs);
  free(indices);
  C.read_local(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  C.write(np, indices, pairs);
  free(pairs);
  free(indices);

  C["ij"] += A["ik"]*B["kj"];
  C["ij"] += (.3*i)*A["ik"]*B["kj"];
#ifndef TEST_SUITE
  double t;

  CTF_Flop_Counter f = CTF_Flop_Counter();
  t = MPI_Wtime();
  for (i=0; i<niter; i++){
    C["ij"] += A["ik"]*B["kj"];
  }
  t = MPI_Wtime() - t;
  int64_t allf = f.count(dw.comm);
  if (rank == 0){
    printf("%lf seconds/GEMM, %lf GF exact, %lf measured by CTF, %lf locally\n",t/niter, 
           niter*2.*((double)m)*((double)n)*((double)k)*1.E-9/t,
           ((double)allf)*1.E-9/t,((double)f.count())*1.E-9/t);
  }
  
#endif
  int pass = 1;
  if (m==n && n==k){ 
    /* verify D=(A*B)*C = A*(B*C) */
    CTF_Matrix D(m, n, NS, dw);
    CTF_Matrix E(m, n, NS, dw);
    if (0 && num_pes > 1){
      MPI_Comm halbcomm;
      MPI_Comm_split(dw.comm, rank%2, rank/2, &halbcomm);
      CTF_World hdw(halbcomm);
      CTF_Matrix hB(n, n, NS, hdw);
      hB["ij"] += B["ij"];
      assert(hB.norm2()>1.E-6);
    
      D["ij"] = (A["ik"]*hB["kj"]);
      D["ij"] = (D["ik"]*C["kj"]);
      E["ij"] = (hB["ik"]*C["kj"]);
      E["ij"] = (A["ik"]*E["kj"]);
    } else {
      D["ij"] = (A["ik"]*B["kj"]);
      D["ij"] = (D["ik"]*C["kj"]);
      E["ij"] = (B["ik"]*C["kj"]);
      E["ij"] = (A["ik"]*E["kj"]);
    }
    D.align(E);
    D.read_local(&np, &indices_BC, &pairs_BC);
    E.read_local(&np, &indices_AB, &pairs_AB);
    for (i=0; i<np; i++){
      if (fabs((pairs_BC[i]-pairs_AB[i])/pairs_AB[i])>1.E-10){
        pass = 0;
        printf("P[%d]: element "PRId64" is of (A*B)*C is %lf,",
                rank,indices_AB[i],pairs_AB[i]);
        printf("but for A*(B*C) it is %lf\n", pairs_BC[i]);
      }
    }
    if (rank == 0){
      MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
      if (pass)
        printf("{ (A[\"ij\"]*B[\"jk\"])*C[\"kl\"] = A[\"ij\"]*(B[\"jk\"]*C[\"kl\"]) } passed\n");
      else 
        printf("{ (A[\"ij\"]*B[\"jk\"])*C[\"kl\"] = A[\"ij\"]*(B[\"jk\"]*C[\"kl\"]) } failed!\n");
    } else 
      MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    free(pairs_AB);
    free(pairs_BC);
    free(indices_AB);
    free(indices_BC);
  }
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
  if (getCmdOption(input_str, input_str+in_num, "-k")){
    k = atoi(getCmdOption(input_str, input_str+in_num, "-k"));
    if (k < 0) k = 7;
  } else k = 7;
  if (getCmdOption(input_str, input_str+in_num, "-niter")){
    niter = atoi(getCmdOption(input_str, input_str+in_num, "-niter"));
    if (niter < 0) niter = 5;
  } else niter = 5;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);

    CTF_Scalar ts(1.0,dw);
    CTF_Idx_Tensor its(&ts,"");
    CTF_Idx_Tensor tts(its); 
    tts.operator*(its);

    int pass;    
    if (rank == 0){
      printf("Non-symmetric: NS = NS*NS gemm:\n");
    }
    pass = gemm(m, n, k, NS, niter, dw);
    assert(pass);
    if (m==n && n==k){ 
      if (rank == 0){
        printf("Symmetric: NS = SY*SY gemm:\n");
      }
      pass = gemm(m, n, k, SY, niter, dw);
      assert(pass);
      if (rank == 0){
        printf("(Anti-)Skew-symmetric: NS = AS*AS gemm:\n");
      }
      pass = gemm(m, n, k, AS, niter, dw);
      assert(pass);
      if (rank == 0){
        printf("Symmetric-hollow: NS = SH*SH gemm:\n");
      }
      pass = gemm(m, n, k, SH, niter, dw);
      assert(pass);
    }
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */


#endif
