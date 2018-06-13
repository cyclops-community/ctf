/** \addtogroup tests 
  * @{ 
  * \defgroup GEMM_4D GEMM_4D
  * @{ 
  * \brief Folded matrix multiplication on 4D tensors
  */

#include <ctf.hpp>
using namespace CTF;

int  gemm_4D(int const    n,
             int const    sym,
             int const    niter,
             World   &dw){
  int rank, i, num_pes;
  int64_t np;
  double * pairs, * pairs_AB, * pairs_BC;
  int64_t * indices, * indices_AB, * indices_BC;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);


  int shapeN4[] = {sym,NS,sym,NS};
  int sizeN4[] = {n,n,n,n};

  //* Creates distributed tensors initialized with zeros
  Tensor<> A(4, sizeN4, shapeN4, dw);
  Tensor<> B(4, sizeN4, shapeN4, dw);
  Tensor<> C(4, sizeN4, shapeN4, dw);

  srand48(13*rank);
  //* Writes noise to local data based on global index
  A.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  A.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  B.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  B.write(np, indices, pairs);
  delete [] pairs;
  free(indices);
  C.get_local_data(&np, &indices, &pairs);
  for (i=0; i<np; i++ ) pairs[i] = drand48()-.5; //(1.E-3)*sin(indices[i]);
  C.write(np, indices, pairs);
  delete [] pairs;
  free(indices);


#ifndef TEST_SUITE
  double time;
  double t = MPI_Wtime();
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
#endif
  
  /* verify D=(A*B)*C = A*(B*C) */
  Tensor<> D(4, sizeN4, shapeN4, dw);
  
  D["ijkl"] = A["ijmn"]*B["mnkl"];
  D["ijkl"] = D["ijmn"]*C["mnkl"];
  C["ijkl"] = B["ijmn"]*C["mnkl"];
  C["ijkl"] = A["ijmn"]*C["mnkl"];
  
  C.align(D);  
  C.get_local_data(&np, &indices_BC, &pairs_BC);
  D.get_local_data(&np, &indices_AB, &pairs_AB);
  int pass = 1;
  for (i=0; i<np; i++){
    if (fabs((double)pairs_BC[i]-(double)pairs_AB[i])>=1.E-6) pass = 0;
  }
  delete [] pairs_AB;
  delete [] pairs_BC;
  free(indices_AB);
  free(indices_BC);
  if (rank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass)
      printf("{ (A[\"ijmn\"]*B[\"mnpq\"])*C[\"pqkl\"] = A[\"ijmn\"]*(B[\"mnpq\"]*C[\"pqkl\"]) } passed\n");
    else 
      printf("{ (A[\"ijmn\"]*B[\"mnpq\"])*C[\"pqkl\"] = A[\"ijmn\"]*(B[\"mnpq\"]*C[\"pqkl\"]) } failed!\n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
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
  int rank, np, niter, n, pass;
  int const in_num = argc;
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



  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Computing C_ijkl = A_ijmn*B_klmn\n");
      printf("Non-symmetric: NS = NS*NS gemm:\n");
    }
    pass = gemm_4D(n, NS, niter, dw);
    assert(pass);
    if (rank == 0){
      printf("Symmetric: SY = SY*SY gemm:\n");
    }
    pass = gemm_4D(n, SY, niter, dw);
    assert(pass);
    if (rank == 0){
      printf("(Anti-)Skew-symmetric: AS = AS*AS gemm:\n");
    }
    pass = gemm_4D(n, AS, niter, dw);
    assert(pass);
    if (rank == 0){
      printf("Symmetric-hollow: SH = SH*SH gemm:\n");
    }
    pass = gemm_4D(n, SH, niter, dw);
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
