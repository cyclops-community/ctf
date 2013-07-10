/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <ctf.hpp>
#include <assert.h>
#include <stdlib.h>


int test_dft_3D(int const     n,
                cCTF_World    &wrld){
  int myRank, numPes;
  int i, j;
  int64_t  np;
  int64_t * idx;
  std::complex<double> * data;
  std::complex<double> imag(0,1);
  
  int len[] = {n,n,n};
  int sym[] = {NS,NS,NS};
  
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  cCTF_Matrix DFT(n, n, SY, wrld);
  cCTF_Matrix IDFT(n, n, SY, wrld);
  cCTF_Tensor MESH(3, len, sym, wrld);

  DFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = (1./n)*exp(-2.*(idx[i]/n)*(idx[i]%n)*(M_PI/n)*imag);
  }
  DFT.write_remote_data(np, idx, data);
  //DFT.print(stdout);
  free(idx);
  free(data); 
  
  IDFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = (1./n)*exp(2.*(idx[i]/n)*(idx[i]%n)*(M_PI/n)*imag);
  }
  IDFT.write_remote_data(np, idx, data);
  //IDFT.print(stdout);
  free(idx);
  free(data); 

  MESH.get_local_data(&np, &idx, &data);
  for (i=0; i<np; i++){
    for (j=0; j<n; j++){
      data[i] += exp(imag*((-2.*M_PI*(j/(double)(n)))
                      *((idx[i]%n) + ((idx[i]/n)%n) +(idx[i]/(n*n)))));
    }
  }
  MESH.write_remote_data(np, idx, data);
  //MESH.print(stdout);
  free(idx);
  free(data); 
  
  MESH["ijk"] = MESH["pqr"]*DFT["ip"]*DFT["jq"]*DFT["kr"];
 
  MESH.get_local_data(&np, &idx, &data);
  //MESH.print(stdout);
  int pass = 1;
  for (i=0; i<np; i++){
    if (idx[i]%n == (idx[i]/n)%n && idx[i]%n == idx[i]/(n*n)){
      if (fabs(data[i].real() - 1.)>=1.E-9) pass = 0;
    } else {
      if (fabs(data[i].real())>=1.E-9) pass = 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  
  if (myRank == 0){
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass)
      printf("{ MESH[\"ijk\"] = MESH[\"pqr\"]*DFT[\"ip\"]*DFT[\"jq\"]*DFT[\"kr\"] } passed\n");
    else
      printf("{ MESH[\"ijk\"] = MESH[\"pqr\"]*DFT[\"ip\"]*DFT[\"jq\"]*DFT[\"kr\"] } failed\n");
  } else 
    MPI_Reduce(&pass, MPI_IN_PLACE, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);


  free(idx);
  free(data);
  return pass;
}

#ifndef TEST_SUITE
/**
 * \brief Forms N-by-N DFT matrix A and inverse-dft iA and checks A*iA=I
 */
int main(int argc, char ** argv){
  int logn;
  int64_t n;
  MPI_Init(&argc, &argv);

  if (argc > 1){
    logn = atoi(argv[1]);
    if (logn<0) logn = 5;
  } else {
    logn = 5;
  }
  n = 1<<logn;

  {
    cCTF_World dw(argc, argv);
    int pass = test_dft_3D(n, dw);
    assert(pass);
  }

  MPI_Finalize();
  
}
#endif
