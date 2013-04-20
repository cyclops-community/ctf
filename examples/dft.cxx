/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <ctf.hpp>
#include <assert.h>
#include <stdlib.h>
  
int test_dft(int64_t const  n,
             cCTF_World    &dw){
  int numPes, myRank;
  int64_t  np, i;
  int64_t * idx;
  std::complex<double> * data;
  std::complex<double> imag(0,1);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  cCTF_World wrld(MPI_COMM_WORLD);
  cCTF_Matrix DFT(n, n, SY, wrld);
  cCTF_Matrix IDFT(n, n, SY, wrld);

  DFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = exp(-2.*(idx[i]/n)*(idx[i]%n)*(M_PI/n)*imag);
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

  /*DFT.contract(std::complex<double> (1.0, 0.0), DFT, "ij", IDFT, "jk", 
               std::complex<double> (0.0, 0.0), "ik");*/
  DFT["ik"] = DFT["ij"]*IDFT["jk"];

 
  DFT.get_local_data(&np, &idx, &data);
  int pass = 1;
  //DFT.print(stdout);
  for (i=0; i<np; i++){
    //printf("data[%lld] = %lf\n",idx[i],data[i].real());
    if (idx[i]/n == idx[i]%n){
      if (fabs(data[i].real() - 1.)>=1.E-9)
        pass = 0;
    } else  {
      if (fabs(data[i].real())>=1.E-9)
        pass = 0;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  
  if (myRank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &pass, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    if (pass)
      printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] } passed\n");
    else
      printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] } failed\n");
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
  int myRank, numPes, logn;
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
    int pass = test_dft(n, dw);
    assert(pass);
  }

  MPI_Finalize();
  
}

#endif
