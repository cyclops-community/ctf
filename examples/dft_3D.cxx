/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup DFT_3D DFT_3D
  * @{ 
  * \brief 3D Discrete Fourier Transform by tensor contractions
  */


#include <ctf.hpp>
using namespace CTF;


int test_dft_3D(int     n,
                World & wrld){
  int myRank;
  int i, j;
  int64_t  np;
  int64_t * idx;
  std::complex<long double> * data;
  std::complex<long double> imag(0,1);
  
  int len[] = {n,n,n};
  int sym[] = {NS,NS,NS};
  
  MPI_Comm_rank(wrld.comm, &myRank);

  CTF::Ring< std::complex<long double> > ldr;

  Matrix < std::complex<long double> >DFT(n, n, SY, wrld, ldr);
  Matrix < std::complex<long double> >IDFT(n, n, SY, wrld, ldr);
  Tensor < std::complex<long double> >MESH(3, len, sym, wrld, ldr);

  DFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = ((long double)1./n)*exp(-2.*(idx[i]/n)*(idx[i]%n)*((long double)M_PI/n)*imag);
  }
  DFT.write(np, idx, data);
  //DFT.print(stdout);
  free(idx);
  delete [] data; 
  
  IDFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = ((long double)1./n)*exp(2.*(idx[i]/n)*(idx[i]%n)*((long double)M_PI/n)*imag);
  }
  IDFT.write(np, idx, data);
  //IDFT.print(stdout);
  free(idx);
  delete [] data; 

  MESH.get_local_data(&np, &idx, &data);
  for (i=0; i<np; i++){
    for (j=0; j<n; j++){
      data[i] += exp(imag*(long double)((-2.*M_PI*(j/(double)(n)))
                      *((idx[i]%n) + ((idx[i]/n)%n) +(idx[i]/(n*n)))));
    }
  }
  MESH.write(np, idx, data);
  //MESH.print(stdout);
  free(idx);
  delete [] data; 
  
  MESH["ijk"] = 1.0*MESH["pqr"]*DFT["ip"]*DFT["jq"]*DFT["kr"];
 
  MESH.get_local_data(&np, &idx, &data);
  //MESH.print(stdout);
  int pass = 1;
  for (i=0; i<np; i++){
    if (idx[i]%n == (idx[i]/n)%n && idx[i]%n == idx[i]/(n*n)){
      if (fabs((double)data[i].real() - 1.)>=1.E-9) pass = 0;
    } else {
      if (fabs((double)data[i].real())>=1.E-9) pass = 0;
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
  delete [] data; 
  return pass;
}

#ifndef TEST_SUITE
/**
 * \brief Forms N-by-N DFT matrix A and inverse-dft iA and checks A*iA=I
 */
int main(int argc, char ** argv){
  int64_t n;
  MPI_Init(&argc, &argv);

  if (argc > 1){
    n = atoi(argv[1]);
  } else {
    n = 6;
  }

  {
    World dw(argc, argv);
    int pass = test_dft_3D(n, dw);
    assert(pass);
  }

  MPI_Finalize();
  
}
/**
 * @} 
 * @}
 */

#endif
