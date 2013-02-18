#include <ctf.hpp>
#include <assert.h>
#include <stdlib.h>

/**
 * \brief Forms N-by-N DFT matrix A and inverse-dft iA and checks A*iA=I
 */
int main(int argc, char ** argv){
  int myRank, numPes, logn, i;
  int64_t  n, np;
  int64_t * idx;
  std::complex<double> * data;
  std::complex<double> imag(0,1);
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if (argc > 1){
    logn = atoi(argv[1]);
    if (logn<0) logn = 5;
  } else {
    logn = 5;
  }
  n = 1<<logn;
  int edge_lens[2] = {n,n};
  int sym[2]       = {SY,NS};

  tCTF_World< std::complex<double> > * wrld = new tCTF_World< std::complex<double> >();
  tCTF_Tensor< std::complex<double> > DFT(2, edge_lens, sym, wrld);
  tCTF_Tensor< std::complex<double> > IDFT(2, edge_lens, sym, wrld);

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

  DFT.contract(std::complex<double> (1.0, 0.0), DFT, "ij", IDFT, "jk", 
               std::complex<double> (0.0, 0.0), "ik");

 
  DFT.get_local_data(&np, &idx, &data);
  //DFT.print(stdout);
  for (i=0; i<np; i++){
    //printf("data[%lld] = %lf\n",idx[i],data[i].real());
    if (idx[i]/n == idx[i]%n)
      assert(fabs(data[i].real() - 1.)<=1.E-9);
    else 
      assert(fabs(data[i].real())<=1.E-9);
  }
  
  if (myRank == 0)
    printf("{ DFT matrix * IDFT matrix = Identity } confirmed\n");

  MPI_Barrier(MPI_COMM_WORLD);

  free(idx);
  free(data);
  
}
