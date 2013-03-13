#include <ctf.hpp>
#include <assert.h>
#include <stdlib.h>




/**
 * \brief Forms N-by-N DFT matrix A and inverse-dft iA and checks A*iA=I
 */
int main(int argc, char ** argv){
  int myRank, numPes, logn, i, j;
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

  tCTF_World< std::complex<double> > * wrld = new tCTF_World< std::complex<double> >();
  tCTF_Matrix< std::complex<double> > DFT(n, n, SY, wrld);
  tCTF_Matrix< std::complex<double> > IDFT(n, n, SY, wrld);
  tCTF_Tensor< std::complex<double> > MESH(3, (int[]){n, n, n}, (int[]){NS, NS, NS}, wrld);

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
  for (i=0; i<np; i++){
    if (idx[i]%n == (idx[i]/n)%n && idx[i]%n == idx[i]/(n*n))
      assert(fabs(data[i].real() - 1.)<=1.E-9);
    else 
      assert(fabs(data[i].real())<=1.E-9);
  }
  
  if (myRank == 0)
    printf("{ 3D_IDFT(3D_DFT(I))) = I } confirmed\n");

  MPI_Barrier(MPI_COMM_WORLD);

  free(idx);
  free(data);
  
}
