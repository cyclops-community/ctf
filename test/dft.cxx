/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup tests 
  * @{ 
  * \defgroup DFT  DFT
  * @{ 
  * \brief Discrete Fourier Transform by matrix multiplication
  */

#include <ctf.hpp>
using namespace CTF;
template <typename int_type>  
int test_dft(int64_t n,
             World   &wrld){
  int numPes, myRank;
  int64_t  np, i;
  int_type * idx;
  std::complex<double> * data;
  std::complex<double> imag(0,1);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  Matrix < std::complex<double> >DFT(n, n, SY, wrld, "DFT", 1);
  Matrix < std::complex<double>  >IDFT(n, n, SY, wrld, "IDFT", 0);

  DFT.get_local_data_aos_idx(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = exp(-2.*(idx[2*i])*(idx[2*i+1])*(M_PI/n)*imag);
  //  printf("[%lld][%lld] (%20.14E,%20.14E)\n",i%n,i/n,data[i].real(),data[i].imag());
  }
  DFT.write_aos_idx(np, idx, data);
  
  IDFT.read_aos_idx(np, idx, data);

  for (i=0; i<np; i++){
    data[i] = (1./n)*exp(2.*(idx[2*i])*(idx[2*i+1])*(M_PI/n)*imag);
  }
  IDFT.write_aos_idx(np, idx, data);
  //IDFT.print(stdout);
  free(idx);
  delete [] data; 

  /*DFT.contract(std::complex<double> (1.0, 0.0), DFT, "ij", IDFT, "jk", 
               std::complex<double> (0.0, 0.0), "ik");*/
  DFT["ik"] = .5*DFT["ij"]*IDFT["jk"];

  Scalar< std::complex<double> > ss(wrld);
  ss[""] = Function< std::complex<double>, std::complex<double>, std::complex<double> >([](std::complex<double> a, std::complex<double> b){ return a+b; })(DFT["ij"],DFT["ij"]);
 
  DFT.get_local_data_aos_idx(&np, &idx, &data);
  int pass = 1;
  //DFT.print(stdout);
  for (i=0; i<np; i++){
    //printf("data[%lld] = %lf\n",idx[i],data[i].real());
    if (idx[2*i] == idx[2*i+1]){
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
    if (typeid(int_type) == typeid(int)){
      if (pass)
        printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] with int index type } passed\n");
      else
        printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] with int index type } failed\n");
    } else {
      if (pass)
        printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] with int64_t index type } passed\n");
      else
        printf("{ DFT[\"ik\"] = DFT[\"ij\"]*IDFT[\"jk\"] with int64_t index type } failed\n");
    }
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
    World dw(argc, argv);
    int pass = test_dft<int>(n, dw);
    assert(pass);
  }
  {
    World dw(argc, argv);
    int pass = test_dft<int64_t>(n, dw);
    assert(pass);
  }


  MPI_Finalize();
  
}
/**
 * @} 
 * @}
 */


#endif
