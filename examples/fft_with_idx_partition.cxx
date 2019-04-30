/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

/** \addtogroup examples 
  * @{ 
  * \defgroup fft_with_idx_partition fft_with_idx_partition
  * @{ 
  * \brief Does FFT along fibers of 3D tensor by making use of user-defined mapping
  */


#include <ctf.hpp>
using namespace CTF;

static const std::complex<double> img(0,1);

void fft(int64_t m, std::complex<double> const * input, std::complex<double> * output){
  if (m==1){
    output[0] = input[0];
  } else {
    assert(m%2 == 0);

    std::complex<double> * u = (std::complex<double>*)malloc(sizeof(std::complex<double>)*m/2);
    std::complex<double> * v = (std::complex<double>*)malloc(sizeof(std::complex<double>)*m/2);

    for (int64_t i=0; i<m/2; i++){
      output[i] = input[2*i];
    }
    fft(m/2, output,     u);

    for (int64_t i=0; i<m/2; i++){
      output[m/2+i] = input[2*i+1];
    }
    fft(m/2, output+m/2, v);

    for (int64_t i=0; i<m/2; i++){
      std::complex<double> z = exp(-2.*i*((double)M_PI/m)*img)*v[i];
      output[i]     = u[i] + z;
      output[m/2+i] = u[i] - z;
    }

    free(v);
    free(u);
  }
}

int fft_with_idx_partition(int64_t n,
                           int64_t m,
                           World & wrld){
  int myRank, numPes;
  int i;
  int64_t  np;
  int64_t * idx;
  std::complex<double> * data;
 
  Matrix< std::complex<double> > DFT(m, m, SY, wrld);

  DFT.get_local_data(&np, &idx, &data);

  for (i=0; i<np; i++){
    data[i] = exp(-2.*(idx[i]/m)*(idx[i]%m)*((double)M_PI/m)*img);
  }
  DFT.write(np, idx, data);
  free(idx);
  delete [] data; 
 
  int64_t len[] = {m,n,n};
  int sym[] = {NS,NS,NS};
  
  MPI_Comm_size(wrld.comm, &numPes);
  MPI_Comm_rank(wrld.comm, &myRank);

  int pr = (int)sqrt(numPes);
  int pc = numPes/pr;
  while (pr*pc != numPes){
    pr++;
    pc = numPes/pr;
  }
  int plens[2] = {pr, pc};
  Partition part(2, plens);

  Tensor< std::complex<double> > A(3, len, sym, wrld, "ijk", part["jk"]);
  Tensor< std::complex<double> > B(3, len, sym, wrld, "ijk", part["jk"]);
  Tensor< std::complex<double> > C(3, len, sym, wrld);

  A.fill_random(std::complex<double>(0.,0.), std::complex<double>(1.,1.));

  C["ijk"] = DFT["il"]*A["ljk"];

  int64_t size_raw_data_A, size_raw_data_B;
  std::complex<double> const * raw_data_A = A.get_raw_data(&size_raw_data_A);
  std::complex<double> * raw_data_B = B.get_raw_data(&size_raw_data_B);
  assert(size_raw_data_A == size_raw_data_B);
  assert(size_raw_data_A % m == 0);

  int64_t nlocfft = size_raw_data_A / m;

  for (int64_t i=0; i<nlocfft; i++){
    fft(m, raw_data_A+i*m, raw_data_B+i*m);
  }
 
  C["ijk"]-=B["ijk"];


  double norm;

  C.norm2(norm);

  bool pass = (norm <= n*n*m*1.e-6);

  if (myRank == 0){
    if (pass)
      printf("{ A[\"fft(i)jk\"] = DFT[\"il\"]*A[\"ljk\"] } passed\n");
    else
      printf("{ A[\"fft(i)jk\"] = DFT[\"il\"]*A[\"ljk\"] } failed\n");
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
  int64_t n, logm, m;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 6;
  } else n = 6;

  if (getCmdOption(input_str, input_str+in_num, "-logm")){
    logm = atoi(getCmdOption(input_str, input_str+in_num, "-logm"));
    if (logm < 0) logm = 8;
  } else logm = 8;

  m = 1<<logm;

  {
    World dw(argc, argv);
    int pass = fft_with_idx_partition(n, m, dw);
    assert(pass);
  }

  MPI_Finalize();
  
}
/**
 * @} 
 * @}
 */

#endif
