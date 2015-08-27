/** \addtogroup examples 
  * @{ 
  * \defgroup fft fft
  * @{ 
  * \brief FFT iterative method using gemv and spmv
  */

#include <ctf.hpp>
using namespace CTF;

namespace CTF_int{
  void factorize(int n, int *nfactor, int **factor);
}

Matrix< std::complex<double> > DFT_matrix(int n, World & wrld){
  int64_t np;
  int64_t * idx;
  std::complex<double> imag(0,1);
  std::complex<double> * data;
  Matrix < std::complex<double> >DFT(n, n, NS, wrld);
  DFT.read_local(&np, &idx, &data);

  for (int64_t i=0; i<np; i++){
    data[i] = exp(-2.*(idx[i]/n)*(idx[i]%n)*(M_PI/n)*imag);
  }
  DFT.write(np, idx, data);
  free(idx);
  free(data);
  return DFT;
}


void fft(Vector< std::complex<double> > & v, int n){
  int64_t np;
  int64_t * idx;
  std::complex<double> * data;
  
  int nfact;
  int * factors;
  CTF_int::factorize(n, &nfact, &factors);
  
  Tensor< std::complex<double> > V(nfact, factors, *v.wrld, *v.sr);

  v.read_local(&np, &idx, &data);

  V.write(np, idx, data);
  char inds[nfact];
  for (int i=0; i<nfact; i++){
    inds[i]='a'+i;
  }
  
  for (int i=0; i<nfact; i++){
    Matrix< std::complex<double> > DFT = DFT_matrix(factors[i],*v.wrld);
    char inds_in[nfact];
    char inds_out[nfact];
    char inds_DFT[2];
    memcpy(inds_in, inds, nfact*sizeof(char));
    inds_in[i] = 'z';
    memcpy(inds_out, inds, nfact*sizeof(char));
    inds_DFT[0] = inds_out[i];
    inds_DFT[1] = 'z';
    V[inds_out] = DFT[inds_DFT]*V[inds_in];
  }
  free(idx);
  free(data);
 
  V.read_local(&np, &idx, &data);

  v.write(np, idx, data);
  
  free(idx);
  free(data);
}



int fft(int     n,
        World & dw){


  Vector< std::complex<double> > a(n, dw);
  Vector< std::complex<double> > da(n, dw);
  Vector< std::complex<double> > fa(n, dw);

  srand48(dw.rank);
  Transform< std::complex<double> > set_rand([](std::complex<double> & d){ d=std::complex<double>(drand48(),drand48()); } );
  set_rand(a["i"]);
  
  Matrix< std::complex<double> > DFT = DFT_matrix(n,dw);
  
  da["i"] = DFT["ij"]*a["j"];

  fa["i"] = a["i"];
  
  fft(fa, n);

  da["i"] -= fa["i"];

  std::complex<double> dnrm;
  Scalar< std::complex<double> > s(dnrm, dw);
  s[""] += da["i"];
  dnrm = s;  
  bool pass = dnrm.real() <= 1.E-6 && dnrm.imag() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ FFT = DFT } passed \n");
    else
      printf("{ FFT = DFT } failed \n");
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
  int rank, np, n, pass;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;


  {
    World dw(argc, argv);

    if (rank == 0){
      printf("Running FFT on random dimension %d vector\n",n);
    }
    pass = fft(n, dw);
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
