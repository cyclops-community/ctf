/** \addtogroup examples 
  * @{ 
  * \defgroup fft fft
  * @{ 
  * \brief FFT iterative method using gemv and spmv
  */

#include <ctf.hpp>
using namespace CTF;

namespace CTF_int{
  // factorizes n into nfactor factors
  void factorize(int n, int *nfactor, int **factor);
}

// gets ith power of nth imaginary root of the identity, w^i_n
std::complex<double> omega(int i, int n){
  std::complex<double> imag(0,1);
  return exp(-2.*i*(M_PI/n)*imag);
}

// gets DFT matrix D(i,j) = w_n^(ij)
Matrix< std::complex<double> > DFT_matrix(int n, World & wrld){
  int64_t np;
  int64_t * idx;
  std::complex<double> * data;
  Matrix < std::complex<double> >DFT(n, n, NS, wrld, "DFT");
  DFT.get_local_data(&np, &idx, &data);

  for (int64_t i=0; i<np; i++){
    data[i] = omega((idx[i]/n)*(idx[i]%n), n);
  }
  DFT.write(np, idx, data);
  free(idx);
  delete [] data;
  return DFT;
}

// gets twiddle factor matirx T = [1, 1; 1, w^1_n]
Matrix< std::complex<double> > twiddle_matrix(int n, World & wrld){
  int64_t np;
  int64_t * idx;
  std::complex<double> * data;
  Matrix < std::complex<double> >T(2, 2, NS, wrld, "T");
  T.get_local_data(&np, &idx, &data);

  for (int64_t i=0; i<np; i++){
    if (idx[i]<3) data[i] = std::complex<double>(1,0);
    else data[i] = omega(1,n);
  }
  T.write(np, idx, data);
  free(idx);
  delete [] data;
  return T;

}

void fft(Vector< std::complex<double> > & v, int n){
  // assert that n is a power of two
  int logn=0;
  while (1<<logn < n){
    logn++;
  }
  assert(1<<logn == n);
  // factorize n, factors will all just be 2 for now
  int nfact;
  int * factors;
  CTF_int::factorize(n, &nfact, &factors);
  assert(nfact == logn);
  
  // Fold v into log_2(n) tensor V
  Tensor< std::complex<double> > V = v.reshape(nfact, factors);

  // define range of indices [a, b, c, ...]
  char inds[nfact+1];
  char rev_inds[nfact];
  for (int i=0; i<nfact; i++){
    inds[i]='a'+i;
    rev_inds[i]='a'+nfact-i-1;
  }
  inds[nfact] = 'a';
  // reverse tensor index order
  // now that each FFT step transforms the lowest level modes, which should be faster
  V[rev_inds] = V[inds];
  for (int i=0; i<nfact; i++){
    inds[nfact+i]='a'+i;
  }
 
  Matrix< std::complex<double> > DFT = DFT_matrix(2,*v.wrld);
  Tensor< std::complex<double> > S_old(1, factors, *v.wrld, *v.sr, "S");
  S_old["i"] = 1;

  char inds_in[nfact];
  char inds_DFT[2];
  // m = 2^{i+1}
  int m = 1;
  // execute FFT from the bottom level of recursion upwards
  for (int i=0; i<nfact; i++){
    m*=2;
    // S(m) when unfolded looks like [ 1     1     1     ... 1         ]
    //                               [ w_m   w_m^2 w_m^3 ... w_m^{m/2} ]
    // it is used to scale the odd elements of the FFT, i.e., we compute recursively FFT(v) via
    // for i=1 to m/2, A_i =       sum_{j=0}^{m/2-1} v_{2j}  *w_{m/2}^{ij} 
    //                 B_i = w_m^i sum_{j=0}^{m/2-1} v_{2j+1}*w_{m/2}^{ij}
    // multiplying V by S will do the scaling of the elements of B_i by w_m^i
    // we do it first, since we want to apply it to the sequences obtain recursively 
    // note that for i=0, m=2 the application of S(2) = [1; 1] is trivial
    Tensor< std::complex<double> > S;
    if (i==0){
      S = S_old;
    } else {
      // we construct S(m) by the Kharti-Rao product of S(m/2) and the twiddle factor [1, 1; 1, w_m]
      S = Tensor< std::complex<double> >(i+1, factors, *v.wrld, *v.sr, "S");
      Matrix< std::complex<double> > T = twiddle_matrix(m, *v.wrld);
      char inds_T[2] = {'a', inds[i]};
      S[inds] = T[inds_T]*S_old[inds+1];
      S_old = S;
    }
    V[inds] *= S[inds];

    // then we multiply by the 2-by-2 DFT matrix, which is just [1, 1; 1, -1]
    // this gives FFT(v) = [A+B; A-B]
    // this combines the recursive sequences or does the DFT at the base case when i=0
    memcpy(inds_in, inds, nfact*sizeof(char));
    inds_in[i] = 'a'-1;
    inds_DFT[0] = inds[i];
    inds_DFT[1] = 'a'-1;
    V[inds] = DFT[inds_DFT]*V[inds_in];
  }

  // we now unfold the tensor back into vector form
  v.reshape(V);
}


int fft(int     n,
        World & dw){

  Vector< std::complex<double> > a(n, dw, "a");
  Vector< std::complex<double> > da(n, dw, "da");

  srand48(2*dw.rank);
  Transform< std::complex<double> > set_rand([](std::complex<double> & d){ d=std::complex<double>(drand48(),drand48()); } );
  set_rand(a["i"]);
  
  Matrix< std::complex<double> > DFT = DFT_matrix(n,dw);
  
  da["i"] = DFT["ij"]*a["j"];

  Vector< std::complex<double> > fa(n, dw, "fa");
  fa["i"] = a["i"];
  
  fft(fa, n);

  da["i"] -= fa["i"];

  std::complex<double> dnrm;
  Scalar< std::complex<double> > s(dnrm, dw);
  s[""] += da["i"]*da["i"];
  dnrm = s;  
  bool pass = fabs(dnrm.real()) <= 1.E-6 && fabs(dnrm.imag()) <= 1.E-6;

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
    if (n < 0) n = 16;
  } else n = 16;

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
