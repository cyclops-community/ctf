/** \addtogroup examples 
  * @{ 
  * \defgroup hosvd hosvd
  * @{ 
  * \brief Calculates a Tucker decomposition via high-order SVD
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
bool hosvd(int n, int R, double sp_frac, World & dw){
  int lens[4] = {n, n+1, n+2, n+3};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(4, is_sparse, lens, dw);

  T.fill_sp_random(-1.,1.,sp_frac);

  Tensor<dtype> U, S, V1, V2, V3, V4;

  T["ijkl"].svd(U["aijk"],S["a"],V1["al"],R+3);
  U["aijk"] *= S["a"];
  U["aijk"].svd(U["abij"],S["b"],V2["bk"],R+2);
  U["abij"] *= S["b"];
  U["abij"].svd(U["abci"],S["c"],V3["cj"],R+1);
  U["abci"] *= S["c"];
  U["abci"].svd(U["abcd"],S["d"],V4["di"],R);
  U["abcd"] *= S["d"];

  double Tnorm;
  T.norm2(Tnorm);
  T["ijkl"] -= U["abcd"]*V1["al"]*V2["bk"]*V3["cj"]*V4["di"];
  double Tnorm2;
  T.norm2(Tnorm2);
  double Rn = ((double)R)/n;
  bool pass = Tnorm2 <= (Tnorm*(1.-Rn*Rn*Rn*Rn) + 1.e-4);
  

  if (pass){
#ifndef TEST_SUITE
    if (dw.rank == 0)
      printf("Passed HoSVD test, Tnorm = %lf, Tnorm2 = %lf\n", Tnorm, Tnorm2);
  } else {
    if (dw.rank == 0)
      printf("FAILED HoSVD test, Tnorm = %lf, Tnorm2 = %lf\n", Tnorm, Tnorm2);
#endif
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
  int n, R;
  double sp;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 7;
  } else n = 7;

  if (getCmdOption(input_str, input_str+in_num, "-R")){
    R = atoi(getCmdOption(input_str, input_str+in_num, "-R"));
    if (R < 0) R = 7;
  } else R = 7;

  if (getCmdOption(input_str, input_str+in_num, "-sp")){
    sp = atof(getCmdOption(input_str, input_str+in_num, "-sp"));
    if (sp < 0.0 || sp > 1.0) sp = .8;
  } else sp = .8;

  {
    World dw;
    if (dw.rank == 0){
      printf("Running sparse (%lf fraction zeros) HoSVD on order 4 tensor with dimension %d and rank %d\n",sp,n,R);
    }

    bool pass;
    pass = hosvd<float>(n,R,sp,dw);
    assert(pass); // failed hosvd<float>
    pass = hosvd<double>(n,R,sp,dw);
    assert(pass); // failed hosvd<double>
    pass = hosvd<std::complex<float>>(n,R,sp,dw);
    assert(pass); // failed hosvd<std::complex<float>>
    pass = hosvd<std::complex<double>>(n,R,sp,dw);
    assert(pass); // failed hosvd<std::complex<double>>
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
