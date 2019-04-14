/** \addtogroup examples 
  * @{ 
  * \defgroup mttkrp mttkrp
  * @{ 
  * \brief Calculates a Tucker decomposition via high-order SVD
  */

#include <ctf.hpp>
#include <float.h>
using namespace CTF;

template <typename dtype>
bool mttkrp(int n, int R, double sp_frac, World & dw){
  int lens[4] = {n, n+1, n+2, n+3};
  bool is_sparse = sp_frac < 1.;

  Tensor<dtype> T(4, is_sparse, lens, dw);

  T.fill_sp_random(-1.,1.,sp_frac);

  Matrix<dtype> U1(lens[0], R, dw);
  Matrix<dtype> U2(lens[1], R, dw);
  Matrix<dtype> U3(lens[2], R, dw);
  Matrix<dtype> U4(lens[3], R, dw);
  U1.fill_random((dtype)0,(dtype)1);
  U2.fill_random((dtype)0,(dtype)1);
  U3.fill_random((dtype)0,(dtype)1);
  U4.fill_random((dtype)0,(dtype)1);

  Matrix<dtype> V1(lens[0], R, dw);
  Matrix<dtype> V2(lens[1], R, dw);
  Matrix<dtype> V3(lens[2], R, dw);
  Matrix<dtype> V4(lens[3], R, dw);

  Matrix<dtype> W1(lens[0], R, dw);
  Matrix<dtype> W2(lens[1], R, dw);
  Matrix<dtype> W3(lens[2], R, dw);
  Matrix<dtype> W4(lens[3], R, dw);
  
  //int lens2[4] = {n, n+1, n+2, R};
  //Tensor<dtype> T2(4, is_sparse, lens2, dw);
  //Tensor<dtype> T3(4, lens2, dw);
  //T2["ijka"] = T["ijkl"]*U4["la"];
  //T3["ijka"] = T["ijkl"]*U4["la"];
  //T2.print();
  //T3.print();
  //T3["ijka"] -= T2["ijka"];
  //double nrm;
  //T3.norm2(nrm);
  //printf("nrm is %lf\n",nrm);

  V1["ia"] = T["ijkl"]*U2["ja"]*U3["ka"]*U4["la"];
  V2["ja"] = T["ijkl"]*U1["ia"]*U3["ka"]*U4["la"];
  V3["ka"] = T["ijkl"]*U1["ia"]*U2["ja"]*U4["la"];
  V4["la"] = T["ijkl"]*U1["ia"]*U2["ja"]*U3["ka"];


  for (int i=0; i<R; i++){
    Vector<dtype> u1(n, dw), u2(lens[1], dw), u3(lens[2], dw), u4(lens[3], dw);
    Vector<dtype> w1(n, dw), w2(lens[1], dw), w3(lens[2], dw), w4(lens[3], dw);
    u1.reshape(U1.slice(i*lens[0],(i+1)*lens[0]-1));
    u2.reshape(U2.slice(i*lens[1],(i+1)*lens[1]-1));
    u3.reshape(U3.slice(i*lens[2],(i+1)*lens[2]-1));
    u4.reshape(U4.slice(i*lens[3],(i+1)*lens[3]-1));

    Tensor<dtype> * mlist1[3] = {&u2, &u3, &u4};
    int modes1[3] = {1,2,3};
    Tensor<dtype> R(T);
    TTTP<dtype>(&R, 3, modes1, mlist1, false);
    w1["i"] += R["ijkl"];
    W1.slice(i*lens[0],(i+1)*lens[0]-1,(dtype)1.,w1,0,lens[0]-1,(dtype)1.);

    Tensor<dtype> * mlist2[3] = {&u1, &u3, &u4};
    int modes2[3] = {0,2,3};
    R = T;
    TTTP<dtype>(&R, 3, modes2, mlist2, false);
    w2["j"] += R["ijkl"];
    W2.slice(i*lens[1],(i+1)*lens[1]-1,(dtype)1.,w2,0,lens[1]-1,(dtype)1.);

    Tensor<dtype> * mlist3[3] = {&u1, &u2, &u4};
    int modes3[3] = {0,1,3};
    R = T;
    TTTP<dtype>(&R, 3, modes3, mlist3, false);
    w3["k"] += R["ijkl"];
    W3.slice(i*lens[2],(i+1)*lens[2]-1,(dtype)1.,w3,0,lens[2]-1,(dtype)1.);

    Tensor<dtype> * mlist4[3] = {&u1, &u2, &u3};
    int modes4[3] = {0,1,2};
    R = T;
    TTTP<dtype>(&R, 3, modes4, mlist4, false);
    w4["l"] += R["ijkl"];
    W4.slice(i*lens[3],(i+1)*lens[3]-1,(dtype)1.,w4,0,lens[3]-1,(dtype)1.);
  }

  W1["ia"] -= V1["ia"];
  W2["ia"] -= V2["ia"];
  W3["ia"] -= V3["ia"];
  W4["ia"] -= V4["ia"];
  double norm1, norm2, norm3, norm4;
  W1.norm2(norm1);
  W2.norm2(norm2);
  W3.norm2(norm3);
  W4.norm2(norm4);
  int64_t sz = T.get_tot_size(false);
  bool pass;
  pass = (norm1/sz < 1.e-5) && (norm2/sz < 1.e-5) && (norm3/sz < 1.e-5) && (norm4/sz < 1.e-5);
  if (dw.rank == 0){
    if (!pass)
      printf("FAILED MTTKRP tests norms are %lf %lf %lf %lf\n",norm1,norm2,norm3,norm4);
    else
      printf("Passed MTTKRP tests.\n");
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
      printf("Running sparse (%lf fraction zeros) MTTKRP on order 4 tensor with dimension %d and rank %d\n",sp,n,R);
    }

    bool pass;
    pass = mttkrp<float>(n,R,sp,dw);
    assert(pass); // failed mttkrp<float>
    pass = mttkrp<double>(n,R,sp,dw);
    assert(pass); // failed mttkrp<double>
    pass = mttkrp<std::complex<float>>(n,R,sp,dw);
    assert(pass); // failed mttkrp<std::complex<float>>
    pass = mttkrp<std::complex<double>>(n,R,sp,dw);
    assert(pass); // failed mttkrp<std::complex<double>>
  }

  MPI_Finalize();
  return 0;
}
/**
 * @} 
 * @}
 */

#endif
