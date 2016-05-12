#include <ctf.hpp>
using namespace CTF;

struct grp{
#ifdef __CUDACC__
  __device__ __host__
#endif
  static double op1(double a, double b){ return b-b/a; };
#ifdef __CUDACC__
  __device__ __host__
#endif
  static void op2(double a, double & b){ b+=a; };
};


void train_off_vec_mat(int64_t n, int64_t m, World & dw){
  for (double sp = .005; sp<.32; sp*=2.){
    Vector<> b(n, dw);
    Vector<> c(m, dw);
    Matrix<> A(m, n, dw);
    Matrix<> B(m, n, dw);
    Matrix<> G(n, n, NS, dw);
  
    srand48(dw.rank);
    b.fill_random(-.5, .5);
    c.fill_random(-.5, .5);
    A.fill_random(-.5, .5);
    B.fill_random(-.5, .5);
    G.fill_random(-.5, .5);
 
    Bivar_Kernel<double,double,double,grp::op1,grp::op2> k1;
    
    if (sp > .099)
      A.sparsify([=](double a){ return fabs(a)<=.5*sp; });
  
    k1(A["ik"],G["kj"],B["ij"]);
    k1(A["ij"],b["j"],c["i"]);
    
  }
}

