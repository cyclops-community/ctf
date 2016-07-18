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
  static double op2_t2(double a, double b){ return a+b; };
  static void op2_red(double const * a, double * b, int n){ 
    #pragma omp parallel for
    for (int i=0; i<n; i++){
      b[i] += a[i];
    }
  }
};


void train_off_vec_mat(int64_t n, int64_t m, World & dw, bool sp_A, bool sp_B, bool sp_C){
  MPI_Op madd;
  MPI_Op_create([](void * a, void * b, int * n, MPI_Datatype*){ 
                  grp::op2_red((double*)a, (double*)b, *n);
                }, 1, &madd);
  Monoid<> mon(0, grp::op2_t2, madd);
  for (double sp = .005; sp<.32; sp*=2.){
    Matrix<> A(m, n, dw, mon);
    Matrix<> B(m, n, dw, mon);
    Matrix<> G(n, n, dw, mon);
    Vector<> b(n, dw, mon);
    Vector<> c(m, dw, mon);
  
    srand48(dw.rank);
    b.fill_random(-.5, .5);
    c.fill_random(-.5, .5);
    A.fill_random(-.5, .5);
    B.fill_random(-.5, .5);
    G.fill_random(-.5, .5);
 
    Bivar_Kernel<double,double,double,grp::op1,grp::op2> k1;
    
    if (sp > .009){
      if (sp_A)
        A.sparsify([=](double a){ return fabs(a)<=.5*sp; });
      if (sp_B){
        G.sparsify([=](double a){ return fabs(a)<=.5*sp; });
        b.sparsify([=](double a){ return fabs(a)<=.5*sp; });
      }
      if (sp_C){
        B.sparsify([=](double a){ return fabs(a)<=.5*sp; });
        c.sparsify([=](double a){ return fabs(a)<=.5*sp; });
      }
    }
  
    k1(A["ik"],G["kj"],B["ij"]);
    k1(A["ij"],b["j"],c["i"]);
    
  }
}

