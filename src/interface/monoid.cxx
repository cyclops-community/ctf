#include "../sparse_formats/csr.h"
#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/mkl_symbs.h"
#include "../shared/util.h"
using namespace CTF_int;
namespace CTF {
/*  template <>
  void Monoid<float,1>::csr_add(int64_t m, int64_t n, char const * a, int const * ja, int const * ia, char const * b, int const * jb, int const * ib, char *& c, int *& jc, int *& ic){
    if (fadd != default_add<float>){
      printf("CTF error: support for CSR addition for this type unavailable\n");
      assert(0);
    }
    alloc(sizeof(int)*(m+1), (void**)&ic);
    bool tA = 'N';
    bool tB = 'N';
    int job = 1;
    int sort = 1;
    float mlid = 1.0;
    int info;
    MKL_SCSRADD(tA, tB, &job, &sort, m, n, (float*)a, ja, ia, &mlid, (float*)b, jb, ib, NULL, NULL, ic, NULL, &info);
    alloc(sizeof(int)*ic[m], (void**)&jc);
    alloc(sizeof(float)*ic[m], (void**)&c);
    int job = 2;
    MKL_SCSRADD(tA, tB, &job, &sort, m, n, (float*)a, ja, ia, &mlid, (float*)b, jb, ib, (float*)c, jc, ic, NULL, &info);
  }*/

  template <>
  char * CTF::Monoid<double,1>::csr_add(char * cA, char * cB, bool is_ccsr) const {
    //FIXME reduce CCSR add to MKL CSR add for shared part
    if (is_ccsr)
      return CTF_int::algstrct::csr_add(cA, cB, is_ccsr);
#if USE_MKL
    TAU_FSTART(mkl_csr_add)
    if (fadd != &default_add<double>){
      return CTF_int::algstrct::csr_add(cA, cB, is_ccsr);
    }
    CSR_Matrix A(cA);
    CSR_Matrix B(cB);
    int * ic;
    int m = A.nrow();
    int n = A.ncol();
    alloc_ptr(sizeof(int)*(m+1), (void**)&ic);
    char tA = 'N';
    int job = 1;
    int sort = 1;
    double mlid = 1.0;
    int info;
    CTF_BLAS::MKL_DCSRADD(&tA, &job, &sort, &m, &n, (double*)A.vals(), A.JA(), A.IA(), &mlid, (double*)B.vals(), B.JA(), B.IA(), NULL, NULL, ic, NULL, &info);
    CSR_Matrix C(ic[m]-1, m, n, this);
    memcpy(C.IA(), ic, sizeof(int)*(m+1));
    cdealloc(ic);
    job = 2;
    CTF_BLAS::MKL_DCSRADD(&tA, &job, &sort, &m, &n, (double*)A.vals(), A.JA(), A.IA(), &mlid, (double*)B.vals(), B.JA(), B.IA(), (double*)C.vals(), C.JA(), C.IA(), NULL, &info);
    TAU_FSTOP(mkl_csr_add)
    return C.all_data;
#else
    return CTF_int::algstrct::csr_add(cA, cB, is_ccsr);
#endif
  }

}
