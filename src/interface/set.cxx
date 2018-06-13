#include "set.h"
#include "../shared/blas_symbs.h"
#include "../shared/mkl_symbs.h"
#include "../shared/util.h"


namespace CTF_int {

#ifdef USE_MPI_CPP
  MPI_Datatype MPI_CTF_BOOL = MPI::BOOL;
  MPI_Datatype MPI_CTF_DOUBLE_COMPLEX = MPI::DOUBLE_COMPLEX;
  MPI_Datatype MPI_CTF_LONG_DOUBLE_COMPLEX = MPI::LONG_DOUBLE_COMPLEX;
#else
  MPI_Datatype MPI_CTF_BOOL = MPI_CXX_BOOL;
  MPI_Datatype MPI_CTF_DOUBLE_COMPLEX = MPI_CXX_DOUBLE_COMPLEX;
  MPI_Datatype MPI_CTF_LONG_DOUBLE_COMPLEX = MPI_CXX_LONG_DOUBLE_COMPLEX;
#endif

#if USE_MKL
  void def_coo_to_csr_fl(int64_t nz, int nrow, float * csr_vs, int * csr_ja, int * csr_ia, float * coo_vs, int * coo_rs, int * coo_cs, bool to_csr){
    int inz = nz;
    int info;

    if (to_csr){
      int job[8]={2,1,1,0,inz,0,0,0};
      CTF_BLAS::MKL_SCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (float*)coo_vs, coo_rs, coo_cs, &info);
    } else {
      int job[8]={0,1,1,0,inz,3,0,0};
      CTF_BLAS::MKL_SCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (float*)coo_vs, coo_rs, coo_cs, &info);
    }
  }
  void def_coo_to_csr_dbl(int64_t nz, int nrow, double * csr_vs, int * csr_ja, int * csr_ia, double * coo_vs, int * coo_rs, int * coo_cs, bool to_csr){
    int inz = nz;
    int info;
    if (to_csr){
      TAU_FSTART(MKL_DCOOCSR);
      int job[8]={2,1,1,0,inz,0,0,0};
      CTF_BLAS::MKL_DCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (double*)coo_vs, coo_rs, coo_cs, &info);
      TAU_FSTOP(MKL_DCOOCSR);
    } else {
      TAU_FSTART(MKL_DCSRCOO);
      int job[8]={0,1,1,0,inz,3,0,0};
      CTF_BLAS::MKL_DCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (double*)coo_vs, coo_rs, coo_cs, &info);
      /*printf("converted %d nonzers to coo\n", inz);
      for (int i=0; i<inz; i++){
        printf("i=%d\n",i);
        printf("vs[i] = %lf\n",coo_vs[i]);
        printf("rs[i] = %d\n",coo_rs[i]);
        printf("cs[i] = %d\n",coo_cs[i]);
      }*/
      ASSERT(inz == nz);
      TAU_FSTOP(MKL_DCSRCOO);
    }
  }

  void def_coo_to_csr_cdbl(int64_t nz, int nrow, std::complex<double> * csr_vs, int * csr_ja, int * csr_ia, std::complex<double> * coo_vs, int * coo_rs, int * coo_cs, bool to_csr){
    int inz = nz;
    int info;
    
    if (to_csr){
      int job[8]={2,1,1,0,inz,0,0,0};
      CTF_BLAS::MKL_ZCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (std::complex<double>*)coo_vs, coo_rs, coo_cs, &info);
    } else {
      int job[8]={0,1,1,0,inz,3,0,0};
      CTF_BLAS::MKL_ZCSRCOO(job, &nrow, csr_vs, csr_ja, csr_ia, &inz, (std::complex<double>*)coo_vs, coo_rs, coo_cs, &info);
    }
  }
#endif

  bool try_mkl_coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_ja, int * csr_ia, char const * coo_vs, int const * coo_rs, int const * coo_cs, int el_size){
#if USE_MKL
    switch (el_size){
      case 4:
        def_coo_to_csr_fl(nz,nrow,(float*)csr_vs,csr_ja,csr_ia,(float*)coo_vs,(int*)coo_rs,(int*)coo_cs,1);
        return true; 
        break;
      case 8:
        def_coo_to_csr_dbl(nz,nrow,(double*)csr_vs,csr_ja,csr_ia,(double*)coo_vs,(int*)coo_rs,(int*)coo_cs,1);
        return true; 
        break;
      case 16:
        def_coo_to_csr_cdbl(nz,nrow,(std::complex<double>*)csr_vs,csr_ja,csr_ia,(std::complex<double>*)coo_vs,(int*)coo_rs,(int*)coo_cs,1);
        return true; 
        break;
    } 
#endif
    return false;
  }


  bool try_mkl_csr_to_coo(int64_t nz, int nrow, char const * csr_vs, int const * csr_ja, int const * csr_ia, char * coo_vs, int * coo_rs, int * coo_cs, int el_size){
#if USE_MKL
    switch (el_size){
      case 4:
        def_coo_to_csr_fl(nz,nrow,(float*)csr_vs,(int*)csr_ja,(int*)csr_ia,(float*)coo_vs,coo_rs,coo_cs,0);
        return true; 
        break;
      case 8:
        def_coo_to_csr_dbl(nz,nrow,(double*)csr_vs,(int*)csr_ja,(int*)csr_ia,(double*)coo_vs,coo_rs,coo_cs,0);
        return true; 
        break;
      case 16:
        def_coo_to_csr_cdbl(nz,nrow,(std::complex<double>*)csr_vs,(int*)csr_ja,(int*)csr_ia,(std::complex<double>*)coo_vs,coo_rs,coo_cs,0);
        return true; 
        break;
    } 
#endif
    return false;
  }
}

namespace CTF {
  template <>
  void CTF::Set<float,true>::copy(int64_t nn, char const * a, int inc_a, char * b, int inc_b) const {
    int n = nn;
    CTF_BLAS::SCOPY(&n, (float const*)a, &inc_a, (float*)b, &inc_b);
  }
  template <>
  void CTF::Set<double,true>::copy(int64_t nn, char const * a, int inc_a, char * b, int inc_b) const {
    int n = nn;
    CTF_BLAS::DCOPY(&n, (double const*)a, &inc_a, (double*)b, &inc_b);
  }
  template <>
  void CTF::Set<std::complex<float>,false>::copy(int64_t nn, char const * a, int inc_a, char * b, int inc_b) const {
    int n = nn;
    CTF_BLAS::DCOPY(&n, (double const*)a, &inc_a, (double*)b, &inc_b);
  }
  template <>
  void CTF::Set<std::complex<double>,false>::copy(int64_t nn, char const * a, int inc_a, char * b, int inc_b) const {
    int n = nn;
    CTF_BLAS::ZCOPY(&n, (std::complex<double> const*)a, &inc_a, (std::complex<double>*)b, &inc_b);
  }

}
