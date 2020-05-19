#ifndef __SET_H__
#define __SET_H__

#include "../tensor/algstrct.h"
#include "functions.h"
//#include <stdint.h>
#include <limits>
#include <inttypes.h>
#include "../shared/memcontrol.h"
#ifdef _OPENMP
#include "omp.h"
#endif

#ifdef _OPENMP
#include "omp.h"
#endif
namespace CTF {
  /**
   * \brief index-value pair used for tensor data input
   */
  template<typename dtype=double>
  class Pair  {
    public:
      /** \brief key, global index [i1,i2,...] specified as i1+len[0]*i2+... */
      int64_t k;

      /** \brief tensor value associated with index */
      dtype d;

      /**
       * \brief constructor builds pair
       * \param[in] k_ key
       * \param[in] d_ value
       */
      Pair(int64_t k_, dtype d_){
        this->k = k_;
        d = d_;
      }

      /**
       * \brief default constructor
       */
      Pair(){
        //k=0;
        //d=0; //(not possible if type has no zero!)
      }

      /**
       * \brief determines pair ordering
       */
      bool operator<(Pair<dtype> other) const {
        return k<other.k;
      }

  };

  template<typename dtype>
  inline bool comp_pair(Pair<dtype> i,
                        Pair<dtype> j) {
    return (i.k<j.k);
  }
}

namespace CTF_int {

  //does conversion using MKL function if it is available
  bool try_mkl_coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_ja, int * csr_ia, char const * coo_vs, int const * coo_rs, int const * coo_cs, int el_size);

  bool try_mkl_csr_to_coo(int64_t nz, int nrow, char const * csr_vs, int const * csr_ja, int const * csr_ia, char * coo_vs, int * coo_rs, int * coo_cs, int el_size);

  template <typename dtype>
  void seq_coo_to_csr(int64_t nz, int nrow, dtype * csr_vs, int * csr_ja, int * csr_ia, dtype const * coo_vs, int const * coo_rs, int const * coo_cs){
    int sz = sizeof(dtype);
    if (sz == 4 || sz == 8 || sz == 16){
      bool b = try_mkl_coo_to_csr(nz, nrow, (char*)csr_vs, csr_ja, csr_ia, (char const*)coo_vs, coo_rs, coo_cs, sz);
      if (b) return;
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      csr_ja[i] = i;
      //printf("csr_ja[%d/%d] = %d\n",i,nz,csr_ja[i]);
    }

    class comp_ref {
      public:
        int const * a;
        comp_ref(int const * a_){ a = a_; }
        bool operator()(int u, int v){
          return a[u] < a[v];
        }
    };

    comp_ref crc(coo_cs);
    std::sort(csr_ja, csr_ja+nz, crc);
    comp_ref crr(coo_rs);
    std::stable_sort(csr_ja, csr_ja+nz, crr);
    // do not copy by value in case values are objects, then csr_vs is uninitialized
    //printf("csr nz = %ld\n",nz);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      //printf("%d, %d, %ld\n",(int)((char*)(coo_vs+csr_ja[i])-(char*)(coo_vs))-csr_ja[i]*sizeof(dtype),sizeof(dtype),csr_ja[i]);
//      memcpy(csr_vs+i, coo_vs+csr_ja[i]-1,sizeof(dtype));
      //memcpy(csr_vs+i, coo_vs+csr_ja[i],sizeof(dtype));
      csr_vs[i] = coo_vs[csr_ja[i]];
//      printf("i %ld csr_ja[i] %d\n", i, csr_ja[i]);
//      printf("i %ld v %lf\n", i, csr_vs[i]);
      //printf("%p %d\n",coo_vs+i,*(int32_t*)(coo_vs+i));
    }
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      csr_ja[i] = coo_cs[csr_ja[i]];
    }
    csr_ia[0] = 1;
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i=1; i<nrow+1; i++){
      csr_ia[i] = 0;
    }
#ifdef _OPENMP
    int * scoo_rs = (int*)CTF_int::alloc(sizeof(int)*nz);
    memcpy(scoo_rs, coo_rs, nz*sizeof(int));
    std::sort(scoo_rs,scoo_rs+nz);
    #pragma omp parallel 
    {
      int tid = omp_get_thread_num();
      int ntd = omp_get_num_threads();
      int64_t i_st = tid*(nz/ntd)+std::min(tid,(int)(nz%ntd));
      int64_t i_end = (tid+1)*(nz/ntd)+std::min((tid+1),(int)(nz%ntd));
      while (i_st > 0 && i_st < nz && scoo_rs[i_st] == scoo_rs[i_st-1]) i_st++;
      while (i_end < nz && scoo_rs[i_end] == scoo_rs[i_end-1]) i_end++;
      for (int64_t i=i_st; i<i_end; i++){
        csr_ia[scoo_rs[i]]++;
      }
    }
    CTF_int::cdealloc(scoo_rs);
#else
    for (int64_t i=0; i<nz; i++){
      //printf("scoo_rs[%d]=%d\n",i,scoo_rs[i]);
      csr_ia[coo_rs[i]]++;
    }
#endif

#ifdef _OPENMP
    //int * csr_ia2 = (int*)CTF_int::alloc(sizeof(int)*(nrow+1));
    //CTF_int::prefix<int>(nrow+1, csr_ia, csr_ia2);
    ////memcpy(csr_ia, csr_ia2, nrow*sizeof(int));
    //#pragma omp parallel for
    //for (int i=0; i<nrow+1; i++){
    //  assert((i==0 && csr_ia2[i] == 0) || csr_ia[i-1] == csr_ia2[i]);
    //  csr_ia[i] += csr_ia2[i];
    //  printf("csr_ia[%d/%d] = %d\n",i,nrow,csr_ia[i]);
    //}
    //CTF_int::cdealloc(csr_ia2);
    CTF_int::parallel_postfix<int>(nrow+1, 1, csr_ia);
#else
    for (int i=0; i<nrow; i++){
      csr_ia[i+1] += csr_ia[i];
      //printf("csr_ia[%d/%d] = %d\n",i,nrow,csr_ia[i]);
    }
#endif
  }

  template <typename dtype>
  void seq_coo_to_ccsr(int64_t nz, int64_t nnz_row, dtype * ccsr_vs, int * ccsr_ja, int * ccsr_ia, dtype const * coo_vs, int64_t const * coo_rs, int64_t const * coo_cs){
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      ccsr_ja[i] = i;
      //printf("ccsr_ja[%d/%d] = %d\n",i,nz,ccsr_ja[i]);
    }

    class comp_ref {
      public:
        int64_t const * a;
        comp_ref(int64_t const * a_){ a = a_; }
        bool operator()(int u, int v){
          return a[u] < a[v];
        }
    };

    comp_ref crc(coo_cs);
    std::sort(ccsr_ja, ccsr_ja+nz, crc);
    comp_ref crr(coo_rs);
    std::stable_sort(ccsr_ja, ccsr_ja+nz, crr);
    // do not copy by value in case values are objects, then ccsr_vs is uninitialized
    //printf("ccsr nz = %ld\n",nz);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      //printf("%d, %d, %ld\n",(int)((char*)(coo_vs+ccsr_ja[i])-(char*)(coo_vs))-ccsr_ja[i]*sizeof(dtype),sizeof(dtype),ccsr_ja[i]);
//      memcpy(ccsr_vs+i, coo_vs+ccsr_ja[i]-1,sizeof(dtype));
      //memcpy(ccsr_vs+i, coo_vs+ccsr_ja[i],sizeof(dtype));
      ccsr_vs[i] = coo_vs[ccsr_ja[i]];
//      printf("i %ld ccsr_ja[i] %d\n", i, ccsr_ja[i]);
//      printf("i %ld v %lf\n", i, ccsr_vs[i]);
      //printf("%p %d\n",coo_vs+i,*(int32_t*)(coo_vs+i));
    }
    ccsr_ia[0] = 1;
    ccsr_ia[1] = 1 + (nz>0);
    //FIXME: parallelize
    int64_t cia = 1;
    for (int64_t i=1; i<nz; i++){
      if (coo_rs[ccsr_ja[i]] > coo_rs[ccsr_ja[i-1]]){
        cia++;
        ccsr_ia[cia] = ccsr_ia[cia-1];
      }
      ccsr_ia[cia]++;
    }

//#ifdef _OPENMP
//    #pragma omp parallel for
//#endif
//    for (int i=0; i<nnz_row; i++){
//      ccsr_ia[i+1] += ccsr_ia[i];
//      //printf("ccsr_ia[%d/%d] = %d\n",i,nrow,ccsr_ia[i]);
//    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      ccsr_ja[i] = coo_cs[ccsr_ja[i]];
    }
  }

  template <typename dtype>
  void seq_csr_to_coo(int64_t nz, int nrow, dtype const * csr_vs, int const * csr_ja, int const * csr_ia, dtype * coo_vs, int * coo_rs, int * coo_cs){
    int sz = sizeof(dtype);
    if (sz == 4 || sz == 8 || sz == 16){
      bool b = try_mkl_csr_to_coo(nz, nrow, (char const*)csr_vs, csr_ja, csr_ia, (char*)coo_vs, coo_rs, coo_cs, sz);
      if (b) return;
    }
    //memcpy(coo_vs, csr_vs, sizeof(dtype)*nz);
    std::copy(csr_vs, csr_vs+nz, coo_vs);
    memcpy(coo_cs, csr_ja, sizeof(int)*nz);
    for (int i=0; i<nrow; i++){
      std::fill(coo_rs+csr_ia[i]-1, coo_rs+csr_ia[i+1]-1, i+1);
    }
  }

  template <typename dtype>
  void def_coo_to_ccsr(int64_t nz, int64_t nnz_row, dtype * ccsr_vs, int * ccsr_ja, int * ccsr_ia, dtype const * coo_vs, int64_t const * coo_rs, int64_t const * coo_cs){
    seq_coo_to_ccsr<dtype>(nz, nnz_row, ccsr_vs, ccsr_ja, ccsr_ia, coo_vs, coo_rs, coo_cs);
  }

  template <typename dtype>
  void def_coo_to_csr(int64_t nz, int nrow, dtype * csr_vs, int * csr_ja, int * csr_ia, dtype const * coo_vs, int const * coo_rs, int const * coo_cs){
    seq_coo_to_csr<dtype>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }

  template <typename dtype>
  void def_csr_to_coo(int64_t nz, int nrow, dtype const * csr_vs, int const * csr_ja, int const * csr_ia, dtype * coo_vs, int * coo_rs, int * coo_cs){
    seq_csr_to_coo<dtype>(nz, nrow, csr_vs, csr_ja, csr_ia, coo_vs, coo_rs, coo_cs);
  }

  template <typename dtype>
  void seq_ccsr_to_coo(int64_t nz, int64_t nnz_row, dtype const * ccsr_vs, int const * ccsr_ja, int const * ccsr_ia, int64_t const * row_enc, dtype * coo_vs, int64_t * coo_rs, int64_t * coo_cs){
    //memcpy(coo_vs, ccsr_vs, sizeof(dtype)*nz);
    std::copy(ccsr_vs, ccsr_vs+nz, coo_vs);
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nz; i++){
      coo_cs[i] = ccsr_ja[i];
    }

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int64_t i=0; i<nnz_row; i++){
      std::fill(coo_rs+ccsr_ia[i]-1, coo_rs+ccsr_ia[i+1]-1, row_enc[i]);
    }
  }

  template <typename dtype>
  void def_coo_to_ccsr(int64_t nz, int64_t nnz_row, dtype * ccsr_vs, int * ccsr_ja, int * ccsr_ia, int const * row_enc, dtype const * coo_vs, int64_t const * coo_rs, int64_t const * coo_cs){
    seq_coo_to_ccsr<dtype>(nz, nnz_row, ccsr_vs, ccsr_ja, ccsr_ia, row_enc, coo_vs, coo_rs, coo_cs);
  }

  template <typename dtype>
  void def_ccsr_to_coo(int64_t nz, int64_t nnz_row, dtype const * ccsr_vs, int const * ccsr_ja, int const * ccsr_ia, int64_t const * row_enc, dtype * coo_vs, int64_t * coo_rs, int64_t * coo_cs){
    seq_ccsr_to_coo<dtype>(nz, nnz_row, ccsr_vs, ccsr_ja, ccsr_ia, row_enc, coo_vs, coo_rs, coo_cs);
  }

  template <typename dtype>
  bool default_isequal(dtype a, dtype b){
    int sz = sizeof(dtype);
    for (int i=0; i<sz; i++){
      if (((char const *)&a)[i] != ((char const *)&b)[i]){
        return false;
      }
    }
    return true;
  }

  template <typename dtype>
  dtype default_addinv(dtype a){
    return -a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_abs(dtype a){
    dtype b = default_addinv<dtype>(a);
    return a>=b ? a : b;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_abs(dtype a){
    printf("CTF ERROR: cannot compute abs unless the set is ordered");
    assert(0);
    return a;
  }

  template <typename dtype, dtype (*abs)(dtype)>
  void char_abs(char const * a,
                char * b){
    ((dtype*)b)[0]=abs(((dtype const*)a)[0]);
  }

  //C++14 support needed for these std::enable_if
  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_min(dtype a, dtype b){
    return a>b ? b : a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_min(dtype a, dtype b){
    printf("CTF ERROR: cannot compute a max unless the set is ordered");
    assert(0);
    return a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_max_lim(){
    return std::numeric_limits<dtype>::max();
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_max_lim(){
    printf("CTF ERROR: cannot compute a max unless the set is ordered");
    assert(0);
    dtype * a = NULL;
    return *a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_min_lim(){
    return std::numeric_limits<dtype>::min();
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_min_lim(){
    printf("CTF ERROR: cannot compute a max unless the set is ordered");
    assert(0);
    dtype * a = NULL;
    return *a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<is_ord, dtype>::type
  default_max(dtype a, dtype b){
    return b>a ? b : a;
  }

  template <typename dtype, bool is_ord>
  inline typename std::enable_if<!is_ord, dtype>::type
  default_max(dtype a, dtype b){
    printf("CTF ERROR: cannot compute a min unless the set is ordered");
    assert(0);
    return a;
  }
  template <typename dtype>
  MPI_Datatype get_default_mdtype(bool & is_custom){
    MPI_Datatype newtype;
    MPI_Type_contiguous(sizeof(dtype), MPI_BYTE, &newtype);
    MPI_Type_commit(&newtype);
    is_custom = true;
    return newtype;
  }

  extern MPI_Datatype MPI_CTF_BOOL;
  extern MPI_Datatype MPI_CTF_DOUBLE_COMPLEX;
  extern MPI_Datatype MPI_CTF_LONG_DOUBLE_COMPLEX;

  template <>
  inline MPI_Datatype get_default_mdtype<bool>(bool & is_custom){ is_custom=false; return MPI_CTF_BOOL; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<double> >(bool & is_custom){ is_custom=false; return MPI_CTF_DOUBLE_COMPLEX; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<long double> >(bool & is_custom){ is_custom=false; return MPI_CTF_LONG_DOUBLE_COMPLEX; }
  template <>
  inline MPI_Datatype get_default_mdtype<char>(bool & is_custom){ is_custom=false; return MPI_CHAR; }
  template <>
  inline MPI_Datatype get_default_mdtype<int>(bool & is_custom){ is_custom=false; return MPI_INT; }
  template <>
  inline MPI_Datatype get_default_mdtype<int64_t>(bool & is_custom){ is_custom=false; return MPI_INT64_T; }
  template <>
  inline MPI_Datatype get_default_mdtype<unsigned int>(bool & is_custom){ is_custom=false; return MPI_UNSIGNED; }
  template <>
  inline MPI_Datatype get_default_mdtype<uint64_t>(bool & is_custom){ is_custom=false; return MPI_UINT64_T; }
  template <>
  inline MPI_Datatype get_default_mdtype<float>(bool & is_custom){ is_custom=false; return MPI_FLOAT; }
  template <>
  inline MPI_Datatype get_default_mdtype<double>(bool & is_custom){ is_custom=false; return MPI_DOUBLE; }
  template <>
  inline MPI_Datatype get_default_mdtype<long double>(bool & is_custom){ is_custom=false; return MPI_LONG_DOUBLE; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<float> >(bool & is_custom){ is_custom=false; return MPI_COMPLEX; }



  template <typename dtype>
  constexpr bool get_default_is_ord(){
    return false;
  }

  #define INST_ORD_TYPE(dtype)                  \
    template <>                                 \
    constexpr bool get_default_is_ord<dtype>(){ \
      return true;                              \
    }

  INST_ORD_TYPE(float)
  INST_ORD_TYPE(double)
  INST_ORD_TYPE(long double)
  INST_ORD_TYPE(bool)
  INST_ORD_TYPE(char)
  INST_ORD_TYPE(int)
  INST_ORD_TYPE(unsigned int)
  INST_ORD_TYPE(int64_t)
  INST_ORD_TYPE(uint64_t)

  #define INST_IET(typ) \
    template <> \
    inline bool default_isequal<typ>(typ a, typ b){ \
      return a==b; \
    } \

  INST_IET(float)
  INST_IET(double)
  INST_IET(std::complex<float>)
  INST_IET(std::complex<double>)
  INST_IET(bool)
  INST_IET(int)
  INST_IET(int16_t)
  INST_IET(int64_t)
  INST_IET(uint16_t)
  INST_IET(uint32_t)
  INST_IET(uint64_t)
  INST_IET(std::complex<long double>)
  INST_IET(long double)


}


namespace CTF {

  /** \brief pair for sorting */
  template <typename dtype>
  struct dtypePair{
    int64_t key;
    dtype data;
    bool operator < (const dtypePair<dtype>& other) const {
      return (key < other.key);
    }
  };

  /**
   * \defgroup algstrct Algebraic Structures
   * \addtogroup algstrct
   * @{
   */

  /**
   * \brief Set class defined by a datatype and a min/max function (if it is partially ordered i.e. is_ord=true)
   *         currently assumes min and max are given by numeric_limits (custom min/max not allowed)
   */
  template <typename dtype=double, bool is_ord=CTF_int::get_default_is_ord<dtype>()>
  class Set : public CTF_int::algstrct {
    public:
      int pair_sz;
      bool is_custom_mdtype;
      MPI_Datatype tmdtype;
      ~Set(){
        if (is_custom_mdtype) MPI_Type_free(&tmdtype);
      }

      Set(Set const & other) : CTF_int::algstrct(other) {
        if (other.is_custom_mdtype){
          tmdtype = CTF_int::get_default_mdtype<dtype>(is_custom_mdtype);
        } else {
          this->tmdtype = other.tmdtype;
          is_custom_mdtype = false;
        }
        pair_sz = sizeof(std::pair<int64_t,dtype>);
        //printf("%ld %ld \n", sizeof(dtype), pair_sz);
        abs = other.abs;
      }

      int pair_size() const {
        //printf("%d %d \n", sizeof(dtype), pair_sz);
        return pair_sz;
      }

      int64_t get_key(char const * a) const {
        return ((std::pair<int64_t,dtype> const *)a)->first;
      }

      char * get_value(char * a) const {
        return (char*)&(((std::pair<int64_t,dtype> const *)a)->second);
      }

      char const * get_const_value(char const * a) const {
        return (char const *)&(((std::pair<int64_t,dtype> const *)a)->second);
      }



      virtual CTF_int::algstrct * clone() const {
        return new Set<dtype, is_ord>(*this);
      }

      bool is_ordered() const { return is_ord; }

      Set() : CTF_int::algstrct(sizeof(dtype)){
        tmdtype = CTF_int::get_default_mdtype<dtype>(is_custom_mdtype);
        set_abs_to_default();
        pair_sz = sizeof(std::pair<int64_t,dtype>);
      }

      void set_abs_to_default(){
        abs = &CTF_int::char_abs< dtype, CTF_int::default_abs<dtype, is_ord> >;
      }

      MPI_Datatype mdtype() const {
        return tmdtype;
      }

      void min(char const * a,
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = CTF_int::default_min<dtype,is_ord>(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void max(char const * a,
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = CTF_int::default_max<dtype,is_ord>(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void min(char * c) const {
        ((dtype*)c)[0] = CTF_int::default_min_lim<dtype,is_ord>();
      }

      void max(char * c) const {
        ((dtype*)c)[0] = CTF_int::default_max_lim<dtype,is_ord>();
      }

      void cast_double(double d, char * c) const {
        //((dtype*)c)[0] = (dtype)d;
        printf("CTF ERROR: double cast not possible for this algebraic structure\n");
        assert(0);
      }

      void cast_int(int64_t i, char * c) const {
        //((dtype*)c)[0] = (dtype)i;
        printf("CTF ERROR: integer cast not possible for this algebraic structure\n");
        assert(0);
      }

      double cast_to_double(char const * c) const {
        printf("CTF ERROR: double cast not possible for this algebraic structure\n");
        IASSERT(0);
        assert(0);
        return 0.0;
      }

      int64_t cast_to_int(char const * c) const {
        printf("CTF ERROR: int cast not possible for this algebraic structure\n");
        assert(0);
        return 0;
      }


      void print(char const * a, FILE * fp=stdout) const {
        for (int i=0; i<el_size; i++){
          fprintf(fp,"%x",a[i]);
        }
      }

      bool isequal(char const * a, char const * b) const {
        if (a == NULL && b == NULL) return true;
        if (a == NULL || b == NULL) return false;
        for (int i=0; i<el_size; i++){
          if (a[i] != b[i]) return false;
        }
        return true;
      }

      void coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_ja, int * csr_ia, char const * coo_vs, int const * coo_rs, int const * coo_cs) const {
        CTF_int::def_coo_to_csr(nz, nrow, (dtype *)csr_vs, csr_ja, csr_ia, (dtype const *) coo_vs, coo_rs, coo_cs);
      }

      void csr_to_coo(int64_t nz, int nrow, char const * csr_vs, int const * csr_ja, int const * csr_ia, char * coo_vs, int * coo_rs, int * coo_cs) const {
        CTF_int::def_csr_to_coo(nz, nrow, (dtype const *)csr_vs, csr_ja, csr_ia, (dtype*) coo_vs, coo_rs, coo_cs);
      }

      void coo_to_ccsr(int64_t nz, int64_t nnz_row, char * ccsr_vs, int * ccsr_ja, int * ccsr_ia, char const * coo_vs, int64_t const * coo_rs, int64_t const * coo_cs) const {
        CTF_int::def_coo_to_ccsr(nz, nnz_row, (dtype *)ccsr_vs, ccsr_ja, ccsr_ia, (dtype const *) coo_vs, coo_rs, coo_cs);
      }

      void ccsr_to_coo(int64_t nz, int64_t nnz_row, char const * csr_vs, int const * csr_ja, int const * csr_ia, int64_t const * row_enc, char * coo_vs, int64_t * coo_rs, int64_t * coo_cs) const {
        CTF_int::def_ccsr_to_coo(nz, nnz_row, (dtype const *)csr_vs, csr_ja, csr_ia, row_enc, (dtype*) coo_vs, coo_rs, coo_cs);
      }

      char * pair_alloc(int64_t n) const {
        //assert(sizeof(std::pair<int64_t,dtype>[n])==(uint64_t)(pair_size()*n));
        CTF_int::memprof_alloc_pre(n*sizeof(std::pair<int64_t,dtype>));
        char * ptr = (char*)(new std::pair<int64_t,dtype>[n]);
        CTF_int::memprof_alloc_post(n*sizeof(std::pair<int64_t,dtype>),(void**)&ptr);
        return ptr;
      }

      char * alloc(int64_t n) const {
        //assert(sizeof(dtype[n])==(uint64_t)(el_size*n));
        CTF_int::memprof_alloc_pre(n*sizeof(dtype));
        char * ptr = (char*)(new dtype[n]);
        CTF_int::memprof_alloc_post(n*sizeof(dtype),(void**)&ptr);
        return ptr;
      }

      void dealloc(char * ptr) const {
        CTF_int::memprof_dealloc(ptr);
        return delete [] (dtype*)ptr;
      }

      void pair_dealloc(char * ptr) const {
        CTF_int::memprof_dealloc(ptr);
        return delete [] (std::pair<int64_t,dtype>*)ptr;
      }


      void sort(int64_t n, char * pairs) const {
        std::sort((dtypePair<dtype>*)pairs,((dtypePair<dtype>*)pairs)+n);
      }

      void copy(char * a, char const * b) const {
        ((dtype *)a)[0] = ((dtype const *)b)[0];
      }

      void copy(char * a, char const * b, int64_t n) const {
        std::copy((dtype const *)b, ((dtype const *)b) + n, (dtype *)a);
      }

      void copy_pair(char * a, char const * b) const {
        ((std::pair<int64_t,dtype> *)a)[0] = ((std::pair<int64_t,dtype> const *)b)[0];
      }

      void copy_pairs(char * a, char const * b, int64_t n) const {
        std::copy((std::pair<int64_t,dtype> const *)b, ((std::pair<int64_t,dtype> const *)b) + n, (std::pair<int64_t,dtype> *)a);
        //std::copy((std::pair<int64_t,dtype> *)a, (std::pair<int64_t,dtype> const *)b, n);
        //for (int64_t i=0; i<n; i++){
          /*printf("i=%ld\n",i);
          this->print((char*)&(((std::pair<int64_t,dtype> const *)a)[i].second));
          this->print((char*)&(((std::pair<int64_t,dtype> const *)b)[i].second));*/
          //((std::pair<int64_t,dtype>*)a)[i] = ((std::pair<int64_t,dtype> const *)b)[i];
          //this->print((char*)&(((std::pair<int64_t,dtype> const *)a)[i].second));
        //}
      }

      void set(char * a, char const * b, int64_t n) const {
        if (n >= 100) {
#ifdef _OPENMP
          dtype *ia = (dtype*)a;
          dtype ib = *((dtype*)b);
          #pragma omp parallel
          {
            int64_t tid = omp_get_thread_num();
            int64_t chunksize = n / omp_get_num_threads();
            dtype *begin = ia + chunksize * tid;
            dtype *end;
            if (tid == omp_get_num_threads() - 1)
              end = ia + n;
            else
              end = begin + chunksize;
            std::fill(begin, end, ib);
          }
          return;
#endif
        }
        std::fill((dtype*)a, ((dtype*)a)+n, *((dtype*)b));
      }

      void set_pair(char * a, int64_t key, char const * b) const {
        ((std::pair<int64_t,dtype> *)a)[0] = std::pair<int64_t,dtype>(key,*((dtype*)b));
      }

      void set_pairs(char * a,  char const * b, int64_t n) const {
        std::fill((std::pair<int64_t,dtype> *)a, (std::pair<int64_t,dtype> *)a + n, *(std::pair<int64_t,dtype> const*)b);
      }

      void copy(int64_t n, char const * a, int inc_a, char * b, int inc_b) const {
        dtype const * da = (dtype const*)a;
        dtype * db = (dtype *)b;
        for (int64_t i=0; i<n; i++){
          db[inc_b*i] = da[inc_a*i];
        }
      }

      void copy(int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char *       b,
                int64_t      lda_b) const {

        dtype const * da = (dtype const*)a;
        dtype * db = (dtype *)b;
        for (int64_t j=0; j<n; j++){
          for (int64_t i=0; i<m; i++){
            db[j*lda_b+i] = da[j*lda_a+i];
          }
        }
      }

    void init(int64_t n, char * arr) const {
      dtype addid = dtype();
      set(arr, (char const *)&addid, n);
    }

    /** \brief initialize n objects to zero
      * \param[in] n number of items
      * \param[in] arr array containing n items, to be set to zero
      */
    virtual void init_shell(int64_t n, char * arr) const {
      dtype dummy = dtype();
      for (int i=0; i<n; i++){
        memcpy(arr+i*el_size,(char*)&dummy,el_size);
      }
    }

    CTF_int::bivar_function * get_elementwise_smaller() const {
      return new Bivar_Function<dtype,dtype,bool>([](dtype a, dtype b){ return !CTF_int::default_isequal<dtype>(CTF_int::default_max<dtype,is_ord>(a,b), a);});  
    }

    CTF_int::bivar_function * get_elementwise_smaller_or_equal() const {
      return new Bivar_Function<dtype,dtype,bool>([](dtype a, dtype b){ return CTF_int::default_isequal<dtype>(CTF_int::default_max<dtype,is_ord>(a,b), b);});  
    }

    CTF_int::bivar_function * get_elementwise_is_equal() const {
      return new Bivar_Function<dtype,dtype,bool>([](dtype a, dtype b){ return CTF_int::default_isequal<dtype>(a, b);});  
    }

    CTF_int::bivar_function * get_elementwise_is_not_equal() const {
      return new Bivar_Function<dtype,dtype,bool>([](dtype a, dtype b){ return !CTF_int::default_isequal<dtype>(a, b);});  
    }

/*
      void copy(int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char const * alpha,
                char *       b,
                int64_t      lda_b,
                char const * beta) const {

        dtype const * da = (dtype const*)a;
        dtype dalpha = *((dtype const*)alpha);
        dtype dbeta = *((dtype const*)beta);
        dtype * db = (dtype *)b;
        for (int64_t j=0; j<n; j++){
          for (int64_t i=0; i<m; i++){
            dbeta*db[j*lda_b+i] += dalpha*da[j*lda_a+i]
          }
        }
      }*/

  };

  //FIXME do below with macros to shorten

  template <>
  inline void Set<float>::cast_double(double d, char * c) const {
    ((float*)c)[0] = (float)d;
  }

  template <>
  inline void Set<double>::cast_double(double d, char * c) const {
    ((double*)c)[0] = d;
  }

  template <>
  inline void Set<long double>::cast_double(double d, char * c) const {
    ((long double*)c)[0] = (long double)d;
  }

  template <>
  inline void Set<int>::cast_double(double d, char * c) const {
    ((int*)c)[0] = (int)d;
  }

  template <>
  inline void Set<uint64_t>::cast_double(double d, char * c) const {
    ((uint64_t*)c)[0] = (uint64_t)d;
  }

  template <>
  inline void Set<int64_t>::cast_double(double d, char * c) const {
    ((int64_t*)c)[0] = (int64_t)d;
  }

  template <>
  inline void Set< std::complex<float>,false >::cast_double(double d, char * c) const {
    ((std::complex<float>*)c)[0] = (std::complex<float>)d;
  }

  template <>
  inline void Set< std::complex<double>,false >::cast_double(double d, char * c) const {
    ((std::complex<double>*)c)[0] = (std::complex<double>)d;
  }

  template <>
  inline void Set< std::complex<long double>,false >::cast_double(double d, char * c) const {
    ((std::complex<long double>*)c)[0] = (std::complex<long double>)d;
  }

  template <>
  inline void Set<float>::cast_int(int64_t d, char * c) const {
    ((float*)c)[0] = (float)d;
  }

  template <>
  inline void Set<double>::cast_int(int64_t d, char * c) const {
    ((double*)c)[0] = (double)d;
  }

  template <>
  inline void Set<long double>::cast_int(int64_t d, char * c) const {
    ((long double*)c)[0] = (long double)d;
  }

  template <>
  inline void Set<int>::cast_int(int64_t d, char * c) const {
    ((int*)c)[0] = (int)d;
  }

  template <>
  inline void Set<uint64_t>::cast_int(int64_t d, char * c) const {
    ((uint64_t*)c)[0] = (uint64_t)d;
  }

  template <>
  inline void Set<int64_t>::cast_int(int64_t d, char * c) const {
    ((int64_t*)c)[0] = (int64_t)d;
  }

  template <>
  inline void Set< std::complex<float>,false >::cast_int(int64_t d, char * c) const {
    ((std::complex<float>*)c)[0] = (std::complex<float>)d;
  }

  template <>
  inline void Set< std::complex<double>,false >::cast_int(int64_t d, char * c) const {
    ((std::complex<double>*)c)[0] = (std::complex<double>)d;
  }

  template <>
  inline void Set< std::complex<long double>,false >::cast_int(int64_t d, char * c) const {
    ((std::complex<long double>*)c)[0] = (std::complex<long double>)d;
  }

  template <>
  inline double Set<float>::cast_to_double(char const * c) const {
    return (double)(((float*)c)[0]);
  }

  template <>
  inline double Set<double>::cast_to_double(char const * c) const {
    return ((double*)c)[0];
  }

  template <>
  inline double Set<int>::cast_to_double(char const * c) const {
    return (double)(((int*)c)[0]);
  }

  template <>
  inline double Set<uint64_t>::cast_to_double(char const * c) const {
    return (double)(((uint64_t*)c)[0]);
  }

  template <>
  inline double Set<int64_t>::cast_to_double(char const * c) const {
    return (double)(((int64_t*)c)[0]);
  }


  template <>
  inline int64_t Set<int64_t>::cast_to_int(char const * c) const {
    return ((int64_t*)c)[0];
  }

  template <>
  inline int64_t Set<int>::cast_to_int(char const * c) const {
    return (int64_t)(((int*)c)[0]);
  }

  template <>
  inline int64_t Set<unsigned int>::cast_to_int(char const * c) const {
    return (int64_t)(((unsigned int*)c)[0]);
  }

  template <>
  inline int64_t Set<uint64_t>::cast_to_int(char const * c) const {
    return (int64_t)(((uint64_t*)c)[0]);
  }

  template <>
  inline int64_t Set<bool>::cast_to_int(char const * c) const {
    return (int64_t)(((bool*)c)[0]);
  }

  template <>
  inline void Set<float>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%11.5E",((float*)a)[0]);
  }

  template <>
  inline void Set<double>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%11.5E",((double*)a)[0]);
  }

  template <>
  inline void Set<int64_t>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%ld",((int64_t*)a)[0]);
  }

  template <>
  inline void Set<uint64_t>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%lu",((uint64_t*)a)[0]);
  }

  template <>
  inline void Set<uint32_t>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%u",((uint32_t*)a)[0]);
  }


  template <>
  inline void Set<int>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%d",((int*)a)[0]);
  }

  template <>
  inline void Set< std::complex<float>,false >::print(char const * a, FILE * fp) const {
    fprintf(fp,"(%11.5E,%11.5E)",((std::complex<float>*)a)[0].real(),((std::complex<float>*)a)[0].imag());
  }

  template <>
  inline void Set< std::complex<double>,false >::print(char const * a, FILE * fp) const {
    fprintf(fp,"(%11.5E,%11.5E)",((std::complex<double>*)a)[0].real(),((std::complex<double>*)a)[0].imag());
  }

  template <>
  inline void Set< std::complex<long double>,false >::print(char const * a, FILE * fp) const {
    fprintf(fp,"(%11.5LE,%11.5LE)",((std::complex<long double>*)a)[0].real(),((std::complex<long double>*)a)[0].imag());
  }

  template <>
  inline bool Set<float>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((float*)a)[0] == ((float*)b)[0];
  }

  template <>
  inline bool Set<double>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((double*)a)[0] == ((double*)b)[0];
  }

  template <>
  inline bool Set<int>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((int*)a)[0] == ((int*)b)[0];
  }

  template <>
  inline bool Set<uint64_t>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((uint64_t*)a)[0] == ((uint64_t*)b)[0];
  }

  template <>
  inline bool Set<int64_t>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((int64_t*)a)[0] == ((int64_t*)b)[0];
  }

  template <>
  inline bool Set<long double>::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return ((long double*)a)[0] == ((long double*)b)[0];
  }

  template <>
  inline bool Set< std::complex<float>,false >::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return (( std::complex<float> *)a)[0] == (( std::complex<float> *)b)[0];
  }

  template <>
  inline bool Set< std::complex<double>,false >::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return (( std::complex<double> *)a)[0] == (( std::complex<double> *)b)[0];
  }

  template <>
  inline bool Set< std::complex<long double>,false >::isequal(char const * a, char const * b) const {
    if (a == NULL && b == NULL) return true;
    if (a == NULL || b == NULL) return false;
    return (( std::complex<long double> *)a)[0] == (( std::complex<long double> *)b)[0];
  }

  /**
   * @}
   */
}
#include "monoid.h"
#endif

