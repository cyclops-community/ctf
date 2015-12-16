#ifndef __SET_H__
#define __SET_H__

#include "../tensor/algstrct.h"
//#include <stdint.h>
#include <limits>
#include <inttypes.h>

namespace CTF_int {

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
  template <>
  inline MPI_Datatype get_default_mdtype<char>(bool & is_custom){ is_custom=false; return MPI_CHAR; }
  template <>
  inline MPI_Datatype get_default_mdtype<bool>(bool & is_custom){ is_custom=false; return MPI_C_BOOL; }
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
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<double> >(bool & is_custom){ is_custom=false; return MPI_DOUBLE_COMPLEX; }
  template <>
  inline MPI_Datatype get_default_mdtype< std::complex<long double> >(bool & is_custom){ is_custom=false; return MPI::LONG_DOUBLE_COMPLEX; }

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

}


namespace CTF {
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
        abs = other.abs;
      }

      virtual CTF_int::algstrct * clone() const {
        return new Set<dtype, is_ord>(*this);
      }

      bool is_ordered() const { return is_ord; }

      Set() : CTF_int::algstrct(sizeof(dtype)){ 
        tmdtype = CTF_int::get_default_mdtype<dtype>(is_custom_mdtype);
        set_abs_to_default();
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
        ((dtype*)c)[0] = std::numeric_limits<dtype>::min();
      }

      void max(char * c) const {
        ((dtype*)c)[0] = std::numeric_limits<dtype>::max();
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
