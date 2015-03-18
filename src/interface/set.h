#ifndef __SET_H__
#define __SET_H__

#include "../tensor/algstrct.h"
//#include <stdint.h>
#include <inttypes.h>

namespace CTF {
  //C++14, nasty
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

  /**
   * Set class defined by a datatype and a min/max function (if it is partially ordered i.e. is_ord=true)
   * FIXME: currently assumes min and max are given by numeric_limits (custom min/max not allowed)
   */
  template <typename dtype=double, bool is_ord=true> 
  class Set : public CTF_int::algstrct {
    public:
      Set(Set const & other) : CTF_int::algstrct(other) {}

      virtual CTF_int::algstrct * clone() const {
        return new Set<dtype, is_ord>(*this);
      }

      Set() : CTF_int::algstrct(sizeof(dtype)){ }

      void min(char const * a, 
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = default_min<dtype,is_ord>(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void max(char const * a, 
               char const * b,
               char *       c) const {
        ((dtype*)c)[0] = default_max<dtype,is_ord>(((dtype*)a)[0],((dtype*)b)[0]);
      }

      void min(char * c) const {
        ((dtype*)c)[0] = std::numeric_limits<dtype>::min();
      }

      void max(char * c) const {
        ((dtype*)c)[0] = std::numeric_limits<dtype>::max();
      }

      void cast_double(double d, char * c) const {
        ((dtype*)c)[0] = (dtype)d;
      }

      void cast_int(int64_t i, char * c) const {
        ((dtype*)c)[0] = (dtype)i;
      }

      double cast_to_double(char const * c) const {
        printf("CTF ERROR: double cast not possible for this algebraic structure\n");
        ASSERT(0);
        return 0.0;
      }

      int64_t cast_to_int(char const * c) const {
        printf("CTF ERROR: int cast not possible for this algebraic structure\n");
        ASSERT(0);
        return 0;
      }


      void print(char const * a, FILE * fp=stdout) const {
        for (int i=0; i<el_size; i++){
          fprintf(fp,"%x",a[i]);
        }
      }
  };


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
    fprintf(fp,"%20.14E",((float*)a)[0]);
  }

  template <>  
  inline void Set<double>::print(char const * a, FILE * fp) const {
    fprintf(fp,"%20.14E",((double*)a)[0]);
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
    fprintf(fp,"(%20.14E,%20.14E)",((std::complex<float>*)a)[0].real(),((std::complex<float>*)a)[0].imag());
  }

  template <>  
  inline void Set< std::complex<double>,false >::print(char const * a, FILE * fp) const {
    fprintf(fp,"(%20.14E,%20.14E)",((std::complex<double>*)a)[0].real(),((std::complex<double>*)a)[0].imag());
  }


}
#include "monoid.h"
#endif
