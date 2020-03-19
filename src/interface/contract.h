#ifndef __CONTRACT_H__
#define __CONTRACT_H__

namespace CTF {
 
  template <typename dtype=double> 
  class Contract {
    public:
      CTF_int::contraction *pctr;

      Contract();
      
      ~Contract();
      
      Contract(dtype                 alpha,
               CTF_int::tensor &     A,
               char const *          idx_A,
               CTF_int::tensor &     B,
               char const *          idx_B,
               dtype                 beta,
               CTF_int::tensor &     C,
               char const *          idx_C);

      Contract(dtype                 alpha,
               CTF_int::tensor &     A,
               char const *          idx_A,
               CTF_int::tensor &     B,
               char const *          idx_B,
               dtype                 beta,
               CTF_int::tensor &     C,
               char const *          idx_C,
               Bivar_Function<dtype> func);
      
      void prepareA(CTF_int::tensor& A,
                    const char *     idx_A);

      void prepareB(CTF_int::tensor& B,
                    const char *     idx_B);
      
      void prepareC(CTF_int::tensor& C,
                    const char *     idx_C);
      
      void execute();
      
      void releaseA();
      
      void releaseB();
      
      void releaseC();
  };
}

#include "contract.cxx"

#endif
