#ifndef __COO_H__
#define __COO_H__

#include "../tensor/algstrct.h"

namespace CTF_int {

  class Bivar_Function;

  int64_t get_coo_size(int64_t nnz, int val_size);

  class COO_Matrix{
    public:
      char * all_data;

      COO_Matrix(int64_t nnz, algstrct const * sr);

      COO_Matrix(char * all_data);

      int64_t nnz() const;

      int64_t size() const;

      char * vals() const;

      int * rows() const;

      int * cols() const;

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A is this COO_Matrix, while B and C are dense
       */
      void coomm(algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, Bivar_Function const * func);

  };
}

#endif
