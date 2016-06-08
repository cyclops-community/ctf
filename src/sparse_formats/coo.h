#ifndef __COO_H__
#define __COO_H__

#include "../tensor/algstrct.h"

namespace CTF_int {

  class bivar_function;

  int64_t get_coo_size(int64_t nnz, int val_size);

  class COO_Matrix{
    public:
      char * all_data;
      
      COO_Matrix(int64_t nnz, algstrct const * sr);

      COO_Matrix(char * all_data);

      int64_t nnz() const;

      int val_size() const;

      int64_t size() const;

      char * vals() const;

      int * rows() const;

      int * cols() const;

      void set_data(int64_t nz, int order, int const * lens, int const * ordering, int nrow_idx, char const * tsr_data, algstrct const * sr, int const * phase);

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A is a COO_Matrix, while B and C are dense
       */
      static void coomm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func);

  };
}

#endif
