#ifndef __CSR_H__
#define __CSR_H__

#include "../tensor/algstrct.h"
#include "coo.h"

namespace CTF_int {

  class bivar_function;

  int64_t get_csr_size(int64_t nnz, int nrow, int val_size);

  class CSR_Matrix{
    public:
      char * all_data;
      
      CSR_Matrix(int64_t nnz, int nrow, int ncol, algstrct const * sr);

      CSR_Matrix(char * all_data);
      
      CSR_Matrix(COO_Matrix const & coom, int nrow, int ncol, algstrct const * sr, char * data=NULL);

      int64_t nnz() const;

      int64_t size() const;

      int nrow() const;
      
      int ncol() const;
      
      int val_size() const;

      char * vals() const;

      int * rows() const;

      int * cols() const;

//      void set_data(int64_t nz, int order, int const * lens, int const * ordering, int nrow_idx, char const * tsr_data, algstrct const * sr, int const * phase);

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A is this CSR_Matrix, while B and C are dense
       */
      void csrmm(algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload);

  };
}

#endif
