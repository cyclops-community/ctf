#ifndef __CSR_H__
#define __CSR_H__

#include "../tensor/algstrct.h"
#include "coo.h"

namespace CTF_int {

  class bivar_function;

  /**
   * \brief computes the size of a serialized CSR matrix 
   * \param[in] nnz number of nonzeros in matrix
   * \param[in] nrow number of rows in matrix
   * \param[in] val_size size of each matrix entry
   */
  int64_t get_csr_size(int64_t nnz, int nrow, int val_size);

  /**
   * \brief abstraction for a serialized sparse matrix stored in column-sparse-row (CSR) layout
   */
  class CSR_Matrix{
    public:
      /** \brief serialized buffer containing all info, index, and values related to matrix */
      char * all_data;
      
      /** \brief constructor allocates all_data */
      CSR_Matrix(int64_t nnz, int nrow, int ncol, algstrct const * sr);

      /** \brief constructor given serialized CSR matrix */
      CSR_Matrix(char * all_data);
      
      CSR_Matrix(){ all_data=NULL; }
      
      /** \brief constructor given coordinate format (COO) matrix */
      CSR_Matrix(COO_Matrix const & coom, int nrow, int ncol, algstrct const * sr, char * data=NULL);

      /** \brief retrieves number of nonzeros out of all_data */
      int64_t nnz() const;

      /** \brief retrieves buffer size out of all_data */
      int64_t size() const;

      /** \brief retrieves number of rows out of all_data */
      int nrow() const;
      
      /** \brief retrieves number of columns out of all_data */
      int ncol() const;
      
      /** \brief retrieves matrix entry size out of all_data */
      int val_size() const;

      /** \brief retrieves array of values out of all_data */
      char * vals() const;

      /** \brief retrieves prefix sum of number of nonzeros for each row (of size nrow()+1) out of all_data */
      int * rows() const;

      /** \brief retrieves column indices of each value in vals stored in sorted form by row */
      int * cols() const;

//      void set_data(int64_t nz, int order, int const * lens, int const * ordering, int nrow_idx, char const * tsr_data, algstrct const * sr, int const * phase);

      /**
       * \brief splits CSR matrix into s submatrices (returned) corresponding to subsets of rows, all parts allocated in one contiguous buffer (passed back in parts_buffer)
       */
      CSR_Matrix * partition(int s, char ** parts_buffer);
      
      /**
       * \brief constructor merges parts into one CSR matrix, assuming they are split by partition() (Above)
       */
      CSR_Matrix(char * const * smnds, int s);

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A is a CSR_Matrix, while B and C are dense
       */
      static void csrmm(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload);
      
      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A and B are CSR_Matrices, while C is dense
       */
      static void csrmultd(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload);


  };
}

#endif
