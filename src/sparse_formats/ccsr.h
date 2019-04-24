#ifndef __CCSR_H__
#define __CCSR_H__

#include "../tensor/algstrct.h"
#include "coo.h"
#include "sparse_matrix.h"

namespace CTF_int {

  class bivar_function;

  /**
   * \brief computes the size of a serialized CCSR matrix 
   * \param[in] nnz number of nonzeros in matrix
   * \param[in] nrow number of nonzero rows in matrix
   * \param[in] val_size size of each matrix entry
   */
  int64_t get_ccsr_size(int64_t nnz, int64_t nnz_row, int val_size);

  /**
   * \brief abstraction for a serialized sparse matrix stored in compressed-compressed-sparse-row (CCSR) layout, i.e. CSR where zero rows are also compressed
   */
  class CCSR_Matrix : public sparse_matrix {
    public:
      /** \brief constructor allocates all_data */
      CCSR_Matrix(int64_t nnz, int64_t nnz_row, int64_t nrow, int64_t ncol, accumulatable const * sr);

      /** \brief constructor given serialized CCSR matrix */
      CCSR_Matrix(char * all_data);
      
      CCSR_Matrix(){ all_data=NULL; }
      
      CCSR_Matrix(CCSR_Matrix const & other){ all_data=other.all_data; }
      
      /** \brief constructor given coordinate format (COO) matrix */
      CCSR_Matrix(tCOO_Matrix<int64_t> const & coom, int64_t nrow, int64_t ncol, algstrct const * sr, char * data=NULL, bool init_data=true);

      /** \brief retrieves number of nonzeros out of all_data */
      int64_t nnz() const;

      /** \brief retrieves buffer size out of all_data */
      int64_t size() const;

      /** \brief retrieves number of rows out of all_data */
      int64_t nrow() const;
      
      /** \brief retrieves number of columns out of all_data */
      int64_t ncol() const;
  
      /** \brief retrieves number of nonzero rows */
      int64_t nnz_row() const;
      
      /** \brief retrieves matrix entry size out of all_data */
      int val_size() const;

      /** \brief retrieves array of values out of all_data */
      char * vals() const;

      /** \brief retrieves prefix sum of number of nonzeros for each row (of size nrow()+1) out of all_data */
      int * IA() const;

      /** \brief retrieves column indices of each value in vals stored in sorted form by row */
      int * JA() const;

      /** \brief retrieves encoding of nonzero rows (each element of array gives true (local) index of row corresponding to each subsequent nonzero row)
      */
      int64_t * nnz_row_encoding() const;

      /**
       * \brief splits CCSR matrix into s submatrices (returned) corresponding to subsets of rows, all parts allocated in one contiguous buffer (passed back in parts_buffer)
       */
      void partition(int s, char ** parts_buffer, sparse_matrix ** parts);
      
      /**
       * \brief merges parts into one CCSR matrix, assuming they are split by partition() (Above)
       */
      void assemble(char * const * smnds, int s);
      
      /**
       * \brief outputs matrix data to stdout, intended for debugging
       * \param[in] sr algebraic structure allowing print
       */
      void print(algstrct const * sr);

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A is a CCSR_Matrix, while B and C are dense
       */
      static void ccsrmm(char const * A, algstrct const * sr_A, int64_t m, int64_t n, int64_t k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char *& C, algstrct const * sr_C, bivar_function const * func, bool do_offload);
      
      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A and B are CCSR_Matrices, while C is dense
       */
      //static void ccsrmultd(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char * C, algstrct const * sr_C, bivar_function const * func, bool do_offload);

      /**
       * \brief computes C = beta*C + func(alpha*A*B) where A, B, and C are CCSR_Matrices, while C is dense
       */
      //static void ccsrmultccsr(char const * A, algstrct const * sr_A, int m, int n, int k, char const * alpha, char const * B, algstrct const * sr_B, char const * beta, char *& C, algstrct const * sr_C, bivar_function const * func, bool do_offload);

      //static void compute_has_col(

      //                int const * JA,
      //                int const * IA,
      //                int const * JB,
      //                int const * IB,
      //                int         i,
      //                int *       has_col);
      
      static char * ccsr_add(char * cA, char * cB, accumulatable const * adder);
  };
}

#endif
