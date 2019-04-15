#ifndef __SPARSE_MATRIX_H__
#define __SPARSE_MATRIX_H__

#include "../tensor/algstrct.h"
#include "coo.h"

namespace CTF_int {
  class sparse_matrix {
    public:
      /** \brief serialized buffer containing all info, index, and values related to matrix */
      char * all_data;

      sparse_matrix(){ all_data = NULL; }
      virtual ~sparse_matrix(){ }

      /** \brief retrieves buffer size out of all_data */
      virtual int64_t size() const;
  
      /**
       * \brief constructor merges parts into one CSR matrix, assuming they are split by partition() (Above)
       */
      virtual void assemble(char * const * smnds, int s);
 
       /*
        * \brief splits CSR matrix into s submatrices (returned) corresponding to subsets of rows, all parts allocated in one contiguous buffer (passed back in parts_buffer)
       */
      virtual void partition(int s, char ** parts_buffer, sparse_matrix ** parts);

  };      
}

#endif
