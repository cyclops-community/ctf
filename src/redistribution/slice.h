#ifndef __SLICE_H__
#define __SLICE_H__

#include "../tensor/algstrct.h"

namespace CTF_int {
  void extract_slice(algstrct const * sr,
                     int order,
                     int64_t * lens,
                     int const * sym,
                     int64_t const * offsets,
                     int64_t const * ends,
                     char const * tensor_data,
                     char * slice_data);

      /**
       * \brief accumulates out a slice (block) of this tensor = B
       *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
       * \param[in] offsets_B bottom left corner of block
       * \param[in] ends_B top right corner of block
       * \param[in] beta scaling factor of this tensor
       * \param[in] A tensor who owns pure-operand slice
       * \param[in] offsets_A bottom left corner of block of A
       * \param[in] ends_A top right corner of block of A
       * \param[in] alpha scaling factor of tensor A
       */
  void push_slice(tensor *        B,
                  int64_t const * offsets_B,
                  int64_t const * ends_B,
                  char const *    beta,
                  tensor *        AA,
                  int64_t const * offsets_A,
                  int64_t const * ends_A,
                  char const *    alpha);
}
#endif

