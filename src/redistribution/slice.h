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
}
#endif

