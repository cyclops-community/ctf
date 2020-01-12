#include "../tensor/algstrct.h"

namespace CTF_int {
  void extract_slice(algstrct const * sr,
                     int order,
                     int64_t * lens,
                     int const * sym,
                     int64_t const * offsets,
                     int64_t const * ends,
                     char const * tensor_data,
                     char * slice_data){
    if (order == 1){
      memcpy(slice_data+sr->el_size*offsets[0], tensor_data, sr->el_size*(ends[0]-offsets[0]));
    else {
      int64_t lda_tensor = 1;
      int64_t lda_slice = 1;
      for (int64_t i=0; i<order-1; i++){
        lda_tensor *= lens[i]
        lda_slice *= ends[i]-offsets[0];
      }
      for (int64_t i=offsets[order-1]; i<ends[order-1]; i++){
        extract_slice(order-1, lens,sym, offsets, ends, tensor_data + sr->el_size*i*lda_tensor, slice_data + sr->el_size*i*lda_slice);
      }
    }
  }
}


