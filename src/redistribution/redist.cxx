#include "redist.h"

namespace CTF_int {
  int can_block_reshuffle(int         order,
                          int const *      old_phase,
                          mapping const *  map){
    int new_phase, j;
    int can_block_resh = 1;
    for (j=0; j<order; j++){
      new_phase  = calc_phase(map+j);
      if (new_phase != old_phase[j]) can_block_resh = 0;
    }
    return can_block_resh;
  }


}
