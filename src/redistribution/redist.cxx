/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "redist.h"

namespace CTF_int {
  int can_block_reshuffle(int         order,
                          int const *      old_phase,
                          mapping const *  map){
    int new_phase, j;
    int can_block_resh = 1;
    for (j=0; j<order; j++){
      new_phase  = map[j].calc_phase();
      if (new_phase != old_phase[j]) can_block_resh = 0;
    }
    return can_block_resh;
  }


}
