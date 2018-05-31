#include "../../include/ctf.hpp"


namespace CTF_int {
  CTF::Ring<double> double_ring = CTF::Ring<double>();
  CTF_int::algstrct const * get_double_ring(){
    return &double_ring;
  }
  CTF::Ring<int64_t> int64_t_ring = CTF::Ring<int64_t>();
  CTF_int::algstrct const * get_int64_t_ring(){
    return &int64_t_ring;
  }
}


