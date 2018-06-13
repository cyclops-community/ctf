#include "../../include/ctf.hpp"


namespace CTF_int {
  CTF::Ring<float> float_ring = CTF::Ring<float>();
  CTF_int::algstrct const * get_float_ring(){
    return &float_ring;
  }
  CTF::Ring<double> double_ring = CTF::Ring<double>();
  CTF_int::algstrct const * get_double_ring(){
    return &double_ring;
  }
  CTF::Ring<int> int_ring = CTF::Ring<int>();
  CTF_int::algstrct const * get_int_ring(){
    return &int_ring;
  }
  CTF::Ring<int64_t> int64_t_ring = CTF::Ring<int64_t>();
  CTF_int::algstrct const * get_int64_t_ring(){
    return &int64_t_ring;
  }
}


