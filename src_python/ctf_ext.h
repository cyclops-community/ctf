
#include "../include/ctf.hpp"
  
namespace CTF_int{

  template <typename dtype>
  void all(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);


  /**
   * \brief sum all 1 values in boolean tensor
   * \param[in] A tensor of boolean values
   * \return number of 1s in A
   */
  int64_t sum_bool_tsr(tensor * A);
}
