
#include "../include/ctf.hpp"
  
namespace CTF_int{
  /**
   * \python all function
   * \param[in] A tensor, param[in] B tensor with bool values created, param[in] index of A, param[in] index of B
   * \return None
   */
  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
  
  void conj_helper(tensor * A, tensor * B);

  /**
   * \python function that get the real number from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the real number from tensor A
   * \return None
   */
  template <typename dtype>
  void get_real(tensor * A, tensor * B);

  /**
   * \python function that get the imagine number from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the imagine number from tensor A
   * \return None
   */
  template <typename dtype>
  void get_imag(tensor * A, tensor * B);
  /**
   * \python any function
   * \param[in] A tensor, param[in] B tensor with bool values created, param[in] index of A, param[in] index of B
   * \return None
   */
  template <typename dtype>
  void any_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
  /**
   * \brief sum all 1 values in boolean tensor
   * \param[in] A tensor of boolean values
   * \return number of 1s in A
   */
  int64_t sum_bool_tsr(tensor * A);
}
