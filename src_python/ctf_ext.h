
#include "../include/ctf.hpp"
  
namespace CTF_int{


  /**
   * \python absolute value function
   * \param[in] A tensor, param[in,out] B tensor (becomes absolute value of A)
   * \return None
   */
  template <typename dtype>
  void abs_helper(tensor * A, tensor * B);

  /**
   * \python pow function
   * \param[in] A tensor, param[in] B tensor, param[in,out] C tensor, param[in] index of A, param[in] index of B, param[in] index of C
   * \return None
   */
  template <typename dtype>
  void pow_helper(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);

  /**
   * \python all function
   * \param[in] A tensor, param[in] B tensor with bool values created, param[in] index of A, param[in] index of B
   * \return None
   */
  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);
  
  template <typename dtype>
  void conj_helper(tensor * A, tensor * B);

  /**
   * \python function that get the real part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the real part from tensor A
   * \return None
   */
  template <typename dtype>
  void get_real(tensor * A, tensor * B);

  /**
   * \python function that get the imaginary part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the imaginary part from tensor A
   * \return None
   */
  template <typename dtype>
  void get_imag(tensor * A, tensor * B);

  /**
   * \python function that set the real part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the real part from tensor A
   * \return None
   */
  template <typename dtype>
  void set_real(tensor * A, tensor * B);

  /**
   * \python function that set the imaginary part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the imaginary part from tensor A
   * \return None
   */
  template <typename dtype>
  void set_imag(tensor * A, tensor * B);

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

  /**
   * \brief extract a sample of the entries (if sparse of the current nonzeros)
  * \param[in] A tensor to sample
  * \param[in] probability keep each entry with probability
  */
  void subsample(tensor * A, double probability);

  void matrix_svd(tensor * A, tensor * U, tensor * S, tensor * VT, int rank);
  void matrix_svd_cmplx(tensor * A, tensor * U, tensor * S, tensor * VT, int rank);
  
  void matrix_qr(tensor * A, tensor * Q, tensor * R);
  void matrix_qr_cmplx(tensor * A, tensor * Q, tensor * R);

  /**
   * \brief convert tensor from one type to another
   * \param[in] type_idx1 index of first ype
   * \param[in] type_idx2 index of second ype
   * \param[in] A tensor to convert
   * \param[in] B tensor to convert to
   */
  void conv_type(int type_idx1, int type_idx2, tensor * A, tensor * B);

  void delete_arr(tensor const * dt, char * arr);
  void delete_pairs(tensor const * dt, char * pairs);
}
