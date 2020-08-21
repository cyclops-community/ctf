
#include "../include/ctf.hpp"
  
namespace CTF_int{


  /**
   * \brief initialize world instance (when CTF library loaded)
   */
  void init_global_world();

  /**
   * \brief delete global world instance (when MPI stopped)
   */
  void delete_global_world();

  /**
   * \brief (back-end for python) absolute value function
   * \param[in] A tensor, param[in,out] B tensor (becomes absolute value of A)
   * \return None
   */
  template <typename dtype>
  void abs_helper(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) pow function
   * \param[in] A tensor, param[in] B tensor, param[in,out] C tensor, param[in] index of A, param[in] index of B, param[in] index of C
   * \return None
   */
  template <typename dtype>
  void pow_helper(tensor * A, tensor * B, tensor * C, char const * idx_A, char const * idx_B, char const * idx_C);

  /**
   * \brief (back-end for python) function that computes floor of part
   * \param[in] A tensor, param[in,out] B tensor stores b_{...} = floor(a_{...})
   */
  template <typename dtype>
  void helper_floor(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that computes ceil of part
   * \param[in] A tensor, param[in,out] B tensor stores b_{...} = ceil(a_{...})
   */
  template <typename dtype>
  void helper_ceil(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that computes round of part
   * \param[in] A tensor, param[in,out] B tensor stores b_{...} = round(a_{...})
   */
  template <typename dtype>
  void helper_round(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) all function
   * \param[in] A tensor, param[in] B tensor with bool values created, param[in] index of A, param[in] index of B
   * \return None
   */
  template <typename dtype>
  void helper_clip(tensor * A, tensor * B, double low, double high);

  /**
   * \brief (back-end for python) function that clips the array
   * \param[in] A tensor, param[in,out] B tensor stores b_{...} = clamp(a_{...}, low, high)
   * \return None
   */
  template <typename dtype>
  void all_helper(tensor * A, tensor * B_bool, char const * idx_A, char const * idx_B);

  /**
   * \brief (back-end for python) conjugation function
   * \param[in] A tensor, param[in] B tensor with bool values created
   * \return None
   */
  template <typename dtype>
  void conj_helper(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that get the real part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the real part from tensor A
   * \return None
   */
  template <typename dtype>
  void get_real(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that get the imaginary part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the imaginary part from tensor A
   * \return None
   */
  template <typename dtype>
  void get_imag(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that set the real part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the real part from tensor A
   * \return None
   */
  template <typename dtype>
  void set_real(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) function that set the imaginary part from complex numbers
   * \param[in] A tensor, param[in] B tensor stores the imaginary part from tensor A
   * \return None
   */
  template <typename dtype>
  void set_imag(tensor * A, tensor * B);

  /**
   * \brief (back-end for python) any function
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
  
  void matrix_cholesky(tensor * A, tensor * L);
  void matrix_cholesky_cmplx(tensor * A, tensor * L);
  void matrix_trsm(tensor * L, tensor * B, tensor * X, bool lower, bool from_left, bool transp_L);
  void matrix_trsm_cmplx(tensor * L, tensor * B, tensor * X, bool lower, bool from_left, bool transp_L);
  void matrix_solve_spd(tensor * M, tensor * B, tensor * X);
  void matrix_solve_spd_cmplx(tensor * M, tensor * B, tensor * X);
  void matrix_qr(tensor * A, tensor * Q, tensor * R);
  void matrix_qr_cmplx(tensor * A, tensor * Q, tensor * R);
  void matrix_eigh(tensor * A, tensor * U, tensor * D);
  void matrix_eigh_cmplx(tensor * A, tensor * U, tensor * D);
  void matrix_svd(tensor * A, tensor * U, tensor * S, tensor * VT, int rank, double threshold);
  void matrix_svd_cmplx(tensor * A, tensor * U, tensor * S, tensor * VT, int rank, double threshold);
  void matrix_svd_rand(tensor * A, tensor * U, tensor * S, tensor * VT, int rank, int iter, int oversamp, tensor * U_init);
  void matrix_svd_rand_cmplx(tensor * A, tensor * U, tensor * S, tensor * VT, int rank, int iter, int oversamp, tensor * U_init);
  void matrix_svd_batch(tensor * A, tensor * U, tensor * S, tensor * VT, int rank);
  void matrix_svd_batch_cmplx(tensor * A, tensor * U, tensor * S, tensor * VT, int rank);
  void tensor_svd(tensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, tensor ** USVT);
  void tensor_svd_cmplx(tensor * dA, char * idx_A, char * idx_U, char * idx_VT, int rank, double threshold, bool use_svd_rand, int num_iter, int oversamp, tensor ** USVT);

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

  template <typename dtype>
  void vec_arange(tensor * t, dtype start, dtype stop, dtype step);
}
