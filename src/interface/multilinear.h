#ifndef __MULTILINEAR_H__
#define __MULTILINEAR_H__


namespace CTF {
  /**
   * \brief Compute updates to entries in tensor A based on matrices or vectors in mat_list (tensor times tensor products).
   *        Takes products of entries of A with multilinear dot products of columns of given matrices.
   *        For ndim=3 and mat_list=[X,Y,Z], this operation is equivalent to einsum("ijk,ia,ja,ka->ijk",A,X,Y,Z).
   *        FIXME: ignores semiring and just multiplies
   * \param[in] num_ops number of operands (matrices or vectors)
   * \param[in] modes modes on which to apply the operands
   * \param[in] mat_list where ith tensor is n_i-by-k or k-by-n_i matrix or vector of dim n_i where n_i is this->lens[mode[i]], should either all be vectors or be matices with same orientation
   * \param[in] aux_mode_first if true k-dim mode is first in all matrices in mat_list
   */
  template <typename dtype>
  void TTTP(Tensor<dtype> * T, int num_ops, int const * modes, Tensor<dtype> ** mat_list, bool aux_mode_first=false);

}

#include "multilinear.cxx"
#endif
