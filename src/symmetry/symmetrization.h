#ifndef __INT_SYMMETRIZATION_H__
#define __INT_SYMMETRIZATION_H__

#include <assert.h>
#include "../tensor/untyped_tensor.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"

namespace CTF_int {
  /**
   * \brief unfolds the data of a tensor
   * \param[in] sym_tsr starting symmetric tensor (where data starts)
   * \param[in] nonsym_tsr new tensor with a potentially unfolded symmetry
   * \param[in] is_C whether the tensor is an output of the operation
   */
  void desymmetrize(tensor * sym_tsr,
                    tensor * nonsym_tsr,
                    bool     is_C);

  /**
   * \brief folds the data of a tensor
   * \param[in] sym_tsr starting symmetric tensor (where data ends)
   * \param[in] nonsym_tsr new tensor with a potentially unfolded symmetry
   */
  void symmetrize(tensor * sym_tsr,
                  tensor * nonsym_tsr);

  /**
   * \brief finds all permutations of a tensor according to a symmetry
   *
   * \param[in] ndim dimension of tensor
   * \param[in] sym symmetry specification of tensor
   * \param[out] nperm number of symmeitrc permutations to do
   * \param[out] perm the permutation
   * \param[out] sign sign of each permutation
   */
  void cmp_sym_perms(int         ndim,
                     int const * sym,
                     int *       nperm,
                     int **      perm,
                     double *    sign);

  /**
   * \brief orders the summation indices of one tensor 
   *        that don't break summation symmetries
   *
   * \param[in] A
   * \param[in] B
   * \param[in] idx_arr inverted summation index map
   * \param[in] off_A offset of A in inverted index map
   * \param[in] off_B offset of B in inverted index map
   * \param[in] idx_A index map of A
   * \param[in] idx_B index map of B
   * \param[in,out] add_sign sign of contraction
   * \param[in,out] mod 1 if sum is permuted
   */
  void order_perm(tensor const * A,
                  tensor const * B,
                  int *          idx_arr,
                  int            off_A,
                  int            off_B,
                  int *          idx_A,
                  int *          idx_B,
                  int &          add_sign,
                  int &          mod);

  /**
   * \brief orders the contraction indices of one tensor 
   *        that don't break contraction symmetries
   *
   * \param[in] A
   * \param[in] B
   * \param[in] C
   * \param[in] idx_arr inverted contraction index map
   * \param[in] off_A offset of A in inverted index map
   * \param[in] off_B offset of B in inverted index map
   * \param[in] off_C offset of C in inverted index map
   * \param[in] idx_A index map of A
   * \param[in] idx_B index map of B
   * \param[in] idx_C index map of C
   * \param[in,out] add_sign sign of contraction
   * \param[in,out] mod 1 if permutation done
   */
  void order_perm(tensor const * A,
                  tensor const * B,
                  tensor const * C,
                  int *          idx_arr,
                  int            off_A,
                  int            off_B,
                  int            off_C,
                  int *          idx_A,
                  int *          idx_B,
                  int *          idx_C,
                  int &          add_sign,
                  int &          mod);


  /**
   * \brief puts a summation map into a nice ordering according to preserved
   *        symmetries, and adds it if it is distinct
   *
   * \param[in,out] perms the permuted summation specifications
   * \param[in,out] signs sign of each summation
   * \param[in] new_perm summation signature
   * \param[in] new_sign alpha
   */
  void add_sym_perm(std::vector<summation>& perms,
                    std::vector<int>&       signs,
                    summation const &       new_perm,
                    int                     new_sign);

  /**
   * \brief puts a contraction map into a nice ordering according to preserved
   *        symmetries, and adds it if it is distinct
   *
   * \param[in,out] perms the permuted contraction specifications
   * \param[in,out] signs sign of each contraction
   * \param[in] new_perm contraction signature
   * \param[in] new_sign alpha
   */
  void add_sym_perm(std::vector<contraction>& perms,
                    std::vector<int>&         signs,
                    contraction const &       new_perm,
                    int                       new_sign);




  /**
   * \brief finds all permutations of a summation 
   *        that must be done for a broken symmetry
   *
   * \param[in] sum summation specification
   * \param[out] perms the permuted summation specifications
   * \param[out] signs sign of each summation
   */
  void get_sym_perms(summation const &       sum,
                     std::vector<summation>& perms,
                     std::vector<int>&       signs);

  /**
   * \brief finds all permutations of a contraction 
   *        that must be done for a broken symmetry
   *
   * \param[in] ctr contraction specification
   * \param[out] perms the permuted contraction specifications
   * \param[out] signs sign of each contraction
   */
  void get_sym_perms(contraction const &       ctr,
                     std::vector<contraction>& perms,
                     std::vector<int>&         signs);



}
#endif
