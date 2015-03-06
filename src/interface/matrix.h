#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace CTF {

  /**
   * \brief Matrix class which encapsulates a 2D tensor 
   * \param[in] dtype specifies tensor element type
   * \param[in] is_ord specifies whether these can be ordered (i.e. operator '<' must be defined)
   */
  template<typename dtype=double, bool is_ord=true> 
  class Matrix : public Tensor<dtype, is_ord> {
    public:
      int nrow, ncol, sym;
      /**
       * \brief constructor for a matrix
       * \param[in] nrow_ number of matrix rows
       * \param[in] ncol_ number of matrix columns
       * \param[in] sym_ symmetry of matrix
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] sr_ defines the tensor arithmetic for this tensor
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       sym,
             World &                   wrld,
             Set<dtype,is_ord> const & sr=Ring<dtype,is_ord>());

      /**
       * \brief constructor for a matrix
       * \param[in] nrow_ number of matrix rows
       * \param[in] ncol_ number of matrix columns
       * \param[in] sym_ symmetry of matrix
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       * \param[in] sr_ defines the tensor arithmetic for this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       sym,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             Set<dtype,is_ord> const & sr=Ring<dtype,is_ord>());


      Matrix<dtype,is_ord> & operator=(const Matrix<dtype,is_ord> & A);
  };
}
#include "matrix.cxx"
#endif
