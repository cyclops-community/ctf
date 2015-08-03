#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace CTF {
  /**
   * \addtogroup CTF
   * @{
   */

  /**
   * \brief Matrix class which encapsulates a 2D tensor 
   * \param[in] dtype specifies tensor element type
   */
  template<typename dtype=double> 
  class Matrix : public Tensor<dtype> {
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
             CTF_int::algstrct const & sr=Ring<dtype>());

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
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief constructor for a matrix
       * \param[in] is_sparse whether matrix should be treated as a sparse (tensor)
       * \param[in] nrow_ number of matrix rows
       * \param[in] ncol_ number of matrix columns
       * \param[in] sym_ symmetry of matrix
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       * \param[in] sr_ defines the tensor arithmetic for this tensor
       */ 
      Matrix(bool                      is_sparse,
             int                       nrow,
             int                       ncol,
             int                       sym,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief constructor for a matrix
       * \param[in] is_sparse whether matrix should be treated as a sparse (tensor)
       * \param[in] nrow_ number of matrix rows
       * \param[in] ncol_ number of matrix columns
       * \param[in] sym_ symmetry of matrix
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] sr_ defines the tensor arithmetic for this tensor
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Matrix(bool                      is_sparse,
             int                       nrow,
             int                       ncol,
             int                       sym,
             World &                   wrld,
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);




      Matrix<dtype> & operator=(const Matrix<dtype> & A);
  };
  /**
   * @}
   */
}
#include "matrix.cxx"
#endif
