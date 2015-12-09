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
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             World &                   wrld,
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] world CTF world where the tensor will live
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] qtf quantifier for qtfmetry or sparsity of matrix
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       qtf=0,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] qtf qtfmetry of matrix
       * \param[in] world CTF world where the tensor will live
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       qtf,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());


      Matrix<dtype> & operator=(const Matrix<dtype> & A);
  };
  /**
   * @}
   */
}
#include "matrix.cxx"
#endif
