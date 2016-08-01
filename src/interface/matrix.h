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
      int nrow, ncol, symm;


      /** 
       * \brief default constructor for a matrix
       * \param[in] A matrix to copy along with its data
       */
      Matrix<dtype>();


      /** 
       * \brief copy constructor for a matrix
       * \param[in] A matrix to copy along with its data
       */
      Matrix<dtype>(Matrix<dtype> const & A);

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
       * \param[in] atr quantifier for sparsity and symmetry of matrix
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       atr=0,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] atr quantifier for sparsity and symmetry of matrix
       * \param[in] world CTF world where the tensor will live
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             int                       atr,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] idx assignment of characters to each dim
       * \param[in] prl mesh processor topology with character labels
       * \param[in] blk local blocking with processor labels
       * \param[in] atr quantifier for sparsity and symmetry of matrix
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int                       nrow,
             int                       ncol,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             int                       atr=0,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);


//      Matrix<dtype> & operator=(const Matrix<dtype> & A);
  };
  /**
   * @}
   */
}
#include "matrix.cxx"
#endif
