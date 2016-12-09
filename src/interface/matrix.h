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
       */
      Matrix();


      /** 
       * \brief copy constructor for a matrix
       * \param[in] A matrix to copy along with its data
       */
      Matrix(Matrix<dtype> const & A);


      /** 
       * \brief casts a tensor to a matrix
       * \param[in] A tensor object of order 2
       */
      Matrix(Tensor<dtype> const & A);

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
       * \param[in] wrld CTF world where the tensor will live
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
       * \param[in] wrld CTF world where the tensor will live
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
       * \brief constructor for a matrix with a given initial cyclic distribution 
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

      /**
       * \brief writes a nonsymmetric matrix from a block-cyclic initial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] pc number of cols in processor grid
       * \param[in] rsrc processor row holding first block row (0-based unlike ScaLAPACK)
       * \param[in] csrc processor col holding first block row (0-based unlike ScaLAPACK)
       * \param[in] lda leading dimension (length of buffer corresponding to row)
       * \param[out] data locally stored values
       */
      void write_mat(int           mb,
                     int           nb,
                     int           pr,
                     int           pc,
                     int           rsrc,
                     int           csrc,
                     int           lda,
                     dtype const * data);


 
      /**
       * \brief constructor for a nonsymmetric matrix with a block-cyclic initial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] pc number of cols in processor grid
       * \param[in] rsrc processor row holding first block row (0-based unlike ScaLAPACK)
       * \param[in] csrc processor col holding first block row (0-based unlike ScaLAPACK)
       * \param[in] lda leading dimension (length of buffer corresponding to row)
       * \param[in] data locally stored values
       * \param[in] wrld CTF world where the tensor will live, must contain pr*pc processors
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Matrix(int                       nrow,
             int                       ncol,
             int                       mb,
             int                       nb,
             int                       pr,
             int                       pc,
             int                       rsrc,
             int                       csrc,
             int                       lda,
             dtype const *             data,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);
     

      /**
       * \brief construct Matrix from ScaLAPACK array descriptor
       *        `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] desc ScaLAPACK descriptor array:
       *                 see ScaLAPACK docs for "Array Descriptor for In-core Dense Matrices"
       * \param[in] data locally stored values
       * \param[in] wrld CTF world where the tensor will live, must contain pr*pc processors
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Matrix(int const *               desc,
             dtype const *             data,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief reads a nonsymmetric matrix into a block-cyclic initial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] pc number of cols in processor grid
       * \param[in] rsrc processor row holding first block row (0-based unlike ScaLAPACK)
       * \param[in] csrc processor col holding first block row (0-based unlike ScaLAPACK)
       * \param[in] lda leading dimension (length of buffer corresponding to row)
       * \param[out] data locally stored values
       */
      void read_mat(int     mb,
                    int     nb,
                    int     pr,
                    int     pc,
                    int     rsrc,
                    int     csrc,
                    int     lda,
                    dtype * data);

 
      /**
       * \brief read Matrix into ScaLAPACK array descriptor
       *        `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] desc ScaLAPACK descriptor array:
       *                 see ScaLAPACK docs for "Array Descriptor for In-core Dense Matrices"
       * \param[in] data locally stored values
       */
      void read_mat(int const * desc,
                    dtype *     data);

      /*
       * \brief prints matrix by row and column (modify print(...) overload in set.h if you would like a different print format)
       */
      void print_matrix();
  };
  /**
   * @}
   */
}
#include "matrix.cxx"
#endif
