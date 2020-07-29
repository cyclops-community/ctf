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
	//template<typename dtype>
	//class Vector : public Tensor<dtype>;

  template<typename dtype=double>
  class Matrix : public Tensor<dtype> {
    public:
      int64_t nrow, ncol;
      int symm;
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
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
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
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
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
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
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
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
             int                       atr,
             World &                   wrld,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      /**
       * \brief constructor for a matrix with a given guessial cyclic distribution
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
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
             char const *              idx,
             Idx_Partition const &     prl,
             Idx_Partition const &     blk=Idx_Partition(),
             int                       atr=0,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief writes a nonsymmetric matrix from a block-cyclic guessial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
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
                     char          layout_order,
                     int           rsrc,
                     int           csrc,
                     int           lda,
                     dtype const * data);



      /**
       * \brief constructor for a nonsymmetric matrix with a block-cyclic guessial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] pc number of cols in processor grid
       * \param[in] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
       * \param[in] rsrc processor row holding first block row (0-based unlike ScaLAPACK)
       * \param[in] csrc processor col holding first block row (0-based unlike ScaLAPACK)
       * \param[in] lda leading dimension (length of buffer corresponding to row)
       * \param[in] data locally stored values
       * \param[in] wrld CTF world where the tensor will live, must contain pr*pc processors
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Matrix(int64_t                   nrow,
             int64_t                   ncol,
             int                       mb,
             int                       nb,
             int                       pr,
             int                       pc,
             char                      layout_order,
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
       * \param[in] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
       * \param[in] wrld CTF world where the tensor will live, must contain pr*pc processors
       * \param[in] sr defines the tensor arithmetic for this tensor
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       */
      Matrix(int const *               desc,
             dtype const *             data,
             char                      layout_order='C',
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>(),
             char const *              name=NULL,
             int                       profile=0);

      /**
       * \brief reads a nonsymmetric matrix into a block-cyclic guessial distribution
       *        this is `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] mb row block dimension
       * \param[in] nb col block dimension
       * \param[in] pr number of rows in processor grid
       * \param[in] pc number of cols in processor grid
       * \param[in] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
       * \param[in] rsrc processor row holding first block row (0-based unlike ScaLAPACK)
       * \param[in] csrc processor col holding first block row (0-based unlike ScaLAPACK)
       * \param[in] lda leading dimension (length of buffer corresponding to row)
       * \param[out] data locally stored values
       */
      void read_mat(int     mb,
                    int     nb,
                    int     pr,
                    int     pc,
                    char    layout_order,
                    int     rsrc,
                    int     csrc,
                    int     lda,
                    dtype * data);


      /**
       * \brief get a ScaLAPACK descriptor for this Matrix, will always be in pure cyclic layout
       * \param[out] ictxt index of newly created context
       * \param[out] desc array of integers of size 9, which will be filled with attributes
       * \param[out] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
       *                 see ScaLAPACK docs for "Array Descriptor for In-core Dense Matrices"
       */
      void get_desc(int & ictxt, int *& desc, char & layout_order);


      /**
       * \brief read Matrix into ScaLAPACK array descriptor
       *        `cheap' if mb=nb=1, nrow%pr=0, ncol%pc=0, rsrc=0, csrc=0, but is done via sparse read/write otherwise
       *        assumes processor grid is row-major (otherwise transpose matrix)
       * \param[in] desc ScaLAPACK descriptor array:
       *                 see ScaLAPACK docs for "Array Descriptor for In-core Dense Matrices"
       * \param[in] data locally stored values
       * \param[out] layout_order 'R' if processor grid is row major 'C' if processor grid is column major
       */
      void read_mat(int const * desc,
                    dtype *     data,
                    char        layout_order='C');

      /*
       * \brief prints matrix by row and column (modify print(...) overload in set.h if you would like a different print format)
       */
      void print_matrix();

      /*
       * \brief extracts upper or lower triangular (trapezoidal) portion of matrix
       * \param[in,out] T lower or upper triangular or trapezoidal matrix, can be of dimensions m-by-n where m=nrow and n=ncol if m>=n and lower=true or m<=n and lower=false or m=n=min(nrow,ncol)
       * \param[in] lower if true take lower triangular part
       * \param[in] keep_diag if true keep diagonal
       */
      void get_tri(Matrix<dtype> & T, bool lower=false, bool keep_diag=true);

      /*
       * \calculates the Cholesky decomposition, assuming this matrix is SPD
       * \param[out] L n-by-n lower-triangular matrix
       * \param[in] lower if true L is lower triangular of false, upper
       */
      void cholesky(Matrix<dtype> & L, bool lower=true);

      /*
       * \calculates Solves symmetric positive definite systems of equations
       * \param[in] M n-by-n symmetric/Hermitian positive definite matrix
       * \param[out] X m-by-k matrix of right hand sides, of same dimensions as this matrix
       * \param[in] from_left if true solve MX=this with m=n and k=nrhs and if false solve XM=this with k=n and m=nrhs
       */
      void solve_spd(Matrix<dtype> & A, Matrix<dtype> & X);

      /*
       * \calculates triangular solve with many right-hand sides, this matrix is the right or left-hand-side
       * \param[in] L n-by-n lower-triangular matrix
       * \param[out] X solution(s) to triangular solve, same shape as this
       * \param[in] lower if true L is lower triangular of false, upper
       * \param[in] from_left if true, solve LX=this if false solve XL=this
       * \param[in] transp_L if true, L solve L^TX=this or XL^T=this
       */
      void solve_tri(Matrix<dtype> & L, Matrix<dtype> & X, bool lower=true, bool from_left=true, bool transp_L=false);

      /*
       * \calculates the reduced QR decomposition, A = Q x R for A of dimensions m by n with m>=n
       * \param[out] Q m-by-n matrix with orthonormal columns
       * \param[out] R n-by-n upper-triangular matrix
       */
      void qr(Matrix<dtype> & Q, Matrix<dtype> & R);

      /*
       * \calculates the singular value decomposition, M = U x S x VT, of matrix using pdgesvd from ScaLAPACK
       * \param[out] U left singular vectors of matrix
       * \param[out] S singular values of matrix
       * \param[out] VT right singular vectors of matrix
       * \param[in] rank rank of output matrices. If rank = 0, will use min(matrix.rows, matrix.columns)
       * \param[in] threshold for truncating singular values of the SVD, determines rank, if threshold ia also used, rank will be set to minimum of rank and number of singular values above threshold
       */
      void svd(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT, int rank = 0, double threshold=0.);

      /*
       * \calculates uses randomized method (orthogonal iteration) to calculate a low-rank singular value decomposition, M = U x S x VT. Is faster, especially for low-rank, but less robust than typical svd.
       * \param[out] U left singular vectors of matrix
       * \param[out] S singular values of matrix
       * \param[out] VT right singular vectors of matrix
       * \param[in] rank rank of output matrices. If rank = 0, will use min(matrix.rows, matrix.columns)
       * \param[in] iter number of orthogonal iterations to perform (higher gives better accuracy)
       * \param[in] oversamp oversampling parameter
       * \param[in,out] U_guess guessial guess for first rank+oversamp singular vectors (matrix with orthogonal columns is also good), on output is final iterate (with oversamp more columns than U)
       */
      void svd_rand(Matrix<dtype> & U, Vector<dtype> & S, Matrix<dtype> & VT, int rank, int iter=1, int oversamp=5, Matrix<dtype> * U_guess=NULL);

      /**
       * \brief calculate symmetric or Hermitian eigensolve, must be called on square and symmetric or Hermitian matrix
       * \param[out] U will be defined as a matrix of eigenvectors of the same dimensions as this matrix
       * \param[out] D will be a vector of eigenvalues of the same dimension/type as this matrix (if this matrix is complex, D will be complex, but each value will have a zero complex part)
       */
      void eigh(Matrix<dtype> & U, Vector<dtype> & D);

      /**
       * \brief transforms matrix to a batch of distributed columns distributed over appropriate subworlds
       * \return vector of distributed vectors containing each column of this matrix as it is distributed in matrix
       */
      std::vector<CTF::Vector<dtype>*> to_vector_batch();
 
  };
  /**
   * @}
   */
}
#include "matrix.cxx"
#endif

