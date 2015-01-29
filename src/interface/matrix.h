#ifndef __MATRIX_H__
#define __MATRIX_H__

namespace CTF {

  /**
   * \brief Matrix class which encapsulates a 2D tensor 
   */
  template<typename dtype=double> 
  class Matrix : public Tensor<dtype> {
    public:
      int nrow, ncol, sym;

      /**
       * \brief constructor for a matrix
       * \param[in] nrow number of matrix rows
       * \param[in] ncol number of matrix columns
       * \param[in] sym symmetry of matrix
       * \param[in] world CTF world where the tensor will live
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Matrix(int          nrow, 
             int          ncol, 
             int          sym,
             World      * wrld,
             char const * name = NULL,
             int          profile = 0);

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
      Matrix(int             nrow, 
             int             ncol, 
             int             sym,
             World      *    wrld,
             Semiring<dtype> sr,
             char const *    name = NULL,
             int             profile = 0);


  };
}
#include "matrix.cxx"
#endif
