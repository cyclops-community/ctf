#ifndef __VECTOR_H__
#define __VECTOR_H__

namespace CTF {

  /**
   * \addtogroup CTF
   * @{
   **/
  /**
   * \brief Vector class which encapsulates a 1D tensor 
   */
  template <typename dtype=double>
  class Vector : public Tensor<dtype> {
    public:
      int len;
      /** 
       * \brief default constructor for a vector
       */
      Vector();

      /** 
       * \brief copy constructor for a matrix
       * \param[in] A matrix to copy along with its data
       */
      Vector<dtype>(Vector<dtype> const & A);

      /** 
       * \brief casts a tensor to a matrix
       * \param[in] A tensor object of order 1
       */
      Vector<dtype>(Tensor<dtype> const & A);

      /**
       * \brief constructor for a vector
       * \param[in] len dimension of vector
       * \param[in] world CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Vector(int                       len,
             World &                   world,
             CTF_int::algstrct const & sr);

      /**
       * \brief constructor for a vector
       * \param[in] len dimension of vector
       * \param[in] world CTF world where the tensor will live
       * \param[in] name an optionary name for the tensor
       * \param[in] profile set to 1 to profile contractions involving this tensor
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Vector(int                       len,
             World &                   world=get_universe(),
             char const *              name=NULL,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief constructor for a vector
       * \param[in] len dimension of vector
       * \param[in] atr quantifier for sparsity and symmetry of matrix (0 -> dense, >0 -> sparse)
       * \param[in] world CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Vector(int                       len,
             int                       atr,
             World &                   world=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>());


      /**
       * \brief constructor for a vector
       * \param[in] len dimension of vector
       * \param[in] atr quantifier for sparsity and symmetry of matrix (0 -> dense, >0 -> sparse)
       * \param[in] world CTF world where the tensor will live
       */ 
      Vector(int                       len,
             int                       atr,
             World &                   world,
             char const *              name,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      //Vector<dtype> & operator=(const Vector<dtype> & A);
  /**
   * @}
   */
  };
}
#include "vector.cxx"
#endif
