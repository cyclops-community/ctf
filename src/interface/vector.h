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
       * \brief constructor for a vector
       * \param[in] len_ dimension of vector
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] sr_ defines the tensor arithmetic for this tensor
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Vector(int                       len,
             World &                   world,
             CTF_int::algstrct const & sr);

      /**
       * \brief constructor for a vector
       * \param[in] len_ dimension of vector
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Vector(int                       len,
             World &                   world=get_universe(),
             char const *              name=NULL,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief constructor for a vector
       * \param[in] is_sparse whether vector should be treated as a sparse (tensor)
       * \param[in] len_ dimension of vector
       * \param[in] world_ CTF world where the tensor will live
       * \param[in] name_ an optionary name for the tensor
       * \param[in] profile_ set to 1 to profile contractions involving this tensor
       */ 
      Vector(bool                      is_sparse,
             int                       len,
             World &                   world=get_universe(),
             char const *              name=NULL,
             int                       profile=0,
             CTF_int::algstrct const & sr=Ring<dtype>());



      Vector<dtype> & operator=(const Vector<dtype> & A);
  /**
   * @}
   */
  };
}
#include "vector.cxx"
#endif
