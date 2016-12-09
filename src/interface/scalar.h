#ifndef __SCALAR_H__
#define __SCALAR_H__
namespace CTF {

  /**
   * \addtogroup CTF
   * @{
   **/
  /**
   * \brief Scalar class which encapsulates a 0D tensor 
   */
  template <typename dtype=double>
  class Scalar : public Tensor<dtype> {
    public:
      /**
       * \brief constructor for a scalar
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Scalar(World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief constructor for a scalar with predefined value
       * \param[in] val scalar value
       * \param[in] wrld CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Scalar(dtype                     val,
             World &                   wrld=get_universe(),
             CTF_int::algstrct const & sr=Ring<dtype>());

      /**
       * \brief returns scalar value
       */
      dtype get_val();
      
      /**
       * \brief sets scalar value
       */
      void set_val(dtype val);

      /**
       * \brief casts into a dtype value
       */
      operator dtype() { return get_val(); }

      Scalar<dtype> & operator=(const Scalar<dtype> & A);

  };

  /**
   * @}
   */
}
#include "scalar.cxx"
#endif
