#ifndef __SCALAR_H__
#define __SCALAR_H__
namespace CTF {

  /**
   * \brief Scalar class which encapsulates a 0D tensor 
   */
  template<typename dtype=double> 
  class Scalar : public Tensor<dtype> {
    public:
      /**
       * \brief constructor for a scalar
       * \param[in] world CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */
      Scalar(World &            wrld,
             Set<dtype> const & sr=Ring<dtype>());

      /**
       * \brief constructor for a scalar with predefined value
       * \param[in] val scalar value
       * \param[in] world CTF world where the tensor will live
       * \param[in] sr defines the tensor arithmetic for this tensor
       */ 
      Scalar(dtype              val,
             World &            wrld,
             Set<dtype> const & sr=Ring<dtype>());

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


  };

}
#include "scalar.cxx"
#endif
