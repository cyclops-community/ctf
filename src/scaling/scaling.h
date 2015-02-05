#ifndef __INT_SCALING_H__
#define __INT_SCALING_H__

#include "../interface/common.h"
#include "sym_seq_scl.h"

namespace CTF_int {
  class tensor; 

  /**
   * \brief class for execution distributed scaling of a tensor
   */
  class scaling {
    public:
      /** \brief operand/output */
      tensor * A;

      /** \brief scaling of A */
      char const * alpha;
    
      /** \brief indices of A */
      int const * idx_map;
      
      /** \brief whether there is a elementwise custom function */
      bool is_custom;

      /** \brief function to execute on elementwise elements */
      endomorphism func;

      /**
       * \brief constructor definining contraction with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_map indices of left operand
       * \param[in] alpha scaling factor alpha * A[idx_map];
                      A[idx_map] = alpha * A[idx_map]
       */
      scaling(tensor *     A,
              int const *  idx_map,
              char const * alpha);
     
      /**
       * \brief constructor definining scaling with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_map indices of left operand
                      func(&A[idx_map])
       */
      scaling(tensor *     A,
              int const *  idx_map,
              endomorphism func,
              char const * alpha=NULL);

      /** \brief run scaling  \return whether success or error */
      int execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();
    
  };

}

#endif
