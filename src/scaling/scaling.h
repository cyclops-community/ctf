#ifndef __INT_SCALING_H__
#define __INT_SCALING_H__


namespace CTF_int {
  class tensor; 

  /**
   * \brief untyped internal class for singly-typed single variable function (Endomorphism)
   */
  class endomorphism {
    public:
      /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to type by extending class
       *                  return result of applying f on value at a
       */
      virtual void apply_f(char * a) { assert(0); }
  };

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
      int const * idx_A;

      /**
       * \brief constructor definining contraction with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] alpha scaling factor alpha * A[idx_A];
                      A[idx_A] = alpha * A[idx_A]
       */
      scaling(tensor * A, 
              int const * idx_A,
              char const * alpha);
     
      /**
       * \brief constructor definining scaling with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
                      func(&A[idx_A])
       */
      scaling(tensor * A, 
              int const * idx_A,
              endomorphism func);

      /** \brief run scaling  */
      void execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();
    
  };

}

#endif
