#ifndef __INT_SUMMATION_H__
#define __INT_SUMMATION_H__

#include "assert.h"

namespace CTF_int {
  class tensor; 

  /**
   * \brief untyped internal class for doubly-typed univariate function
   */
  class univar_function {
    public:
      /**
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to type by extending class
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      virtual void apply_f(char const * a, char * b) { assert(0); }
  };


  /**
   * \brief class for execution distributed summation of tensors
   */
  class summation {
     public:
      /** \brief left operand */
      tensor * A;
      /** \brief output */
      tensor * B;

      /** \brief scaling of A */
      char const * alpha;
      /** \biref scaling of existing B */
      char const * beta;
    
      /** \brief indices of left operand */
      int const * idx_A;
      /** \brief indices of output */
      int const * idx_B;

      /**
       * \brief constructor definining summation with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] alpha scaling factor alpha * A[idx_A];
       * \param[in] B ouput operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] beta scaling factor of ouput 
                      C[idx_B] = beta*B[idx_B] + alpha * A[idx_A]
       */
      summation(tensor * A, 
                int const * idx_A,
                char const * alpha, 
                tensor * B, 
                int const * idx_B,
                char const * beta);
     
      /**
       * \brief constructor definining summation with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B ouput operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] func custom elementwise function 
                      func(A[idx_A],&B[idx_B])
       */
      summation(tensor * A, 
                int const * idx_A,
                tensor * B, 
                int const * idx_B,
                univar_function func);

      /** \brief run summation  */
      void execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();
   
  };
}

#endif
