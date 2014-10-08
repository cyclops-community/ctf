#ifndef __INT_CONTRACTION_H__
#define __INT_CONTRACTION_H__

#include "assert.h"
#include "ctr_comm.h"

namespace CTF_int {

  /**
   * \brief untyped internal class for triply-typed bivariate function
   */
  class bivar_function {
    public:
      /**
       * \brief apply function f to values stored at a and b
       * \param[in] a pointer to first operand that will be cast to type by extending class
       * \param[in] b pointer to second operand that will be cast to type by extending class
       * \param[in,out] result: c=&f(*a,*b) 
       */
      virtual void apply_f(char const * a, char const * b, char * c) { assert(0); }
  };

  /**
   * \brief class for execution distributed contraction of tensors
   */
  class contraction {
    public:
      /** \brief left operand */
      tensor * A;
      /** \brief right operand */
      tensor * B;
      /** \brief output */
      tensor * C;

      /** \brief scaling of A*B */
      char const * alpha;
      /** \biref scaling of existing C */
      char const * beta;
    
      /** \brief indices of left operand */
      int const * idx_A;
      /** \brief indices of right operand */
      int const * idx_B;
      /** \brief indices of output */
      int const * idx_C;

      /**
       * \brief constructor definining contraction with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] alpha scaling factor alpha * A[idx_A] * B[idx_B];
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] beta scaling factor of ouput 
                      C[idx_C] = beta*C[idx_C] 
                                + alpha * A[idx_A] * B[idx_B];
       */
      contraction(tensor * A, 
                  int const * idx_A,
                  tensor * B, 
                  int const * idx_B,
                  char const * alpha, 
                  tensor * C, 
                  int const * idx_C,
                  char const * beta);
     
      /**
       * \brief constructor definining contraction with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] func custom elementwise function 
                      func(A[idx_A],B[idx_B],&C[idx_C])
       */
      contraction(tensor * A, 
                  int const * idx_A,
                  tensor * B, 
                  int const * idx_B,
                  tensor * C, 
                  int const * idx_C,
                  bivar_function func);

      /** \brief run contraction */
      void execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();
  };

}

#endif
