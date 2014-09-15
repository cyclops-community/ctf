
#ifndef __INT_FUNCTIONS_H__
#define __INT_FUNCTIONS_H__

#include "assert.h"

/**
 * \brief untyped internal class for singly-typed single variable function (Endomorphism)
 */
class Int_Endomorphism {
  public:
    /**
     * \brief apply function f to value stored at a
     * \param[in,out] a pointer to operand that will be cast to type by extending class
     *                  return result of applying f on value at a
     */
    virtual void apply_f(char * a) { assert(0); }
};

/**
 * \brief untyped internal class for doubly-typed univariate function
 */
class Int_Univar_Function {
  public:
    /**
     * \brief apply function f to value stored at a
     * \param[in] a pointer to operand that will be cast to type by extending class
     * \param[in,out] result &f(*a) of applying f on value of (different type) on a
     */
    virtual void apply_f(char const * a, char * b) { assert(0); }
};

/**
 * \brief untyped internal class for triply-typed bivariate function
 */
class Int_Bivar_Function {
  public:
    /**
     * \brief apply function f to values stored at a and b
     * \param[in] a pointer to first operand that will be cast to type by extending class
     * \param[in] b pointer to second operand that will be cast to type by extending class
     * \param[in,out] result: c=&f(*a,*b) 
     */
    virtual void apply_f(char const * a, char const * b, char * c) { assert(0); }
};

#endif
