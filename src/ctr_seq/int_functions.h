
#ifndef __INT_fUNCTIONS_H__
#define __INT_fUNCTIONS_H__

#include "assert.h"

namespace CTF_int {

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

}

#endif
