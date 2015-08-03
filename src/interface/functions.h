#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "../scaling/scaling.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"


namespace CTF {

/**
 * @defgroup CTF_func CTF functions
 * \brief user-defined function interface
 * @addtogroup CTF_func
 * @{
 */
  class Idx_Tensor;

  /**
   * \brief custom scalar function on tensor: e.g. A["ij"] = f(A["ij"])
   */
  template<typename dtype=double>
  class Endomorphism : public CTF_int::endomorphism {
    public:
      /**
       * \brief function signature for element-wise operation a=f(a)
       */
      dtype (*f)(dtype);
     
      /**
       * \brief constructor takes function pointer
       * \param[in] f_ scalar function: (type) -> (type)
       */
      Endomorphism(dtype (*f_)(dtype)){ f = f_; }

      /**
       * \brief default constructor
       */
      Endomorphism(){}

      /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to dtype
       *                  is set to result of applying f on value at a
       */
      void apply_f(char * a) const { ((dtype*)a)[0]=f(((dtype*)a)[0]); }
  };

  /**
   * \brief custom function f : X -> Y to be applied to tensor elemetns: 
   *          e.g. B["ij"] = f(A["ij"])
   */
  template<typename dtype_B=double, typename dtype_A=dtype_B>
  class Univar_Function : public CTF_int::univar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute b=f(a)
       */
      dtype_B (*f)(dtype_A);
      
      /**
       * \brief constructor takes function pointers to compute B=f(A));
       * \param[in] f_ linear function (type_A)->(type_B)
       */
      Univar_Function(dtype_B (*f_)(dtype_A)){ f = f_; }

      /** 
       * \brief evaluate B=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * return f(A) output tensor 
       */
      //Idx_Tensor operator()(Idx_Tensor const  & A);
      
      /**
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char * b) const { ((dtype_B*)b)[0]=f(((dtype_A*)a)[0]); }
      
      /**
       * \brief compute b = b+f(a)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       * \param[in] sr_B algebraic structure for b, needed to do add
       */
      void acc_f(char const * a, char * b, CTF_int::algstrct const * sr_B) const {
        dtype_B tb=f(((dtype_A*)a)[0]); 
        sr_B->add(b, (char const *)&tb, b);
      }

  };


  /**
   * \brief custom function f : (X * Y) -> X applied on two tensors as summation: 
   *          e.g. B["ij"] = f(A["ij"],B["ij"])
   */
  template<typename dtype_B=double, typename dtype_A=dtype_B>
  class Univar_Accumulator : public CTF_int::univar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute b=f(a)
       */
      void (*f)(dtype_A, dtype_B &);
      
      /**
       * \brief constructor takes function pointers to compute B=f(A));
       * \param[in] f_ linear function (type_A)->(type_B)
       */
      Univar_Accumulator(void (*f_)(dtype_A, dtype_B&)){ f = f_; }

      /** 
       * \brief evaluate B=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * return f(A) output tensor 
       */
      //Idx_Tensor operator()(Idx_Tensor const  & A);
      
      /**
       * \brief apply function f to value stored at a, for an accumulator, this is the same as acc_f below
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char * b) const { acc_f(a,b,NULL); }

       /**
       * \brief compute f(a,b)
       * \param[in] a pointer to the accumulated operand 
       * \param[in,out] value that is accumulated to
       * \param[in] sr_B algebraic structure for b, here is ignored
       */
      void acc_f(char const * a, char * b, CTF_int::algstrct const * sr_B) const {
        f(((dtype_A*)a)[0], ((dtype_B*)b)[0]);
      }

      bool is_accumulator() const { return true; }
  };


  /**
   * \brief custom bilinear function on two tensors: 
   *          e.g. C["ij"] = f(A["ik"],B["kj"])
   */
  template<typename dtype_C=double, typename dtype_A=dtype_C, typename dtype_B=dtype_C>
  class Bivar_Function : public CTF_int::bivar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute C=f(A,B)
       */
      dtype_C (*f)(dtype_A, dtype_B);
     
      /**
       * \brief constructor takes function pointers to compute C=f(A,B);
       * \param[in] f_ bilinear function (type_A,type_B)->(type_C)
       */
      Bivar_Function(dtype_C (*f_)(dtype_A, dtype_B)){ f=f_; }

      /**
       * \brief default constructor sets function pointer to NULL
       */
      Bivar_Function();

      /** 
       * \brief evaluate C=f(A,B) 
       * \param[in] A left operand tensor with pre-defined indices 
       * \param[in] B right operand tensor with pre-defined indices
       * \return C output tensor
      */
      //Idx_Tensor operator()(Idx_Tensor const  & A, 
      //                      Idx_Tensor const  & B);
      
      /**
       * \brief apply function f to values stored at a and b
       * \param[in] a pointer to first operand that will be cast to dtype 
       * \param[in] b pointer to second operand that will be cast to dtype 
       * \param[in,out] result: c=&f(*a,*b) 
       */
      void apply_f(char const * a, char const * b, char * c) const { 
        ((dtype_C*)c)[0]=f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]); 
      }
  };

/**
 * @}
 */
}

#endif

