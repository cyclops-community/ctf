#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "../scaling/scaling.h"
#include "../summation/summation.h"
#include "../contraction/contraction.h"

namespace CTF {

  template <typename dtype> class Idx_Tensor;

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
       * \param[in] f scalar function: (type) -> (type)
       */
      Endomorphism(dtype (*f)(dtype));

      /** 
       * \brief evaluate A=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * \return f(A)
      */
      Idx_Tensor<dtype> operator()(Idx_Tensor<dtype> const & A);

      /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to dtype
       *                  is set to result of applying f on value at a
       */
      void apply_f(char * a){ return ((dtype*)a)[0]=f(((dtype*)a)[0]); };
  };

  /**
   * \brief custom linear function on two tensors: 
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
       * \param[in] f linear function (type_A)->(type_B)
       * \param[in] fadd associative addition function (type_B,type_B)->(type_B)
       */
      Univar_Function(dtype_B (*f)(dtype_A));

      /** 
       * \brief evaluate B=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * return f(A) output tensor 
       */
      Idx_Tensor<dtype_B> operator()(Idx_Tensor<dtype_A> const  & A);
      
      /**
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char * b) { ((dtype_B*)b)[0]=f(((dtype_A*)a)[0]); }

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
       * \param[in] f bilinear function (type_A,type_B)->(type_C)
       * \param[in] fadd associative addition function (type_B,type_B)->(type_B)
       */
      Bivar_Function(dtype_C (*f)(dtype_A, dtype_B));

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
      Idx_Tensor<dtype_C> operator()(Idx_Tensor<dtype_A> const  & A, 
                                     Idx_Tensor<dtype_B> const  & B);
      
      /**
       * \brief apply function f to values stored at a and b
       * \param[in] a pointer to first operand that will be cast to dtype 
       * \param[in] b pointer to second operand that will be cast to dtype 
       * \param[in,out] result: c=&f(*a,*b) 
       */
      void apply_f(char const * a, char const * b, char * c){ 
        ((dtype_C*)c)[0]=f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]); 
      }
  };

}

#endif

