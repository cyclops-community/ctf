#ifndef __CTF_FUNCTIONS_H__
#define __CTF_FUNCTIONS_H__

template <typename dtype> class CTF_Idx_Tensor;

/**
 * \brief custom scalar function on tensor: e.g. A["ij"] = f(A["ij"])
 */
template<typename dtype=double>
class CTF_Fscalar  {
  public:
    /**
     * \brief function signature for element-wise operation a=f(a)
     */
    dtype (*f)(dtype);
   
    /**
     * \brief constructor takes function pointer
     * \param[in] f scalar function: (type) -> (type)
     */
    CTF_Fscalar(dtype (*f)(dtype));

    /** 
     * \brief evaluate A=f(A) 
     * \param[in] A operand tensor with pre-defined indices 
     * \return f(A)
    */
    CTF_Idx_Tensor<dtype> operator()(CTF_Idx_Tensor<dtype> const & A);
};

/**
 * \brief custom linear function on two tensors: 
 *          e.g. B["ij"] = f(A["ij"])
 */
template<typename dtype_B=double, typename dtype_A=dtype_B>
class CTF_Flinear {
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
    CTF_Flinear(dtype_B (*f)(dtype_A));

    /** 
     * \brief evaluate B=f(A) 
     * \param[in] A operand tensor with pre-defined indices 
     * return f(A) output tensor 
     */
    CTF_Idx_Tensor<dtype_B> operator()(CTF_Idx_Tensor<dtype_A> const  & A);

};

/**
 * \brief custom bilinear function on two tensors: 
 *          e.g. C["ij"] = f(A["ik"],B["kj"])
 */
template<typename dtype_C=double, typename dtype_A=dtype_C, typename dtype_B=dtype_C>
class CTF_Fbilinear {
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
    CTF_Fbilinear(dtype_C (*f)(dtype_A, dtype_B));

    /**
     * \brief default constructor sets function pointer to NULL
     */
    CTF_Fbilinear();

    /** 
     * \brief evaluate C=f(A,B) 
     * \param[in] A left operand tensor with pre-defined indices 
     * \param[in] B right operand tensor with pre-defined indices
     * \return C output tensor
    */
    CTF_Idx_Tensor<dtype_C> operator()(CTF_Idx_Tensor<dtype_A> const  & A, 
                                       CTF_Idx_Tensor<dtype_B> const  & B);
};


#endif

