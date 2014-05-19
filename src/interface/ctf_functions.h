#ifndef __CTF_FUNCTIONS_H__
#define __CTF_FUNCTIONS_H__

/**
 * \brief custom scalar function on tensor: e.g. A["ij"] = f(A["ij"])
 */
template<typename dtype>
class tCTF_Fscalar  {
  public:
    /**
     * \brief function signature for element-wise operation a=f(a)
     */
    dtype (*f)(dtype);
   
    /**
     * \brief constructor takes function pointer
     * \param[in] f scalar function: (type) -> (type)
     */
    tCTF_Fscalar(dtype (*f)(dtype));

    /** 
     * \brief evaluate A=f(A) 
     * \param[in] A operand tensor with pre-defined indices 
    */
    operator()(tCTF_IdxTensor<dtype_A> & A);
};

/**
 * \brief custom linear function on two tensors: 
 *          e.g. B["ij"] = fadd(B["ij"],flin(A["ij"]))
 */
template<typename dtype_A, template dtype_B>
class tCTF_Flinear {
  public:
    /**
     * \brief function signature for element-wise multiplication, compute b=f(a)
     */
    dtype_B (*f)(dtype_A);
    
    /**
     * \brief function signature for element-wise summation, compute b=add(b_1,b_2)
     *        must be associative
     */
    dtype_B (*fadd)(dtype_B, dtype_B);

    /**
     * \brief constructor takes function pointers to compute B=fadd(B,f(A,B));
     * \param[in] f linear function (type_A)->(type_B)
     * \param[in] fadd associative addition function (type_B,type_B)->(type_B)
     */
    tCTF_Flinear(dtype_B (*flin)(dtype_A),
                 dtype_B (*fadd)(dtype_B, dtype_B));

    /** 
     * \brief evaluate B=fadd(B,flin(A)) 
     * \param[in] A operand tensor with pre-defined indices 
     * \param[in] B output tensor with pre-defined indices and scaling factor
    */
    operator()(tCTF_IdxTensor<dtype_A> const  & A, 
               tCTF_IdxTensor<dtype_B>        & B);

};

/**
 * \brief custom bilinear function on two tensors: 
 *          e.g. C["ij"] = fadd(C["ij"],f(A["ik"],B["kj"]))
 */
template<typename dtype_A, typename dtype_B, typename dtype_C>
class tCTF_Fbilinear {
  public:
    /**
     * \brief function signature for element-wise multiplication, compute C=f(A,B)
     */
    dtype_C (*f)(dtype_A, dtype_B);
    
    /**
     * \brief function signature for element-wise summation, compute C=add(C_1,C_2)
     */
    dtype_C (*fadd)(dtype_C, dtype_C);

    /**
     * \brief constructor takes function pointers to compute C=fadd(C,f(A,B));
     * \param[in] f bilinear function (type_A,type_B)->(type_C)
     * \param[in] fadd associative addition function (type_B,type_B)->(type_B)
     */
    tCTF_Fbilinear(dtype_C (*f)(dtype_A, dtype_B),
                   dtype_C (*fadd)(dtype_C, dtype_C));

    /** 
     * \brief evaluate C=fadd(C,f(A,B)) 
     * \param[in] A left operand tensor with pre-defined indices 
     * \param[in] B right operand tensor with pre-defined indices
     * \param[in] C output tensor with pre-defined indices and scaling factor
    */
    operator()(tCTF_IdxTensor<dtype_A> const  & A, 
               tCTF_IdxTensor<dtype_B> const  & B,
               tCTF_IdxTensor<dtype_C>        & C);
};


#endif

