#ifndef __CTF_EXPRESSION_H__
#define __CTF_EXPRESSION_H__

#include "ctf_tensor.h"
#include <map>
#include <set>

/**
 * \defgroup expression Tensor expression compiler
 * @{
 */

template <typename dtype> class CTF_Term;
template <typename dtype> class CTF_Sum_Term;
template <typename dtype> class CTF_Contract_Term;

/**
 * \brief comparison function for sets of tensor pointers
 * This ensures the set iteration order is consistent across nodes
 */
template<typename dtype>
struct tensor_tid_less {
  bool operator()(CTF_Tensor<dtype>* A, CTF_Tensor<dtype>* B) {
    if (A == NULL && B != NULL) {
      return true;
    } else if (A == NULL || B == NULL) {
      return false;
    }
    return A->tid < B->tid;
  }
};


/**
 * \brief a tensor with an index map associated with it (necessary for overloaded operators)
 */
template<typename dtype=double>
class CTF_Idx_Tensor : public CTF_Term<dtype> {
  public:
    CTF_Tensor<dtype> * parent;
    char * idx_map;
    int is_intm;

  public:

  
    // dervied clone calls copy constructor
    CTF_Term<dtype> * clone(std::map< CTF_Tensor<dtype>*, CTF_Tensor<dtype>* >* remap = NULL) const;

    /**
     * \brief constructor takes in a parent tensor and its indices 
     * \param[in] parent_ the parent tensor
     * \param[in] idx_map_ the indices assigned ot this tensor
     * \param[in] copy if set to 1, create copy of parent
     */
    CTF_Idx_Tensor(CTF_Tensor<dtype>* parent_, 
                   const char *       idx_map_,
                   int                copy = 0);
    
    /**
     * \brief copy constructor
     * \param[in] B tensor to copy
     * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
     */
    CTF_Idx_Tensor(CTF_Idx_Tensor<dtype> const & B,
                   int copy = 0,
                   std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL);

    CTF_Idx_Tensor();
    
    CTF_Idx_Tensor(dtype val);
    
    ~CTF_Idx_Tensor();
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(CTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost of a contraction
     * \param[in] output tensor to write results into and its indices
     */
    int64_t  estimate_cost(CTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> estimate_cost(int64_t  & cost) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set< CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(CTF_Term<dtype> const & B);
    void operator=(CTF_Idx_Tensor<dtype> const & B);

    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator+=(CTF_Term<dtype> const & B);
    
    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator-=(CTF_Term<dtype> const & B);
    
    /**
     * \brief A -> A*B contract two tensors
     * \param[in] B tensor on the right hand side
     */
    void operator*=(CTF_Term<dtype> const & B);

    /**
     * \brief TODO A -> A * B^-1
     * \param[in] B
     */
    //void operator/(CTF_IdxTensor& tsr);
    
    /**
     * \brief execute ips into output with scale beta
     */    
    //void run(CTF_Idx_Tensor<dtype>* output, dtype  beta);

    /*operator CTF_Term<dtype>* (){
      CTF_Idx_Tensor * tsr = new CTF_Idx_Tensor(*this);
      return tsr;
    }*/
    /**
     * \brief figures out what world this term lives on
     */
    CTF_World * where_am_i() const;
};


/**
 * \brief a term is an abstract object representing some expression of tensors
 */
template<typename dtype=double>
class CTF_Term {
  public:
    dtype scale;
   
    CTF_Term();
    virtual ~CTF_Term(){};

    /**
     * \brief base classes must implement this copy function to retrieve pointer
     */ 
    virtual CTF_Term * clone(std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL) const = 0;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual void execute(CTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost of a contraction/sum/.. term
     * \param[in] output tensor to write results into and its indices
     */
    virtual int64_t  estimate_cost(CTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param\[in,out] cost the cost of the operatiob
     * \return output tensor to write results into and its indices
     */
    virtual CTF_Idx_Tensor<dtype> estimate_cost(int64_t  & cost) const = 0;
    
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual CTF_Idx_Tensor<dtype> execute() const = 0;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    virtual void get_inputs(std::set<CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const = 0;

    /**
     * \brief constructs a new term which multiplies by tensor A
     * \param[in] A term to multiply by
     */
    CTF_Contract_Term<dtype> operator*(CTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    CTF_Sum_Term<dtype> operator+(CTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    CTF_Sum_Term<dtype> operator-(CTF_Term<dtype> const & A) const;
    
    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(CTF_Term<dtype> const & B) { execute() = B; };
    void operator=(CTF_Idx_Tensor<dtype> const & B) { execute() = B; };
    void operator+=(CTF_Term<dtype> const & B) { execute() += B; };
    void operator-=(CTF_Term<dtype> const & B) { execute() -= B; };
    void operator*=(CTF_Term<dtype> const & B) { execute() *= B; };

    /**
     * \brief multiples by a constant
     * \param[in] scl scaling factor to multiply term by
     */
    CTF_Contract_Term<dtype> operator*(dtype scl) const;

    /**
     * \brief figures out what world this term lives on
     */
    virtual CTF_World * where_am_i() const = 0;

    /**
     * \brief casts into a double if dimension of evaluated expression is 0
     */
    operator dtype() const;
};

template<typename dtype=double>
class CTF_Sum_Term : public CTF_Term<dtype> {
  public:
    std::vector< CTF_Term<dtype>* > operands;

    // default constructor
    CTF_Sum_Term() : CTF_Term<dtype>() {}

    // destructor frees operands
    ~CTF_Sum_Term();
  
    // copy constructor
    CTF_Sum_Term(CTF_Sum_Term<dtype> const & other,
        std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL);

    // dervied clone calls copy constructor
    CTF_Term<dtype>* clone(std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * construct sum term corresponding to a single tensor
     * \param[in] output tensor to write results into and its indices
     */ 
    //CTF_Sum_Term<dtype>(CTF_Idx_Tensor<dtype> const & tsr);

    /**
     * \brief evalues the expression by summing operands into output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(CTF_Idx_Tensor<dtype> output) const;

  
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a sum term
     * \param[in] output tensor to write results into and its indices
     */
    int64_t  estimate_cost(CTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> estimate_cost(int64_t  & cost) const;
    
    
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    CTF_Sum_Term<dtype> operator+(CTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    CTF_Sum_Term<dtype> operator-(CTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    CTF_World * where_am_i() const;
};

template<typename dtype> static
CTF_Contract_Term<dtype> operator*(double d, CTF_Term<dtype> const & tsr){
  return (tsr*d);
}

/**
 * \brief An experession representing a contraction of a set of tensors contained in operands 
 */
template<typename dtype=double>
class CTF_Contract_Term : public CTF_Term<dtype> {
  public:
    std::vector< CTF_Term<dtype>* > operands;

    // \brief default constructor
    CTF_Contract_Term() : CTF_Term<dtype>() {}

    // \brief destructor frees operands
    ~CTF_Contract_Term();
  
    // \brief copy constructor
    CTF_Contract_Term(CTF_Contract_Term<dtype> const & other,
        std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL);

    // \brief dervied clone calls copy constructor
    CTF_Term<dtype> * clone(std::map<CTF_Tensor<dtype>*, CTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * \brief override execution to  to contract operands and add them to output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(CTF_Idx_Tensor<dtype> output) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<CTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a contract term
     * \param[in] output tensor to write results into and its indices
     */
    int64_t  estimate_cost(CTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    CTF_Idx_Tensor<dtype> estimate_cost(int64_t  & cost) const;
    
    
    /**
     * \brief override contraction to grow vector rather than create recursive terms
     * \param[in] A term to multiply by
     */
    CTF_Contract_Term<dtype> operator*(CTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    CTF_World * where_am_i() const;
};
/**
 * @}
 */

#include "ctf_idx_tensor.cxx"
#include "ctf_term.cxx"

#endif
