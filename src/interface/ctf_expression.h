#ifndef __CTF_EXPRESSION_H__
#define __CTF_EXPRESSION_H__


/**
 * \defgroup expression Tensor expression compiler
 * @{
 */


/**
 * \brief a tensor with an index map associated with it (necessary for overloaded operators)
 */
template<typename dtype>
class tCTF_Idx_Tensor : public tCTF_Term<dtype> {
  public:
    tCTF_Tensor<dtype> * parent;
    char * idx_map;
    int is_intm;

  public:

  
    // dervied clone calls copy constructor
    tCTF_Term<dtype> * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * \brief constructor takes in a parent tensor and its indices 
     * \param[in] parent_ the parent tensor
     * \param[in] idx_map_ the indices assigned ot this tensor
     * \param[in] copy if set to 1, create copy of parent
     */
    tCTF_Idx_Tensor(tCTF_Tensor<dtype>* parent_, 
                    const char *        idx_map_,
                    int                 copy = 0);
    
    /**
     * \brief copy constructor
     * \param[in] B tensor to copy
     * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
     */
    tCTF_Idx_Tensor(tCTF_Idx_Tensor<dtype> const & B,
                    int copy = 0,
                    std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    tCTF_Idx_Tensor();
    
    tCTF_Idx_Tensor(dtype val);
    
    ~tCTF_Idx_Tensor();
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost of a contraction
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(tCTF_Term<dtype> const & B);
    void operator=(tCTF_Idx_Tensor<dtype> const & B);

    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator+=(tCTF_Term<dtype> const & B);
    
    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator-=(tCTF_Term<dtype> const & B);
    
    /**
     * \brief A -> A*B contract two tensors
     * \param[in] B tensor on the right hand side
     */
    void operator*=(tCTF_Term<dtype> const & B);

    /**
     * \brief TODO A -> A * B^-1
     * \param[in] B
     */
    //void operator/(tCTF_IdxTensor& tsr);
    
    /**
     * \brief execute ips into output with scale beta
     */    
    //void run(tCTF_Idx_Tensor<dtype>* output, dtype  beta);

    /*operator tCTF_Term<dtype>* (){
      tCTF_Idx_Tensor * tsr = new tCTF_Idx_Tensor(*this);
      return tsr;
    }*/
    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};


/**
 * \brief a term is an abstract object representing some expression of tensors
 */
template<typename dtype>
class tCTF_Term {
  public:
    dtype scale;
   
    tCTF_Term();
    virtual ~tCTF_Term(){};

    /**
     * \brief base classes must implement this copy function to retrieve pointer
     */ 
    virtual tCTF_Term * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const = 0;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual void execute(tCTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost of a contraction/sum/.. term
     * \param[in] output tensor to write results into and its indices
     */
    virtual long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param\[in,out] cost the cost of the operatiob
     * \return output tensor to write results into and its indices
     */
    virtual tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const = 0;
    
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual tCTF_Idx_Tensor<dtype> execute() const = 0;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    virtual void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const = 0;

    /**
     * \brief constructs a new term which multiplies by tensor A
     * \param[in] A term to multiply by
     */
    tCTF_Contract_Term<dtype> operator*(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    tCTF_Sum_Term<dtype> operator+(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    tCTF_Sum_Term<dtype> operator-(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(tCTF_Term<dtype> const & B) { execute() = B; };
    void operator=(tCTF_Idx_Tensor<dtype> const & B) { execute() = B; };
    void operator+=(tCTF_Term<dtype> const & B) { execute() += B; };
    void operator-=(tCTF_Term<dtype> const & B) { execute() -= B; };
    void operator*=(tCTF_Term<dtype> const & B) { execute() *= B; };

    /**
     * \brief multiples by a constant
     * \param[in] scl scaling factor to multiply term by
     */
    tCTF_Contract_Term<dtype> operator*(dtype scl) const;

    /**
     * \brief figures out what world this term lives on
     */
    virtual tCTF_World<dtype> * where_am_i() const = 0;

    /**
     * \brief casts into a double if dimension of evaluated expression is 0
     */
    operator dtype() const;
};

template<typename dtype, typename dtype_A>
class tCTF_Sum_Term : public tCTF_Term<dtype> {
  public:
    std::vector< tCTF_Term<dtype>* > operands;

    // default constructor
    tCTF_Sum_Term() : tCTF_Term<dtype>() {}

    // destructor frees operands
    ~tCTF_Sum_Term();
  
    // copy constructor
    tCTF_Sum_Term(tCTF_Sum_Term<dtype> const & other,
        std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    // dervied clone calls copy constructor
    tCTF_Term<dtype>* clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * construct sum term corresponding to a single tensor
     * \param[in] output tensor to write results into and its indices
     */ 
    //tCTF_Sum_Term<dtype>(tCTF_Idx_Tensor<dtype> const & tsr);

    /**
     * \brief evalues the expression by summing operands into output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;

  
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a sum term
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    tCTF_Sum_Term<dtype> operator+(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    tCTF_Sum_Term<dtype> operator-(tCTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};

template<typename dtype> static
tCTF_Contract_Term<dtype> operator*(double d, tCTF_Term<dtype> const & tsr){
  return (tsr*d);
}

/**
 * \brief An experession representing a contraction of a set of tensors contained in operands 
 */
template<typename dtype>
class tCTF_Contract_Term : public tCTF_Term<dtype> {
  public:
    std::vector< tCTF_Term<dtype>* > operands;

    // \brief default constructor
    tCTF_Contract_Term() : tCTF_Term<dtype>() {}

    // \brief destructor frees operands
    ~tCTF_Contract_Term();
  
    // \brief copy constructor
    tCTF_Contract_Term(tCTF_Contract_Term<dtype> const & other,
        std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    // \brief dervied clone calls copy constructor
    tCTF_Term<dtype> * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * \brief override execution to  to contract operands and add them to output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a contract term
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    
    /**
     * \brief override contraction to grow vector rather than create recursive terms
     * \param[in] A term to multiply by
     */
    tCTF_Contract_Term<dtype> operator*(tCTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};
/**
 * @}
 */



#endif
