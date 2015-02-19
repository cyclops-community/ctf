#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

#include <map>
#include <set>

/**
 * \defgroup expression Tensor expression compiler
 * @{
 */

namespace CTF {
  template <typename dtype, bool is_ord> class Tensor;

  template <typename dtype, bool is_ord> class Term;
  template <typename dtype, bool is_ord> class Sum_Term;
  template <typename dtype, bool is_ord> class Contract_Term;

  /**
   * \brief comparison function for sets of tensor pointers
   * This ensures the set iteration order is consistent across nodes
   */
  template<typename dtype, bool is_ord>
  struct tensor_tid_less {
    bool operator()(Tensor<dtype, is_ord>* A, Tensor<dtype, is_ord>* B) {
      if (A == NULL && B != NULL) {
        return true;
      } else if (A == NULL || B == NULL) {
        return false;
      }
      ASSERT(0);//FIXME
      //return A->tid < B->tid;
      return -1;
    }
  };


  /**
   * \brief a tensor with an index map associated with it (necessary for overloaded operators)
   */
  template<typename dtype=double, bool is_ord=true>
  class Idx_Tensor : public Term<dtype, is_ord> {
    public:
      Tensor<dtype, is_ord> * parent;
      char * idx_map;
      int is_intm;

    public:

    
      // derived clone calls copy constructor
      Term<dtype, is_ord> * clone(std::map< Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>* >* remap = NULL) const;

      /**
       * \brief constructor takes in a parent tensor and its indices 
       * \param[in] parent_ the parent tensor
       * \param[in] idx_map_ the indices assigned ot this tensor
       * \param[in] copy if set to 1, create copy of parent
       */
      Idx_Tensor(Tensor<dtype, is_ord> * parent_, 
                 const char *    idx_map_,
                 int             copy = 0);
      
      /**
       * \brief copy constructor
       * \param[in] B tensor to copy
       * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
       */
      Idx_Tensor(Idx_Tensor<dtype, is_ord> const & B,
                 int copy = 0,
                 std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL);

      Idx_Tensor();
      
      Idx_Tensor(dtype val);
      
      ~Idx_Tensor();
      
      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> execute() const;
      
      /**
       * \brief evalues the expression, which just scales by default
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(Idx_Tensor<dtype, is_ord> output) const;
      
      /**
       * \brief estimates the cost of a contraction
       * \param[in] output tensor to write results into and its indices
       */
      int64_t estimate_cost(Idx_Tensor<dtype, is_ord> output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> estimate_cost(int64_t  & cost) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set< Tensor<dtype, is_ord>*, tensor_tid_less<dtype, is_ord> >* inputs_set) const;

      /**
       * \brief A = B, compute any operations on operand B and set
       * \param[in] B tensor on the right hand side
       */
      void operator=(Term<dtype, is_ord> const & B);
      void operator=(Idx_Tensor<dtype, is_ord> const & B);

      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator+=(Term<dtype, is_ord> const & B);
      
      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator-=(Term<dtype, is_ord> const & B);
      
      /**
       * \brief A -> A*B contract two tensors
       * \param[in] B tensor on the right hand side
       */
      void operator*=(Term<dtype, is_ord> const & B);

      /**
       * \brief TODO A -> A * B^-1
       * \param[in] B
       */
      //void operator/(IdxTensor& tsr);
      
      /**
       * \brief execute ips into output with scale beta
       */    
      //void run(Idx_Tensor<dtype, is_ord>* output, dtype  beta);

      /*operator Term<dtype, is_ord>* (){
        Idx_Tensor * tsr = new Idx_Tensor(*this);
        return tsr;
      }*/
      /**
       * \brief figures out what world this term lives on
       */
      World * where_am_i() const;
  };


  /**
   * \brief a term is an abstract object representing some expression of tensors
   */
  template<typename dtype=double, bool is_ord=true>
  class Term {
    public:
      dtype scale;
     
      Term();
      virtual ~Term(){};

      /**
       * \brief base classes must implement this copy function to retrieve pointer
       */ 
      virtual Term * clone(std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL) const = 0;
      
      /**
       * \brief evalues the expression, which just scales by default
       * \param[in,out] output tensor to write results into and its indices
       */
      virtual void execute(Idx_Tensor<dtype, is_ord> output) const = 0;
      
      /**
       * \brief estimates the cost of a contraction/sum/.. term
       * \param[in] output tensor to write results into and its indices
       */
      virtual int64_t  estimate_cost(Idx_Tensor<dtype, is_ord> output) const = 0;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param\[in,out] cost the cost of the operatiob
       * \return output tensor to write results into and its indices
       */
      virtual Idx_Tensor<dtype, is_ord> estimate_cost(int64_t  & cost) const = 0;
      
      
      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      virtual Idx_Tensor<dtype, is_ord> execute() const = 0;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      virtual void get_inputs(std::set<Tensor<dtype, is_ord>*, tensor_tid_less<dtype, is_ord> >* inputs_set) const = 0;

      /**
       * \brief constructs a new term which multiplies by tensor A
       * \param[in] A term to multiply by
       */
      Contract_Term<dtype, is_ord> operator*(Term<dtype, is_ord> const & A) const;
      
      /**
       * \brief constructs a new term by addition of two terms
       * \param[in] A term to add to output
       */
      Sum_Term<dtype, is_ord> operator+(Term<dtype, is_ord> const & A) const;
      
      /**
       * \brief constructs a new term by subtracting term A
       * \param[in] A subtracted term
       */
      Sum_Term<dtype, is_ord> operator-(Term<dtype, is_ord> const & A) const;
      
      /**
       * \brief A = B, compute any operations on operand B and set
       * \param[in] B tensor on the right hand side
       */
      void operator=(Term<dtype, is_ord> const & B) { execute() = B; };
      void operator=(Idx_Tensor<dtype, is_ord> const & B) { execute() = B; };
      void operator+=(Term<dtype, is_ord> const & B) { execute() += B; };
      void operator-=(Term<dtype, is_ord> const & B) { execute() -= B; };
      void operator*=(Term<dtype, is_ord> const & B) { execute() *= B; };

      /**
       * \brief multiples by a constant
       * \param[in] scl scaling factor to multiply term by
       */
      Contract_Term<dtype, is_ord> operator*(dtype scl) const;

      /**
       * \brief figures out what world this term lives on
       */
      virtual World * where_am_i() const = 0;

      /**
       * \brief casts into a double if dimension of evaluated expression is 0
       */
      operator dtype() const;
  };

  template<typename dtype=double, bool is_ord=true>
  class Sum_Term : public Term<dtype, is_ord> {
    public:
      std::vector< Term<dtype, is_ord>* > operands;

      // default constructor
      Sum_Term() : Term<dtype, is_ord>() {}

      // destructor frees operands
      ~Sum_Term();
    
      // copy constructor
      Sum_Term(Sum_Term<dtype, is_ord> const & other,
          std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL);

      // dervied clone calls copy constructor
      Term<dtype, is_ord>* clone(std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL) const;

      /**
       * construct sum term corresponding to a single tensor
       * \param[in] output tensor to write results into and its indices
       */ 
      //Sum_Term<dtype, is_ord>(Idx_Tensor<dtype, is_ord> const & tsr);

      /**
       * \brief evalues the expression by summing operands into output
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(Idx_Tensor<dtype, is_ord> output) const;
    
      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> execute() const;
      
      /**
       * \brief estimates the cost of a sum term
       * \param[in] output tensor to write results into and its indices
       */
      int64_t  estimate_cost(Idx_Tensor<dtype, is_ord> output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> estimate_cost(int64_t  & cost) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set<Tensor<dtype, is_ord>*, tensor_tid_less<dtype, is_ord> >* inputs_set) const;

      /**
       * \brief constructs a new term by addition of two terms
       * \param[in] A term to add to output
       */
      Sum_Term<dtype, is_ord> operator+(Term<dtype, is_ord> const & A) const;
      
      /**
       * \brief constructs a new term by subtracting term A
       * \param[in] A subtracted term
       */
      Sum_Term<dtype, is_ord> operator-(Term<dtype, is_ord> const & A) const;

      /**
       * \brief figures out what world this term lives on
       */
      World * where_am_i() const;
  };

  template<typename dtype, bool is_ord> static
  Contract_Term<dtype, is_ord> operator*(double d, Term<dtype, is_ord> const & tsr){
    return (tsr*d);
  }

  /**
   * \brief An experession representing a contraction of a set of tensors contained in operands 
   */
  template<typename dtype=double, bool is_ord=true>
  class Contract_Term : public Term<dtype, is_ord> {
    public:
      std::vector< Term<dtype, is_ord>* > operands;

      // \brief default constructor
      Contract_Term() : Term<dtype, is_ord>() {}

      // \brief destructor frees operands
      ~Contract_Term();
    
      // \brief copy constructor
      Contract_Term(Contract_Term<dtype, is_ord> const & other,
          std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL);

      // \brief dervied clone calls copy constructor
      Term<dtype, is_ord> * clone(std::map<Tensor<dtype, is_ord>*, Tensor<dtype, is_ord>*>* remap = NULL) const;

      /**
       * \brief override execution to  to contract operands and add them to output
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(Idx_Tensor<dtype, is_ord> output) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set<Tensor<dtype, is_ord>*, tensor_tid_less<dtype, is_ord> >* inputs_set) const;

      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> execute() const;
      
      /**
       * \brief estimates the cost of a contract term
       * \param[in] output tensor to write results into and its indices
       */
      int64_t  estimate_cost(Idx_Tensor<dtype, is_ord> output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor<dtype, is_ord> estimate_cost(int64_t  & cost) const;
      
      
      /**
       * \brief override contraction to grow vector rather than create recursive terms
       * \param[in] A term to multiply by
       */
      Contract_Term<dtype, is_ord> operator*(Term<dtype, is_ord> const & A) const;

      /**
       * \brief figures out what world this term lives on
       */
      World * where_am_i() const;
  };
  /**
   * @}
   */

}
#include "idx_tensor.cxx"
#include "term.cxx"

#endif
