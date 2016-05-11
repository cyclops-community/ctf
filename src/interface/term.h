#ifndef __TERM_H__
#define __TERM_H__

#include <map>
#include <set>
#include "../tensor/untyped_tensor.h"

namespace CTF {
  class Idx_Tensor;
}

namespace CTF_int {
  /**
   * \defgroup expression Tensor expression compiler
   * \addtogroup expression
   * @{
   */
  class Sum_Term;
  class Contract_Term;

  /**
   * \brief comparison function for sets of tensor pointers
   * This ensures the set iteration order is consistent across nodes
   */
  struct tensor_name_less {
    bool operator()(CTF::Idx_Tensor* A, CTF::Idx_Tensor* B);
  };


  /**
   * \brief a term is an abstract object representing some expression of tensors
   */
  class Term {
    public:
      char * scale;
      algstrct * sr;
     
      Term(algstrct const * sr);

      virtual ~Term();

      /**
       * \brief base classes must implement this copy function to retrieve pointer
       */ 
      virtual Term * clone(std::map<tensor*, tensor*>* remap = NULL) const = 0;
      
      /**
       * \brief evalues the expression, which just scales by default
       * \param[in,out] output tensor to write results into and its indices
       */
      virtual void execute(CTF::Idx_Tensor output) const = 0;
      
      /**
       * \brief estimates the cost of a contraction/sum/.. term
       * \param[in] output tensor to write results into and its indices
       */
      virtual double  estimate_time(CTF::Idx_Tensor output) const = 0;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param\[in,out] cost the cost of the operatiob
       * \return output tensor to write results into and its indices
       */
      virtual CTF::Idx_Tensor estimate_time(double  & cost) const = 0;
      
      
      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      virtual CTF::Idx_Tensor execute() const = 0;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      virtual void get_inputs(std::set<CTF::Idx_Tensor*, tensor_name_less >* inputs_set) const = 0;

      /**
       * \brief constructs a new term which multiplies by tensor A
       * \param[in] A term to multiply by
       */
      Contract_Term operator*(Term const & A) const;
      /**
       * \brief multiples by a constant
       * \param[in] scl scaling factor to multiply term by
       */
      Contract_Term operator*(int64_t scl) const;
      Contract_Term operator*(double scl) const;
      
      /**
       * \brief constructs a new term by addition of two terms
       * \param[in] A term to add to output
       */
      Sum_Term operator+(Term const & A) const;
      Sum_Term operator+(double scl) const;
      Sum_Term operator+(int64_t scl) const;
      
      /**
       * \brief constructs a new term by subtracting term A
       * \param[in] A subtracted term
       */
      Sum_Term operator-(Term const & A) const;
      
      Sum_Term operator-(double scl) const;
      Sum_Term operator-(int64_t scl) const;
      
      Term & operator-();
      
      /**
       * \brief A = B, compute any operations on operand B and set
       * \param[in] B tensor on the right hand side
       */
      void operator=(CTF::Idx_Tensor const & B);
      void operator=(Term const & B);
      void operator+=(Term const & B);
      void operator-=(Term const & B);
      void operator*=(Term const & B);

      void operator=(double scl);
      void operator+=(double scl);
      void operator-=(double scl);
      void operator*=(double scl);

      void operator=(int64_t scl);
      void operator+=(int64_t scl);
      void operator-=(int64_t scl);
      void operator*=(int64_t scl);

      void operator=(int scl);
      void operator+=(int scl);
      void operator-=(int scl);
      void operator*=(int scl);
      /**
       * \brief figures out what world this term lives on
       */
      virtual CTF::World * where_am_i() const = 0;

      /**
       * \brief cast to double (works only if tensor type is castable to double)
       *        allows a scalar output
       */
      operator double() const;
 
      /**
       * \brief cast to int64_t (works only if tensor type is castable to int64_t)
       *        allows a scalar output
       */     
      operator int64_t() const;


  };

  class Sum_Term : public Term {
    public:
      std::vector< Term* > operands;

      /**
       * \brief creates sum term for B+A
       * \param[in] B left hand side
       * \param[in] B right hand side
       */
      Sum_Term(Term * B, Term * A);

      // destructor frees operands
      ~Sum_Term();
    
      // copy constructor
      Sum_Term(Sum_Term const & other,
               std::map<tensor*, tensor*>* remap = NULL);

      // dervied clone calls copy constructor
      Term* clone(std::map<tensor*, tensor*>* remap = NULL) const;

      /**
       * construct sum term corresponding to a single tensor
       * \param[in] output tensor to write results into and its indices
       */ 
      //Sum_Term(CTF::Idx_Tensor const & tsr);

      /**
       * \brief evalues the expression by summing operands into output
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(CTF::Idx_Tensor output) const;
    
      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      CTF::Idx_Tensor execute() const;
      
      /**
       * \brief estimates the cost of a sum term
       * \param[in] output tensor to write results into and its indices
       */
      double  estimate_time(CTF::Idx_Tensor output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      CTF::Idx_Tensor estimate_time(double  & cost) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set<CTF::Idx_Tensor*, tensor_name_less >* inputs_set) const;

      /**
       * \brief constructs a new term by addition of two terms
       * \param[in] A term to add to output
       */
      Sum_Term operator+(Term const & A) const;
      
      /**
       * \brief constructs a new term by subtracting term A
       * \param[in] A subtracted term
       */
      Sum_Term operator-(Term const & A) const;
 
      /**
       * \brief negates term
       */
//      Sum_Term operator-() const;


      /**
       * \brief figures out what world this term lives on
       */
      CTF::World * where_am_i() const;
  };

  /**
   * \brief An experession representing a contraction of a set of tensors contained in operands 
   */
  class Contract_Term : public Term {
    public:
      std::vector< Term* > operands;

 
      /**
       * \brief creates sum term for B+A
       * \param[in] B left hand side
       * \param[in] B right hand side
       */
      Contract_Term(Term * B, Term * A);


      // \brief destructor frees operands
      ~Contract_Term();
    
      // \brief copy constructor
      Contract_Term(Contract_Term const & other,
          std::map<tensor*, tensor*>* remap = NULL);

      // \brief dervied clone calls copy constructor
      Term * clone(std::map<tensor*, tensor*>* remap = NULL) const;

      /**
       * \brief override execution to  to contract operands and add them to output
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(CTF::Idx_Tensor output) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set<CTF::Idx_Tensor*, tensor_name_less >* inputs_set) const;

      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      CTF::Idx_Tensor execute() const;
      
      /**
       * \brief estimates the cost of a contract term
       * \param[in] output tensor to write results into and its indices
       */
      double  estimate_time(CTF::Idx_Tensor output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      CTF::Idx_Tensor estimate_time(double  & cost) const;
      
      
      /**
       * \brief override contraction to grow vector rather than create recursive terms
       * \param[in] A term to multiply by
       */
      Contract_Term operator*(Term const & A) const;
 
      /**
       * \brief negates term
       */
//      Contract_Term operator-() const;

      /**
       * \brief figures out what world this term lives on
       */
      CTF::World * where_am_i() const;
  };


  //FIXME: what if noncommutative?
  inline CTF_int::Contract_Term operator*(double const & d, CTF_int::Term const & tsr){
    return (tsr*d);
  }

  //FIXME: what if noncommutative?
  inline CTF_int::Contract_Term operator*(int64_t const & i, CTF_int::Term const & tsr){
    return (tsr*i);
  }

  void operator-=(double & d, CTF_int::Term const & tsr);

  void operator+=(double & d, CTF_int::Term const & tsr);

  void operator-=(int64_t & d, CTF_int::Term const & tsr);

  void operator+=(int64_t & d, CTF_int::Term const & tsr);


/**
 * @}
 */

}

#endif
