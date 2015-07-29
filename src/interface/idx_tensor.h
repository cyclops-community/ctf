#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

#include "term.h"

namespace CTF {
  /**
   * \addtogroup expression
   * @{
   */
  /**
   * \brief a tensor with an index map associated with it (necessary for overloaded operators)
   */
  class Idx_Tensor : public CTF_int::Term {
    public:
      CTF_int::tensor * parent;
      char * idx_map;
      int is_intm;

    
      // derived clone calls copy constructor
      CTF_int::Term * clone(std::map< CTF_int::tensor*, CTF_int::tensor* >* remap = NULL) const;

      /**
       * \brief constructor takes in a parent tensor and its indices 
       * \param[in] parent_ the parent tensor
       * \param[in] idx_map_ the indices assigned ot this tensor
       * \param[in] copy if set to 1, create copy of parent
       */
      Idx_Tensor(CTF_int::tensor * parent_,
                 const char *      idx_map_,
                 int               copy=0);
      
      /**
       * \brief copy constructor
       * \param[in] B tensor to copy
       * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
       * \param[in] remap redistribution dependency map
       */
      Idx_Tensor(Idx_Tensor const &                            B,
                 int                                           copy=0,
                 std::map<CTF_int::tensor*, CTF_int::tensor*>* remap=NULL);

      /**
       * \brief constructor for scalar
       * \param[in] sr ring/semiring
       */
      Idx_Tensor(CTF_int::algstrct const * sr);
      Idx_Tensor(CTF_int::algstrct const * sr, double scl);
      Idx_Tensor(CTF_int::algstrct const * sr, int64_t scl);
      ~Idx_Tensor();
      
      /**
       * \brief constructor for scalar
       * \param[in] val double value
       */
      //Idx_Tensor(double val);

      /**
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor execute() const;
      
      /**
       * \brief evalues the expression, which just scales by default
       * \param[in,out] output tensor to write results into and its indices
       */
      void execute(Idx_Tensor output) const;
      
      /**
       * \brief estimates the cost of a contraction
       * \param[in] output tensor to write results into and its indices
       */
      double estimate_time(Idx_Tensor output) const;
      
      /**
       * \brief estimates the cost the expression to produce an intermediate with 
       *        all expression indices remaining
       * \param[in,out] output tensor to write results into and its indices
       */
      Idx_Tensor estimate_time(double  & cost) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set< CTF_int::tensor*, CTF_int::tensor_tid_less >* inputs_set) const;

      /**
       * \brief A = B, compute any operations on operand B and set
       * \param[in] B tensor on the right hand side
       */
      void operator=(CTF_int::Term const & B);
      void operator=(Idx_Tensor const & B);

      //same as in parent (Term) but not inherited in C++
      void operator=(double scl);
      void operator=(int64_t scl);

      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator+=(CTF_int::Term const & B);
      
      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator-=(CTF_int::Term const & B);
      
      /**
       * \brief A -> A*B contract two tensors
       * \param[in] B tensor on the right hand side
       */
      void operator*=(CTF_int::Term const & B);

      /**
       * \brief TODO A -> A * B^-1
       * \param[in] B
       */
      //void operator/(IdxTensor& tsr);
      
      /**
       * \brief execute ips into output with scale beta
       */    
      //void run(Idx_Tensor* output, dtype  beta);

      /*operator CTF_int::Term* (){
        Idx_Tensor * tsr = new Idx_Tensor(*this);
        return tsr;
      }*/
      /**
       * \brief figures out what world this term lives on
       */
      World * where_am_i() const;
  };

  /**
   * @}
   */
}


#endif
