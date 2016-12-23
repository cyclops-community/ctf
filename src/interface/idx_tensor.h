#ifndef __EXPRESSION_H__
#define __EXPRESSION_H__

#include "term.h"
#include "functions.h"

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
       * \param[in] other tensor to copy
       * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
       * \param[in] remap redistribution dependency map
       */
      Idx_Tensor(CTF::Idx_Tensor const &                       other,
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
       * \brief evalues the expression to produce an intermediate with 
       *        all expression indices remaining
       * \return output tensor to write results into and its indices
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
       * \param[out] cost estimate of time in sec
       */
      Idx_Tensor estimate_time(double  & cost) const;
      
      /**
      * \brief appends the tensors this depends on to the input set
      */
      void get_inputs(std::set<Idx_Tensor*, CTF_int::tensor_name_less >* inputs_set) const;

      /**
       * \brief A = B, compute any operations on operand B and set
       * \param[in] B tensor on the right hand side
       */
      void operator=(CTF_int::Term const & B);
      void operator=(Idx_Tensor const & B);

      //same as in parent (Term) but not inherited in C++
      void operator=(double scl);
      void operator+=(double scl);
      void operator-=(double scl);
      void operator*=(double scl);
      void multeq(double scl);
      void operator=(int64_t scl);
      void operator+=(int64_t scl);
      void operator-=(int64_t scl);
      void operator*=(int64_t scl);
      void operator=(int scl);
      void operator+=(int scl);
      void operator-=(int scl);
      void operator*=(int scl);

      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator+=(CTF_int::Term const & B);
      void operator<<(CTF_int::Term const & B);
      
      /**
       * \brief A += B, compute any operations on operand B and add
       * \param[in] B tensor on the right hand side
       */
      void operator-=(CTF_int::Term const & B);

      /**
       * \brief negates term
       */
//      Idx_Tensor operator-() const;
 
      
      /**
       * \brief A -> A*B contract two tensors
       * \param[in] B tensor on the right hand side
       */
      void operator*=(CTF_int::Term const & B);

      /**
       * brief TODO A -> A * B^-1
       * param[in] B
       */
      //void operator/(IdxTensor& tsr);
      
      /**
       * brief execute ips into output with scale beta
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

  template<typename dtype>
  class Typ_Idx_Tensor;

  
  template<typename dtype_A, typename dtype_B>
  class Univar_Transform;
  
  template<typename dtype_A, typename dtype_B, typename dtype_C>
  class Typ_Contract_Term;
  
  template<typename dtype_A, typename dtype_B>
  class Typ_Sum_Term : public CTF_int::Sum_Term {
    public:

      Typ_Sum_Term(Typ_Idx_Tensor<dtype_A> * A, Typ_Idx_Tensor<dtype_B> * B) : CTF_int::Sum_Term(A,B) {}

      void operator()(std::function<void(dtype_A, dtype_B&)> f){
        (Transform<dtype_A,dtype_B>(f))(*operands[0],*operands[1]);
      }

      void operator,(std::function<void(dtype_A, dtype_B&)> f){
        (Transform<dtype_A,dtype_B>(f))(*operands[0],*operands[1]);
      }

      void operator()(std::function<dtype_B(dtype_A)> f){
        ((Function<dtype_A,dtype_B>(f))(*operands[0])).execute(operands[1]->execute());
      }

      void operator,(std::function<dtype_B(dtype_A)> f){
        ((Function<dtype_A,dtype_B>(f))(*operands[0])).execute(operands[1]->execute());
      }
 
      
      template<typename dtype_C>
      Typ_Contract_Term<dtype_A,dtype_B,dtype_C> operator&(Typ_Idx_Tensor<dtype_C> C){
        return Typ_Contract_Term<dtype_A,dtype_B,dtype_C>(&C, *this);
      }
  };
  
  template<typename dtype_A, typename dtype_B, typename dtype_C>
  class Typ_Contract_Term : public CTF_int::Contract_Term {
    public:
      Typ_Idx_Tensor<dtype_C> * C;
      Typ_Contract_Term(Typ_Idx_Tensor<dtype_C> * C_, 
                        Typ_Sum_Term<dtype_A,dtype_B> S) 
          : Contract_Term(S.operands[0]->clone(), 
                          S.operands[1]->clone()) {
        C = C_;
      }
      
      void operator,(std::function<dtype_C(dtype_A, dtype_B)> f){
        ((Function<dtype_A,dtype_B,dtype_C>(f))(*operands[1],*operands[0])).execute(*C);
      }
      
      void operator,(std::function<void(dtype_A, dtype_B, dtype_C&)> f){
        ((Transform<dtype_A,dtype_B,dtype_C>(f)))(*operands[1],*operands[0],*C);
      }
        
  };

  template<typename dtype>
  class Typ_AIdx_Tensor;

  template<typename dtype>
  class Typ_Idx_Tensor : public Idx_Tensor {
    public:
      ~Typ_Idx_Tensor(){}

      /**
       * \brief constructor takes in a parent tensor and its indices 
       * \param[in] parent_ the parent tensor
       * \param[in] idx_map_ the indices assigned ot this tensor
       * \param[in] copy if set to 1, create copy of parent
       */
      Typ_Idx_Tensor(CTF_int::tensor * parent_,
                     const char *      idx_map_,
                     int               copy=0) : Idx_Tensor(parent_, idx_map_, copy) {}
      
      /**
       * \brief copy constructor
       * \param[in] B tensor to copy
       * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
       * \param[in] remap redistribution dependency map
       */
      Typ_Idx_Tensor(Typ_Idx_Tensor const &                            B,
                     int                                           copy=0,
                     std::map<CTF_int::tensor*, CTF_int::tensor*>* remap=NULL) : Idx_Tensor(B, copy, remap) {}


      Typ_Idx_Tensor<dtype> * tclone() const { return new Typ_Idx_Tensor<dtype>(*this); }

      CTF_int::Term * clone(std::map< CTF_int::tensor*, CTF_int::tensor* >* remap = NULL) const { return new Typ_Idx_Tensor<dtype>(*this, 0, remap); }

      void operator=(CTF_int::Term const & B){ Idx_Tensor::operator=(B); }
      void operator=(Idx_Tensor const & B){ Idx_Tensor::operator=(B); }
      void operator=(double scl){ Idx_Tensor::operator=(scl); }
      void operator=(int64_t scl){ Idx_Tensor::operator=(scl); }
      void operator=(int scl){ Idx_Tensor::operator=(scl); }



      template <typename dtype_B>
      Typ_Sum_Term<dtype, dtype_B> operator&(Typ_Idx_Tensor<dtype_B> B){
        return Typ_Sum_Term<dtype, dtype_B>(this->tclone(), B.tclone());
      }
 
      template <typename dtype_A, typename dtype_B>
      Typ_Contract_Term<dtype_A, dtype_B, dtype> operator+=(Typ_Sum_Term<dtype_A,dtype_B> t){
        return Typ_Contract_Term<dtype_A,dtype_B,dtype>(this->tclone(), t);
      }
     /* 
      template <typename dtype_A, typename dtype_B>
      Typ_Contract_Term<dtype_A, dtype_B, dtype> operator&=(Typ_Sum_Term<dtype_A,dtype_B> t){
        sr->safecopy(scale,sr->addid());
        return Typ_Contract_Term<dtype_A,dtype_B,dtype>(this->clone(), t);
      }*/
      
      template <typename dtype_A, typename dtype_B>
      Typ_Contract_Term<dtype_A, dtype_B, dtype> operator=(Typ_Sum_Term<dtype_A,dtype_B> t){
        sr->safecopy(scale,sr->addid());
        return Typ_Contract_Term<dtype_A,dtype_B,dtype>(this, t);
      }

      Typ_AIdx_Tensor<dtype> operator~(){
        return Typ_AIdx_Tensor<dtype>(*this);
      }
 
      
      template <typename dtype_A>
      Typ_Sum_Term<dtype_A, dtype> operator=(Typ_AIdx_Tensor<dtype_A> t){
        sr->safecopy(scale,sr->addid());
        return Typ_Sum_Term<dtype_A,dtype>(t.tclone(), this->tclone());
      }
      
      template <typename dtype_A>
      Typ_Sum_Term<dtype_A, dtype> operator+=(Typ_AIdx_Tensor<dtype_A> t){
        return Typ_Sum_Term<dtype_A,dtype>(t.tclone(), this->tclone());
      }

      void operator+=(CTF_int::Term const & B){ Idx_Tensor::operator+=(B); }
      void operator+=(Idx_Tensor const & B){ Idx_Tensor::operator+=(B); }
      void operator+=(double scl){ Idx_Tensor::operator+=(scl); }
      void operator+=(int64_t scl){ Idx_Tensor::operator+=(scl); }
      void operator+=(int scl){ Idx_Tensor::operator+=(scl); }


      void operator,(std::function<void(dtype&)> f){
        ((Transform<dtype>(f)))(*this);
      }
      
      void operator()(std::function<void(dtype&)> f){
        ((Transform<dtype>(f)))(*this);
      }
        
  };

  template<typename dtype>
  class Typ_AIdx_Tensor : public Typ_Idx_Tensor<dtype> {
    public:
      Typ_AIdx_Tensor(Typ_Idx_Tensor<dtype> const & A) : Typ_Idx_Tensor<dtype>(A) {};
  };
  /**
   * @}
   */
}
#endif
