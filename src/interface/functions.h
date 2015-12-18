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
      //dtype (*f)(dtype);
      std::function<void(dtype&)> f;
     
      /**
       * \brief constructor takes function pointer
       * \param[in] f_ scalar function: (type) -> (type)
       */
      Endomorphism(std::function<void(dtype&)> f_){ f = f_; }
      /**
       * \brief default constructor
       */
      Endomorphism(){}

      /**
       * \brief apply function f to value stored at a
       * \param[in,out] a pointer to operand that will be cast to dtype
       *                  is set to result of applying f on value at a
       */
      void apply_f(char * a) const { f(((dtype*)a)[0]); }
  };


  /**
   * \brief custom function f : X -> Y to be applied to tensor elemetns: 
   *          e.g. B["ij"] = f(A["ij"])
   */
  template<typename dtype_A=double, typename dtype_B=dtype_A>
  class Univar_Function : public CTF_int::univar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute b=f(a)
       */
      //dtype_B (*f)(dtype_A);
      std::function<dtype_B(dtype_A)> f;
      
      /**
       * \brief constructor takes function pointers to compute B=f(A));
       * \param[in] f_ linear function (type_A)->(type_B)
       */
      Univar_Function(std::function<dtype_B(dtype_A)> f_){ f = f_; }

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
  template<typename dtype_A=double, typename dtype_B=dtype_A>
  class Univar_Transform : public CTF_int::univar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute b=f(a)
       */
      //void (*f)(dtype_A, dtype_B &);
      std::function<void(dtype_A, dtype_B &)> f;
      
      /**
       * \brief constructor takes function pointers to compute B=f(A));
       * \param[in] f_ linear function (type_A)->(type_B)
       */
      Univar_Transform(std::function<void(dtype_A, dtype_B &)> f_){ f = f_; }

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
  template<typename dtype_A=double, typename dtype_B=dtype_A, typename dtype_C=dtype_A>
  class Bivar_Function : public CTF_int::bivar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute C=f(A,B)
       */
      //dtype_C (*f)(dtype_A, dtype_B);
      std::function<dtype_C (dtype_A, dtype_B)> f;
     
      /**
       * \brief constructor takes function pointers to compute C=f(A,B);
       * \param[in] f_ bilinear function (type_A,type_B)->(type_C)
       */
      Bivar_Function(std::function<dtype_C (dtype_A, dtype_B)> f_){ f=f_; }

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
       * \brief compute c = f(a,b)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to operand that will be cast to dtype 
       * \param[in,out] result c+f(*a,b) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char const * b, char * c) const { 
        ((dtype_C*)c)[0] = f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]); 
      }

      /**
       * \brief compute c = c+ f(a,b)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to operand that will be cast to dtype 
       * \param[in,out] result c+f(*a,b) of applying f on value of (different type) on a
       * \param[in] sr_C algebraic structure for b, needed to do add
       */
      void acc_f(char const * a, char const * b, char * c, CTF_int::algstrct const * sr_C) const { 
        dtype_C tmp;
        tmp = f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]);
        sr_C->add(c, (char const *)&tmp, c); 
      }


  };

  /**
   * \brief custom function f : (X * Y) -> X applied on two tensors as summation: 
   *          e.g. B["ij"] = f(A["ij"],B["ij"])
   */
  template<typename dtype_A=double, typename dtype_B=dtype_A, typename dtype_C=dtype_A>
  class Bivar_Transform : public CTF_int::bivar_function {
    public:
      /**
       * \brief function signature for element-wise multiplication, compute b=f(a)
       */
      //void (*f)(dtype_A, dtype_B &);
      std::function<void(dtype_A, dtype_B, dtype_C &)> f;
      
      /**
       * \brief constructor takes function pointers to compute B=f(A));
       * \param[in] f_ linear function (type_A)->(type_B)
       */
      Bivar_Transform(std::function<void(dtype_A, dtype_B, dtype_C &)> f_){ f = f_; }

      /** 
       * \brief evaluate B=f(A) 
       * \param[in] A operand tensor with pre-defined indices 
       * return f(A) output tensor 
       */
      //Idx_Tensor operator()(Idx_Tensor const  & A);
       /**
       * \brief compute f(a,b)
       * \param[in] a pointer to the accumulated operand 
       * \param[in,out] value that is accumulated to
       * \param[in] sr_B algebraic structure for b, here is ignored
       */
      void acc_f(char const * a, char const * b, char * c, CTF_int::algstrct const * sr_B) const {
        f(((dtype_A*)a)[0], ((dtype_B*)b)[0], ((dtype_C*)c)[0]);
      }
      
      /**
       * \brief apply function f to value stored at a, for an accumulator, this is the same as acc_f below
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char const * b, char * c) const { acc_f(a,b,c,NULL); }


      bool is_accumulator() const { return true; }
  };




  template<typename dtype_A=double, typename dtype_B=dtype_A, typename dtype_C=dtype_A>
  class Function {
    public:
      bool is_univar;
      Univar_Function<dtype_A, dtype_B> * univar;
      bool is_bivar;
      Bivar_Function<dtype_A, dtype_B, dtype_C> * bivar;

      Function(std::function<dtype_B(dtype_A)> f_){
        is_univar = true;
        is_bivar = false;
        univar = new Univar_Function<dtype_A, dtype_B>(f_);
      }

 
      Function(std::function<dtype_C(dtype_A,dtype_B)> f_){
        is_univar = false;
        is_bivar = true;
        bivar = new Bivar_Function<dtype_A, dtype_B, dtype_C>(f_);
      }

      CTF_int::Unifun_Term operator()(CTF_int::Term const & A) const {
        assert(is_univar);
        return univar->operator()(A);
      }
 
      CTF_int::Bifun_Term operator()(CTF_int::Term const & A, CTF_int::Term const & B) const {
        assert(is_bivar);
        return bivar->operator()(A,B);
      }
      
      operator Univar_Function<dtype_A, dtype_B>() const {
        assert(is_univar);
        return *univar;
      }
      
      operator Bivar_Function<dtype_A, dtype_B, dtype_C>() const {
        assert(is_bivar);
        return *bivar;
      }

      ~Function(){
        if (is_univar) delete(univar);
        if (is_bivar) delete(bivar);
      }
  };
  
  template<typename dtype_A=double, typename dtype_B=dtype_A, typename dtype_C=dtype_A>
  class Transform {
    public:
      bool is_endo;
      Endomorphism<dtype_A> * endo;
      bool is_univar;
      Univar_Transform<dtype_A, dtype_B> * univar;
      bool is_bivar;
      Bivar_Transform<dtype_A, dtype_B, dtype_C> * bivar;

      Transform(std::function<void(dtype_A&)> f_){
        is_endo = true;
        is_univar = false;
        is_bivar = false;
        endo = new Endomorphism<dtype_A>(f_);
      }
      
      Transform(std::function<void(dtype_A, dtype_B&)> f_){
        is_endo = false;
        is_univar = true;
        is_bivar = false;
        univar = new Univar_Transform<dtype_A, dtype_B>(f_);
      }
      
      Transform(std::function<void(dtype_A, dtype_B, dtype_C&)> f_){
        is_endo = false;
        is_univar = false;
        is_bivar = true;
        bivar = new Bivar_Transform<dtype_A, dtype_B, dtype_C>(f_);
      }


      ~Transform(){
        if (is_endo) delete endo;
        if (is_univar) delete univar;
        if (is_bivar) delete bivar;
      }

      void operator()(CTF_int::Term const & A) const {
        assert(is_endo);
        endo->operator()(A);
      }
 
      void operator()(CTF_int::Term const & A, CTF_int::Term const & B) const {
        assert(is_univar);
        univar->operator()(A,B);
      }
 
      void operator()(CTF_int::Term const & A, CTF_int::Term const & B, CTF_int::Term const & C) const {
        assert(is_bivar);
        bivar->operator()(A,B,C);
      }
      
      operator Bivar_Transform<dtype_A, dtype_B, dtype_C>(){
        assert(is_bivar);
        return *bivar;
      }

      operator Univar_Transform<dtype_A, dtype_B>(){
        assert(is_univar);
        return *univar;
      }
      
      operator Endomorphism<dtype_A>(){
        assert(is_endo);
        return *endo;
      }
      
      bool is_accumulator() const { return true; }
  };

/**
 * @}
 */
}

#endif

