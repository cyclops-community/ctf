/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CTR_COMM_H__
#define __CTR_COMM_H__

#include "../tensor/algstrct.h"
#include "../interface/fun_term.h"

namespace CTF_int{
  class contraction;
  
  /**
   * \brief untyped internal class for triply-typed bivariate function
   */
  class bivar_function {
    public:
      bool has_gemm;
      bool commutative;

      /**
       * \brief apply function f to values stored at a and b
       * \param[in] a pointer to first operand that will be cast to type by extending class
       * \param[in] b pointer to second operand that will be cast to type by extending class
       * \param[in,out] result: c=&f(*a,*b) 
       */
      virtual void apply_f(char const * a, char const * b, char * c) const = 0;
      
      /**
       * \brief compute c = c+f(a,b)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to operand that will be cast to dtype 
       * \param[in,out] result c+f(*a,b) of applying f on value of (different type) on a
       * \param[in] sr_C algebraic structure for b, needed to do add
       */
      virtual void acc_f(char const * a, char const * b, char * c, CTF_int::algstrct const * sr_C) const = 0;


      /** 
       * \brief evaluate C+=f(A,B)  or f(A,B,C) if transform
       * \param[in] A operand tensor with pre-defined indices 
       * \param[in] B operand tensor with pre-defined indices 
       * \param[in] C output tensor with pre-defined indices 
      */
      void operator()(Term const & A, Term const & B, Term const & C) const;

      /** 
       * \brief evaluate f(A,B) 
       * \param[in] A operand tensor with pre-defined indices 
       * \param[in] B operand tensor with pre-defined indices 
       * \return Bifun_Term that evaluates f(A)
      */
      Bifun_Term operator()(Term const & A, Term const & B) const;
      
      bivar_function(){ has_gemm = false; commutative = false; }
      bivar_function(bool is_comm){ has_gemm = false; commutative = is_comm; }

      virtual ~bivar_function(){}
      
      virtual bool is_accumulator() const { return false; }

      virtual void cgemm(char         tA,
                         char         tB,
                         int          m,
                         int          n,
                         int          k,
                         char const * A,
                         char const * B,
                         char *       C)  const {};


  };


  /**
   * \addtogroup nest_dist Nested distributed contraction and summation routines
   * @{
   */

  class ctr {
    public: 
      algstrct const * sr_A;
      algstrct const * sr_B;
      algstrct const * sr_C;
      char const * beta;
      int num_lyr; /* number of copies of this matrix being computed on */
      int idx_lyr; /* the index of this copy */

      virtual void run(char * A, char * B, char * C) { printf("SHOULD NOTR\n"); };
      virtual void print() { };
      virtual int64_t mem_fp() { return 0; };
      virtual int64_t mem_rec() { return mem_fp(); };
      virtual double est_time_fp(int nlyr) { return 0; };
      virtual double est_time_rec(int nlyr) { return est_time_fp(nlyr); };
      virtual ctr * clone() { return NULL; };
      
      /**
       * \brief deallocates generic ctr object
       */
      virtual ~ctr();
    
      /**
       * \brief copies generic ctr object
       */
      ctr(ctr * other);
      
      /**
       * \brief main constructor, defines variable based on contraction class
       */
      ctr(contraction const * c);
  };

  class ctr_replicate : public ctr {
    public: 
      int ncdt_A; /* number of processor dimensions to replicate A along */
      int ncdt_B; /* number of processor dimensions to replicate B along */
      int ncdt_C; /* number of processor dimensions to replicate C along */
      int64_t size_A; /* size of A blocks */
      int64_t size_B; /* size of B blocks */
      int64_t size_C; /* size of C blocks */

      CommData ** cdt_A;
      CommData ** cdt_B;
      CommData ** cdt_C;
      /* Class to be called on sub-blocks */
      ctr * rec_ctr;
      
      void run(char * A, char * B, char * C);
      /**
       * \brief returns the number of bytes of buffer space
       *  we need 
       * \return bytes needed
       */
      int64_t mem_fp();
      /**
       * \brief returns the number of bytes need by each processor in this kernel and its recursive calls
       * \return bytes needed for recursive contraction
       */
      int64_t mem_rec();
      /**
       * \brief returns the execution time the local part this kernel is estimated to take
       * \return time in sec
       */
      double est_time_fp(int nlyr);
      /**
       * \brief returns the execution time this kernel and its recursive calls are estimated to take
       * \return time in sec
       */
      double est_time_rec(int nlyr);
      void print();
      ctr * clone();

      ctr_replicate(ctr * other);
      ~ctr_replicate();
      ctr_replicate(contraction const * c,
                    int const *         phys_mapped,
                    int64_t             blk_sz_A,
                    int64_t             blk_sz_B,
                    int64_t             blk_sz_C);
  };
  /**
   * @}
   */


}
#endif // __CTR_COMM_H__
