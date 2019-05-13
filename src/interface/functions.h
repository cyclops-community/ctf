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
       * \brief apply function f to value stored at a
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] b result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char * b) const { ((dtype_B*)b)[0]=f(((dtype_A*)a)[0]); }
      
      /**
       * \brief compute b = b+f(a)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] b result &f(*a) of applying f on value of (different type) on a
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
       * \brief apply function f to value stored at a, for an accumulator, this is the same as acc_f below
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in,out] b result &f(*a) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char * b) const { acc_f(a,b,NULL); }

       /**
       * \brief compute f(a,b)
       * \param[in] a pointer to the accumulated operand 
       * \param[in,out] b value that is accumulated to
       * \param[in] sr_B algebraic structure for b, here is ignored
       */
      void acc_f(char const * a, char * b, CTF_int::algstrct const * sr_B) const {
        f(((dtype_A*)a)[0], ((dtype_B*)b)[0]);
      }

      bool is_accumulator() const { return true; }
  };


  /**
   * \brief custom bivariate function on two tensors: 
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
       * \param[in] f_ bivariate function (type_A,type_B)->(type_C)
       */
      Bivar_Function(std::function<dtype_C (dtype_A, dtype_B)> f_)
        : CTF_int::bivar_function(){
        f=f_; commutative=0; 
      }
      
      /**
       * \brief constructor takes function pointers to compute C=f(A,B);
       * \param[in] f_ bivariate function (type_A,type_B)->(type_C)
       * \param[in] is_comm whether function is commutative
       */
      Bivar_Function(std::function<dtype_C (dtype_A, dtype_B)> f_, 
                     bool                                      is_comm)
        : CTF_int::bivar_function(is_comm){
        f=f_;
      }

      /**
       * \brief default constructor sets function pointer to NULL
       */
      Bivar_Function();


      /**
       * \brief compute c = f(a,b)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to operand that will be cast to dtype 
       * \param[in,out] c result c+f(*a,b) of applying f on value of (different type) on a
       */
      void apply_f(char const * a, char const * b, char * c) const { 
        ((dtype_C*)c)[0] = f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]); 
      }

      /**
       * \brief compute c = c+ f(a,b)
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to operand that will be cast to dtype 
       * \param[in,out] c result c+f(*a,b) of applying f on value of (different type) on a
       * \param[in] sr_C algebraic structure for b, needed to do add
       */
      void acc_f(char const * a, char const * b, char * c, CTF_int::algstrct const * sr_C) const { 
        dtype_C tmp;
        tmp = f(((dtype_A const*)a)[0],((dtype_B const*)b)[0]);
        sr_C->add(c, (char const *)&tmp, c); 
      }


      // FIXME: below kernels replicate code from src/interface/semiring.h
      void csrmm(int              m,
                 int              n,
                 int              k,
                 dtype_A const *  A,
                 int const *      JA,
                 int const *      IA,
                 int64_t          nnz_A,
                 dtype_B const *  B,
                 dtype_C *        C,
                 CTF_int::algstrct const * sr_C) const {
        //TAU_FSTART(3type_csrmm);
  #ifdef _OPENMP
        #pragma omp parallel for
  #endif
        for (int row_A=0; row_A<m; row_A++){
  #ifdef _OPENMP
          #pragma omp parallel for
  #endif
          for (int col_B=0; col_B<n; col_B++){
            for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
              int col_A = JA[i_A]-1;
              dtype_C tmp = f(A[i_A],B[col_B*k+col_A]);
              sr_C->add((char const *)&C[col_B*m+row_A],(char const*)&tmp,(char *)&C[col_B*m+row_A]);

            }
          }
        }
        //TAU_FSTOP(3type_csrmm);
      }


      void csrmultd
            (int              m,
             int              n,
             int              k,
             dtype_A const *  A,
             int const *      JA,
             int const *      IA,
             int64_t          nnz_A,
             dtype_B const *  B,
             int const *      JB,
             int const *      IB,
             int64_t          nnz_B,
             dtype_C *        C,
             CTF_int::algstrct const * sr_C) const {
  #ifdef _OPENMP
        #pragma omp parallel for
  #endif
        for (int row_A=0; row_A<m; row_A++){
          for (int i_A=IA[row_A]-1; i_A<IA[row_A+1]-1; i_A++){
            int row_B = JA[i_A]-1; //=col_A
            for (int i_B=IB[row_B]-1; i_B<IB[row_B+1]-1; i_B++){
              int col_B = JB[i_B]-1;
              dtype_C tmp = f(A[i_A],B[i_B]);
              sr_C->add((char const*)&C[col_B*m+row_A],(char const*)&tmp,(char *)&C[col_B*m+row_A]);
            }
          }
        }
      }


      void csrmultcsr
                (int              m,
                 int              n,
                 int              k,
                 dtype_A const *  A,
                 int const *      JA,
                 int const *      IA,
                 int64_t          nnz_A,
                 dtype_B const *  B,
                 int const *      JB,
                 int const *      IB,
                 int64_t          nnz_B,
                 char *&          C_CSR,
                 CTF_int::algstrct const * sr_C) const {
        int * IC = (int*)CTF_int::alloc(sizeof(int)*(m+1));
        int * has_col = (int*)CTF_int::alloc(sizeof(int)*n);
        IC[0] = 1;
        for (int i=0; i<m; i++){
          memset(has_col, 0, sizeof(int)*n);
          IC[i+1] = IC[i];
          CTF_int::CSR_Matrix::compute_has_col(JA, IA, JB, IB, i, has_col);
          for (int j=0; j<n; j++){
            IC[i+1] += has_col[j];
          }
        }
        CTF_int::CSR_Matrix C(IC[m]-1, m, n, sr_C);
        dtype_C * vC = (dtype_C*)C.vals();
        int * JC = C.JA();
        memcpy(C.IA(), IC, sizeof(int)*(m+1));
        CTF_int::cdealloc(IC);
        IC = C.IA();
        int64_t * rev_col = (int64_t*)CTF_int::alloc(sizeof(int64_t)*n);
        for (int i=0; i<m; i++){
          memset(has_col, 0, sizeof(int)*n);
          CTF_int::CSR_Matrix::compute_has_col(JA, IA, JB, IB, i, has_col);
          int vs = 0;
          for (int j=0; j<n; j++){
            if (has_col[j]){
              JC[IC[i]+vs-1] = j+1;
              rev_col[j] = IC[i]+vs-1;
              vs++;
            }
          }
          memset(has_col, 0, sizeof(int)*n);
          for (int j=0; j<IA[i+1]-IA[i]; j++){
            int row_B = JA[IA[i]+j-1]-1;
            int idx_A = IA[i]+j-1;
            for (int l=0; l<IB[row_B+1]-IB[row_B]; l++){
              int idx_B = IB[row_B]+l-1;
              if (has_col[JB[idx_B]-1]){
                dtype_C tmp = f(A[idx_A],B[idx_B]);
                sr_C->add((char const *)&vC[rev_col[JB[idx_B]-1]], (char const *)&tmp, (char *)&vC[rev_col[JB[idx_B]-1]]);  
              } else {
                vC[rev_col[JB[idx_B]-1]] = f(A[idx_A],B[idx_B]);
              }
              has_col[JB[idx_B]-1] = 1;  
            }
          }
        }
        CTF_int::CSR_Matrix C_in(C_CSR);
        if (C_CSR == NULL || C_in.nnz() == 0){
          C_CSR = C.all_data;
        } else {
          char * ans = CTF_int::CSR_Matrix::csr_add(C_CSR, C.all_data, sr_C);
          CTF_int::cdealloc(C.all_data);
          C_CSR = ans;
        }
        CTF_int::cdealloc(has_col);
        CTF_int::cdealloc(rev_col);
      }

      void fcsrmm(int              m,
                  int              n,
                  int              k,
                  char const *     A,
                  int const *      JA,
                  int const *      IA,
                  int64_t          nnz_A,
                  char const *     B,
                  char *           C,
                  CTF_int::algstrct const * sr_C) const {
        csrmm(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B, (dtype_C *)C, sr_C);
      }

      void fcsrmultd
                   (int              m,
                    int              n,
                    int              k,
                    char const *     A,
                    int const *      JA,
                    int const *      IA,
                    int64_t          nnz_A,
                    char const *     B,
                    int const *      JB,
                    int const *      IB,
                    int64_t          nnz_B,
                    char *           C,
                    CTF_int::algstrct const * sr_C) const {
        csrmultd(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B,JB,IB,nnz_B,(dtype_C *)C,sr_C);
      }

      void fcsrmultcsr
                (int              m,
                 int              n,
                 int              k,
                 char const *     A,
                 int const *      JA,
                 int const *      IA,
                 int64_t          nnz_A,
                 char const *     B,
                 int const *      JB,
                 int const *      IB,
                 int64_t          nnz_B,
                 char *&          C_CSR,
                 CTF_int::algstrct const * sr_C) const {
        csrmultcsr(m,n,k,(dtype_A const *)A,JA,IA,nnz_A,(dtype_B const *)B, JB, IB, nnz_B, C_CSR, sr_C);
      }




  };

  /**
   * \brief custom function f : (X * Y * Z) -> Z applied on three tensors as contraction: 
   *          e.g. f(A["ij"],B["ij"],C["ij"])
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
      Bivar_Transform(std::function<void(dtype_A, dtype_B, dtype_C &)> f_)
        : CTF_int::bivar_function() {
        f = f_; 
      }

      /**
       * \brief constructor takes function pointers to compute C=f(A,B);
       * \param[in] f_ bivariate function (type_A,type_B)->(type_C)
       * \param[in] is_comm whether function is commutative
       */
      Bivar_Transform(std::function<void(dtype_A, dtype_B, dtype_C &)> f_,
                      bool                                             is_comm)
        : CTF_int::bivar_function(is_comm){
        f=f_; 
      }

       /**
       * \brief compute f(a,b)
       * \param[in] a pointer to first operand 
       * \param[in] b pointer to second operand 
       * \param[in,out] c value that is accumulated to
       * \param[in] sr_B algebraic structure for b, here is ignored
       */
      void acc_f(char const * a, char const * b, char * c, CTF_int::algstrct const * sr_B) const {
        f(((dtype_A*)a)[0], ((dtype_B*)b)[0], ((dtype_C*)c)[0]);
      }
      
      /**
       * \brief apply function f to value stored at a, for an accumulator, this is the same as acc_f below
       * \param[in] a pointer to operand that will be cast to dtype 
       * \param[in] b pointer to second operand that will be cast to dtype 
       * \param[in,out] c result &f(*a,*b) of applying f on value of (different type) on a
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

 
      Function(std::function<dtype_C(dtype_A,dtype_B)> f_, bool is_comm=false){
        is_univar = false;
        is_bivar = true;
        bivar = new Bivar_Function<dtype_A, dtype_B, dtype_C>(f_,is_comm);
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

