#ifndef __INT_SEMIRING_H__
#define __INT_SEMIRING_H__

#include "../interface/common.h"
#include "cblas.h"

namespace CTF_int {
  /**
   * \brief algstrct (algebraic structure) defines the elementwise operations computed 
   *         in each tensor contraction, virtual classes defined in derived typed classes or algstrctcpy
   */
  class algstrct {
    public: 
      /** \brief size of each element of algstrct in bytes */
      int el_size;

      /** \brief b = \max(a,addinv(a)) */
      void (*abs)(char const * a, 
                  char *       b);

      /** \brief gets pair size el_size plus the key size */
      int pair_size() const { return el_size + sizeof(int64_t); }


      /**
       * \brief default constructor
       */
      algstrct(){};

      /**
       * \brief copy constructor
       * \param[in] other another algstrct to copy from
       */
      //algstrct(algstrct const &other);

      /**
       * \brief constructor creates algstrct with all parameters
       * \param[in] el_size number of bytes in each element in the algstrct set
             */
      algstrct(int el_size);

      /**
       * \brief destructor
       */
      ~algstrct();

      /**
       * \brief ''copy constructor''
       */
      virtual algstrct * clone() const {
        return new algstrct(el_size);
      }

      /** \brief MPI addition operation for reductions */
      virtual MPI_Op addmop() const;

      /** \brief MPI datatype (only used in reductions) */
      virtual MPI_Datatype mdtype() const;

      /** \brief identity element for addition i.e. 0 */
      virtual char const * addid() const;

      /** \brief identity element for multiplication i.e. 1 */
      virtual char const * mulid() const;

      /** \brief b = -a */
      virtual void addinv(char const * a, char * b) const;

      /** \brief c = a+b */
      virtual void add(char const * a, 
                       char const * b,
                       char *       c) const;
      
      /** \brief c = a*b */
      virtual void mul(char const * a, 
                       char const * b,
                       char *       c) const;

      /** \brief c = min(a,b) */
      virtual void min(char const * a, 
                       char const * b,
                       char *       c)  const;

      /** \brief c = max(a,b) */
      virtual void max(char const * a, 
                       char const * b,
                       char *       c)  const;

      /** \brief c = minimum possible value */
      virtual void min(char * c)  const;

      /** \brief c = maximum possible value */
      virtual void max(char * c)  const;

      /** \brief c = &i */
      virtual void cast_int(int64_t i, char * c) const;

      /** \brief c = &d */
      virtual void cast_double(double d, char * c) const;

      /** \brief X["i"]=alpha*X["i"]; */
      virtual void scal(int          n,
                        char const * alpha,
                        char       * X,
                        int          incX)  const;

      /** \brief Y["i"]+=alpha*X["i"]; */
      virtual void axpy(int          n,
                        char const * alpha,
                        char const * X,
                        int          incX,
                        char       * Y,
                        int          incY)  const;

      /** \brief beta*C["ij"]=alpha*A^tA["ik"]*B^tB["kj"]; */
      virtual void gemm(char         tA,
                        char         tB,
                        int          m,
                        int          n,
                        int          k,
                        char const * alpha,
                        char const * A,
                        char const * B,
                        char const * beta,
                        char *       C)  const;

      /** \brief returns true if algstrct elements a and b are equal */
      bool isequal(char const * a, char const * b) const;
    
      /** \brief compute b=beta*b + alpha*a */
      void acc(char * b, char const * beta, char const * a, char const * alpha) const;
      
      /** \brief copies element b to element a */
      void copy(char * a, char const * b) const;
      
      /** \brief copies n elements from array b to array a */
      void copy(char * a, char const * b, int64_t n) const;
      
      /** \brief copies n elements TO array b with increment inc_a FROM array a with increment inc_b */
      void copy(int64_t n, char const * a, int64_t inc_a, char * b, int64_t inc_b) const;
      
      /** \brief copies m-by-n submatrix from a with lda_a to b with lda_b  */
      void copy(int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char *       b,
                int64_t      lda_b) const;
      
      /** \brief copies m-by-n submatrix from a with lda_a and scaling alpha to b with lda_b and scaling by 1 */
      void copy(int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char const * alpha,
                char *       b,
                int64_t      lda_b,
                char const * beta) const;
      
      /** \brief copies pair b to element a */
      void copy_pair(char * a, char const * b) const;
      
      /** \brief copies n pair from array b to array a */
      void copy_pairs(char * a, char const * b, int64_t n) const;

      /** \brief sets n elements of array a to value b */
      void set(char * a, char const * b, int64_t n) const;
      
      /** \brief sets 1 elements of pair a to value and key */
      void set_pair(char * a, int64_t key, char const * vb) const;

      /** \brief sets n elements of array of pairs a to value b */
      void set_pairs(char * a, char const * b, int64_t n) const;
      
      /** \brief gets key from pair */
      int64_t get_key(char const * a) const;
      
      /** \brief gets pair to value from pair */
      char const * get_value(char const * a) const;
  };

  class PairIterator;

  class ConstPairIterator {
    public:
      algstrct const * sr;
      char const * ptr;
      
      /** \brief conversion constructor for iterator to constant buffer of pairs */    
      ConstPairIterator(PairIterator const & pi);

      /** \brief constructor for iterator of constant buffer of pairs */    
      ConstPairIterator(algstrct const * sr_, char const * ptr_);

      /** \brief indexing moves by \param[in] n pairs */
      ConstPairIterator operator[](int n) const;

      /** \brief returns key of pair at head of ptr */
      int64_t k() const;
    
      /** \brief returns value of pair at head of ptr */
      char const * d() const;
    
      /** 
       * \brief sets data to what this operator points to
       * \param[in,out] buf data to set 
       * \param[in] n number of pairs to set
       */
      void read(char * buf, int64_t n=1) const;
      
      /** 
       * \brief sets value to the value pointed by the iterator
       * \param[in,out] buf pair to set
       */
      void read_val(char * buf) const;
  };


  class PairIterator {
    public:
      algstrct const * sr;
      char * ptr;
    
      /** \brief constructor for iterator of buffer of pairs */    
      PairIterator(algstrct const * sr_, char * ptr_);

      /** \brief indexing moves by \param[in] n pairs */
      PairIterator operator[](int n) const;

      /** \brief returns key of pair at head of ptr */
      int64_t k() const;
    
      /** \brief returns value of pair at head of ptr */
      char const * d() const;
    
      /** 
       * \brief sets external data to what this operator points to
       * \param[in,out] buf data to set 
       * \param[in] n number of pairs to set
       */
      void read(char * buf, int64_t n=1) const;
      
      /** 
       * \brief sets external value to the value pointed by the iterator
       * \param[in,out] buf pair to set
       */
      void read_val(char * buf) const;

      /** 
       * \brief sets internal pairs to provided data
       * \param[in] buf provided data to copy from
       * \param[in] n number of pairs to set
       */
      void write(char const * buf, int64_t n=1);

      /** 
       * \brief sets internal pairs to data from another iterator
       * \param[in] iter to copy from
       * \param[in] n number of pairs to set
       */
      void write(PairIterator const iter, int64_t n=1);

      /** 
       * \brief sets internal pairs to data from another iterator
       * \param[in] iter to copy from
       * \param[in] n number of pairs to set
       */
      void write(ConstPairIterator const iter, int64_t n=1);

      /** 
       * \brief sets value of head pair to what is in buf
       * \param[in] buf value to read into iterator head
       */
      void write_val(char const * buf);

      /** 
       * \brief sets key of head pair to key
       * \param[in] key to set
       */
      void write_key(int64_t key);

      /**
       * \brief sorts set of pairs using std::sort 
       */
      void sort(int64_t n);
      
      /**
       * \brief searches for pair op via std::lower_bound
       */
      int64_t lower_bound(int64_t n, ConstPairIterator op);
  };

  void sgemm(char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             float          alpha,
             float  const * A,
             float  const * B,
             float          beta,
             float  *       C);

  void dgemm(char           tA,
             char           tB,
             int            m,
             int            n,
             int            k,
             double         alpha,
             double const * A,
             double const * B,
             double         beta,
             double *       C);

  void cgemm(char                        tA,
             char                        tB,
             int                         m,
             int                         n,
             int                         k,
             std::complex<float>         alpha,
             std::complex<float> const * A,
             std::complex<float> const * B,
             std::complex<float>         beta,
             std::complex<float> *       C);

  void zgemm(char                         tA,
             char                         tB,
             int                          m,
             int                          n,
             int                          k,
             std::complex<double>         alpha,
             std::complex<double> const * A,
             std::complex<double> const * B,
             std::complex<double>         beta,
             std::complex<double> *       C);

}

#endif
