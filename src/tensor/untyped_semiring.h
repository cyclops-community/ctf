#ifndef __INT_SEMIRING_H__
#define __INT_SEMIRING_H__

#include "../interface/common.h"
#include "cblas.h"

namespace CTF_int {

#if 0
  /**
   * \brief char * -based index-value pair used for tensor data input
   */
  class pair {
    public: 
      int64_t k;

      /**
       * \brief tensor value of this key-value pair is a char *
       */
      char * v;

      pair() {}

      /**
       * \brief compares key to other pair to determine which index appears first
       * \param[in] other pair to compare with
       * \return true if this key smaller than other's key
       */
      bool operator< (const pair& other) const{
        return k < other.k;
      }
  /*  bool operator==(const pair& other) const{
        return (k == other.k && d == other.d);
      }
      bool operator!=(const pair& other) const{
        return !(*this == other);
      }*/
      virtual int size() { assert(0); }
  };
#endif

  /**
   * \brief semirings defined the elementwise operations computed 
   *         in each tensor contraction
   */
  class semiring {
    public: 
      /** \brief size of each element of semiring in bytes */
      int el_size;
      /** \brief true if an additive inverse is provided */
      bool is_ring;
      /** \brief identity element for addition i.e. 0 */
      char * addid;
      /** \brief identity element for multiplication i.e. 1 */
      char * mulid;
      /** \brief MPI addition operation */
      MPI_Op addmop;
      /** \brief MPI datatype */
      MPI_Datatype mdtype;

      /** \brief gets pair size el_size plus the key size */
      int pair_size() const { return el_size + sizeof(int64_t); }

      /** \brief b = -a */
      void (*addinv)(char const * a, 
                     char * b);

      /** \brief c = a+b */
      void (*add)(char const * a, 
                  char const * b,
                  char *       c);
      
      /** \brief c = a*b */
      void (*mul)(char const * a, 
                  char const * b,
                  char *       c);

      /** \brief X["i"]=alpha*X["i"]; */
      void (*scal)(int          n,
                   char const * alpha,
                   char const * X,
                   int          incX);

      /** \brief Y["i"]+=alpha*X["i"]; */
      void (*axpy)(int          n,
                   char const * alpha,
                   char const * X,
                   int          incX,
                   char       * Y,
                   int          incY);

      /** \brief beta*C["ij"]=alpha*A^tA["ik"]*B^tB["kj"]; */
      void (*gemm)(char         tA,
                   char         tB,
                   int          m,
                   int          n,
                   int          k,
                   char const * alpha,
                   char const * A,
                   char const * B,
                   char const * beta,
                   char *       C);

    public:
      /**
       * \brief default constructor
       */
      semiring();


      /**
       * \brief copy constructor
       * \param[in] other another semiring to copy from
       */
      semiring(semiring const &other);

      /**
       * \brief constructor creates semiring with all parameters
       * \param[in] el_size number of bytes in each element in the semiring set
       * \param[in] addid additive identity element 
       *              (e.g. 0.0 for the (+,*) semiring over doubles)
       * \param[in] mulid multiplicative identity element 
       *              (e.g. 1.0 for the (+,*) semiring over doubles)
       * \param[in] addmop addition operation to pass to MPI reductions
       * \param[in] add function pointer to add c=a+b on semiring
       * \param[in] mul function pointer to multiply c=a*b on semiring
       * \param[in] addinv function pointer to additive inverse b=-a
       * \param[in] gemm function pointer to multiply blocks C, A, and B on semiring
       * \param[in] axpy function pointer to add A to B on semiring
       * \param[in] scal function pointer to scale A on semiring
       */
      semiring(    int          el_size, 
                   char const * addid,
                   char const * mulid,
                   MPI_Op       addmop,
                   void (*add )(char const * a,
                                char const * b,
                                char       * c),
                   void (*mul )(char const * a,
                                char const * b,
                                char       * c),
                   void (*addinv)(char const * a,
                                  char  * b)=NULL,
                   void (*gemm)(char         tA,
                                char         tB,
                                int          m,
                                int          n,
                                int          k,
                                char const * alpha,
                                char const * A,
                                char const * B,
                                char const * beta,
                                char *       C)=NULL,
                   void (*axpy)(int          n,
                                char const * alpha,
                                char const * X,
                                int          incX,
                                char       * Y,
                                int          incY)=NULL,
                   void (*scal)(int          n,
                                char const * alpha,
                                char const * X,
                                int          incX)=NULL);
      /**
       * \brief destructor frees addid and mulid
       */
      ~semiring();
      
      /** \brief returns true if semiring elements a and b are equal */
      bool isequal(char const * a, char const * b) const;
      
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
      semiring const * sr;
      char const * ptr;
      
      /** \brief conversion constructor for iterator to constant buffer of pairs */    
      ConstPairIterator(PairIterator const & pi);

      /** \brief constructor for iterator of constant buffer of pairs */    
      ConstPairIterator(semiring const * sr_, char const * ptr_);

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
      semiring const * sr;
      char * ptr;
    
      /** \brief constructor for iterator of buffer of pairs */    
      PairIterator(semiring const * sr_, char * ptr_);

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
