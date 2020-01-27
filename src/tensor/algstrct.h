#ifndef __INT_SEMIRING_H__
#define __INT_SEMIRING_H__

#include "../interface/common.h"

namespace CTF_int {

  class bivar_function;

  /**
   * \brief abstract class that knows how to add
   */
  class accumulatable {
    public:
      /** \brief size of each element of algstrct in bytes */
      int el_size;

      /** \brief b+=a */
      virtual void accum(char const * a, 
                         char *       b) const { assert(0); }

      /** \brief initialize n objects to zero
        * \param[in] n number of items
        * \param[in] arr array containing n items, to be set to zero
        */
      virtual void init_shell(int64_t n, char * arr) const { assert(0); };

  };

  /**
   * \brief algstrct (algebraic structure) defines the elementwise operations computed 
   *         in each tensor contraction, virtual classes defined in derived typed classes or algstrctcpy
   */
  class algstrct : public accumulatable {
    public: 
      /** \brief whether there was a custom COO CSRMM kernel provided for this algebraic structure */
      bool has_coo_ker;
      /** brief datatype for pairs, always custom create3d */
//      MPI_Datatype pmdtype;

      /** \brief b = max(a,addinv(a)) */
      void (*abs)(char const * a, 
                  char *       b);

      /** \brief gets pair size el_size plus the key size */
      virtual int pair_size() const { return el_size + sizeof(int64_t); }


      /**
       * \brief default constructor
       */
      algstrct(){ has_coo_ker = false; }

      /**
       * brief copy constructor
       * param[in] other another algstrct to copy from
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
      virtual ~algstrct();

      /**
       * \brief ''copy constructor''
       */
      virtual algstrct * clone() const = 0;
//        return new algstrct(el_size);

      virtual bool is_ordered() const = 0;

      /** \brief MPI addition operation for reductions */
      virtual MPI_Op addmop() const;

      /** \brief MPI datatype */
      virtual MPI_Datatype mdtype() const;
      
      /** \brief MPI datatype for pairs */
//      MPI_Datatype pair_mdtype();

      /** \brief identity element for addition i.e. 0 */
      virtual char const * addid() const;

      /** \brief identity element for multiplication i.e. 1 */
      virtual char const * mulid() const;

      /** \brief b = -a */
      virtual void addinv(char const * a, char * b) const;
      
      /** \brief b = -a, with checks for NULL and alloc as necessary */
      virtual void safeaddinv(char const * a, char *& b) const;

      /** \brief c = a+b */
      virtual void add(char const * a, 
                       char const * b,
                       char *       c) const;

      /** \brief b+=a */
      virtual void accum(char const * a, char * b) const;

      /** returns whether multiplication operator is present */
      virtual bool has_mul() const { return false; }
      
      /** \brief c = a*b */
      virtual void mul(char const * a, 
                       char const * b,
                       char *       c) const;

      /** \brief c = a*b, with NULL treated as mulid */
      virtual void safemul(char const * a, 
                           char const * b,
                           char *&      c) const;

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

      /** \brief return (int64_t)*c */
      virtual int64_t cast_to_int(char const * c) const;

      /** \brief return (double)*c */
      virtual double cast_to_double(char const * c) const;

      /** \brief prints the value */
      virtual void print(char const * a, FILE * fp=stdout) const;

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

      /** \brief beta*C["ijl"]=alpha*A^tA["ikl"]*B^tB["kjl"]; */
      virtual void gemm_batch(char         tA,
                              char         tB,
                              int          l,
                              int          m,
                              int          n,
                              int          k,
                              char const * alpha,
                              char const * A,
                              char const * B,
                              char const * beta,
                              char *       C)  const;

      virtual void offload_gemm(char         tA,
                                char         tB,
                                int          m,
                                int          n,
                                int          k,
                                char const * alpha,
                                char const * A,
                                char const * B,
                                char const * beta,
                                char *       C)  const;

      virtual bool is_offloadable() const;

      /** \brief sparse version of gemm using coordinate format for A */
      virtual void coomm(int                    m,
                         int                    n,
                         int                    k,
                         char const *           alpha,
                         char const *           A,
                         int const *            rows_A,
                         int const *            cols_A,
                         int64_t                nnz_A,
                         char const *           B,
                         char const *           beta,
                         char *                 C,
                         bivar_function const * func) const;

      /** \brief sparse version of gemm using CSR format for A */
      virtual void csrmm(int                    m,
                         int                    n,
                         int                    k,
                         char const *           alpha,
                         char const *           A,
                         int const *            JA,
                         int const *            IA,
                         int64_t                nnz_A,
                         char const *           B,
                         char const *           beta,
                         char *                 C,
                         bivar_function const * func) const;

      /** \brief sparse version of gemm using CCSR format for A */
      virtual void ccsrmm(int64_t         m,
                          int64_t         n,
                          int64_t         k,
                          int64_t         nnz_row,
                          char const *    alpha,
                          char const *    A,
                          int const *     JA,
                          int const *     IA,
                          int64_t const * row_enc,
                          int64_t         nnz_A,
                          char const *    B,
                          char const *    beta,
                          char *&         C,
                          CTF_int::bivar_function const * func) const;


      /** \brief sparse version of gemm using CSR format for A and B*/
      virtual void csrmultd
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *       C) const;
      virtual void csrmultcsr
                (int          m,
                 int          n,
                 int          k,
                 char const * alpha,
                 char const * A,
                 int const *  JA,
                 int const *  IA,
                 int64_t      nnz_A,
                 char const * B,
                 int const *  JB,
                 int const *  IB,
                 int64_t      nnz_B,
                 char const * beta,
                 char *&      C_CSR) const;

      /** \brief returns true if algstrct elements a and b are equal */
      virtual bool isequal(char const * a, char const * b) const;

      /** \brief converts coordinate sparse matrix layout to CSR layout */
      virtual void coo_to_csr(int64_t nz, int nrow, char * csr_vs, int * csr_cs, int * csr_rs, char const * coo_vs, int const * coo_rs, int const * coo_cs) const;
      
      /** \brief converts coordinate sparse matrix layout to CSR layout */
      virtual void coo_to_ccsr(int64_t nz, int64_t nnz_row, char * csr_vs, int * csr_cs, int * csr_rs, char const * coo_vs, int64_t const * coo_rs, int64_t const * coo_cs) const;

      /** \brief converts CSR sparse matrix layout to coordinate (COO) layout */
      virtual void csr_to_coo(int64_t nz, int nrow, char const * csr_vs, int const * csr_ja, int const * csr_ia, char * coo_vs, int * coo_rs, int * coo_cs) const;

      /** \brief converts CCSR sparse matrix layout to coordinate (COO) layout */
      virtual void ccsr_to_coo(int64_t nz, int64_t nnz_row, char const * csr_vs, int const * csr_ja, int const * csr_ia, int64_t const * row_enc, char * coo_vs, int64_t * coo_rs, int64_t * coo_cs) const;

      /** \brief adds CSR matrices A (stored in cA) and B (stored in cB) to create matric C (pointer to all_data returned), C data allocated internally */
      virtual char * csr_add(char * cA, char * cB, bool is_ccsr) const;

      /** \brief reduces CSR matrices stored in cA on each processor in cm and returns result on processor root */
      virtual char * csr_reduce(char * cA, int root, MPI_Comm cm, bool is_ccsr) const;

      /** \brief allocate space for n (int64_t,dtype) pairs, necessary for object types 
        * \param[in] n number of pairs
        * \return array containing n pair items
        */
      virtual char * pair_alloc(int64_t n) const;

      /** \brief allocate space for n items, necessary for object types 
        * \param[in] n number of items
        * \return array containing n items
        */
      virtual char * alloc(int64_t n) const;
     
      /** \brief gets key from pair */
      virtual int64_t get_key(char const * a) const;
      
      /** \brief gets pair to value from pair */
      virtual char * get_value(char * a) const;
      virtual char const * get_const_value(char const * a) const;
      /**
        * \brief deallocate given pointer containing contiguous array of values
        * \param[in] ptr array to deallocate
        */
      virtual void dealloc(char * ptr) const;
 
      /**
        * \brief deallocate given pointer containing contiguous array of pairs
        * \param[in] ptr array to deallocate
        */
      virtual void pair_dealloc(char * ptr) const;
      
      /** \brief initialize n objects to zero
        * \param[in] n number of items
        * \param[in] arr array containing n items, to be set to zero
        */
      virtual void init(int64_t n, char * arr) const;

      /**
       * \brief sorts n sets of pairs using std::sort 
       */
      virtual void sort(int64_t n, char * pairs) const;
 
      /** estimate time in seconds necessary for CSR reduction with input of size msg_sz */
      double estimate_csr_red_time(int64_t msg_sz, CommData const * cdt) const;

      /** \brief compute b=beta*b + alpha*a */
      void acc(char * b, char const * beta, char const * a, char const * alpha) const;
      
      /** \brief compute c=c + alpha*a*b */
      void accmul(char * c, char const * a, char const * b, char const * alpha) const;
      
      /** \brief copies element b to element a */
      virtual void copy(char * a, char const * b) const;
      
      /** \brief copies element b to element a, , with checks for NULL and alloc as necessary */
      void safecopy(char *& a, char const * b) const;
      
      /** \brief copies n elements from array b to array a */
      virtual void copy(char * a, char const * b, int64_t n) const;
      
      /** \brief copies n elements TO array b with increment inc_a FROM array a with increment inc_b */
      virtual void copy(int64_t n, char const * a, int inc_a, char * b, int inc_b) const;
      
      /** \brief copies m-by-n submatrix from a with lda_a to b with lda_b  */
      virtual void copy(
                int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char *       b,
                int64_t      lda_b) const;
      
      /** \brief copies m-by-n submatrix from a with lda_a and scaling alpha to b with lda_b and scaling by 1 */
      virtual void copy(
                int64_t      m,
                int64_t      n,
                char const * a,
                int64_t      lda_a,
                char const * alpha,
                char *       b,
                int64_t      lda_b,
                char const * beta) const;

      /** \brief copies pair b to element a */
      virtual void copy_pair(char * a, char const * b) const;
      
      /** \brief copies n pair from array b to array a */
      virtual void copy_pairs(char * a, char const * b, int64_t n) const;

      /** \brief sets n elements of array a to value b */
      virtual void set(char * a, char const * b, int64_t n) const;
      
      /** \brief sets 1 elements of pair a to value and key */
      virtual void set_pair(char * a, int64_t key, char const * vb) const;

      /** \brief sets n elements of array of pairs a to value b */
      virtual void set_pairs(char * a, char const * b, int64_t n) const;

      /** \brief accumulates tensor to local slice */
      virtual void accumulate_local_slice(int order,
                                          int64_t * lens,
                                          int64_t * lens_slice,
                                          int const * sym,
                                          int64_t const * offsets,
                                          int64_t const * ends,
                                          char const * slice_data,
                                          char const * alpha,
                                          char * tensor_data,
                                          char const * beta) const;

      /**
       * \brief returns function for elementwise comparison
       */
      virtual bivar_function * get_elementwise_smaller() const;

      /**
       * \brief returns function for elementwise comparison
       */
      virtual bivar_function * get_elementwise_smaller_or_equal() const;

      /**
       * \brief returns function for elementwise comparison
       */
      virtual bivar_function * get_elementwise_is_equal() const;

      /**
       * \brief returns function for elementwise comparison
       */
      virtual bivar_function * get_elementwise_is_not_equal() const;
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

      
      /**
       * \brief permutes keys of n pairs
       */
      void permute(int64_t n, int order, int64_t const * old_lens, int64_t const * new_lda, PairIterator wA);
      
      /**
       * \brief pins keys of n pairs
       */
      void pin(int64_t n, int order, int64_t const * lens, int const * divisor, PairIterator pi_new);
      

  };
  //http://stackoverflow.com/questions/630950/pure-virtual-destructor-in-c
  //inline algstrct::~algstrct(){}

  /**
   * \brief depins keys of n pairs
   */
  void depin(algstrct const * sr, int order, int64_t const * lens, int const * divisor, int nvirt, int const * virt_dim, int const * phys_rank, char * X, int64_t & new_nnz_B, int64_t * nnz_blk, char *& new_B, bool check_padding);

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
      char * d() const;
    
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

}


#endif
