/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __tCTF_HPP__
#define __tCTF_HPP__

#define CTF_VERSION 100

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <deque>
#include <set>
#include <map>
#include "../src/dist_tensor/cyclopstf.hpp"

/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
#if (!defined NS && !defined SY && !defined SH)
#define NS 0
#define SY 1
#define AS 2
#define SH 3
#endif

typedef long_int int64_t;

template<typename dtype> class tCTF_fscl;
template<typename dtype> class tCTF_fsum;
template<typename dtype> class tCTF_fctr;
template<typename dtype> class tCTF_Idx_Tensor;
template<typename dtype> class tCTF_Sparse_Tensor;
template<typename dtype> class tCTF_Term;
template<typename dtype> class tCTF_Sum_Term;
template<typename dtype> class tCTF_Contract_Term;

/**
 * \defgroup CTF CTF: main C++ interface
 * @{
 */


/**
 * \brief reduction types for tensor data (enum actually defined in ../src/dist_tensor/cyclopstf.hpp)
 */
//enum CTF_OP { CTF_OP_SUM, CTF_OP_SUMABS, CTF_OP_SQNRM2,
//              CTF_OP_MAX, CTF_OP_MIN, CTF_OP_MAXABS, CTF_OP_MINABS };

/**
 * \brief an instance of the tCTF library (world) on a MPI communicator
 */
template<typename dtype>
class tCTF_World {
  public:
    MPI_Comm comm;
    tCTF<dtype> * ctf;

  public:
    /**
     * \brief creates tCTF library on comm_ that can output profile data 
     *        into a file with a name based on the main args
     * \param[in] comm_ MPI communicator associated with this CTF instance
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    tCTF_World(int argc, char * const * argv);

    /**
     * \brief creates tCTF library on comm_ that can output profile data 
     *        into a file with a name based on the main args
     * \param[in] comm_ MPI communicator associated with this CTF instance
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    tCTF_World(MPI_Comm       comm_ = MPI_COMM_WORLD,
               int            argc = 0,
               char * const * argv = NULL);

    /**
     * \brief creates tCTF library on comm_
     * \param[in] ndim number of torus network dimensions
     * \param[in] lens lengths of torus network dimensions
     * \param[in] comm MPI global context for this CTF World
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    tCTF_World(int            ndim, 
               int const *    lens, 
               MPI_Comm       comm_ = MPI_COMM_WORLD,
               int            argc = 0,
               char * const * argv = NULL);

    /**
     * \brief frees tCTF library
     */
    ~tCTF_World();
};



/**
 * \brief an instance of a tensor within a tCTF world
 */
template<typename dtype>
class tCTF_Tensor {
  public:
    int tid, ndim;
    int * sym, * len;
    char * idx_map;
    char const * name;
    tCTF_World<dtype> * world;

  public:
    /**
     * \breif default constructor
     */
    tCTF_Tensor();

    /**
     * \brief copies a tensor (setting data to zero or copying A)
     * \param[in] A tensor to copy
     * \param[in] copy whether to copy the data of A into the new tensor
     */
    tCTF_Tensor(tCTF_Tensor const &   A,
                bool                  copy = true);

    /**
     * \brief copies a tensor filled with zeros
     * \param[in] ndim number of dimensions of tensor
     * \param[in] len edge lengths of tensor
     * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
     * \param[in] world_ a world for the tensor to live in
     * \param[in] name an optionary name for the tensor
     * \param[in] profile set to 1 to profile contractions involving this tensor
     */
    tCTF_Tensor(int                  ndim_,
                int const *          len_,
                int const *          sym_,
                tCTF_World<dtype> &  world_,
#if DEBUG < 3
                char const *  name_ = NULL,
                int           profile_ = 0
#else
                char const *  name_ = "X",
                int           profile_ = 1
#endif
                 );
    
    /**
     * \brief creates a zeroed out copy (data not copied) of a tensor in a different world
     * \param[in] A tensor whose characteristics to copy
     * \param[in] world_ a world for the tensor we are creating to live in, can be different from A
     */
    tCTF_Tensor(tCTF_Tensor const & A,
                tCTF_World<dtype> & world_);

    /**
     * \brief gives the values associated with any set of indices
     * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
     * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
     * and the column index is second for matrices, which means they are column major. 
     * \param[in] npair number of values to fetch
     * \param[in] global_idx index within global tensor of each value to fetch
     * \param[in,out] data a prealloced pointer to the data with the specified indices
     */
    void read(long_int          npair, 
              long_int const *  global_idx, 
              dtype *           data) const;
    
    /**
     * \brief gives the values associated with any set of indices
     * \param[in] npair number of values to fetch
     * \param[in,out] pairs a prealloced pointer to key-value pairs
     */
    void read(long_int          npair,
              tkv_pair<dtype> * pairs) const;
    
    /**
     * \brief sparse read: A[global_idx[i]] = alpha*A[global_idx[i]]+beta*data[i]
     * \param[in] npair number of values to read into tensor
     * \param[in] alpha scaling factor on read data
     * \param[in] beta scaling factor on value in initial values vector
     * \param[in] global_idx global index within tensor of value to add
     * \param[in] data values to add to the tensor
     */
    void read(long_int         npair, 
              dtype            alpha, 
              dtype            beta,
              long_int const * global_idx,
              dtype *          data) const;

    /**
     * \brief sparse read: pairs[i].d = alpha*A[pairs[i].k]+beta*pairs[i].d
     * \param[in] npair number of values to read into tensor
     * \param[in] alpha scaling factor on read data
     * \param[in] beta scaling factor on value in initial pairs vector
     * \param[in] pairs key-value pairs to add to the tensor
     */
    void read(long_int          npair,
              dtype             alpha,
              dtype             beta,
              tkv_pair<dtype> * pairs) const;
   

    /**
     * \brief writes in values associated with any set of indices
     * The sparse data is defined in coordinate format. The tensor index (i,j,k,l) of a tensor with edge lengths
     * {m,n,p,q} is associated with the global index g via the formula g=i+j*m+k*m*n+l*m*n*p. The row index is first
     * and the column index is second for matrices, which means they are column major. 
     * \param[in] npair number of values to write into tensor
     * \param[in] global_idx global index within tensor of value to write
     * \param[in] data values to  write to the indices
     */
    void write(long_int         npair, 
               long_int const * global_idx, 
               dtype const    * data);

    /**
     * \brief writes in values associated with any set of indices
     * \param[in] npair number of values to write into tensor
     * \param[in] pairs key-value pairs to write to the tensor
     */
    void write(long_int                 npair,
               tkv_pair<dtype> const *  pairs);
    
    /**
     * \brief sparse add: A[global_idx[i]] = beta*A[global_idx[i]]+alpha*data[i]
     * \param[in] npair number of values to write into tensor
     * \param[in] alpha scaling factor on value to add
     * \param[in] beta scaling factor on original data
     * \param[in] global_idx global index within tensor of value to add
     * \param[in] data values to add to the tensor
     */
    void write(long_int         npair, 
               dtype            alpha, 
               dtype            beta,
               long_int const * global_idx,
               dtype const *    data);

    /**
     * \brief sparse add: A[pairs[i].k] = alpha*A[pairs[i].k]+beta*pairs[i].d
     * \param[in] npair number of values to write into tensor
     * \param[in] alpha scaling factor on value to add
     * \param[in] beta scaling factor on original data
     * \param[in] pairs key-value pairs to add to the tensor
     */
    void write(long_int                npair,
               dtype                   alpha,
               dtype                   beta,
               tkv_pair<dtype> const * pairs);
   
    /**
     * \brief contracts C[idx_C] = beta*C[idx_C] + alpha*A[idx_A]*B[idx_B]
     *        if fseq defined computes fseq(alpha,A[idx_A],B[idx_B],beta*C[idx_C])
     * \param[in] alpha A*B scaling factor
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \param[in] beta C scaling factor
     * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
     * \param[in] fseq sequential operation to execute, default is multiply-add
     */
    void contract(dtype                    alpha, 
                  const tCTF_Tensor&       A, 
                  char const *             idx_A,
                  const tCTF_Tensor&       B, 
                  char const *             idx_B,
                  dtype                    beta,
                  char const *             idx_C,
                  tCTF_fctr<dtype>         fseq = tCTF_fctr<dtype>());

    /**
     * \brief estimate the cost of a contraction C[idx_C] = A[idx_A]*B[idx_B]
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
     * \return cost as a int64_t type, currently a rought estimate of flops/processor
     */
    int64_t estimate_cost(const tCTF_Tensor & A,
                          char const *        idx_A,
                          const tCTF_Tensor & B,
                          char const *        idx_B,
                          char const *        idx_C);
    
    /**
     * \brief estimate the cost of a sum B[idx_B] = A[idx_A]
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \return cost as a int64_t type, currently a rought estimate of flops/processor
     */
    int64_t estimate_cost(const tCTF_Tensor & A,
                          char const *        idx_A,
                          char const *        idx_B);


    
    /**
     * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
     *        if fseq defined computes fseq(alpha,A[idx_A],beta*B[idx_B])
     * \param[in] alpha A scaling factor
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
     * \param[in] beta B scaling factor
     * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
     * \param[in] fseq sequential operation to execute, default is multiply-add
     */
    void sum(dtype                   alpha, 
             const tCTF_Tensor&      A, 
             char const *            idx_A,
             dtype                   beta,
             char const *            idx_B,
             tCTF_fsum<dtype>        fseq = tCTF_fsum<dtype>());
    
    /**
     * \brief scales A[idx_A] = alpha*A[idx_A]
     *        if fseq defined computes fseq(alpha,A[idx_A])
     * \param[in] alpha A scaling factor
     * \param[in] idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
     * \param[in] fseq sequential operation to execute, default is multiply-add
     */
    void scale(dtype                   alpha, 
               char const *            idx_A,
               tCTF_fscl<dtype>        fseq = tCTF_fscl<dtype>());

    /**
     * \brief cuts out a slice (block) of this tensor A[offsets,ends)
     * \param[in] offsets bottom left corner of block
     * \param[in] ends top right corner of block
     * \return new tensor corresponding to requested slice
     */
    tCTF_Tensor slice(int const * offsets,
                      int const * ends) const;
    
    tCTF_Tensor slice(long_int corner_off,
                      long_int corner_end) const;
    
    /**
     * \brief cuts out a slice (block) of this tensor A[offsets,ends)
     * \param[in] offsets bottom left corner of block
     * \param[in] ends top right corner of block
     * \return new tensor corresponding to requested slice which lives on
     *          oworld
     */
    tCTF_Tensor slice(int const *         offsets,
                      int const *         ends,
                      tCTF_World<dtype> * oworld) const;

    tCTF_Tensor slice(long_int            corner_off,
                      long_int            corner_end,
                      tCTF_World<dtype> * oworld) const;
    
    
    /**
     * \brief cuts out a slice (block) of this tensor = B
     *   B[offsets,ends)=beta*B[offsets,ends) + alpha*A[offsets_A,ends_A)
     * \param[in] offsets bottom left corner of block
     * \param[in] ends top right corner of block
     * \param[in] alpha scaling factor of this tensor
     * \param[in] offsets bottom left corner of block of A
     * \param[in] ends top right corner of block of A
     * \param[in] alpha scaling factor of tensor A
     */
    void slice(int const *    offsets,
               int const *    ends,
               dtype          beta,
               tCTF_Tensor const & A,
               int const *    offsets_A,
               int const *    ends_A,
               dtype          alpha) const;
    
    void slice(long_int       corner_off,
               long_int       corner_end,
               dtype          beta,
               tCTF_Tensor const & A,
               long_int       corner_off_A,
               long_int       corner_end_A,
               dtype          alpha) const;

    /**
     * \brief Apply permutation to matrix, potentially extracting a slice
     *              B[i,j,...] 
     *                = beta*B[...] + alpha*A[perms_A[0][i],perms_A[1][j],...]
     *
     * \param[in] beta scaling factor for values of tensor B (this)
     * \param[in] A specification of operand tensor A must live on 
                    the same CTF_World or a subset of the CTF_World on which B lives
     * \param[in] perms_A specifies permutations for tensor A, e.g. A[perms_A[0][i],perms_A[1][j]]
     *                    if a subarray NULL, no permutation applied to this index,
     *                    if an entry is -1, the corresponding entries of the tensor are skipped 
                            (A must then be smaller than B)
     * \param[in] alpha scaling factor for A tensor
     */
    void permute(dtype          beta,
                 tCTF_Tensor &  A,
                 int * const *  perms_A,
                 dtype          alpha);

    /**
     * \brief Apply permutation to matrix, potentially extracting a slice
     *              B[perms_B[0][i],perms_B[0][j],...] 
     *                = beta*B[...] + alpha*A[i,j,...]
     *
     * \param[in] perms_B specifies permutations for tensor B, e.g. B[perms_B[0][i],perms_B[1][j]]
     *                    if a subarray NULL, no permutation applied to this index,
     *                    if an entry is -1, the corresponding entries of the tensor are skipped 
     *                       (A must then be smaller than B)
     * \param[in] beta scaling factor for values of tensor B (this)
     * \param[in] A specification of operand tensor A must live on 
                    the same CTF_World or a superset of the CTF_World on which B lives
     * \param[in] alpha scaling factor for A tensor
     */
    void permute(int * const *  perms_B,
                 dtype          beta,
                 tCTF_Tensor &  A,
                 dtype          alpha);
    
   /**
     * \brief accumulates this tensor to a tensor object defined on a different world
     * \param[in] tsr a tensor object of the same characteristic as this tensor, 
     *             but on a different CTF_world/MPI_comm
     * \param[in] alpha scaling factor for this tensor (default 1.0)
     * \param[in] beta scaling factor for tensor tsr (default 1.0)
     */
    void add_to_subworld(tCTF_Tensor<dtype> * tsr,
                         dtype alpha,
                         dtype beta) const;
    void add_to_subworld(tCTF_Tensor<dtype> * tsr) const;
    
   /**
     * \brief accumulates this tensor from a tensor object defined on a different world
     * \param[in] tsr a tensor object of the same characteristic as this tensor, 
     *             but on a different CTF_world/MPI_comm
     * \param[in] alpha scaling factor for tensor tsr (default 1.0)
     * \param[in] beta scaling factor for this tensor (default 1.0)
     */
    void add_from_subworld(tCTF_Tensor<dtype> * tsr,
                           dtype alpha,
                           dtype beta) const;
    void add_from_subworld(tCTF_Tensor<dtype> * tsr) const;
    

    /**
     * \brief aligns data mapping with tensor A
     * \param[in] A align with this tensor
     */
    void align(tCTF_Tensor const & A);

    /**
     * \brief performs a reduction on the tensor
     * \param[in] op reduction operation (see top of this cyclopstf.hpp for choices)
     */    
    dtype reduce(CTF_OP op);
    
    /**
     * \brief computes the entrywise 1-norm of the tensor
     */    
    dtype norm1(){ return reduce(CTF_OP_NORM1); };

    /**
     * \brief computes the frobenius norm of the tensor
     */    
    dtype norm2(){ return reduce(CTF_OP_NORM2); };

    /**
     * \brief finds the max absolute value element of the tensor
     */    
    dtype norm_infty(){ return reduce(CTF_OP_MAXABS); };

    /**
     * \brief gives the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to local data
     */
    dtype * get_raw_data(long_int * size);

    /**
     * \brief gives a read-only copy of the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to read-only copy of local data
     */
    const dtype * raw_data(long_int * size) const;

    /**
     * \brief gives the global indices and values associated with the local data
     * \param[out] npair number of local values
     * \param[out] global_idx index within global tensor of each data value
     * \param[out] data pointer to local values in the order of the indices
     */
    void read_local(long_int *   npair, 
                    long_int **  global_idx, 
                    dtype **     data) const;

    /**
     * \brief gives the global indices and values associated with the local data
     * \param[out] npair number of local values
     * \param[out] pairs pointer to local key-value pairs
     */
    void read_local(long_int *         npair,
                    tkv_pair<dtype> ** pairs) const;

    /**
     * \brief collects the entire tensor data on each process (not memory scalable)
     * \param[out] npair number of values in the tensor
     * \param[out] data pointer to the data of the entire tensor
     */
    void read_all(long_int * npair, 
                  dtype **   data) const;
    
    /**
     * \brief collects the entire tensor data on each process (not memory scalable)
     * \param[in,out] preallocated data pointer to the data of the entire tensor
     */
    long_int read_all(dtype * data) const;

    /**
     * \brief obtains a small number of the biggest elements of the 
     *        tensor in sorted order (e.g. eigenvalues)
     * \param[in] n number of elements to collect
     * \param[in] data output data (should be preallocated to size at least n)
     *
     * WARNING: currently functional only for dtype=double
     */
    void get_max_abs(int        n,
                     dtype *    data);

    /**
     * \brief turns on profiling for tensor
     */
    void profile_on();
    
    /**
     * \brief turns off profiling for tensor
     */
    void profile_off();

    /**
     * \brief sets tensor name
     * \param[in] name new for tensor
     */
    void set_name(char const * name);

    /**
     * \brief sets all values in the tensor to val
     */
    tCTF_Tensor& operator=(dtype val);
    
    /**
     * \brief sets the tensor
     */
    void operator=(tCTF_Tensor<dtype> A);
    
    /**
     * \brief associated an index map with the tensor for future operation
     * \param[in] idx_map_ index assignment for this tensor
     */
    tCTF_Idx_Tensor<dtype> operator[](char const * idx_map_);
    
    /**
     * \brief gives handle to sparse index subset of tensors
     * \param[in] indices, vector of indices to sparse tensor
     */
    tCTF_Sparse_Tensor<dtype> operator[](std::vector<long_int> indices);
    
    /**
     * \brief prints tensor data to file using process 0
     * \param[in] fp file to print to e.g. stdout
     * \param[in] cutoff do not print values of absolute value smaller than this
     */
    void print(FILE * fp = stdout, double cutoff = -1.0) const;

    /**
     * \brief prints two sets of tensor data side-by-side to file using process 0
     * \param[in] fp file to print to e.g. stdout
     * \param[in] A tensor to compare against
     * \param[in] cutoff do not print values of absolute value smaller than this
     */
    void compare(const tCTF_Tensor<dtype>& A, FILE * fp = stdout, double cutoff = -1.0) const;

    /**
     * \brief frees tCTF tensor
     */
    ~tCTF_Tensor();
};

/**
 * \brief comparison function for sets of tensor pointers
 * This ensures the set iteration order is consistent across nodes
 */
template<typename dtype>
struct tensor_tid_less {
  bool operator()(tCTF_Tensor<dtype>* A, tCTF_Tensor<dtype>* B) {
    if (A == NULL && B != NULL) {
      return true;
    } else if (A == NULL || B == NULL) {
      return false;
    }
    return A->tid < B->tid;
  }
};

/**
 * \brief Matrix class which encapsulates a 2D tensor 
 */
template<typename dtype> 
class tCTF_Matrix : public tCTF_Tensor<dtype> {
  public:
    int nrow, ncol, sym;

    /**
     * \brief constructor for a matrix
     * \param[in] nrow number of matrix rows
     * \param[in] ncol number of matrix columns
     * \param[in] sym symmetry of matrix
     * \param[in] world CTF world where the tensor will live
     * \param[in] name_ an optionary name for the tensor
     * \param[in] profile_ set to 1 to profile contractions involving this tensor
     */ 
    tCTF_Matrix(int                 nrow_, 
                int                 ncol_, 
                int                 sym_,
                tCTF_World<dtype> & world,
                char const *        name_ = NULL,
                int                 profile_ = 0);

};

/**
 * \brief Vector class which encapsulates a 1D tensor 
 */
template<typename dtype> 
class tCTF_Vector : public tCTF_Tensor<dtype> {
  public:
    int len;

    /**
     * \brief constructor for a vector
     * \param[in] len_ dimension of vector
     * \param[in] world CTF world where the tensor will live
     * \param[in] name_ an optionary name for the tensor
     * \param[in] profile_ set to 1 to profile contractions involving this tensor
     */ 
    tCTF_Vector(int                 len_,
                tCTF_World<dtype> & world,
                char const *        name_ = NULL,
                int                 profile_ = 0);
};

/**
 * \brief Scalar class which encapsulates a 0D tensor 
 */
template<typename dtype> 
class tCTF_Scalar : public tCTF_Tensor<dtype> {
  public:

    /**
     * \brief constructor for a scalar
     * \param[in] world CTF world where the tensor will live
     */ 
    tCTF_Scalar(tCTF_World<dtype> & world);
    
    /**
     * \brief constructor for a scalar with predefined value
     * \param[in] val scalar value
     * \param[in] world CTF world where the tensor will live
     */ 
    tCTF_Scalar(dtype               val,
                tCTF_World<dtype> & world);

    /**
     * \brief returns scalar value
     */
    dtype get_val();
    
    /**
     * \brief sets scalar value
     */
    void set_val(dtype val);

    /**
     * \brief casts into a dtype value
     */
    operator dtype() { return get_val(); }
};

/**
 * \brief a tensor with an index map associated with it (necessary for overloaded operators)
 */
template<typename dtype>
class tCTF_Idx_Tensor : public tCTF_Term<dtype> {
  public:
    tCTF_Tensor<dtype> * parent;
    char * idx_map;
    int is_intm;

  public:

  
    // dervied clone calls copy constructor
    tCTF_Term<dtype> * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * \brief constructor takes in a parent tensor and its indices 
     * \param[in] parent_ the parent tensor
     * \param[in] idx_map_ the indices assigned ot this tensor
     * \param[in] copy if set to 1, create copy of parent
     */
    tCTF_Idx_Tensor(tCTF_Tensor<dtype>* parent_, 
                    const char *        idx_map_,
                    int                 copy = 0);
    
    /**
     * \brief copy constructor
     * \param[in] B tensor to copy
     * \param[in] copy if 1 then copy the parent tensor of B into a new tensor
     */
    tCTF_Idx_Tensor(tCTF_Idx_Tensor<dtype> const & B,
                    int copy = 0,
                    std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    tCTF_Idx_Tensor();
    
    tCTF_Idx_Tensor(dtype val);
    
    ~tCTF_Idx_Tensor();
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost of a contraction
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(tCTF_Term<dtype> const & B);
    void operator=(tCTF_Idx_Tensor<dtype> const & B);

    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator+=(tCTF_Term<dtype> const & B);
    
    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator-=(tCTF_Term<dtype> const & B);
    
    /**
     * \brief A -> A*B contract two tensors
     * \param[in] B tensor on the right hand side
     */
    void operator*=(tCTF_Term<dtype> const & B);

    /**
     * \brief TODO A -> A * B^-1
     * \param[in] B
     */
    //void operator/(tCTF_IdxTensor& tsr);
    
    /**
     * \brief execute ips into output with scale beta
     */    
    //void run(tCTF_Idx_Tensor<dtype>* output, dtype  beta);

    /*operator tCTF_Term<dtype>* (){
      tCTF_Idx_Tensor * tsr = new tCTF_Idx_Tensor(*this);
      return tsr;
    }*/
    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};


/**
 * \brief element-wise function for scaling the elements of a tensor via the scale() function
 */
template<typename dtype>
class tCTF_fscl {
  public:
    /**
     * \brief function signature for element-wise scale operation
     * \param[in] alpha scaling value, defined in scale call 
     *            but subject to internal change due to symmetry
     * \param[in,out] a element from tensor A
     **/
    void  (*func_ptr)(dtype   alpha, 
                      dtype & a);
  public:
    tCTF_fscl() { func_ptr = NULL; }
};
/**
 * \brief Interface for custom element-wise function for tensor summation to be used with sum() call on tensors
 */
template<typename dtype>
class tCTF_fsum {
  public:
    /**
     * \brief function signature for element-wise summation operation
     * \param[in] alpha scaling value, defined in summation call 
     *            but subject to internal change due to symmetry
     * \param[in] a element from summand tensor A
     * \param[in,out] b element from summand tensor B
     **/
    void  (*func_ptr)(dtype   alpha, 
                      dtype   a,
                      dtype  &b);
  public:
    tCTF_fsum() { func_ptr = NULL; }
};

/**
 * \brief Interface for custom element-wise function for tensor contraction to be used with contract() call on tensors
 */
template<typename dtype>
class tCTF_fctr {
  public:
    /**
     * \brief function signature for element-wise contraction operation
     * \param[in] alpha scaling value, defined in contraction call 
     *            but subject to internal change due to symmetry
     * \param[in] a element from contraction tensor A
     * \param[in] b element from contraction tensor B
     * \param[in,out] c element from contraction tensor C
     **/
    void  (*func_ptr)(dtype  alpha, 
                      dtype  a, 
                      dtype  b,
                      dtype &c);
  public:
    tCTF_fctr() { func_ptr = NULL; }
};


/**
 * \brief a sparse subset of a tensor 
 */
template<typename dtype>
class tCTF_Sparse_Tensor {
  public:
    tCTF_Tensor<dtype> * parent;
    std::vector<long_int> indices;
    dtype scale;

    /** 
      * \brief base constructor 
      */
    tCTF_Sparse_Tensor();
    
    /**
     * \brief initialize a tensor which corresponds to a set of indices 
     * \param[in] indices a vector of global indices to tensor values
     * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
     */
    tCTF_Sparse_Tensor(std::vector<long_int> indices,
                       tCTF_Tensor<dtype> * parent);

    /**
     * \brief initialize a tensor which corresponds to a set of indices 
     * \param[in] number of values this sparse tensor will have locally
     * \param[in] indices an array of global indices to tensor values
     * \param[in] parent dense distributed tensor to which this sparse tensor belongs to
     */
    tCTF_Sparse_Tensor(long_int              n,
                       long_int *            indices,
                       tCTF_Tensor<dtype> * parent);

    /**
     * \brief set the sparse set of indices on the parent tensor to values
     *        forall(j) i = indices[j]; parent[i] = beta*parent[i] + alpha*values[j];
     * \param[in] alpha scaling factor on values array 
     * \param[in] values data, should be of same size as the number of indices (n)
     * \param[in] beta scaling factor to apply to previously existing data
     */
    void write(dtype              alpha, 
               dtype *            values,
               dtype              beta); 

    // C++ overload special-cases of above method
    void operator=(std::vector<dtype> values); 
    void operator+=(std::vector<dtype> values); 
    void operator-=(std::vector<dtype> values); 
    void operator=(dtype * values); 
    void operator+=(dtype * values); 
    void operator-=(dtype * values); 

    /**
     * \brief read the sparse set of indices on the parent tensor to values
     *        forall(j) i = indices[j]; values[j] = alpha*parent[i] + beta*values[j];
     * \param[in] alpha scaling factor on parent array 
     * \param[in] values data, should be preallocated to the same size as the number of indices (n)
     * \param[in] beta scaling factor to apply to previously existing data in values
     */
    void read(dtype              alpha, 
              dtype *            values,
              dtype              beta); 

    // C++ overload special-cases of above method
    operator std::vector<dtype>();
    operator dtype*();
};



/* these typedefs yield a non-tempalated interface for double and complex<double> */
// \brief a world for double precision tensor types 
typedef tCTF<double>                        CTF;
typedef tCTF_Idx_Tensor<double>             CTF_Idx_Tensor;
typedef tCTF_Tensor<double>                 CTF_Tensor;
typedef tCTF_Sparse_Tensor<double>          CTF_Sparse_Tensor;
typedef tCTF_Matrix<double>                 CTF_Matrix;
typedef tCTF_Vector<double>                 CTF_Vector;
typedef tCTF_Scalar<double>                 CTF_Scalar;
typedef tCTF_World<double>                  CTF_World;
typedef tCTF_fscl<double>                   CTF_fscl;
typedef tCTF_fsum<double>                   CTF_fsum;
typedef tCTF_fctr<double>                   CTF_fctr;
#ifdef CTF_COMPLEX
// \brief a world for complex double precision tensor types 
typedef tCTF< std::complex<double> >        cCTF;
typedef tCTF_Idx_Tensor< std::complex<double> > cCTF_Idx_Tensor;
typedef tCTF_Tensor< std::complex<double> > cCTF_Tensor;
typedef tCTF_Sparse_Tensor< std::complex<double> > cCTF_Sparse_Tensor;
typedef tCTF_Matrix< std::complex<double> > cCTF_Matrix;
typedef tCTF_Vector< std::complex<double> > cCTF_Vector;
typedef tCTF_Scalar< std::complex<double> > cCTF_Scalar;
typedef tCTF_World< std::complex<double> >  cCTF_World;
typedef tCTF_fscl< std::complex<double> >   cCTF_fscl;
typedef tCTF_fsum< std::complex<double> >   cCTF_fsum;
typedef tCTF_fctr< std::complex<double> >   cCTF_fctr;
#endif

/**
 * @}
 */

/**
 * \defgroup expression Tensor expression compiler
 * @{
 */


/**
 * \brief a term is an abstract object representing some expression of tensors
 */
template<typename dtype>
class tCTF_Term {
  public:
    dtype scale;
   
    tCTF_Term();
    virtual ~tCTF_Term(){};

    /**
     * \brief base classes must implement this copy function to retrieve pointer
     */ 
    virtual tCTF_Term * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const = 0;
    
    /**
     * \brief evalues the expression, which just scales by default
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual void execute(tCTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost of a contraction/sum/.. term
     * \param[in] output tensor to write results into and its indices
     */
    virtual long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const = 0;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param\[in,out] cost the cost of the operatiob
     * \return output tensor to write results into and its indices
     */
    virtual tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const = 0;
    
    
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    virtual tCTF_Idx_Tensor<dtype> execute() const = 0;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    virtual void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const = 0;

    /**
     * \brief constructs a new term which multiplies by tensor A
     * \param[in] A term to multiply by
     */
    tCTF_Contract_Term<dtype> operator*(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    tCTF_Sum_Term<dtype> operator+(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    tCTF_Sum_Term<dtype> operator-(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(tCTF_Term<dtype> const & B) { execute() = B; };
    void operator=(tCTF_Idx_Tensor<dtype> const & B) { execute() = B; };
    void operator+=(tCTF_Term<dtype> const & B) { execute() += B; };
    void operator-=(tCTF_Term<dtype> const & B) { execute() -= B; };
    void operator*=(tCTF_Term<dtype> const & B) { execute() *= B; };

    /**
     * \brief multiples by a constant
     * \param[in] scl scaling factor to multiply term by
     */
    tCTF_Contract_Term<dtype> operator*(dtype scl) const;

    /**
     * \brief figures out what world this term lives on
     */
    virtual tCTF_World<dtype> * where_am_i() const = 0;

    /**
     * \brief casts into a double if dimension of evaluated expression is 0
     */
    operator dtype() const;
};

template<typename dtype>
class tCTF_Sum_Term : public tCTF_Term<dtype> {
  public:
    std::vector< tCTF_Term<dtype>* > operands;

    // default constructor
    tCTF_Sum_Term() : tCTF_Term<dtype>() {}

    // destructor frees operands
    ~tCTF_Sum_Term();
  
    // copy constructor
    tCTF_Sum_Term(tCTF_Sum_Term<dtype> const & other,
        std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    // dervied clone calls copy constructor
    tCTF_Term<dtype>* clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * construct sum term corresponding to a single tensor
     * \param[in] output tensor to write results into and its indices
     */ 
    //tCTF_Sum_Term<dtype>(tCTF_Idx_Tensor<dtype> const & tsr);

    /**
     * \brief evalues the expression by summing operands into output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;

  
    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a sum term
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief constructs a new term by addition of two terms
     * \param[in] A term to add to output
     */
    tCTF_Sum_Term<dtype> operator+(tCTF_Term<dtype> const & A) const;
    
    /**
     * \brief constructs a new term by subtracting term A
     * \param[in] A subtracted term
     */
    tCTF_Sum_Term<dtype> operator-(tCTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};

template<typename dtype> static
tCTF_Contract_Term<dtype> operator*(double d, tCTF_Term<dtype> const & tsr){
  return (tsr*d);
}

/**
 * \brief An experession representing a contraction of a set of tensors contained in operands 
 */
template<typename dtype>
class tCTF_Contract_Term : public tCTF_Term<dtype> {
  public:
    std::vector< tCTF_Term<dtype>* > operands;

    // \brief default constructor
    tCTF_Contract_Term() : tCTF_Term<dtype>() {}

    // \brief destructor frees operands
    ~tCTF_Contract_Term();
  
    // \brief copy constructor
    tCTF_Contract_Term(tCTF_Contract_Term<dtype> const & other,
        std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

    // \brief dervied clone calls copy constructor
    tCTF_Term<dtype> * clone(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL) const;

    /**
     * \brief override execution to  to contract operands and add them to output
     * \param[in,out] output tensor to write results into and its indices
     */
    void execute(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
    * \brief appends the tensors this depends on to the input set
    */
    void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

    /**
     * \brief evalues the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> execute() const;
    
    /**
     * \brief estimates the cost of a contract term
     * \param[in] output tensor to write results into and its indices
     */
    long_int estimate_cost(tCTF_Idx_Tensor<dtype> output) const;
    
    /**
     * \brief estimates the cost the expression to produce an intermediate with 
     *        all expression indices remaining
     * \param[in,out] output tensor to write results into and its indices
     */
    tCTF_Idx_Tensor<dtype> estimate_cost(long_int & cost) const;
    
    
    /**
     * \brief override contraction to grow vector rather than create recursive terms
     * \param[in] A term to multiply by
     */
    tCTF_Contract_Term<dtype> operator*(tCTF_Term<dtype> const & A) const;

    /**
     * \brief figures out what world this term lives on
     */
    tCTF_World<dtype> * where_am_i() const;
};
/**
 * @}
 */

/**
 * \defgroup scheduler Dynamic scheduler.
 * @{
 */
enum tCTF_TensorOperationTypes {
  TENSOR_OP_NONE,
  TENSOR_OP_SET,
  TENSOR_OP_SUM,
  TENSOR_OP_SUBTRACT,
  TENSOR_OP_MULTIPLY };

/**
 * \brief Provides a untemplated base class for tensor operations.
 */
class tCTF_TensorOperationBase {
public:
  virtual ~tCTF_TensorOperationBase() {}
};

/**
 * \brief A tensor operation, containing all the data (op, lhs, rhs) required
 * to run it. Also provides methods to get a list of inputs and outputs, as well
 * as successor and dependency information used in scheduling.
 */
template<typename dtype>
class tCTF_TensorOperation : public tCTF_TensorOperationBase {
public:
	/**
	 * \brief Constructor, create the tensor operation lhs op= rhs
	 */
	tCTF_TensorOperation(tCTF_TensorOperationTypes op,
			tCTF_Idx_Tensor<dtype>* lhs,
			const tCTF_Term<dtype>* rhs) :
			  dependency_count(0),
			  op(op),
			  lhs(lhs),
			  rhs(rhs),
			  cached_estimated_cost(0) {}

  /**
   * \brief appends the tensors this writes to to the input set
   */
  void get_outputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* outputs_set) const;

	/**
	 * \brief appends the tensors this depends on (reads from, including the output
	 * if a previous value is required) to the input set
	 */
	void get_inputs(std::set<tCTF_Tensor<dtype>*, tensor_tid_less<dtype> >* inputs_set) const;

	/**
	 * \brief runs this operation, but does NOT handle dependency scheduling
	 * optionally takes a remapping of tensors
	 */
	void execute(std::map<tCTF_Tensor<dtype>*, tCTF_Tensor<dtype>*>* remap = NULL);

	/**
	 *\brief provides an estimated runtime cost
	 */
	long_int estimate_cost();

	bool is_dummy() {
	  return op == TENSOR_OP_NONE;
	}

  /**
   * Schedule Recording Variables
   */
	// Number of dependencies I have
  int dependency_count;
  // List of all successors - operations that depend on me
  std::vector<tCTF_TensorOperation<dtype>* > successors;
  std::vector<tCTF_TensorOperation<dtype>* > reads;

  /**
   * Schedule Execution Variables
   */
  int dependency_left;

  /**
   * Debugging Helpers
   */
  const char* name() {
    return lhs->parent->name;
  }

protected:
	tCTF_TensorOperationTypes op;
	tCTF_Idx_Tensor<dtype>* lhs;
	const tCTF_Term<dtype>* rhs;

	long_int cached_estimated_cost;
};

// untemplatized scheduler abstract base class to assist in global operations
class tCTF_ScheduleBase {
public:
	virtual void add_operation(tCTF_TensorOperationBase* op) = 0;
};

extern tCTF_ScheduleBase* global_schedule;

struct tCTF_ScheduleTimer {
  double comm_down_time;
  double exec_time;
  double imbalance_wall_time;
  double imbalance_acuum_time;
  double comm_up_time;
  double total_time;

  tCTF_ScheduleTimer():
    comm_down_time(0),
    exec_time(0),
    imbalance_wall_time(0),
    imbalance_acuum_time(0),
    comm_up_time(0),
    total_time(0) {}

  void operator+=(tCTF_ScheduleTimer const & B) {
    comm_down_time += B.comm_down_time;
    exec_time += B.exec_time;
    imbalance_wall_time += B.imbalance_wall_time;
    imbalance_acuum_time += B.imbalance_acuum_time;
    comm_up_time += B.comm_up_time;
    total_time += B.total_time;
  }
};

template<typename dtype>
class tCTF_Schedule : public tCTF_ScheduleBase {
public:
  /**
   * \brief Constructor, optionally specifying a world to restrict processor
   * allocations to
   */
  tCTF_Schedule(tCTF_World<dtype>* world = NULL) :
    world(world),
    partitions(0) {}

	/**
	 * \brief Starts recording all tensor operations to this schedule
	 * (instead of executing them immediately)
	 */
	void record();

	/**
	 * \brief Executes the schedule and implicitly terminates recording
	 */
	tCTF_ScheduleTimer execute();

  /**
   * \brief Executes a slide of the ready_queue, partitioning it among the
   * processors in the grid
   */
  inline tCTF_ScheduleTimer partition_and_execute();

	/**
	 * \brief Call when a tensor op finishes, this adds newly enabled ops to the ready queue
	 */
	inline void schedule_op_successors(tCTF_TensorOperation<dtype>* op);

	/**
	 * \brief Adds a tensor operation to this schedule.
	 * THIS IS CALL ORDER DEPENDENT - operations will *appear* to execute
	 * sequentially in the order they were added.
	 */
	void add_operation_typed(tCTF_TensorOperation<dtype>* op);
	void add_operation(tCTF_TensorOperationBase* op);

	/**
	 * Testing functionality
	 */
	void set_max_partitions(int in_partitions) {
	  partitions = in_partitions;
	}

protected:
	tCTF_World<dtype>* world;

	/**
	 * Internal scheduling operation overview:
	 * DAG Structure:
	 *  Each task maintains:
	 *    dependency_count: the number of dependencies that the task has
	 *    dependency_left: the number of dependencies left before this task can
	 *      execute
	 *    successors: a vector of tasks which has this as a dependency
	 *  On completing a task, it decrements the dependency_left of all
	 *  successors. Once the count reaches zero, the task is added to the ready
	 *  queue and can be scheduled for execution.
	 *  To allow one schedule to be executed many times, dependency_count is
	 *  only modified by recording tasks, and is copied to dependency_left when
	 *  the schedule starts executing.
	 *
	 * DAG Construction:
	 *  A map from tensors pointers to operations is maintained, which contains
	 *  the latest operation that writes to a tensor.
	 *  When a new operation is added, it checks this map for all dependencies.
	 *  If a dependency has no entry yet, then it is considered satisfied.
	 *  Otherwise, it depends on the current entry - and the latest write
	 *  operation adds this task as a successor.
	 *  Then, the latest_write for this operation is updated.
	 */

	/**
	 * Schedule Recording Variables
	 */
	// Tasks with no dependencies, which can be executed at the start
	std::deque<tCTF_TensorOperation<dtype>*> root_tasks;

  // For debugging purposes - the steps in the original input order
  std::deque<tCTF_TensorOperation<dtype>*> steps_original;

  // Last operation writing to the key tensor
  std::map<tCTF_Tensor<dtype>*, tCTF_TensorOperation<dtype>*> latest_write;

  /**
   * Schedule Execution Variables
   */
  // Ready queue of tasks with all dependencies satisfied
  std::deque<tCTF_TensorOperation<dtype>*> ready_tasks;

  /**
   * Testing variables
   */
  int partitions;

};
/**
 * @}
 */


#define MAX_NAME_LENGTH 53
    
class CTF_Function_timer{
  public:
    char name[MAX_NAME_LENGTH];
    double start_time;
    double start_excl_time;
    double acc_time;
    double acc_excl_time;
    int calls;

    double total_time;
    double total_excl_time;
    int total_calls;

  public: 
    CTF_Function_timer(char const * name_, 
                   double const start_time_,
                   double const start_excl_time_);
    void compute_totals(MPI_Comm comm);
    bool operator<(CTF_Function_timer const & w) const ;
    void print(FILE *         output, 
               MPI_Comm const comm, 
               int const      rank,
               int const      np);
};


/**
 * \defgroup timer Timing and cost measurement
 * @{
 *//**
 * \brief local process walltime measurement
 */
class CTF_Timer{
  public:
    char const * timer_name;
    int index;
    int exited;
    int original;
  
  public:
    CTF_Timer(char const * name);
    ~CTF_Timer();
    void stop();
    void start();
    void exit();
    
};

/**
 * \brief epoch during which to measure timers
 */
class CTF_Timer_epoch{
  private:
    CTF_Timer * tmr_inner;
    CTF_Timer * tmr_outer;
    std::vector<CTF_Function_timer> saved_function_timers;
    double save_excl_time;
    double save_complete_time; 
  public:
    char const * name;
    //create epoch called name
    CTF_Timer_epoch(char const * name_);
    
    //clears timers and begins epoch
    void begin();

    //prints timers and clears them
    void end();
};


/**
 * \brief a term is an abstract object representing some expression of tensors
 */

/**
 * \brief measures flops done in a code region
 */
class CTF_Flop_Counter{
  public:
    long_int start_count;

  public:
    /**
     * \brief constructor, starts counter
     */
    CTF_Flop_Counter();
    ~CTF_Flop_Counter();

    /**
     * \brief restarts counter
     */
    void zero();

    /**
     * \brief get total flop count over all counters in comm
     */
    long_int count(MPI_Comm comm = MPI_COMM_SELF);

};

/**
 * @}
 */



#endif

