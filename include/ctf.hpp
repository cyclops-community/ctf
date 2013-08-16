/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef _tCTF_HPP_
#define _tCTF_HPP_

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
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

typedef int64_t lont_int;

/**
 * \brief reduction types for tensor data (enum actually defined in ../src/dist_tensor/cyclopstf.hpp)
 */
//enum CTF_OP { CTF_OP_SUM, CTF_OP_SUMABS, CTF_OP_SQNRM2,
//              CTF_OP_MAX, CTF_OP_MIN, CTF_OP_MAXABS, CTF_OP_MINABS };

/* custom element-wise function for tensor scale */
template<typename dtype>
class tCTF_fscl {
  public:
    /**
     * \brief function signature for element-wise scale operation
     * \param[in] alpha scaling value, defined in scale call 
     *            but subject to internal change due to symmetry
     * \param[in,out] a element from tensor A
     **/
    void  (*func_ptr)(dtype const alpha, 
                      dtype &     a);
  public:
    tCTF_fscl() { func_ptr = NULL; }
};
/* custom element-wise function for tensor sum */
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
    void  (*func_ptr)(dtype const alpha, 
                      dtype const a,
                      dtype &     b);
  public:
    tCTF_fsum() { func_ptr = NULL; }
};
/* custom element-wise function for tensor contraction */
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
    void  (*func_ptr)(dtype const alpha, 
                      dtype const a, 
                      dtype const b,
                      dtype &     c);
  public:
    tCTF_fctr() { func_ptr = NULL; }
};

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
    tCTF_World(int const      argc,
               char * const * argv);

    /**
     * \brief creates tCTF library on comm_ that can output profile data 
     *        into a file with a name based on the main args
     * \param[in] comm_ MPI communicator associated with this CTF instance
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    tCTF_World(MPI_Comm       comm_ = MPI_COMM_WORLD,
               int const      argc = 0,
               char * const * argv = NULL);

    /**
     * \brief creates tCTF library on comm_
     * \param[in] ndim number of torus network dimensions
     * \param[in] lens lengths of torus network dimensions
     * \param[in] comm MPI global context for this CTF World
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    tCTF_World(int const      ndim, 
               int const *    lens, 
               MPI_Comm       comm_ = MPI_COMM_WORLD,
               int const      argc = 0,
               char * const * argv = NULL);

    /**
     * \brief frees tCTF library
     */
    ~tCTF_World();
};

template<typename dtype>
class tCTF_Idx_Tensor;

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
    tCTF_Tensor * ctr_nbr;
    tCTF_World<dtype> * world;

  public:
    /**
     * \breif default constructor sets nothing 
     */
    tCTF_Tensor(){};

    /**
     * \brief copies a tensor (setting data to zero or copying A)
     * \param[in] A tensor to copy
     * \param[in] copy whether to copy the data of A into the new tensor
     */
    tCTF_Tensor(tCTF_Tensor const &   A,
                bool const            copy = true);

    /**
     * \brief copies a tensor filled with zeros
     * \param[in] ndim number of dimensions of tensor
     * \param[in] len edge lengths of tensor
     * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
     * \param[in] world_ a world for the tensor to live in
     * \param[in] name an optionary name for the tensor
     * \param[in] profile set to 1 to profile contractions involving this tensor
     */
    tCTF_Tensor(int const            ndim_,
                int const *          len_,
                int const *          sym_,
                tCTF_World<dtype> &  world_,
                char const *         name_ = NULL,
                int const            profile_ = 0);
    
    /**
     * \brief gives the values associated with any set of indices
     * \param[in] npair number of values to fetch
     * \param[in] global_idx index within global tensor of each value to fetch
     * \param[in,out] data a prealloced pointer to the data with the specified indices
     */
    void get_remote_data(long_int const    npair, 
                         long_int const *  global_idx, 
                         dtype *          data) const;
    
    /**
     * \brief writes in values associated with any set of indices
     * \param[in] npair number of values to write into tensor
     * \param[in] global_idx global index within tensor of value to write
     * \param[in] data values to  write to the indices
     */
    void write_remote_data(long_int const   npair, 
                           long_int const * global_idx, 
                           dtype const   * data);
   
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
    void contract(dtype const              alpha, 
                  const tCTF_Tensor&       A, 
                  char const *             idx_A,
                  const tCTF_Tensor&       B, 
                  char const *             idx_B,
                  dtype const              beta,
                  char const *             idx_C,
                  tCTF_fctr<dtype>         fseq = tCTF_fctr<dtype>());
    
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
    void sum(dtype const             alpha, 
             const tCTF_Tensor&      A, 
             char const *            idx_A,
             dtype const             beta,
             char const *            idx_B,
             tCTF_fsum<dtype>        fseq = tCTF_fsum<dtype>());
    
    /**
     * \brief scales A[idx_A] = alpha*A[idx_A]
     *        if fseq defined computes fseq(alpha,A[idx_A])
     * \param[in] alpha A scaling factor
     * \param[in] idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
     * \param[in] fseq sequential operation to execute, default is multiply-add
     */
    void scale(dtype const             alpha, 
               char const *            idx_A,
               tCTF_fscl<dtype>        fseq = tCTF_fscl<dtype>());

    /**
     * \brief cuts out a slice (block) of this tensor A[offsets,ends)
     * \param[in] offsets bottom left corner of block
     * \param[in] ends top right corner of block
     * \return new tensor corresponding to requested slice
     */
    tCTF_Tensor slice(int const * offsets,
                      int const * ends);
    
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
    void sum_slice(int const *    offsets,
                   int const *    ends,
                   double         beta,
                   tCTF_Tensor &  A,
                   int const *    offsets_A,
                   int const *    ends_A,
                   double         alpha);

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
    void get_local_data(long_int *   npair, 
                        long_int **  global_idx, 
                        dtype **    data) const;

    /**
     * \brief collects the entire tensor data on each process (not memory scalable)
     * \param[out] npair number of values in the tensor
     * \param[out] data pointer to the data of the entire tensor
     */
    void get_all_data(long_int * npair, 
                      dtype **  data) const;

    /**
     * \brief sparse add: A[global_idx[i]] = alpha*A[global_idx[i]]+beta*data[i]
     * \param[in] npair number of values to write into tensor
     * \param[in] alpha scaling factor on original data
     * \param[in] beta scaling factor on value to add
     * \param[in] global_idx global index within tensor of value to add
     * \param[in] data values to add to the tensor
     */
    void add_remote_data(long_int const    npair, 
                         dtype const      alpha, 
                         dtype const      beta,
                         long_int const *  global_idx,
                         dtype const *    data);
    /**
     * \brief obtains a small number of the biggest elements of the 
     *        tensor in sorted order (e.g. eigenvalues)
     * \param[in] n number of elements to collect
     * \param[in] data output data (should be preallocated to size at least n)
     *
     * WARNING: currently functional only for dtype=double
     */
    void get_max_abs(int const  n,
                     dtype *    data);

    // \brief turns on profiling for tensor
    void profile_on();
    
    // \brief turns off profiling for tensor
    void profile_off();

    /**
     * \brief sets tensor name
     * \param[in] name new for tensor
     */
    void set_name(char const * name);

    /**
     * \brief sets all values in the tensor to val
     */
    tCTF_Tensor& operator=(dtype const val);
    
    /**
     * \brief sets the tensor
     */
    void operator=(tCTF_Tensor<dtype> A);
    
    
    /**
     * \brief associated an index map with the tensor for future operation
     * \param[in] idx_map_ index assignment for this tensor
     */
    tCTF_Idx_Tensor<dtype>& operator[](char const * idx_map_);
    
    /**
     * \brief prints tensor data to file using process 0
     * \param[in] fp file to print to e.g. stdout
     */
    void print(FILE * fp = stdout) const;

    /**
     * \brief frees tCTF tensor
     */
    ~tCTF_Tensor();
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
    tCTF_Matrix(int const           nrow_, 
                int const           ncol_, 
                int const           sym_,
                tCTF_World<dtype> & world,
                char const *        name_ = NULL,
                int const           profile_ = 0);

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
    tCTF_Vector(int const           len_,
                tCTF_World<dtype> & world,
                char const *        name_ = NULL,
                int const           profile_ = 0);
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
    tCTF_Scalar(dtype const         val,
                tCTF_World<dtype> & world);

    /**
     * \brief returns scalar value
     */
    dtype get_val();
    
    /**
     * \brief sets scalar value
     */
    void set_val(dtype const val);

    /**
     * \brief casts into a dtype value
     */
    operator dtype() { return get_val(); }
};

template<typename dtype> static
tCTF_Idx_Tensor<dtype>& operator*(double d, tCTF_Idx_Tensor<dtype>& tsr){
  return tsr*d;
}

template tCTF_Idx_Tensor<double>& 
            operator*(double d, tCTF_Idx_Tensor<double> & tsr);
template tCTF_Idx_Tensor< std::complex<double> >& 
            operator*(double  d, tCTF_Idx_Tensor< std::complex<double> > & tsr);


/**
 * \brief a tensor with an index map associated with it (necessary for overloaded operators)
 */
template<typename dtype>
class tCTF_Idx_Tensor {
  public:
    tCTF_Tensor<dtype> * parent;
    char * idx_map;
    int has_contract, has_scale, has_sum, is_intm;
    double scale;
    tCTF_Idx_Tensor<dtype> * NBR;

  public:
    /**
     * \brief constructor takes in a parent tensor and its indices 
     * \param[in] parent_ the parent tensor
     * \param[in] idx_map_ the indices assigned ot this tensor
     */
    tCTF_Idx_Tensor(tCTF_Tensor<dtype>* parent_, const char * idx_map_);

    ~tCTF_Idx_Tensor();
    
    /**
     * \brief A = B, compute any operations on operand B and set
     * \param[in] B tensor on the right hand side
     */
    void operator=(tCTF_Idx_Tensor<dtype>& B);

    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator+=(tCTF_Idx_Tensor<dtype>& B);
    
    /**
     * \brief A += B, compute any operations on operand B and add
     * \param[in] B tensor on the right hand side
     */
    void operator-=(tCTF_Idx_Tensor<dtype>& B);
    
    /**
     * \brief A -> A*B contract two tensors
     * \param[in] B tensor on the right hand side
     */
    void operator*=(tCTF_Idx_Tensor<dtype>& B);

    /**
     * \brief C -> A*B contract two tensors
     * \param[in] B tensor on the right hand side
     */
    tCTF_Idx_Tensor<dtype>& operator*(tCTF_Idx_Tensor<dtype>& B);

    /**
     * \brief A -> A+B sums two tensors
     * \param[in] B tensor on the right hand side
     */
    tCTF_Idx_Tensor<dtype>& operator+(tCTF_Idx_Tensor<dtype>& B);
    
    /**
     * \brief A -> A-B subtacts two tensors
     * \param[in] tsr tensor on the right hand side
     */
    tCTF_Idx_Tensor<dtype>& operator-(tCTF_Idx_Tensor<dtype>& B);
    
    /**
     * \brief A -> A-B subtacts two tensors
     * \param[in] tsr tensor on the right hand side
     */
    tCTF_Idx_Tensor<dtype>& operator*(double const scl);


    /**
     * \brief TODO A -> A * B^-1
     * \param[in] B
     */
    //void operator/(tCTF_IdxTensor& tsr);
    
    /**
     * \brief casts into a double if dimension of evaluated expression is 0
     */
    operator dtype();

    /**
     * \brief execute ips into output with scale beta
     */    
    void run(tCTF_Idx_Tensor<dtype>* output, double beta);

};

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


/* these typedefs yield a non-tempalated interface for double and complex<double> */
typedef tCTF<double>                        CTF;
typedef tCTF_Tensor<double>                 CTF_Tensor;
typedef tCTF_Matrix<double>                 CTF_Matrix;
typedef tCTF_Vector<double>                 CTF_Vector;
typedef tCTF_Scalar<double>                 CTF_Scalar;
typedef tCTF_World<double>                  CTF_World;
typedef tCTF_fscl<double>                   CTF_fscl;
typedef tCTF_fsum<double>                   CTF_fsum;
typedef tCTF_fctr<double>                   CTF_fctr;
typedef tCTF< std::complex<double> >        cCTF;
typedef tCTF_Tensor< std::complex<double> > cCTF_Tensor;
typedef tCTF_Matrix< std::complex<double> > cCTF_Matrix;
typedef tCTF_Vector< std::complex<double> > cCTF_Vector;
typedef tCTF_Scalar< std::complex<double> > cCTF_Scalar;
typedef tCTF_World< std::complex<double> >  cCTF_World;
typedef tCTF_fscl< std::complex<double> >   cCTF_fscl;
typedef tCTF_fsum< std::complex<double> >   cCTF_fsum;
typedef tCTF_fctr< std::complex<double> >   cCTF_fctr;
#endif
