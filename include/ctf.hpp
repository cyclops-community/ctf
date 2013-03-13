#ifndef _tCTF_HPP_
#define _tCTF_HPP_

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include "../src/dist_tensor/cyclopstf.hpp"

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
     * \brief creates tCTF library on comm_
     */
    tCTF_World(MPI_Comm comm_ = MPI_COMM_WORLD);

    /**
     * \brief creates tCTF library on comm_
     * \param[in] ndim number of torus network dimensions
     * \param[in] lens lengths of torus network dimensions
     */
    tCTF_World(int const    ndim, 
               int const *  lens, 
               MPI_Comm     comm_ = MPI_COMM_WORLD);

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
    tCTF_Tensor * ctr_nbr;
    tCTF_World<dtype> * world;

  public:
    /**
     * \brief copies a tensor (setting data to zero or copying A)
     * \param[in] A tensor to copy
     * \param[in] copy whether to copy the data of A into the new tensor
     */
    tCTF_Tensor(tCTF_Tensor const &   A,
                bool const            copy = false);

    /**
     * \brief copies a tensor filled with zeros
     * \param[in] ndim number of dimensions of tensor
     * \param[in] len edge lengths of tensor
     * \param[in] sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
     * \param[in] world_ a world for the tensor to live in
     */
    tCTF_Tensor(int const            ndim_,
                int const *          len_,
                int const *          sym_,
                tCTF_World<dtype> *  world_);
    
    /**
     * \brief gives the values associated with any set of indices
     * \param[in] npair number of values to fetch
     * \param[in] global_idx index within global tensor of each value to fetch
     * \param[in,out] data a prealloced pointer to the data with the specified indices
     */
    void get_remote_data(int64_t const npair, int64_t const * global_idx, dtype * data) const;
    
    /**
     * \brief writes in values associated with any set of indices
     * \param[in] npair number of values to write into tensor
     * \param[in] global_idx global index within tensor of value to write
     * \param[in] data values to  write to the indices
     */
    void write_remote_data(int64_t const npair, int64_t const * global_idx, dtype const * data) const;
   
    /**
     * \brief contracts C[idx_C] = beta*C[idx_C] + alpha*A[idx_A]*B[idx_B]
     * \param[in] alpha A*B scaling factor
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \param[in] beta C scaling factor
     * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
     */
    void contract(const dtype alpha, const tCTF_Tensor& A, const char * idx_A,
                                     const tCTF_Tensor& B, const char * idx_B,
                  const dtype beta,                        const char * idx_C);

    /**
     * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
     * \param[in] alpha A scaling factor
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in sum, e.g. "ij" -> A_{ij}
     * \param[in] beta B scaling factor
     * \param[in] idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
     */
    void sum(const dtype alpha, const tCTF_Tensor& A, const char * idx_A,
             const dtype beta,                        const char * idx_B);
    
    /**
     * \brief scales A[idx_A] = alpha*A[idx_A]
     * \brief alpha A scaling factor
     * \brief idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
     */
    void scale(const dtype alpha, const char * idx_A);

    /**
     * \brief aligns data mapping with tensor A
     * \param[in] A align with this tensor
     */
    void align(const tCTF_Tensor& A);

    /**
     * \brief performs a reduction on the tensor
     * \par[in]am op reduction operation (see top of this cyclopstf.hpp for choices)
     */    
    dtype reduce(CTF_OP op);

    /**
     * \brief gives the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to local data
     */
    dtype * get_raw_data(int64_t * size);

    /**
     * \brief gives a read-only copy of the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to read-only copy of local data
     */
    const dtype * raw_data(int64_t * size) const;

    /**
     * \brief gives the global indices and values associated with the local data
     * \param[out] npair number of local values
     * \param[out] global_idx index within global tensor of each data value
     * \param[out] data pointer to local values in the order of the indices
     */
    void get_local_data(int64_t * npair, int64_t ** global_idx, dtype ** data) const;

    /**
     * \brief collects the entire tensor data on each process (not memory scalable)
     * \param[out] npair number of values in the tensor
     * \param[out] data pointer to the data of the entire tensor
     */
    void get_all_data(int64_t * npair, dtype ** data) const;

    /**
     * \brief sparse add: A[global_idx[i]] = alpha*A[global_idx[i]]+beta*data[i]
     * \param[in] npair number of values to write into tensor
     * \param[in] alpha scaling factor on original data
     * \param[in] beta scaling factor on value to add
     * \param[in] global_idx global index within tensor of value to add
     * \param[in] data values to add to the tensor
     */
    void add_remote_data(int64_t const    npair, 
                         double const     alpha, 
                         double const     beta,
                         int64_t const *  global_idx,
                         dtype const *    data);

    /**
     * \brief sets all values in the tensor to val
     */
    tCTF_Tensor& operator=(const dtype val);
    
    /**
     * \brief associated an index map with the tensor for future operation
     */
    tCTF_Idx_Tensor<dtype>& operator[](const char * idx_map_);
    
    /**
     * \brief prints tensor data to file using process 0
     */
    void print(FILE * fp) const;

    /**
     * \brief frees tCTF tensor
     */
    ~tCTF_Tensor();
};

template<typename dtype> static
tCTF_Idx_Tensor<dtype>& operator*(double d, tCTF_Idx_Tensor<dtype>& tsr){
  return tsr*d;
}

template tCTF_Idx_Tensor<double>& operator*(double d, tCTF_Idx_Tensor<double> & tsr);
template tCTF_Idx_Tensor< std::complex<double> >& operator*(double  d, tCTF_Idx_Tensor< std::complex<double> > & tsr);


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
     * \brief execute ips into output with scale beta
     */    
    void run(tCTF_Idx_Tensor<dtype>* output, double beta);

};

typedef tCTF<double> CTF;
typedef tCTF_Tensor<double> CTF_Tensor;
typedef tCTF_World<double> CTF_World;
#endif
