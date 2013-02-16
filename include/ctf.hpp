#ifndef _CTF_HPP_
#define _CTF_HPP_

#include "mpi.h"
#include <stdio.h>
#include <stdint.h>
#include "../src/dist_tensor/cyclopstf.hpp"

/**
 * \brief an instance of the CTF library (world) on a MPI communicator
 */
class CTF_World {
  public:
    MPI_Comm comm;
    CTF * ctf;

  public:
    /**
     * \brief creates CTF library on comm_
     */
    CTF_World(MPI_Comm comm_ = MPI_COMM_WORLD);

    /**
     * \brief creates CTF library on comm_
     * \param ndim number of torus network dimensions
     * \param lens lengths of torus network dimensions
     */
    CTF_World(int const ndim, int const * lens, MPI_Comm comm_ = MPI_COMM_WORLD);

    /**
     * \brief frees CTF library
     */
    ~CTF_World();
};

/**
 * \brief an instance of a tensor within a CTF world
 */
class CTF_Tensor {
  public:
    int tid;
    int ndim;
    int * sym;
    int * len;
    CTF_World * world;

  public:
    /**
     * \brief copies a tensor (setting data to zero or copying A)
     * \param A tensor to copy
     * \param copy whether to copy the data of A into the new tensor
     */
    CTF_Tensor(CTF_Tensor const &  A,
               bool const          copy = false);

    /**
     * \brief copies a tensor filled with zeros
     * \param ndim number of dimensions of tensor
     * \param len edge lengths of tensor
     * \param sym symmetries of tensor (e.g. symmetric matrix -> sym={SY, NS})
     * \param world_ a world for the tensor to live in
     */
    CTF_Tensor(int const            ndim_,
               int const *          len_,
               int const *          sym_,
               CTF_World *          world_);
    
    /**
     * \brief gives the values associated with any set of indices
     * \param[in] npair number of values to fetch
     * \param[in] global_idx index within global tensor of each value to fetch
     * \param[in,out] data a prealloced pointer to the data with the specified indices
     */
    void get_remote_data(int const npair, int64_t const * global_idx, double * data) const;
    
    /**
     * \brief writes in values associated with any set of indices
     * \param[in] npair number of values to write into tensor
     * \param[in] global_idx global index within tensor of value to write
     * \param[in] data values to  write to the indices
     */
    void write_remote_data(int const npair, int64_t const * global_idx, double const * data) const;
   
    /**
     * \brief contracts C[idx_C] = beta*C[idx_C] + alpha*A[idx_A]*B[idx_B]
     * \brief alpha A*B scaling factor
     * \brief A first operand tensor
     * \brief idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \brief B second operand tensor
     * \brief idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \brief beta C scaling factor
     * \brief idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
     */
    void contract(const double alpha, const CTF_Tensor& A, const char * idx_A,
                                      const CTF_Tensor& B, const char * idx_B,
                  const double beta,                       const char * idx_C);

    /**
     * \brief sums B[idx_B] = beta*B[idx_B] + alpha*A[idx_A]
     * \brief alpha A scaling factor
     * \brief A first operand tensor
     * \brief idx_A indices of A in sum, e.g. "ij" -> A_{ij}
     * \brief beta B scaling factor
     * \brief idx_B indices of B (this tensor), e.g. "ij" -> B_{ij}
     */
    void sum(const double alpha, const CTF_Tensor& A, const char * idx_A,
             const double beta,                       const char * idx_B);
    
    /**
     * \brief scales A[idx_A] = alpha*A[idx_A]
     * \brief alpha A scaling factor
     * \brief idx_A indices of A (this tensor), e.g. "ij" -> A_{ij}
     */
    void scale(const double alpha, const char * idx_A);

    /**
     * \brief performs a reduction on the tensor
     * \param op reduction operation (see top of this cyclopstf.hpp for choices)
     */    
    double reduce(CTF_OP op);

    /**
     * \brief gives the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to local data
     */
    double * get_raw_data(int * size);

    /**
     * \brief gives a read-only copy of the raw current local data with padding included
     * \param[out] size of local data chunk
     * \return pointer to read-only copy of local data
     */
    const double * raw_data(int * size) const;

    /**
     * \brief gives the global indices and values associated with the local data
     * \param[out] npair number of local values
     * \param[out] global_idx index within global tensor of each data value
     * \param[out] data pointer to local values in the order of the indices
     */
    void get_local_data(int * npair, int64_t ** global_idx, double ** data) const;

    /**
     * \brief collects the entire tensor data on each process (not memory scalable)
     * \param[out] npair number of values in the tensor
     * \param[out] data pointer to the data of the entire tensor
     */
    void get_all_data(int * npair, double ** data) const;

    /**
     * \brief sparse add: A[global_idx[i]] = alpha*A[global_idx[i]]+beta*data[i]
     * \param[in] npair number of values to write into tensor
     * \param[in] alpha scaling factor on original data
     * \param[in] beta scaling factor on value to add
     * \param[in] global_idx global index within tensor of value to add
     * \param[in] data values to add to the tensor
     */
    void add_remote_data(int const        npair, 
                         double const     alpha, 
                         double const     beta,
                         int64_t const *  global_idx,
                         double const *   data);

    /**
     * \brief sets all values in the tensor to val
     */
    CTF_Tensor& operator=(const double val);
    
    /**
     * \brief prints tensor data to file using process 0
     */
    void print(FILE * fp) const;

    /**
     * \brief frees CTF tensor
     */
    ~CTF_Tensor();
};
#endif
