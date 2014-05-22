/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_WORLD_HPP__
#define __INT_WORLD_HPP__

#include "mpi.h"
#include <stdint.h>
#include <stdio.h>
#include <complex>

/* READ ME!
 * ALL BELOW FUNCTIONS MUST BE CALLED BY ALL MEMBERS OF MPI_COMM_WORLD
 * all functions return SUCCESS if successfully compeleted
 *
 * Usage:
 * call init_dist_tensor_lib() to initialize library
 * call define_tensor() to specify an empty tensor
 * call write_*_tensor() to input tensor data
 * call contract() to execute contraction (data may get remapped)
 * call read_*_tensor() to read tensor data
 * call clean_tensors() to clean up internal handles
 */
/**
 * \addtogroup CTF CTF: main C++ interface
 * @{
 */
/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
//#if (!defined NS && !defined SY && !defined SH)
//#define NS 0
//#define SY 1
//#define AS 2
//#define SH 3
//#endif
//typedef int64_t int64_t ;
//typedef int64_t  key;
//
//static const char * SY_strings[4] = {"NS", "SY", "AS", "SH"};
/**
 * @}
 */

/**
 * \defgroup internal Tensor mapping and redistribution internals
 * @{
 */
/* Force redistributions always by setting to 1 (use 2.5D algorithms) */
#define REDIST 0
//#define VERIFY 0
#define VERIFY_REMAP 0
#define FOLD_TSR 1
#define PERFORM_DESYM 1
#define ALLOW_NVIRT 1024
#define DIAG_RESCALE
#define USE_SYM_SUM
#define HOME_CONTRACT
#define USE_BLOCK_RESHUFFLE

typedef struct ctr_type {
  int   tid_A;
  int   tid_B;
  int   tid_C;
  int * idx_map_A; /* map indices of tensor A to contraction */
  int * idx_map_B; /* map indices of tensor B to contraction */
  int * idx_map_C; /* map indices of tensor C to contraction */
} ctr_type_t;

typedef struct sum_type {
  int   tid_A;
  int   tid_B;
  int * idx_map_A; /* map indices of tensor A to sum */
  int * idx_map_B; /* map indices of tensor B to sum */
} sum_type_t;

enum { SUCCESS, ERROR, NEGATIVE };

enum MACHINE { MACHINE_GENERIC, MACHINE_BGP, MACHINE_BGQ,
                   MACHINE_8D, NO_TOPOLOGY };

class dist_tensor;

class Int_Scalar{
  public:
    int el_size;
    char * value;

  
}

class Int_World{
  private:
    dist_tensor * dt;
    int initialized;

  public:


    Int_World();
    ~Int_World();

    /* initializes library. Sets topology to be a torus of edge lengths equal to the
       factorization of np. Main args can be sset for profiler output. */
    int init(MPI_Comm const global_context,
             int const      rank,
             int const      np,
             MACHINE        mach = MACHINE_GENERIC,
             int const      argc = 0,
             const char * const * argv = NULL);


    /* initializes library. Sets topology to be a mesh of dimension ndim with
       edge lengths dim_len. */
    int init(MPI_Comm const global_context,
             int const      rank,
             int const      np,
             int const      ndim,
             int const *    dim_len,
             int const      argc = 0,
             const char * const * argv = NULL);


    /* return MPI_Comm global_context */
    MPI_Comm get_MPI_Comm();

    /* return MPI processor rank */
    int get_rank();

    /* return number of MPI processes in the defined global context */
    int get_num_pes();

    /* define a tensor and retrive handle */
    int define_tensor(int const     ndim,
                      int const *   edge_len,
                      int const *   sym,
                      int *         tensor_id,
#if DEBUG < 3
                      char const *  name = NULL,
                      int           profile = 0
#else
                      char const *  name = "X",
                      int           profile = 1
#endif
                      );

    /* Create identical tensor with identical data if copy_data=1 */
    int clone_tensor(int const  tensor_id,
                     int const  copy_data,
                     int *      new_tensor_id);

    /* get information about a tensor */
    int info_tensor(int const tensor_id,
                    int *     ndim,
                    int **    edge_len,
                    int **    sym) const;

    /* set the tensor name */
    int set_name(int const tensor_id, char const * name);

    /* get the tensor name */
    int get_name(int const tensor_id, char const ** name);

    /* turn on profiling */
    int profile_on(int const tensor_id);

    /* turn off profiling */
    int profile_off(int const tensor_id);

    /* get dimension of a tensor */
    int get_dimension(int const tensor_id, int *ndim) const;

    /* get lengths of a tensor */
    int get_lengths(int const tensor_id, int **edge_len) const;

    /* get symmetry of a tensor */
    int get_symmetry(int const tensor_id, int **sym) const;

    /* get raw data pointer WARNING: includes padding */
    int get_raw_data(int const tensor_id, dtype ** data, int64_t  * size);

    /* Input tensor data with <key, value> pairs where key is the
       global index for the value. */
    int write_tensor(int const                tensor_id,
                     int64_t  const           num_pair,
                     pair<dtype> const *  mapped_data);

    /* Add tensor data new=alpha*new+beta*old
       with <key, value> pairs where key is the
       global index for the value. */
    int write_tensor(int const                tensor_id,
                     int64_t  const           num_pair,
                     Int_Scalar              alpha,
                     Int_Scalar              beta,
                     pair<dtype> const *  mapped_data);

    /**
     * Permutes a tensor along each dimension skips if perm set to -1, generalizes slice.
     *        one of permutation_A or permutation_B has to be set to NULL, if multiworld read, then
     *        the parent world tensor should not be being permuted
     */
    int permute_tensor(int const              tid_A,
                       int * const *          permutation_A,
                       Int_Scalar             alpha,
                       Int_World *          tC_A,
                       int const              tid_B,
                       int * const *          permutation_B,
                       Int_Scalar             beta,
                       Int_World *          tC_B);

   /**
     * \brief accumulates this tensor to a tensor object defined on a different world
     * \param[in] tid id of tensor on this CTF instance
     * \param[in] tid_sub id of tensor on a subcomm of this CTF inst
     * \param[in] tC_sub CTF instance on a mpi subcomm
     * \param[in] alpha scaling factor for this tensor
     * \param[in] beta scaling factor for tensor tsr
     */
    int  add_to_subworld(int          tid,
                         int          tid_sub,
                         Int_World *tC_sub,
                         Int_Scalar       alpha,
                         Int_Scalar       beta);
   /**
     * \brief accumulates this tensor from a tensor object defined on a different world
     * \param[in] tsr a tensor object of the same characteristic as this tensor, 
     * \param[in] tid id of tensor on this CTF instance
     * \param[in] tid_sub id of tensor on a subcomm of this CTF inst
     * \param[in] tC_sub CTF instance on a mpi subcomm
     * \param[in] alpha scaling factor for this tensor
     * \param[in] beta scaling factor for tensor tsr
     */
    int  add_from_subworld(int          tid,
                           int          tid_sub,
                           Int_World *tC_sub,
                           Int_Scalar       alpha,
                           Int_Scalar       beta);
    
    /* Add tensor data from A to a block of B, 
       B[offsets_B:ends_B] = beta*B[offsets_B:ends_B] 
                          + alpha*A[offsets_A:ends_A] */
    int slice_tensor(int const    tid_A,
                     int const *  offsets_A,
                     int const *  ends_A,
                     Int_Scalar   alpha,
                     int const    tid_B,
                     int const *  offsets_B,
                     int const *  ends_B,
                     Int_Scalar   beta);

    /* Same as above, except tid_A lives on dt_other_A */
    int slice_tensor(int const      tid_A,
                     int const *    offsets_A,
                     int const *    ends_A,
                     Int_Scalar     alpha,
                     Int_World *  dt_other_A,
                     int const      tid_B,
                     int const *    offsets_B,
                     int const *    ends_B,
                     Int_Scalar     beta);

    /* Same as above, except tid_B lives on dt_other_B */
    int slice_tensor(int const      tid_A,
                     int const *    offsets_A,
                     int const *    ends_A,
                     Int_Scalar     alpha,
                     int const      tid_B,
                     int const *    offsets_B,
                     int const *    ends_B,
                     Int_Scalar     beta,
                     Int_World *  dt_other_B);


    /* read a block from tensor_id,
       new_tensor_id = tensor_id[offsets:ends] */
/*    int read_block_tensor(int const   tensor_id,
                          int const * offsets,
                          int const * ends,
                          int *       new_tensor_id);*/


    /* read tensor data with <key, value> pairs where key is the
       global index for the value, which gets filled in with
       beta times the old values plus alpha times the values read from the tensor. */
    int read_tensor(int const               tensor_id,
                    int64_t  const          num_pair,
                    Int_Scalar              alpha,
                    Int_Scalar              beta,
                    pair<dtype> * const mapped_data);

    /* read tensor data with <key, value> pairs where key is the
       global index for the value, which gets filled in. */
    int read_tensor(int const               tensor_id,
                    int64_t  const          num_pair,
                    pair<dtype> * const mapped_data);


    /* read entire tensor with each processor (in packed layout).
       WARNING: will use a lot of memory. */
    int allread_tensor(int const  tensor_id,
                       int64_t  * num_pair,
                       dtype **   all_data);

    /* read entire tensor with each processor to preallocated buffer
       (in packed layout).
       WARNING: will use a lot of memory. */
    int allread_tensor(int const  tensor_id,
                       int64_t  * num_pair,
                       dtype *    all_data);


    /* map input tensor local data to zero. */
    int set_zero_tensor(int tensor_id);

    /* read tensor data pairs local to processor. */
    int read_local_tensor(int const           tensor_id,
                          int64_t  *          num_pair,
                          pair<dtype> **  mapped_data);

    /* contracts tensors alpha*A*B + beta*C -> C,
       uses standard symmetric contraction sequential kernel */
    int contract(ctr_type_t const * type,
                 Int_Scalar             alpha,
                 Int_Scalar             beta);

    /* contracts tensors alpha*A*B + beta*C -> C,
       seq_func used to perform sequential op */
    int contract(ctr_type_t const *     type,
                 fseq_tsr_ctr<dtype> const  func_ptr,
                 Int_Scalar                 alpha,
                 Int_Scalar                 beta);

    /* contracts tensors alpha*A*B + beta*C -> C,
       seq_func used to perform element-wise sequential op */
    int contract(ctr_type_t const *     type,
                 Bivar_Function      const  felm,
                 Int_Scalar                 alpha,
                 Int_Scalar                 beta);

    /* DAXPY: a*A + B -> B. */
    int sum_tensors(Int_Scalar   alpha,
                    int const    tid_A,
                    int const    tid_B);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(sum_type_t const *  type,
                    Int_Scalar              alpha,
                    Int_Scalar              beta);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(Int_Scalar                alpha,
                    Int_Scalar                beta,
                    int const                 tid_A,
                    int const                 tid_B,
                    int const *               idx_map_A,
                    int const *               idx_map_B,
                    fseq_tsr_sum<dtype> const func_ptr);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(sum_type_t const *    type,
                    Int_Scalar                alpha,
                    Int_Scalar                beta,
                    fseq_tsr_sum<dtype> const func_ptr);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(Int_Scalar                alpha,
                    Int_Scalar                beta,
                    int const                 tid_A,
                    int const                 tid_B,
                    int const *               idx_map_A,
                    int const *               idx_map_B,
                    Univar_Function     const felm);

    /* copy A into B. Realloc if necessary */
    int copy_tensor(int const tid_A, int const tid_B);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(Int_Scalar  alpha, int const tid);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(Int_Scalar                 alpha,
                     int const                  tid,
                     int const *                idx_map_A);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(Int_Scalar                 alpha,
                     int const                  tid,
                     int const *                idx_map_A,
                     fseq_tsr_scl<dtype> const  func_ptr);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(Int_Scalar                 alpha,
                     int const                  tid,
                     int const *                idx_map_A,
                     Endomorphosim       const  felm);

    /**
     * \brief estimate the cost of a contraction C[idx_C] = A[idx_A]*B[idx_B]
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \param[in] beta C scaling factor
     * \param[in] idx_C indices of C (this tensor),  e.g. "ij" -> C_{ij}
     * \return cost as a int64_t type, currently a rought estimate of flops/processor
     */
    int64_t estimate_cost(int tid_A,
                          int const *        idx_A,
                          int tid_B,
                          int const *        idx_B,
                          int tid_C,
                          int const *        idx_C);
    
    /**
     * \brief estimate the cost of a sum B[idx_B] = A[idx_A]
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \return cost as a int64_t type, currently a rought estimate of flops/processor
     */
    int64_t estimate_cost(int tid_A,
                          int const *        idx_A,
                          int tid_B,
                          int const *        idx_B);


    /* aligns tensor mapping of tid_A to that of tid_B */
    int align(int const    tid_A,
              int const    tid_B);

    /* product will contain the dot prodiuct if tsr_A and tsr_B */
    int dot_tensor(int const tid_A, int const tid_B, dtype *product);

    /* reduce data of tid_A with the given OP */
    int reduce_tensor(int const tid, OP op, dtype * result);

    /* map data of tid_A with the given function */
    int map_tensor(int const tid,
                   dtype (*map_func)(int const ndim, int const * indices,
                                     dtype const elem));

    /* obtains the largest n elements (in absolute value) of the tensor */
    int get_max_abs(int const tid, int const n, dtype * data);

    /* Prints a tensor on one processor. */
    int print_tensor(FILE * stream, int const tid, double cutoff = -1.0);

    /* Compares two tensors on one processor. */
    int compare_tensor(FILE * stream, int const tid_A, int const tid_B, double cutoff = -1.0);

    /* Prints contraction type. */
    int print_ctr(ctr_type_t const * ctype,
                  Int_Scalar             alpha,
                  Int_Scalar             beta) const;

    /* Prints sum type. */
    int print_sum(sum_type_t const * stype,
                  Int_Scalar             alpha,
                  Int_Scalar             beta) const;

    /* Deletes all tensor handles. Invalidates all tensor ids. */
    int clean_tensors();

    /* Deletes a tensor handle. Invalidates all tensor ids. */
    int clean_tensor(int const tid);

    /* Exits library and deletes all internal data */
    int exit();

    /* ScaLAPACK back-end */
    int pgemm( char const         TRANSA,
               char const         TRANSB,
               int const          M,
               int const          N,
               int const          K,
               dtype const        ALPHA,
               dtype *            A,
               int const          IA,
               int const          JA,
               int const *        DESCA,
               dtype *            B,
               int const          IB,
               int const          JB,
               int const *        DESCB,
               dtype const        BETA,
               dtype *            C,
               int const          IC,
               int const          JC,
               int const *        DESCC);

    /* define matrix from ScaLAPACK descriptor */
    int def_scala_mat(int const * DESCA, dtype const * data, int * tid);

    /* reads a ScaLAPACK matrix to the original data pointer */
    int read_scala_mat(int const tid, dtype * data);

    int pgemm( char const         TRANSA,
               char const         TRANSB,
               dtype const        ALPHA,
               int const          tid_A,
               int const          tid_B,
               dtype const        BETA,
               int const          tid_C);

};


/**
 * @}
 */



#endif ////__INT_WORLD_HPP__

