/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __DIST_TENSOR_H__
#define __DIST_TENSOR_H__

#include "cyclopstf.hpp"
#include <stdint.h>
#include <stdio.h>
#include <complex>

/* READ ME! 
 * ALL BELOW FUNCTIONS MUST BE CALLED BY ALL MEMBERS OF MPI_COMM_WORLD 
 * all functions return DIST_TENSOR_SUCCESS if successfully compeleted 
 *
 * Usage:
 * call init_dist_tensor_lib() to initialize library
 * call define_tensor() to specify an empty tensor
 * call write_*_tensor() to input tensor data
 * call contract() to execute contraction (data may get remapped)
 * call read_*_tensor() to read tensor data 
 * call clean_tensors() to clean up internal handles
 */

/* Function signature for sub-tensor summation recrusive call */
typedef int  (*CTF_seq_tsr_scl) (double const   alpha,
                                 double *       A,
                                 int const      ndim_A,
                                 int const *    edge_len_A,
                                 int const *    lda_A,
                                 int const *    sym_A,
                                 int const *    idx_map_A);

/* Function signature for sub-tensor summation recrusive call */
typedef int  (*CTF_seq_tsr_sum) (double const   alpha,
                                 double const * A,
                                 int const      ndim_A,
                                 int const *    edge_len_A,
                                 int const *    lda_A,
                                 int const *    sym_A,
                                 int const *    idx_map_A,
                                 double const   beta,
                                 double *       B,
                                 int const      ndim_B,
                                 int const *    edge_len_B,
                                 int const *    lda_B,
                                 int const *    sym_B,
                                 int const *    idx_map_B);

/* Function signature for sub-tensor contraction recrusive call */
typedef int  (*CTF_seq_tsr_ctr) (double const   alpha,
                                 double const * A,
                                 int const      ndim_A,
                                 int const *    edge_len_A,
                                 int const *    lda_A,
                                 int const *    sym_A,
                                 int const *    idx_map_A,
                                 double const * B,
                                 int const      ndim_B,
                                 int const *    edge_len_B,
                                 int const *    lda_B,
                                 int const *    sym_B,
                                 int const *    idx_map_B,
                                 double const   beta,
                                 double *       C,
                                 int const      ndim_C,
                                 int const *    edge_len_C,
                                 int const *    lda_C,
                                 int const *    sym_C,
                                 int const *    idx_map_C);



/* initializes library. Sets topology to be a torus of edge lengths equal to the 
 factorization of np. */
int CTF_init(MPI_Comm const     global_context,
             int const          rank, 
             int const          np);

/* initializes library. Sets topology to be that of a predefined machine 'mach'. */
int CTF_init(MPI_Comm const     global_context,
             CTF_MACHINE        mach,
             int const          rank, 
             int const          np,
             int const          inner_size = DEF_INNER_SIZE);

/* initializes library. Sets topology to be a mesh of dimension ndim with
   edge lengths dim_len.  */
int CTF_init(MPI_Comm const     global_context,
             int const          rank, 
             int const          np, 
             int const          ndim, 
             int const *        dim_len);

/* initializes library. Sets topology to be a torus of edge lengths equal to the 
 factorization of np. */
int CTF_init_complex(MPI_Comm const     global_context,
                     int const          rank, 
                     int const          np);

/* initializes library. Sets topology to be that of a predefined machine 'mach'. */
int CTF_init_complex(MPI_Comm const     global_context,
                     CTF_MACHINE        mach,
                     int const          rank, 
                     int const          np);

/* initializes library. Sets topology to be a mesh of dimension ndim with
   edge lengths dim_len.  */
int CTF_init_complex(MPI_Comm const     global_context,
                     int const          rank, 
                     int const          np, 
                     int const          ndim, 
                     int const *        dim_len);



/* define a tensor and retrive handle */
int CTF_define_tensor(int const ndim,       /* input: number of tensor dimensions */
                      int const *       edge_len,   /* input: global edge lengths */
                      int const *       sym,        /* input: symmetry relations */
                      int *             tensor_id); /* output: tensor index */

/* get dimension of a tensor */
int CTF_get_dimension(int const tensor_id, int *ndim);

/* get lengths of a tensor */
int CTF_get_lengths(int const tensor_id, int **edge_len);

/* get symmetry of a tensor */
int CTF_get_symmetry(int const tensor_id, int **sym);

/* get information about a tensor */
int CTF_info_tensor(int const tensor_id,
                    int * ndim,
                    int ** edge_len,
                    int ** sym);

/* Input tensor data with <key, value> pairs where key is the
   global index for the value. */
int CTF_write_tensor(int const          tensor_id, 
                     int64_t const      num_pair,  
                     kv_pair * const    mapped_data);

/* read tensor data with <key, value> pairs where key is the
   global index for the value, which gets filled in. */
int CTF_read_tensor(int const           tensor_id, 
                    int64_t const       num_pair, 
                    kv_pair * const     mapped_data);

/* read entire tensor with each processor (in packed layout).
   WARNING: will use a lot of memory. */
int CTF_allread_tensor(int const        tensor_id, 
                       int64_t *        num_pair, 
                       double **        all_data);


/* map input tensor local data to zero. */
int CTF_set_zero_tensor(int tensor_id);

/* input tensor local data or set buffer for contract answer. */
/* WARNING: Do not dealloc or modify tsr_data after giving it to this func */
/* WARNING: you probably should not be using this, I might deprecate it.
            (user doesn't know the padding) */
/* DEPRECATED */
/*int CTF_set_local_tensor(int const    tensor_id, 
                         int const      num_val, 
                         double *       tsr_data);*/

/* read tensor data pairs local to processor. */
int CTF_read_local_tensor(int const     tensor_id, 
                          int64_t *     num_pair,  
                          kv_pair **    mapped_data);

/* contracts tensors alpha*A*B + beta*C -> C,
   uses standard symmetric contraction sequential kernel */
int CTF_contract(CTF_ctr_type_t const * type, 
                 double const           alpha,
                 double const           beta);

/* contracts tensors alpha*A*B + beta*C -> C,
   accepts custom-sized buffer-space,
   uses standard symmetric contraction sequential kernel */
int CTF_contract(CTF_ctr_type_t const * type, 
                 double *               buffer, 
                 int const              buffer_len, 
                 double const           alpha,
                 double const           beta);


/* contracts tensors alpha*A*B + beta*C -> C,
   accepts custom-sized buffer-space (set to NULL for dynamic allocs),
   seq_func used to perform sequential op */
int CTF_contract(CTF_ctr_type_t const * type, /* defines contraction */
                 double *               buffer, /* currently unused */
                 int const              buffer_len, /* currently unused */
                 CTF_seq_tsr_ctr const  func_ptr, /* sequential ctr func pointer */
                 double const           alpha,
                 double const           beta);

/* DAXPY: a*A + B -> B. */
int CTF_sum_tensors(double const        alpha,
                    int const           tid_A,
                    int const           tid_B);
    
/* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
int CTF_sum_tensors(double const                alpha,
                    double const                beta,
                    int const                   tid_A,
                    int const                   tid_B,
                    int const *                 idx_map_A,
                    int const *                 idx_map_B,
                    fseq_tsr_sum<double> const  func_ptr);
    
/* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
int CTF_sum_tensors(CTF_sum_type_t const *      type,
                    double const                alpha,
                    double const                beta);

/* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
int CTF_sum_tensors(CTF_sum_type_t const *      type,
                    double const                alpha,
                    double const                beta,
                    CTF_seq_tsr_sum const       func_ptr);

    /* copy A into B. Realloc if necessary */
int CTF_copy_tensor(int const tid_A, int const tid_B);

/* scale tensor by alpha. A <- a*A */
int CTF_scale_tensor(double const alpha, int const tid);

/* scale tensor by alpha. idx_map(A) <- a*idx_map(A) */
int CTF_scale_tensor(double const               alpha, 
                     int const                  tid, 
                     int const *                idx_map,
                     CTF_seq_tsr_scl const      func_ptr);

/* product will contain the dot prodiuct if tsr_A and tsr_B */
int CTF_dot_tensor(int const tid_A, int const tid_B, double *product);

/* reduce data of tid_A with the given OP */
int CTF_reduce_tensor(int const tid, CTF_OP op, double * result);

/* map data of tid_A with the given function */
int CTF_map_tensor(int const tid, 
                   double (*map_func)(int const ndim, int const * indices, 
                                                              double const elem));

/* Prints a tensor on one processor. */
int CTF_print_tensor(FILE * stream, int const tid);

/* Prints contraction type. */
int CTF_print_ctr(CTF_ctr_type_t const * ctype,
                  double const           alpha,
                  double const           beta);

/* Prints sum type. */
int CTF_print_sum(CTF_sum_type_t const * stype,
                  double const           alpha,
                  double const           beta);

/* Deletes all tensor handles. Invalidates all tensor ids. */
int CTF_clean_tensors();

/* Deletes a tensor handle. Invalidates all tensor ids. */
int CTF_clean_tensor(int const tid);

/* Exits library and deletes all internal data */
int CTF_exit();

/* ScaLAPACK PDGEMM back-end */
void CTF_pdgemm(char const      TRANSA, 
                char const      TRANSB, 
                int const       M, 
                int const       N, 
                int const       K, 
                double const    ALPHA,
                double *        A, 
                int const       IA, 
                int const       JA, 
                int const *     DESCA, 
                double *        B, 
                int const       IB, 
                int const       JB, 
                int const *     DESCB, 
                double const    BETA,
                double *        C, 
                int const       IC, 
                int const       JC, 
                int const *     DESCC);

/* ScaLAPACK back-end */
void CTF_pzgemm(char const                      TRANSA, 
                char const                      TRANSB, 
                int const                       M, 
                int const                       N, 
                int const                       K, 
                std::complex<double> const      ALPHA,
                std::complex<double> *          A, 
                int const                       IA, 
                int const                       JA, 
                int const *                     DESCA, 
                std::complex<double> *          B, 
                int const                       IB, 
                int const                       JB, 
                int const *                     DESCB, 
                std::complex<double> const      BETA,
                std::complex<double> *          C, 
                int const                       IC, 
                int const                       JC, 
                int const *                     DESCC);

/* define matrix from ScaLAPACK descriptor */
int CTF_def_scala_mat(int const * DESCA, double const * data, int * tid);

/* define matrix from ScaLAPACK descriptor */
int CTF_def_scala_mat(int const * DESCA, std::complex<double> const * data, int * tid);

/* reads a ScaLAPACK matrix to the original data pointer */
int CTF_read_scala_mat(int const tid, std::complex<double> * data);
    
/* reads a ScaLAPACK matrix to the original data pointer */
int CTF_read_scala_mat(int const tid, double * data);


#endif ////__DIST_TENSOR_H__

