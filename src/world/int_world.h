/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __INT_WORLD_H__
#define __INT_WORLD_H__

#include "mpi.h"
#include <stdint.h>
#include <stdio.h>
#include "../ctr_seq/int_functions.h"
#include "../tensor/int_tensor.h"
#include "../interface/common.h"


namespace CTF_int {
  // \brief orchestrating center, defined by an MPI comm and a topology
  //          which keeps track of all derived topologies, tensors, mappings, and operations
  class world{
    public:

      /** 
       * \brief constructor
       */
      world();
      /** 
       * \brief destructor
       */
      ~world();

      /* \brief return MPI_Comm global_context */
      MPI_Comm get_MPI_Comm();

      /* \brief return MPI processor rank */
      int get_rank();

      /* \brief return number of MPI processes in the defined global context */
      int get_num_pes();

      /**
       * \brief  defines a tensor and retrieves handle
       *
       * \param[in] sr semiring defining type of tensor
       * \param[in] order number of tensor dimensions
       * \param[in] edge_len global edge lengths of tensor
       * \param[in] sym symmetry relations of tensor
       * \param[out] tensor_id the tensor index (handle)
       * \param[in] name string name for tensor (optionary)
       * \param[in] profile wether to make profile calls for the tensor
       */
      int define_tensor(semiring      sr, 
                        int           order,
                        int const *   edge_len,
                        int const *   sym,
                        int *         tensor_id,
                        bool          alloc_data = 1,
  #if DEBUG < 3
                        char const *  name = NULL,
                        bool          profile = 0
  #else
                        char const *  name = "X",
                        bool          profile = 1
  #endif
                        );

      /* Create identical tensor with identical data if copy_data=1 */
      int clone_tensor(int        tensor_id,
                       bool       copy_data,
                       int *      new_tensor_id,
                       bool       alloc_data = 1);

      /* contracts tensors alpha*A*B + beta*C -> C,
         uses standard symmetric contraction sequential kernel */
      int contract(ctr_type_t const * type,
                   char const *       alpha,
                   char const *       beta);

      /* contracts tensors alpha*A*B + beta*C -> C,
         seq_func used to perform sequential op */
  /*    int contract(ctr_type_t const *     type,
                   fseq_tsr_ctr<dtype> const  func_ptr,
                   char const *           alpha,
                   char const *           beta);
  */
      /* contracts tensors alpha*A*B + beta*C -> C,
         seq_func used to perform element-wise sequential op */
      int contract(ctr_type_t const *     type,
                   bivar_function         felm,
                   char const *           alpha,
                   char const *           beta);

      /* DAXPY: a*A + B -> B. */
      int sum_tensors(char const * alpha,
                      int          tid_A,
                      int          tid_B);

      /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
      int sum_tensors(sum_type_t const *  type,
                      char const *    alpha,
                      char const *    beta);

      /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    /*  int sum_tensors(char const *          alpha,
                      char const *          beta,
                      int                       tid_A,
                      int                       tid_B,
                      int const *               idx_map_A,
                      int const *               idx_map_B,
                      fseq_tsr_sum<dtype> const func_ptr);
  */
      /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
     /* int sum_tensors(sum_type_t const *    type,
                      char const *          alpha,
                      char const *          beta,
                      fseq_tsr_sum<dtype> const func_ptr);
  */
      /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
      int sum_tensors(char const *        alpha,
                      char const *        beta,
                      int                 tid_A,
                      int                 tid_B,
                      int const *         idx_map_A,
                      int const *         idx_map_B,
                      univar_function     felm);
      /* scale tensor by alpha. A <- a*A */
      int scale_tensor(char const * alpha, int tid);

      /* scale tensor by alpha. A <- a*A */
      int scale_tensor(char const *               alpha,
                       int                        tid,
                       int const *                idx_map_A);

      /* scale tensor by alpha. A <- a*A */
  /*    int scale_tensor(char const *           alpha,
                       int                        tid,
                       int const *                idx_map_A,
                       fseq_tsr_scl<dtype> const  func_ptr);
  */
      /* scale tensor by alpha. A <- a*A */
      int scale_tensor(char const * alpha,
                       int          tid,
                       int const *  idx_map_A,
                       endomorphism felm);

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

      /* Prints contraction type. */
      int print_ctr(ctr_type_t const * ctype,
                    char const *       alpha,
                    char const *       beta) const;

      /* Prints sum type. */
      int print_sum(sum_type_t const * stype,
                    char const *       alpha,
                    char const *       beta) const;

      /* Deletes all tensor handles. Invalidates all tensor ids. */
      int clean_tensors();

      /* Deletes a tensor handle. Invalidates all tensor ids. */
      int clean_tensor(int tid);

      /* Exits library and deletes all internal data */
      int exit();

      /* ScaLAPACK back-end */
      /*int pgemm( char const         TRANSA,
                 char const         TRANSB,
                 int                M,
                 int                N,
                 int                K,
                 dtype const        ALPHA,
                 dtype *            A,
                 int                IA,
                 int                JA,
                 int const *        DESCA,
                 dtype *            B,
                 int                IB,
                 int                JB,
                 int const *        DESCB,
                 dtype const        BETA,
                 dtype *            C,
                 int                IC,
                 int                JC,
                 int const *        DESCC);
  */
      /* define matrix from ScaLAPACK descriptor */
    //  int def_scala_mat(int const * DESCA, dtype const * data, int * tid);

      /* reads a ScaLAPACK matrix to the original data pointer */
  //    int read_scala_mat(int tid, dtype * data);

    /*  int pgemm( char const         TRANSA,
                 char const         TRANSB,
                 dtype const        ALPHA,
                 int                tid_A,
                 int                tid_B,
                 dtype const        BETA,
                 int                tid_C);
  */
  };

}

/**
 * @}
 */

#endif ////__INT_WORLD_HPP__

