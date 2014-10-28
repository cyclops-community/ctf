#ifndef __INT_CONTRACTION_H__
#define __INT_CONTRACTION_H__

#include "assert.h"
#include "ctr_comm.h"

namespace CTF_int {
  class tensor; 

  /**
   * \brief class for execution distributed contraction of tensors
   */
  class contraction {
    public:
      /** \brief left operand */
      tensor * A;
      /** \brief right operand */
      tensor * B;
      /** \brief output */
      tensor * C;

      /** \brief scaling of A*B */
      char const * alpha;
      /** \biref scaling of existing C */
      char const * beta;
    
      /** \brief indices of left operand */
      int const * idx_A;
      /** \brief indices of right operand */
      int const * idx_B;
      /** \brief indices of output */
      int const * idx_C;

      /**
       * \brief constructor definining contraction with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] alpha scaling factor alpha * A[idx_A] * B[idx_B];
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] beta scaling factor of ouput 
                      C[idx_C] = beta*C[idx_C] 
                                + alpha * A[idx_A] * B[idx_B];
       */
      contraction(tensor * A, 
                  int const * idx_A,
                  tensor * B, 
                  int const * idx_B,
                  char const * alpha, 
                  tensor * C, 
                  int const * idx_C,
                  char const * beta);
     
      /**
       * \brief constructor definining contraction with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] func custom elementwise function 
                      func(A[idx_A],B[idx_B],&C[idx_C])
       */
      contraction(tensor * A, 
                  int const * idx_A,
                  tensor * B, 
                  int const * idx_B,
                  tensor * C, 
                  int const * idx_C,
                  bivar_function func);

      /** \brief run contraction */
      void execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();

      /**
       * \brief calculate the dimensions of the matrix 
       *    the contraction gets reduced to
       *
       * \param[in] ordering_A the dimensional-ordering of the inner mapping of A
       * \param[in] ordering_B the dimensional-ordering of the inner mapping of B
       * \param[in] tsr_A tensor A
       * \param[in] tsr_B tensor B
       * \param[in] tsr_C tensor C
       * \param[out] inner_prm parameters includng n,m,k
       */
      void calc_fold_nmk( int const *             ordering_A, 
                          int const *             ordering_B, 
                          tensor const *   tsr_A, 
                          tensor const *   tsr_B,
                          tensor const *   tsr_C,
                          iparam *                inner_prm);



      /**
       * \brief finds and return all contraction indices which can be folded into
       *    dgemm, for which they must (1) not break symmetry (2) belong to 
       *    exactly two of (A,B,C).
       * \param[out] num_fold number of indices that can be folded
       * \param[out] fold_idx indices that can be folded
       */
      void get_fold_indices(int *                   num_fold,
                            int **                  fold_idx);
      
      /**
       * \brief determines whether this contraction can be folded
       * \return whether we can fold this contraction
       */
      int can_fold();


      /**
       * \brief find ordering of indices of tensor to reduce to DGEMM
       *
       * \param[out] new_ordering_A the new ordering for indices of A
       * \param[out] new_ordering_B the new ordering for indices of B
       * \param[out] new_ordering_C the new ordering for indices of C
       */
      void get_len_ordering(
                int **      new_ordering_A,
                int **      new_ordering_B,
                int **      new_ordering_C);
  };
}

#endif
