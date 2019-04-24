#ifndef __INT_SUMMATION_H__
#define __INT_SUMMATION_H__

#include <assert.h>
#include "sum_tsr.h"
#include "spsum_tsr.h"

namespace CTF_int {
  class tensor; 
  class topology; 

  /**
   * \brief class for execution distributed summation of tensors
   */
  class summation {
     public:
      /** \brief left operand */
      tensor * A;
      /** \brief output */
      tensor * B;

      /** \brief scaling of A */
      char const * alpha;
      /** \brief scaling of existing B */
      char const * beta;
    
      /** \brief indices of left operand */
      int * idx_A;
      /** \brief indices of output */
      int * idx_B;
      /** \brief whether there is a elementwise custom function */
      bool is_custom;
      /** \brief function to execute on elements */
      univar_function const * func;

      /** \brief lazy constructor */
//      summation(){ idx_A = NULL; idx_B = NULL; alpha=NULL; beta=NULL; is_custom=0; };
      
      /** \brief destructor */
      ~summation();

      /** \brief copy constructor \param[in] other object to copy */
      summation(summation const & other);

      /**
       * \brief constructor definining summation with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] alpha scaling factor alpha * A[idx_A]; (can be NULL)
       * \param[in] B ouput operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] beta scaling factor of ouput (can be NULL)
                      C[idx_B] = beta*B[idx_B] + alpha * A[idx_A]
       */
      summation(tensor *     A,
                int const *  idx_A,
                char const * alpha,
                tensor *     B,
                int const *  idx_B,
                char const * beta);
      summation(tensor *     A,
                char const * idx_A,
                char const * alpha,
                tensor *     B,
                char const * idx_B,
                char const * beta);
     
      /**
       * \brief constructor definining summation with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] alpha scaling factor alpha * A[idx_A]; (can be NULL)
       * \param[in] B ouput operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] func custom elementwise function 
                      func(A[idx_A],&B[idx_B])
       * \param[in] beta scaling factor of ouput (can be NULL)
                      C[idx_B] = beta*B[idx_B] + alpha * A[idx_A]
       */
      summation(tensor *                A,
                int const *             idx_A,
                char const *            alpha,
                tensor *                B,
                int const *             idx_B,
                char const *            beta,
                univar_function const * func);
      summation(tensor *                A,
                char const *            idx_A,
                char const *            alpha,
                tensor *                B,
                char const *            idx_B,
                char const *            beta,
                univar_function const * func);

      /** \brief run summation  
        * \param[in] run_diag if true runs diagonal iterators (otherwise calls itself with run_diag=true later
        */
      void execute(bool run_diag=false);
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();
   
      /**
       * \brief returns 1 if summations have same tensors and index map
       * \param[in] os summation object to compare this with
       */
      int is_equal(summation const & os);


      /**
       * \brief PDAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
       * Treats symmetric as lower triangular
       * \param[in] run_diag if 1 run diagonal sum
       */
      int sum_tensors(bool run_diag);

      /** \brief print contraction details */
      void print();

      /**
       * \brief a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
       *        performs all necessary symmetric permutations removes/returns A/B to home buffer
       * \param[in] run_diag if 1 run diagonal sum
       * \param[in] handle_sym if true, use sym_sum
       */
      int home_sum_tsr(bool run_diag, bool handle_sym=true);

    private:
      /**
       * \brief finds and return all summation indices which can be folded into
       *    dgemm,for which they must (1) not break symmetry (2) belong to 
       *    exactly two of (A,B).
       * \param[in] type contraction specification
       * \param[out] num_fold number of indices that can be folded
       * \param[out] fold_idx indices that can be folded
       */
      void get_fold_indices(int *  num_fold,
                            int ** fold_idx);
    
      /**
       * \brief determines whether this summation can be folded
       * \return whether we can fold this summation
       */
      int can_fold();
 
      /**
       * \brief creates a summation object which defines the local folded summations
       * \param[out] fold_sum the summation object created
       * \param[out] all_fdim_A number of dimensions of A folded
       * \param[out] all_fdim_B number of dimensions of B folded
       * \param[out] all_flen_A lengths of dimensions of A folded
       * \param[out] all_flen_B lengths of dimensions of B folded
       */
      void get_fold_sum(summation *& fold_sum,
                        int &        all_fdim_A,
                        int &        all_fdim_B,
                        int64_t *&   all_flen_A,
                        int64_t *&   all_flen_B);


      /**
       * \brief fold tensors into matrices for summation
       * \return inner stride (daxpy size)
       */
      int map_fold();

      /**
       * \brief estimates time it takes to transpose tensors in order to fold them
       * \return time in seconds
       */
      double est_time_fold();


      /**
       * \brief find ordering of indices of tensor to reduce to DAXPY
       *
       * \param[out] new_ordering_A the new ordering for indices of A
       * \param[out] new_ordering_B the new ordering for indices of B
       */
      void get_len_ordering(int ** new_ordering_A,
                            int ** new_ordering_B);
  

      /**
       * \brief constructs function pointer to sum tensors A and B,B = B*beta+alpha*A
       * \return tsum summation class pointer to run
      */
      tsum * construct_sum(int inner_stride=-1);

      /**
       * \brief constructs function pointer to sum tensors A and B at least one of which is sparse,
       *            B = B*beta+alpha*A
       * \param[in] virt_dim dimensions of grid of blocks owned by each process
       * \param[in] phys_mapped dimension 2*num_phys_dim, keeps track of which dimensions A and B are mapped to
       * \return tspsum summation class pointer to run
      */
      tspsum * construct_sparse_sum(int const * phys_mapped);

      /**
       * \brief constructs function pointer to sum tensors A and B both of which are dense,
       *            B = B*beta+alpha*A
       * \param[in] phys_mapped dimension 2*num_phys_dim, keeps track of which dimensions A and B are mapped to
       * \return tsum summation class pointer to run
      */
      tsum * construct_dense_sum(int         inner_stride,
                                 int const * phys_mapped);


      /**
       * \brief a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B).
       *        performs all necessary symmetric permutations
       * \param[in] run_diag if 1 run diagonal sum
       */
      int sym_sum_tsr(bool run_diag);

      /**
       * \brief unfolds a broken symmetry in a summation by defining new tensors
       * \param[out] new_sum new summations specification (new tsrss)
       * \return 3*idx+tsr_type if finds broken sym,-1 otherwise
       */
      int unfold_broken_sym(summation ** new_sum);

      /**
       * \brief checks the edge lengths specfied for this sum match
       *          throws error if not
       * \return whether contraction indices are consistent with lengths
       */
      bool check_consistency();


      /**
       * \brief checks whether mapping of tensors to topology is valid for this summation 
       * \return 1 if valid 0 if not
      */
      int check_mapping();

      /**
       * \brief map the indices which are common in a sum
       *
       * \param topo topology to map to
       * \return status corresponding to mapping success or failure
       */
      int map_sum_indices(topology const * topo);

      /**
       * \brief find best possible mapping for summation and redistribute tensors to this mapping
       * \return SUCCESS if valid mapping found, ERROR if not enough memory or another issue
       */
      int map();
      
      /**
       * \brief performs a sum when one or both of the tensors are sparse
       */
      void sp_sum();
  };
}

#endif
