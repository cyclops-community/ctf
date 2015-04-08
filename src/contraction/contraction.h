#ifndef __INT_CONTRACTION_H__
#define __INT_CONTRACTION_H__

#include "assert.h"
#include "ctr_comm.h"

namespace CTF_int {
  class tensor; 
  class topology; 

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
      int * idx_A;
      /** \brief indices of right operand */
      int * idx_B;
      /** \brief indices of output */
      int * idx_C;
      /** \brief whether there is a elementwise custom function */
      bool is_custom;
      /** \brief function to execute on elements */
      bivar_function * func;

      /** \brief lazy constructor */
      contraction(){ idx_A = NULL; idx_B = NULL; idx_C=NULL; is_custom=0; alpha=NULL; beta=NULL; };
      
      /** \brief destructor */
      ~contraction();

      /** \brief copy constructor \param[in] other object to copy */
      contraction(contraction const & other);

      /**
       * \brief constructor definining contraction with C's mul and add ops
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] alpha scaling factor (can be NULL) alpha * A[idx_A] * B[idx_B];
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] beta scaling factor of output (can be NULL)
                      C[idx_C] = beta*C[idx_C] 
                                + alpha * A[idx_A] * B[idx_B];
       */
      /*contraction(tensor *     A,
                  int const *  idx_A,
                  tensor *     B,
                  int const *  idx_B,
                  char const * alpha,
                  tensor *     C,
                  int const *  idx_C,
                  char const * beta);*/
     
      /**
       * \brief constructor definining contraction with custom function
       * \param[in] A left operand tensor
       * \param[in] idx_A indices of left operand
       * \param[in] B right operand tensor
       * \param[in] idx_B indices of right operand
       * \param[in] alpha scaling factor (can be NULL) alpha * A[idx_A] * B[idx_B];
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] beta scaling factor of output (can be NULL)
                      C[idx_C] = beta*C[idx_C] 
                                + alpha * A[idx_A] * B[idx_B];
       * \param[in] func custom elementwise function 
                      func(A[idx_A],B[idx_B],&C[idx_C])
       */
      contraction(tensor *         A,
                  int const *      idx_A,
                  tensor *         B,
                  int const *      idx_B,
                  char const *     alpha,
                  tensor *         C,
                  int const *      idx_C,
                  char const *     beta,
                  bivar_function * func=NULL);
      contraction(tensor *         A,
                  char const *     idx_A,
                  tensor *         B,
                  char const *     idx_B,
                  char const *     alpha,
                  tensor *         C,
                  char const *     idx_C,
                  char const *     beta,
                  bivar_function * func=NULL);


      /** \brief run contraction */
      void execute();
      
      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();

      /**
       * \brief returns 1 if contractions have same tensors and index map
       * \param[in] os contraction object to compare this with
       */
      int is_equal(contraction const & os);

    private:
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
      void calc_fold_nmk( int const *    ordering_A,
                          int const *    ordering_B,
                          iparam *       inner_prm);


      /**
       * \brief finds and return all contraction indices which can be folded into
       *    dgemm, for which they must (1) not break symmetry (2) belong to 
       *    exactly two of (A,B,C).
       * \param[out] num_fold number of indices that can be folded
       * \param[out] fold_idx indices that can be folded
       */
      void get_fold_indices(int *  num_fold,
                            int ** fold_idx);
      
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
                int ** new_ordering_A,
                int ** new_ordering_B,
                int ** new_ordering_C);

      /**
       * \brief folds tensors for contraction
       * \return inner_prm parameters includng n,m,k
       */
      iparam map_fold();

      /**
       * \brief unfolds a broken symmetry in a contraction by defining new tensors
       * \param[out] new_contraction new contractions specification (new tsrss)
       * \return 3*idx+tsr_type if finds broken sym,-1 otherwise
       */
      int unfold_broken_sym(contraction ** new_contraction);

      /**
       * \brief checks the edge lengths specfied for this contraction match
       *          throws error if not
       */
      void check_consistency();

      /**
       * \brief checks whether mapping of tensors to topology is valid for this contraction 
       * \return 1 if valid 0 if not
      */
      int check_mapping();

      /**
       * \brief maps tensors to topology 
       *        with certain tensor ordering e.g. BCA
       *
       * \param topo topology to map to 
       * \param order order of tensors (BCA, ACB, ABC, etc.)
       * \param idx_ctr buffer for contraction index storage
       * \param idx_extra buffer for extra index storage
       * \param idx_no_ctr buffer for non-contracted index storage
       * \param idx_weigh buffer for weigh index storage
       */
      int map_to_topology(topology * topo,
                          int        order);
/*                          int *      idx_arr,
                          int *      idx_ctr,
                          int *      idx_extra,
                          int *      idx_no_ctr,
                          int *      idx_weigh);*/

      /**
       * \brief attempts to remap 3 tensors to the same topology if possible
       */
      int try_topo_morph();

      /**
       * \brief find best possible mapping for contraction and redistribute tensors to this mapping
       * \param[out] ctrf contraction class to run
       * \param[in] do_remap whether to redistribute tensors
       * \return SUCCESS if valid mapping found, ERROR if not enough memory or another issue
       */
      int map(ctr ** ctrf, bool do_remap=1);
 
      /**
        * \brief contracts tensors alpha*A*B+beta*C -> C.
        * \param[in] is_inner whether the tensors have two levels of blocking
        *                     0->no blocking 1->folding
        * \param[in] inner_params parameters for inner contraction
        * \param[out] nvirt_all total virtualization factor
        * \param[in] is_used whether this ctr pointer will actually be run
        * \return ctr contraction class to run
        */
      ctr * construct_ctr(int            is_inner=0,
                          iparam const * inner_params=NULL,
                          int *          nvirt_all=NULL,
                          int            is_used=1);

      /**
       * \brief contracts tensors alpha*A*B+beta*C -> C
       * \return completion status
       */
      int contract();

      /**
       * \brief contracts tensors alpha*A*B+beta*C -> C performs all symmetric permutations
       * \return completion status
       */
      int sym_contract();

      /**
       * \brief contracts tensors alpha*A*B+beta*C -> C,
       *          returns tensors to initial buffer location
       * \return completion status
       */
      int home_contract();

      /** \brief print contraction details */
      void print();
  };


  /**
   * \brief sets up a ctr_2d_general (2D SUMMA) level where A is not communicated
   *        function will be called with A/B/C permuted depending on desired alg
   *
   * \param[in] is_used whether this ctr will actually be run
   * \param[in] global_comm comm for this CTF instance
   * \param[in] i index in the total index map currently worked on
   * \param[in,out] virt_dim virtual processor grid lengths
   * \param[out] cg_edge_len edge lengths of ctr_2d_gen object to set
   * \param[in,out] total_iter the total number of ctr_2d_gen iterations
   * \param[in] A A tensor
   * \param[in] i_A the index in A to which index i corresponds
   * \param[out] cg_cdt_A the communicator for A to be set for ctr_2d_gen
   * \param[out] cg_ctr_lda_A parameter of ctr_2d_gen corresponding to upper lda for lda_cpy
   * \param[out] cg_ctr_sub_lda_A parameter of ctr_2d_gen corresponding to lower lda for lda_cpy
   * \param[out] cg_move_A tells ctr_2d_gen whether A should be communicated
   * \param[in,out] blk_len_A lengths of local A piece after this ctr_2d_gen level
   * \param[in,out] blk_sz_A size of local A piece after this ctr_2d_gen level
   * \param[in] virt_blk_edge_len_A edge lengths of virtual blocks of A
   * \param[in] load_phase_A tells the offloader how often A buffer changes for ctr_2d_gen
   *
   * ... the other parameters are specified the same as for _A but this time for _B and _C
   */
  int  ctr_2d_gen_build(int                        is_used,
                        CommData                   global_comm,
                        int                        i,
                        int *                      virt_dim,
                        int &                      cg_edge_len,
                        int &                      total_iter,
                        tensor *                   A,
                        int                        i_A,
                        CommData *&                cg_cdt_A,
                        int64_t &                  cg_ctr_lda_A,
                        int64_t &                  cg_ctr_sub_lda_A,
                        bool &                     cg_move_A,
                        int *                      blk_len_A,
                        int64_t &                  blk_sz_A,
                        int const *                virt_blk_len_A,
                        int &                      load_phase_A,
                        tensor *                   B,
                        int                        i_B,
                        CommData *&                cg_cdt_B,
                        int64_t &                  cg_ctr_lda_B,
                        int64_t &                  cg_ctr_sub_lda_B,
                        bool &                     cg_move_B,
                        int *                      blk_len_B,
                        int64_t &                  blk_sz_B,
                        int const *                virt_blk_len_B,
                        int &                      load_phase_B,
                        tensor *                   C,
                        int                        i_C,
                        CommData *&                cg_cdt_C,
                        int64_t &                  cg_ctr_lda_C,
                        int64_t &                  cg_ctr_sub_lda_C,
                        bool &                     cg_move_C,
                        int *                      blk_len_C,
                        int64_t &                  blk_sz_C,
                        int const *                virt_blk_len_C,
                        int &                      load_phase_C);
}

#endif
