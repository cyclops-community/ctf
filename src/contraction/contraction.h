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
      bivar_function func;

      /** \brief lazy constructor */
      contraction(){ idx_A = NULL; idx_B = NULL; idx_C=NULL; };
      
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
       * \param[in] alpha scaling factor alpha * A[idx_A] * B[idx_B];
       * \param[in] C ouput operand tensor
       * \param[in] idx_C indices of right operand
       * \param[in] beta scaling factor of ouput 
                      C[idx_C] = beta*C[idx_C] 
                                + alpha * A[idx_A] * B[idx_B];
       */
      contraction(tensor *     A,
                  int const *  idx_A,
                  tensor *     B,
                  int const *  idx_B,
                  char const * alpha,
                  tensor *     C,
                  int const *  idx_C,
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
      contraction(tensor *       A,
                  int const *    idx_A,
                  tensor *       B,
                  int const *    idx_B,
                  tensor *       C,
                  int const *    idx_C,
                  bivar_function func);

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
                          tensor const * tsr_A,
                          tensor const * tsr_B,
                          tensor const * tsr_C,
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
       * \brief map the indices over which we will be weighing
       *
       * \param idx_arr array of index mappings of size order*3 that
       *        lists the indices (or -1) of A,B,C corresponding to every global index
       * \param idx_weigh specification of which indices are being contracted
       * \param num_tot total number of indices
       * \param num_weigh number of indices being contracted over
       * \param topo topology to map to
       */
      int map_weigh_indices(int const *      idx_arr,
                            int const *      idx_weigh,
                            int              num_tot,
                            int              num_weigh,
                            topology const * topo);

    /**
       * \brief map the indices over which we will be contracting
       *
       * \param idx_arr array of index mappings of size order*3 that
       *        lists the indices (or -1) of A,B,C 
       *        corresponding to every global index
       * \param idx_ctr specification of which indices are being contracted
       * \param num_tot total number of indices
       * \param num_ctr number of indices being contracted over
       * \param topo topology to map to
       */
      int map_ctr_indices(int const *      idx_arr,
                          int const *      idx_ctr,
                          int              num_tot,
                          int              num_ctr,
                          topology const * topo);

      /**
       * \brief map the indices over which we will not be contracting
       *
       * \param idx_arr array of index mappings of size order*3 that
       *        lists the indices (or -1) of A,B,C 
       *        corresponding to every global index
       * \param idx_noctr specification of which indices are not being contracted
       * \param num_tot total number of indices
       * \param num_noctr number of indices not being contracted over
       * \param topo topology to map to
       */
      int map_no_ctr_indices(int const *              idx_arr,
                             int const *              idx_no_ctr,
                             int                      num_tot,
                             int                      num_no_ctr,
                             topology const *         topo);

      /**
       * \brief map the indices which are indexed only for A or B or C
       *
       * \param idx_arr array of index mappings of size order*3 that
       *        lists the indices (or -1) of A,B,C 
       *        corresponding to every global index
       * \param idx_extra specification of which indices are not being contracted
       * \param num_extra number of indices not being contracted over
       */
      int map_extra_indices(int const * idx_arr,
                            int const * idx_extra,
                            int         num_extra);

      /**
       * \brief maps tensors to topology 
       *        with certain tensor ordering e.g. BCA
       *
       * \param itopo topology index
       * \param order order of tensors (BCA, ACB, ABC, etc.)
       * \param idx_ctr buffer for contraction index storage
       * \param idx_extra buffer for extra index storage
       * \param idx_no_ctr buffer for non-contracted index storage
       * \param idx_weigh buffer for weigh index storage
       */
      int map_to_topology(int   itopo,
                          int   order,
                          int * idx_arr,
                          int * idx_ctr,
                          int * idx_extra,
                          int * idx_no_ctr,
                          int * idx_weigh);

      /**
       * \brief attempts to remap 3 tensors to the same topology if possible
       */
      int try_topo_morph();

      /**
       * \brief find best possible mapping for contraction and redistribute tensors to this mapping
       * \param[out] ctrf contraction class to run
       * \return SUCCESS if valid mapping found, ERROR if not enough memory or another issue
       */
      int map(ctr ** ctrf);
 
      /**
        * \brief contracts tensors alpha*A*B+beta*C -> C.
        * \param[in] is_inner whether the tensors have two levels of blocking
        *                     0->no blocking 1->inner_blocking 2->folding
        * \param[in] inner_params parameters for inner contraction
        * \param[out] nvirt_all total virtualization factor
        * \param[in] is_used whether this ctr pointer will actually be run
        * \return ctr contraction class to run
        */
      ctr * construct_ctr(int            is_inner=0,
                          iparam const * inner_params=NULL,
                          int *          nvirt_C=NULL,
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
  };
}

#endif
