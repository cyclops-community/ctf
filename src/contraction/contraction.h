#ifndef __INT_CONTRACTION_H__
#define __INT_CONTRACTION_H__

#include <assert.h>
#include "ctr_tsr.h"

namespace CTF_int {
  class tensor; 
  class topology; 
  class distribution;
  class mapping;

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
      /** \brief scaling of existing C */
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
      bivar_function const * func;

      /** \brief predefined output nonzero density */
      double output_nnz_frac = -1.;

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
      contraction(tensor *               A,
                  int const *            idx_A,
                  tensor *               B,
                  int const *            idx_B,
                  char const *           alpha,
                  tensor *               C,
                  int const *            idx_C,
                  char const *           beta,
                  bivar_function const * func=NULL);
      contraction(tensor *               A,
                  char const *           idx_A,
                  tensor *               B,
                  char const *           idx_B,
                  char const *           alpha,
                  tensor *               C,
                  char const *           idx_C,
                  char const *           beta,
                  bivar_function const * func=NULL);


      /** \brief run contraction */
      void execute();
      
      /** \brief set output sparsity fraction
        * \param[in] nnz_frac density of nonzeros, in [0,1]
        */
      void set_output_nnz_frac(double nnz_frac);

      /** \brief predicts sparsityu fraction of output */
      double estimate_output_nnz_frac();

      /** \brief predicts number of flops, treatign all tensors as dense even if they are sparse */
      double estimate_num_dense_flops();

      /** \brief predicts number of flops assuming random distribution of nonzeros in sparse tensors */
      double estimate_num_flops();

      /** \brief predicts amount of data movement */
      double estimate_bw();


      /** \brief predicts execution time in seconds using performance models */
      double estimate_time();

      /**
       * \brief returns 1 if contractions have same tensors and index map
       * \param[in] os contraction object to compare this with
       */
      int is_equal(contraction const & os);

      /** \brief print contraction details */
      void print();

    private:
      /**
       * \brief returns true if one of the tensors is sparse 
       */
      bool is_sparse();

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
       * \brief creates a contraction object which defines the local folded contractions
       * \param[out] fold_ctr the contraction object created
       * \param[out] all_fdim_A number of dimensions of A folded
       * \param[out] all_fdim_B number of dimensions of B folded
       * \param[out] all_fdim_C number of dimensions of C folded
       * \param[out] all_flen_A lengths of dimensions of A folded
       * \param[out] all_flen_B lengths of dimensions of B folded
       * \param[out] all_flen_C lengths of dimensions of C folded
       */
      void get_fold_ctr(contraction *& fold_ctr,
                        int &          all_fdim_A,
                        int &          all_fdim_B,
                        int &          all_fdim_C,
                        int64_t *&     all_flen_A,
                        int64_t *&     all_flen_B,
                        int64_t *&     all_flen_C);


      /**
       * \brief picks the dimension ordering which can be expressed as gemm and requires fewest/cheapest transpositions
       * \param[in] fold_ctr the folded contraction object
       * \param[in] all_fdim_A number of dimensions of A folded
       * \param[in] all_fdim_B number of dimensions of B folded
       * \param[in] all_fdim_C number of dimensions of C folded
       * \param[in] all_flen_A lengths of dimensions of A folded
       * \param[in] all_flen_B lengths of dimensions of B folded
       * \param[in] all_flen_C lengths of dimensions of C folded
       * \param[out] bperm_order in [0,5] corresponds to ordering of {A,B,C}
       * \param[out] btime estimated execution time of the selected permutation
       * \param[out] iprm the local paramaters associated with this perm
       */
      void select_ctr_perm(contraction const * fold_ctr,
                           int                 all_fdim_A,
                           int                 all_fdim_B,
                           int                 all_fdim_C,
                           int64_t const *     all_flen_A,
                           int64_t const *     all_flen_B,
                           int64_t const *     all_flen_C,
                           int &               bperm_order,
                           double &            btime,
                           iparam &            iprm);

      /**
       * \brief folds tensors for contraction
       * \param[in] do_transp if false then do not transpose data
       * \return inner_prm parameters includng n,m,k
       */
      iparam map_fold(bool do_transp=true);
      
      /**
       * \brief estimates the time need to fold tensors for contraction
       * \return estimated time for the transposes needed to fold contraction
       */
      double est_time_fold();


      /**
       * \brief unfolds a broken symmetry in a contraction by defining new tensors
       * \param[out] new_contraction new contractions specification (new tsrss)
       * \return 3*idx+tsr_type if finds broken sym,-1 otherwise
       */
      int unfold_broken_sym(contraction ** new_contraction);

      /**
       * \brief checks the edge lengths specfied for this contraction match
       *          throws error if not
       * \return returns whether contraction is consistent
       */
      bool check_consistency();

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
       * \brief get number of mappings variants of contraction for a given topology
       * \param[in] topo topology we want to map to
       */
      int get_num_map_variants(topology const * topo);

      int get_num_map_variants(topology const * topo,
                               int &            nmax_ctr_2d,
                               int &            nAB,
                               int &            nAC,
                               int &            nBC);

      bool switch_topo_perm();

      /**
       * \brief maps tensors to topology 
       *        with certain choice of mapping of topo dims to tensor dims
       *
       * \param topo topology to map to 
       * \param index of mapping choice, parsed inside
       *       A->order*B->order*C->order+A->order*B->order+A->order*C->order+B->order*C->order+A->order+B->order+C->order+1 choices per dimension 
       * \param return true if this could be a valid mapping
       */
      bool exh_map_to_topo(topology const * topo,
                           int              variant);
      /**
       * \brief attempts to remap 3 tensors to the same topology if possible
       */
      int try_topo_morph();

      void calc_nnz_frac(double & nnz_frac_A, double & nnz_frac_B, double & nnz_frac_C);

      void detail_estimate_mem_and_time(distribution const * dA, distribution const * dB, distribution const * dC, topology * old_topo_A, topology * old_topo_B, topology * old_topo_C, mapping const * old_map_A, mapping const * old_map_B, mapping const * old_map_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C, int64_t & memuse, double & est_time);

      void get_best_sel_map(distribution const * dA, distribution const * dB, distribution const * dC, topology * old_topo_A, topology * old_topo_B, topology * old_topo_C, mapping const * old_map_A, mapping const * old_map_B, mapping const * old_map_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C, int64_t & idx, double & time);

      void get_best_exh_map(distribution const * dA, distribution const * dB, distribution const * dC, topology * old_topo_A, topology * old_topo_B, topology * old_topo_C, mapping const * old_map_A, mapping const * old_map_B, mapping const * old_map_C, double nnz_frac_A, double nnz_frac_B, double nnz_frac_C, int64_t & idx, double & time, double init_best_time);

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

      ctr * construct_dense_ctr(int            is_inner,
                                iparam const * inner_params,
                                int *          nvirt_all,
                                int            is_used,
                                int const *    phys_mapped);

      ctr * construct_sparse_ctr(int            is_inner,
                                 iparam const * inner_params,
                                 int *          nvirt_all,
                                 int            is_used,
                                 int const *    phys_mapped);



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

      /**
       * \brief applies scaling factor to diagonals for symmetric groups that are contracted over
       */
      void prescale_operands();

      /**
       * \brief returns true if prescale_operands has real work to do
       */
      bool need_prescale_operands();
  };

  class contraction_signature {
    public:
      int order_A;
      int order_B;
      int order_C;
      int64_t * lens_A;
      int64_t * lens_B; 
      int64_t * lens_C;
      int * idx_A;
      int * idx_B;
      int * idx_C;
      int * sym_A;
      int * sym_B;
      int * sym_C;
      int is_sparse_A;
      int is_sparse_B;
      int is_sparse_C;
      int nnz_tot_A;
      int nnz_tot_B;
      int nnz_tot_C;
      topology * topo_A;
      topology * topo_B;
      topology * topo_C;
      mapping * edge_map_A;
      mapping * edge_map_B;
      mapping * edge_map_C;

      contraction_signature(contraction const & ctr);
      contraction_signature(contraction_signature const & other);
      ~contraction_signature();
      bool operator<(contraction_signature const & other) const;
  };

  class topo_info {
    public:
      int64_t ttopo;
      bool is_exh;

      topo_info(int64_t tt, bool ie);
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
                        int64_t &                  cg_edge_len,
                        int &                      total_iter,
                        tensor *                   A,
                        int                        i_A,
                        CommData *&                cg_cdt_A,
                        int64_t &                  cg_ctr_lda_A,
                        int64_t &                  cg_ctr_sub_lda_A,
                        bool &                     cg_move_A,
                        int64_t *                  blk_len_A,
                        int64_t &                  blk_sz_A,
                        int const *                virt_blk_len_A,
                        int &                      load_phase_A,
                        tensor *                   B,
                        int                        i_B,
                        CommData *&                cg_cdt_B,
                        int64_t &                  cg_ctr_lda_B,
                        int64_t &                  cg_ctr_sub_lda_B,
                        bool &                     cg_move_B,
                        int64_t *                  blk_len_B,
                        int64_t &                  blk_sz_B,
                        int const *                virt_blk_len_B,
                        int &                      load_phase_B,
                        tensor *                   C,
                        int                        i_C,
                        CommData *&                cg_cdt_C,
                        int64_t &                  cg_ctr_lda_C,
                        int64_t &                  cg_ctr_sub_lda_C,
                        bool &                     cg_move_C,
                        int64_t *                  blk_len_C,
                        int64_t &                  blk_sz_C,
                        int const *                virt_blk_len_C,
                        int &                      load_phase_C);


}

#endif
