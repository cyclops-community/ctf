/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#if 0 //ndef __DIST_TENSOR_INTERNAL_H__
#define __DIST_TENSOR_INTERNAL_H__

#include "../shared/util.h"
#include "../ctr_comm/scale_tsr.h"
#include "../ctr_comm/sum_tsr.h"
#include "../ctr_comm/ctr_tsr.h"
#include "../ctr_comm/ctr_comm.h"
#include "../ctr_seq/seq_tsr.h"
#include "../ctr_seq/int_functions.h"
#include "../ctr_seq/int_semiring.h"
#include "int_world.h"
#include <limits.h>
#include <stdint.h>

#define NREQ    4
#define NBCAST  4

typedef int64_t int64_t;
typedef int64_t key;

static const char * SY_strings[4] = {"NS", "SY", "AS", "SH"};

class tkv_pair {
  private:
    char * d;
  public: 
    key k;
    tkv_pair() {}
    tkv_pair(key k_, char const * d_, int len) {
      k = k_;
      memcpy(d, d_, len); 
    } 
    bool operator< (const tkv_pair& other) const{
      return k < other.k;
    }
/*  bool operator==(const tkv_pair& other) const{
    return (k == other.k && d == other.d);
  }
  bool operator!=(const tkv_pair& other) const{
    return !(*this == other);
  }*/
};

//template<typename dtype>
inline bool comp_tkv_pair(tkv_pair i, tkv_pair j) {
  return (i.k<j.k);
}
/**
 * \defgroup internal Tensor mapping and redistribution internals
 * @{
 */

class dist_tensor{


  public:

    ~dist_tensor();

    int dist_cleanup();
    CommData get_global_comm();
    void set_global_comm(CommData   cdt);
    CommData get_phys_comm();
    void set_phys_comm(CommData *   cdt, int ndim, int fold=1);
    int get_phys_ndim();
    int * get_phys_lda();
    std::vector< tensor* > * get_tensors();

    int initialize(CommData   cdt_global,
                   int          ndim,
                   int const *  dim_len);


    int define_tensor(int          ndim,
                      int const *  edge_len,
                      int const *  sym,
                      int *        tensor_id,
                      int          alloc_data = 1,
                      char const * name = NULL,
                      int          profile = 0);


    int set_tsr_data(int    tensor_id,
                     int    num_val,
                     char * tsr_data);

    topology * get_topo(int itopo);
    int get_dim(int tensor_id) const;
    int * get_edge_len(int tensor_id) const;
    int * get_sym(int tensor_id) const;
    char * get_raw_data(int tensor_id, int64_t * size);
    
    /* set the tensor name */
    int set_name(int tensor_id, char const * name);
    
    /* get the tensor name */
    int get_name(int tensor_id, char const ** name);
    
    /* turn on profiling */
    int profile_on(int tensor_id);
    
    /* turn off profiling */
    int profile_off(int tensor_id);


    int get_tsr_info(int tensor_id,
                     int * ndim,
                     int ** edge_len,
                     int ** sym) const;

    int permute_tensor(int           tid_A,
                       int * const * permutation_A,
                       char const *  alpha,
                       dist_tensor * dt_A,
                       int           tid_B,
                       int * const * permutation_B,
                       char const *  beta,
                       dist_tensor * dt_B);
    
    void orient_subworld(int           ndim,
                        int            tid_sub,
                        dist_tensor *  dt_sub,
                        int &          bw_mirror_rank,
                        int &          fw_mirror_rank,
                        distribution & odst,
                        char * *       sub_buffer_);

    int  add_to_subworld(int           tid,
                         int           tid_sub,
                         dist_tensor * dt_sub,
                         char const *  alpha,
                         char const *  beta);
    
    int  add_from_subworld(int           tid,
                           int           tid_sub,
                           dist_tensor * dt_sub,
                           char const *  alpha,
                           char const *  beta);
    
    /* Add tensor data from A to a block of B, 
       B[offsets_B:ends_B] = beta*B[offsets_B:ends_B] 
                          + alpha*A[offsets_A:ends_A] */
    int slice_tensor(int           tid_A,
                     int const *   offsets_A,
                     int const *   ends_A,
                     char const *  alpha,
                     dist_tensor * dt_A,
                     int           tid_B,
                     int const *   offsets_B,
                     int const *   ends_B,
                     char const *  beta,
                     dist_tensor * dt_B);
    
    
    
    int write_pairs(int              tensor_id,
                    int64_t          num_pair,
                    char const *     alpha,
                    char const *     beta,
                    tkv_pair * const mapped_data,
                    char const       rw);

    int read_local_pairs(int          tensor_id,
                         int64_t *    num_pair,
                         tkv_pair **  mapped_data);

    int allread_tsr(int        tensor_id,
                    int64_t * num_val,
                    char * *   all_data,
                    int        is_prealloced=0);


    tsum* construct_sum(char const *          alpha,
                        char const *          beta,
                        int                   tid_A,
                        int const *           idx_A,
                        int                   tid_B,
                        int const *           idx_B,
                        Int_Univar_Function * func_ptr,
                        int                   inr_str=-1);
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
    int64_t estimate_cost(int         tid_A,
                          int const * idx_A,
                          int         tid_B,
                          int const * idx_B,
                          int         tid_C,
                          int const * idx_C);
    
    /**
     * \brief estimate the cost of a sum B[idx_B] = A[idx_A]
     * \param[in] A first operand tensor
     * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
     * \param[in] B second operand tensor
     * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
     * \return cost as a int64_t type, currently a rought estimate of flops/processor
     */
    int64_t estimate_cost(int         tid_A,
                          int const * idx_A,
                          int         tid_B,
                          int const * idx_B);
    

    int check_contraction(CTF_ctr_type_t const * type);
    
    int check_sum(CTF_sum_type_t const * type);

    int check_sum(int         tid_A, 
                  int         tid_B, 
                  int const * idx_map_A, 
                  int const * idx_map_B);
    
    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sym_sum_tsr(char const *           alpha,
                    char const *           beta,
                    CTF_sum_type_t const * type,
                    Int_Univar_Function *  func_ptr,
                    int                    run_diag = 0);
    
    int sym_sum_tsr(char const *           alpha_,
                    char const *           beta,
                    int                    tid_A,
                    int                    tid_B,
                    int const *            idx_map_A,
                    int const *            idx_map_B,
                    Int_Univar_Function *  func_ptr,
                    int                    run_diag);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int home_sum_tsr(char const *          alpha,
                     char const *          beta,
                     int                   tid_A,
                     int                   tid_B,
                     int const *           idx_map_A,
                     int const *           idx_map_B,
                     Int_Univar_Function * func_ptr,
                     int                   run_diag = 0);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(char const *          alpha,
                    char const *          beta,
                    int                   tid_A,
                    int                   tid_B,
                    int const *           idx_map_A,
                    int const *           idx_map_B,
                    Int_Univar_Function * func_ptr,
                    int                   run_diag = 0);

    ctr * construct_contraction(CTF_ctr_type_t const * type,
                                Int_Bivar_Function *   func_ptr,
                                char const *           alpha,
                                char const *           beta,
                                int                    is_inner = 0,
                                iparam const *         inner_params = NULL,
                                int *                  nvirt_C = NULL,
                                int                    is_used = 1);

/*    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B);

    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B,
                                int ndim_C, int* idx_C, int* sym_C);

    dtype overcounting_factor(int ndim_A, int* idx_A, int* sym_A,
                            int ndim_B, int* idx_B, int* sym_B,
                            int ndim_C, int* idx_C, int* sym_C);
*/
    int home_contract(CTF_ctr_type_t const * type,
                      Int_Bivar_Function *   func_ptr,
                      char const *           alpha,
                      char const *           beta);

    int sym_contract( CTF_ctr_type_t const * type,
                      Int_Bivar_Function *   func_ptr,
                      char const *           alpha,
                      char const *           beta);

    int contract(CTF_ctr_type_t const * type,
                 Int_Bivar_Function *   func_ptr,
                 char const *           alpha,
                 char const *           beta);

    int map_tensors(CTF_ctr_type_t const * type,
                    Int_Bivar_Function *   func_ptr,
                    char const *           alpha,
                    char const *           beta,
                    ctr **                 ctrf,
                    int                    do_remap=1);

    int map_sum_indices(int const *             idx_arr,
                        int const *             idx_sum,
                        int                     num_tot,
                        int                     num_sum,
                        int                     tid_A,
                        int                     tid_B,
                        topology const *        topo,
                        int                     idx_num);

    int map_weigh_indices(int const *             idx_arr,
                          int const *             idx_weigh,
                          int                     num_tot,
                          int                     num_weigh,
                          int                     tid_A,
                          int                     tid_B,
                          int                     tid_C,
                          topology const *        topo);

    int map_ctr_indices(int const *             idx_arr,
                        int const *             idx_ctr,
                        int                     num_tot,
                        int                     num_ctr,
                        int                     tid_A,
                        int                     tid_B,
                        topology const *        topo);

    int map_no_ctr_indices(int const *          idx_arr,
                           int const *          idx_ctr,
                           int                  num_tot,
                           int                  num_ctr,
                           int                  tid_A,
                           int                  tid_B,
                           int                  tid_C,
                           topology const *     topo);

    int map_extra_indices(int const *   idx_arr,
                          int const *   idx_extra,
                          int           num_extra,
                          int           tid_A,
                          int           tid_B,
                          int           tid_C);

    int map_self_indices(int            tid,
                                           int const*   idx_map);

    int check_contraction_mapping(CTF_ctr_type_t const * type);

    int check_sum_mapping(int           tid_A,
                          int const *   idx_A,
                          int           tid_B,
                          int const *   idx_B);

    int check_self_mapping(int          tid,
                           int const *  idx_map);

    int check_pair_mapping(int tid_A, int tid_B);

    int map_tensor_pair(int tid_A, int tid_B);

    int map_tensor_pair(int             tid_A,
                        int const *     idx_map_A,
                        int             tid_B,
                        int const *     idx_map_B);

    int daxpy_local_tensor_pair(char const * alpha, int tid_A, int tid_B);


    int cpy_tsr(int tid_A, int tid_B);

    int clone_tensor(int tid, int copy_data,
                     int * new_tid, int alloc_data=1);

    int scale_tsr(char const *alpha, int tid);

    int scale_tsr(char const *       alpha,
                  int                tid,
                  int const *        idx_map,
                  Int_Endomorphism * func_ptr);

    int dot_loc_tsr(int tid_A, int tid_B, char * product);

    int red_tsr(int tid, CTF_OP op, char * result);

    int del_tsr(int tid);

    int map_tsr(int tid,
                char * (*map_func)(int ndim, int const * indices,
                                   char * elem));

    int get_max_abs(int        tid,
                    int        n,
                    char *     data);

    int set_zero_tsr(int tensor_id);

    int print_tsr(FILE * stream, int tid, double cutoff = -1.0);

    int compare_tsr(FILE * stream, int tid_A, int tid_B, double cutoff = -1.0);

    int print_map(FILE * stream, int tid,
                  int all=1) const;

    int print_ctr(CTF_ctr_type_t const * ctype,
                  char const *           alpha,
                  char const *           beta) const;

    int print_sum(CTF_sum_type_t const * stype,
                  char const *           alpha,
                  char const *           beta) const;

    int zero_out_padding(int tensor_id);

    int try_topo_morph( int         tid_A,
                        int         tid_B,
                        int         tid_C);

    int map_to_topology(int         tid_A,
                        int         tid_B,
                        int         tid_C,
                        int const * idx_map_A,
                        int const * idx_map_B,
                        int const * idx_map_C,
                        int         itopo,
                        int         order,
                        int *       idx_arr,
                        int *       idx_ctr,
                        int *       idx_extra,
                        int *       idx_no_ctr,
                        int *       idx_weigh);

    int map_inner(CTF_ctr_type_t const * type,
                  iparam * inner_params);

    int map_to_inr_topo(int         tid_A,
                        int         tid_B,
                        int         tid_C,
                        int const * idx_map_A,
                        int const * idx_map_B,
                        int const * idx_map_C,
                        int         itopo,
                        int         order,
                        int *       idx_ctr,
                        int *       idx_extra,
                        int *       idx_no_ctr);

    void unmap_inner(tensor * tsr);

    void get_new_ordering(CTF_ctr_type_t const * type,
                          int **                 new_odering_A,
                          int **                 new_odering_B,
                          int **                 new_odering_C);

    int remap_inr_tsr(tensor *    otsr,
                      tensor *    itsr,
                      int64_t     old_size,
                      int const * old_phase,
                      int const * old_rank,
                      int const * old_virt_dim,
                      int const * old_pe_lda,
                      int         was_cyclic,
                      int const * old_padding,
                      int const * old_edge_len,
                      int const * ordering);

    void get_fold_indices(CTF_sum_type_t const *type,
                          int *                 num_fold,
                          int **                fold_idx);

    void get_fold_indices(CTF_ctr_type_t const *type,
                          int *                 num_fold,
                          int **                fold_idx);

    int can_fold(CTF_sum_type_t const * type);

    int can_fold(CTF_ctr_type_t const * type);

    void fold_tsr(tensor *    tsr,
                  int         nfold,
                  int const * fold_idx,
                  int const * idx_map,
                  int *       all_fdim,
                  int **      all_flen);

    void unfold_tsr(tensor * tsr);

    int map_fold(CTF_sum_type_t const * type,
                 int *                  inner_stride);

    int map_fold(CTF_ctr_type_t const * type,
                 iparam *               inner_prm);

    int unfold_broken_sym(CTF_ctr_type_t const * type,
                          CTF_ctr_type_t *       new_type);

    int unfold_broken_sym(CTF_sum_type_t const * type,
                          CTF_sum_type_t *       new_type);

    void dealias(int sym_tid, int nonsym_tid);

    void desymmetrize(int sym_tid,
                      int nonsym_tid,
                      int is_C);

    void symmetrize(int sym_tid, int nonsym_tid);

    void copy_type(CTF_ctr_type_t const * old_type,
                   CTF_ctr_type_t *       new_type);
    
    void free_type(CTF_ctr_type_t * old_type);
    
    int is_equal_type(CTF_ctr_type_t const * old_type,
                      CTF_ctr_type_t const * new_type);

    void order_perm(tensor const * tsr_A,
                    tensor const * tsr_B,
                    tensor const * tsr_C,
                    int *          idx_arr,
                    int            off_A,
                    int            off_B,
                    int            off_C,
                    int *          idx_map_A,
                    int *          idx_map_B,
                    int *          idx_map_C,
                    double &       add_sign,
                    int &          mod);

    void get_sym_perms(CTF_ctr_type_t const *       type,
                       char const *                 alpha,
                       std::vector<CTF_ctr_type_t>& perms,
                       std::vector&                 signs);

    void add_sym_perm(std::vector<CTF_ctr_type_t>& perms,
                      std::vector&                 signs, 
                      CTF_ctr_type_t const *       new_perm,
                      char *                       new_sign);
    
    void copy_type(CTF_sum_type_t const * old_type,
                   CTF_sum_type_t *       new_type);
    
    void free_type(CTF_sum_type_t * old_type);
    
    int is_equal_type(CTF_sum_type_t const * old_type,
                      CTF_sum_type_t const * new_type);

    void order_perm(tensor const * tsr_A,
                    tensor const * tsr_B,
                    int *          idx_arr,
                    int            off_A,
                    int            off_B,
                    int *          idx_map_A,
                    int *          idx_map_B,
                    double &       add_sign,
                    int &          mod);

    void get_sym_perms(CTF_sum_type_t const *       type,
                       char const *                 alpha,
                       std::vector<CTF_sum_type_t>& perms,
                       std::vector&                 signs);

    void add_sym_perm(std::vector<CTF_sum_type_t> & perms,
                      std::vector&                  signs, 
                      CTF_sum_type_t const *        new_perm,
                      char *                        new_sign);

    void get_len_ordering(CTF_sum_type_t const * type,
                          int **                 new_ordering_A,
                          int **                 new_ordering_B);

    void get_len_ordering(CTF_ctr_type_t const * type,
                          int **                 new_ordering_A,
                          int **                 new_ordering_B,
                          int **                 new_ordering_C);

    int extract_diag(int         tid,
                     int const * idx_map,
                     int         rw,
                     int *       tid_new,
                     int **      idx_map_new);

    void contract_mst();

    int elementalize(int      tid,
                     int      x_rank,
                     int      x_np,
                     int      y_rank,
                     int      y_np,
                     int64_t blk_sz,
                     char *   data);

    /* ScaLAPACK back-end */
    int load_matrix(char *      DATA,
                    int const * DESC,
                    int *       tid,
                    int *       need_free);

    int pgemm(char           TRANSA,
              char           TRANSB,
              int            M,
              int            N,
              int            K,
              char *         ALPHA,
              char *         A,
              int            IA,
              int            JA,
              int const *    DESCA,
              char *         B,
              int            IB,
              int            JB,
              int const *    DESCB,
              char *         BETA,
              char *         C,
              int            IC,
              int            JC,
              int const *    DESCC,
              CTF_ctr_type * pct,
              //fseq_tsr_ctr *pfs,
              int *          need_free);

};


/**
 * @}
 */
#endif
