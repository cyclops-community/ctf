/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __DIST_TENSOR_INTERNAL_H__
#define __DIST_TENSOR_INTERNAL_H__

#include "cyclopstf.hpp"
#include "../ctr_comm/scale_tsr.h"
#include "../ctr_comm/sum_tsr.h"
#include "../ctr_comm/ctr_tsr.h"
#include "../ctr_comm/ctr_comm.h"
#include "../ctr_seq/sym_seq_sum_inner.hxx"
#include "../ctr_seq/sym_seq_ctr_inner.hxx"
#include "../ctr_seq/sym_seq_scl_ref.hxx"
#include "../ctr_seq/sym_seq_sum_ref.hxx"
#include "../ctr_seq/sym_seq_ctr_ref.hxx"
#include "../ctr_seq/sym_seq_scl_cust.hxx"
#include "../ctr_seq/sym_seq_sum_cust.hxx"
#include "../ctr_seq/sym_seq_ctr_cust.hxx"
#if VERIFY
#include "../unit_test/unit_test.h"
#endif
#include <limits.h>
#include <stdint.h>

#define NREQ    4
#define NBCAST  4


enum {
  NOT_MAPPED,
  PHYSICAL_MAP,
  VIRTUAL_MAP
};

struct mapping {
  int type;
  int np;
  int cdt;
  int has_child;
  mapping * child;
};

/* Only supporting mesh/torus topologies */
struct topology {
  int ndim;
  CommData_t ** dim_comm;
  int * lda;
};


template<typename dtype>
struct tensor {
  int ndim;
  int * edge_len;
  int is_padded;
  int * padding;
  int is_scp_padded;
  int * scp_padding; /* to be used by scalapack wrapper */
  int * sym;
  int * sym_table; /* can be compressed into bitmap */
  int is_mapped;
  int is_alloced;
  int itopo;
  mapping * edge_map;
  long_int size;
  int is_inner_mapped;
  int is_folded;
  int * inner_ordering;
  int rec_tid;
  int is_cyclic;
  int is_matrix;
  int is_data_aliased;
  int slay;
  int has_zero_edge_len;
  union {
    dtype * data;
    tkv_pair <dtype> * pairs;
  };
  dtype * home_buffer;
  long_int home_size;
  int is_home;
  int has_home;
  char const * name;
  int profile;
};




template<typename dtype>
int padded_reshuffle(int const          tid,
                     int const          ndim,
                     int const          nval,
                     int const *        old_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const          is_old_pad,
                     int const *        old_padding,
                     int const *        new_edge_len,
                     int const *        new_phase,
                     int const *        new_rank,
                     int const *        new_pe_lda,
                     int const          is_new_pad,
                     int const *        new_padding,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     dtype *            tsr_data,
                     dtype **           tsr_cyclic_data,
                     CommData_t *       ord_glb_comm);

template<typename dtype>
int cyclic_reshuffle(int const          ndim,
                     int const          nval,
                     int const *        old_edge_len,
                     int const *        sym,
                     int const *        old_phase,
                     int const *        old_rank,
                     int const *        old_pe_lda,
                     int const          is_old_pad,
                     int const *        old_padding,
                     int const *        new_edge_len,
                     int const *        new_phase,
                     int const *        new_rank,
                     int const *        new_pe_lda,
                     int const          is_new_pad,
                     int const *        new_padding,
                     int const *        old_virt_dim,
                     int const *        new_virt_dim,
                     dtype **           tsr_data,
                     dtype **           tsr_cyclic_data,
                     CommData_t *       ord_glb_comm,
                     int const          was_cyclic = 0,
                     int const          is_cyclic = 0);

template<typename dtype>
class seq_tsr_ctr : public ctr<dtype> {
  public:
    dtype alpha;
    int ndim_A;
    int * edge_len_A;
    int const * idx_map_A;
    int * sym_A;
    int ndim_B;
    int * edge_len_B;
    int const * idx_map_B;
    int * sym_B;
    int ndim_C;
    int * edge_len_C;
    int const * idx_map_C;
    int * sym_C;
    fseq_tsr_ctr<dtype> func_ptr;

    int is_inner;
    iparam inner_params;
    
    int is_custom;
    fseq_elm_ctr<dtype> custom_params;

    void run();
    void print();
    long_int mem_fp();
    ctr<dtype> * clone();

    seq_tsr_ctr(ctr<dtype> * other);
    ~seq_tsr_ctr(){ CTF_free(edge_len_A), CTF_free(edge_len_B), CTF_free(edge_len_C), 
                    CTF_free(sym_A), CTF_free(sym_B), CTF_free(sym_C); }
    seq_tsr_ctr(){}
};

template<typename dtype>
class seq_tsr_sum : public tsum<dtype> {
  public:
    int ndim_A;
    int * edge_len_A;
    int const * idx_map_A;
    int * sym_A;
    int ndim_B;
    int * edge_len_B;
    int const * idx_map_B;
    int * sym_B;
    fseq_tsr_sum<dtype> func_ptr;

    int is_inner;
    int inr_stride;
    
    int is_custom;
    fseq_elm_sum<dtype> custom_params;

    void run();
    void print();
    long_int mem_fp();
    tsum<dtype> * clone();

    seq_tsr_sum(tsum<dtype> * other);
    ~seq_tsr_sum(){ CTF_free(edge_len_A), CTF_free(edge_len_B), 
                    CTF_free(sym_A), CTF_free(sym_B); };
    seq_tsr_sum(){}
};

template<typename dtype>
class seq_tsr_scl : public scl<dtype> {
  public:
    int ndim;
    int * edge_len;
    int const * idx_map;
    int const * sym;
    fseq_tsr_scl<dtype> func_ptr;

    int is_custom;
    fseq_elm_scl<dtype> custom_params;

    void run();
    void print();
    long_int mem_fp();
    scl<dtype> * clone();

    seq_tsr_scl(scl<dtype> * other);
    ~seq_tsr_scl(){ CTF_free(edge_len); };
    seq_tsr_scl(){}
};


template<typename dtype>
class dist_tensor{

  protected:
    /* internal library state */
    CommData_t * global_comm;
    int num_phys_dims;
    CommData_t * phys_comm;
    int * phys_lda;
    std::vector< tensor<dtype>* > tensors;
    std::vector<topology> topovec;
    std::vector<topology> rejected_topos;

    int inner_size;
    std::vector<topology> inner_topovec;
    

  public:

    ~dist_tensor();

    int dist_cleanup();
    CommData_t * get_global_comm();
    void set_global_comm(CommData_t * cdt);
    CommData_t * get_phys_comm();
    void set_phys_comm(CommData_t ** cdt, int const ndim);
    void set_inner_comm(CommData_t ** cdt, int const ndim);
    int get_phys_ndim();
    int * get_phys_lda();
    std::vector< tensor<dtype>* > * get_tensors();

    int initialize(CommData_t * cdt_global,
                   int const    ndim,
                   int const *  dim_len,
                   int const    inner_sz);

    int init_inner_topology(int const inner_sz);

    int define_tensor(int const         ndim,
                      int const *       edge_len,
                      int const *       sym,
                      int *             tensor_id,
                      int const         alloc_data = 1,
                      char const *      name = NULL,
                      int               profile = 0);


    int set_tsr_data(int const  tensor_id,
                     int const  num_val,
                     dtype * tsr_data);

    topology * get_topo(int const itopo);
    int get_dim(int const tensor_id) const;
    int * get_edge_len(int const tensor_id) const;
    int * get_sym(int const tensor_id) const;
    dtype * get_raw_data(int const tensor_id, long_int * size);
    
    /* set the tensor name */
    int set_name(int const tensor_id, char const * name);
    
    /* get the tensor name */
    int get_name(int const tensor_id, char const ** name);
    
    /* turn on profiling */
    int profile_on(int const tensor_id);
    
    /* turn off profiling */
    int profile_off(int const tensor_id);


    int get_tsr_info(int const tensor_id,
                     int * ndim,
                     int ** edge_len,
                     int ** sym) const;

    int slice_tensor(int const    tid_A,
                     int const *  offsets_A,
                     int const *  ends_A,
                     double const alpha,
                     int const    tid_B,
                     int const *  offsets_B,
                     int const *  ends_B,
                     double const beta);
    
    int write_pairs(int const                 tensor_id,
                    long_int const            num_pair,
                    dtype const               alpha,
                    dtype const               beta,
                    tkv_pair<dtype> * const   mapped_data,
                    char const                rw);

    int read_local_pairs(int const      tensor_id,
                         long_int *     num_pair,
                         tkv_pair<dtype> **     mapped_data);

    int allread_tsr(int const   tensor_id,
                    long_int *  num_val,
                    dtype **    all_data);


    tsum<dtype>* construct_sum( dtype const                     alpha,
                                dtype const                     beta,
                                int const                       tid_A,
                                int const *                     idx_A,
                                int const                       tid_B,
                                int const *                     idx_B,
                                fseq_tsr_sum<dtype> const       ftsr,
                                fseq_elm_sum<dtype> const       felm,
                                int const                       inr_str=-1);

    int check_contraction(CTF_ctr_type_t const * type);
    
    int check_sum(CTF_sum_type_t const * type);

    int check_sum(int const   tid_A, 
                  int const   tid_B, 
                  int const * idx_map_A, 
                  int const * idx_map_B);
    
    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sym_sum_tsr(dtype const                 alpha,
                    dtype const                 beta,
                    CTF_sum_type_t const *      type,
                    fseq_tsr_sum<dtype> const   ftsr,
                    fseq_elm_sum<dtype> const   felm,
                    int const                   run_diag = 0);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int home_sum_tsr( dtype const                 alpha,
                      dtype const                 beta,
                      int const                   tid_A,
                      int const                   tid_B,
                      int const *                 idx_map_A,
                      int const *                 idx_map_B,
                      fseq_tsr_sum<dtype> const   ftsr,
                      fseq_elm_sum<dtype> const   felm,
                      int const                   run_diag = 0);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(dtype const                 alpha,
                    dtype const                 beta,
                    int const                   tid_A,
                    int const                   tid_B,
                    int const *                 idx_map_A,
                    int const *                 idx_map_B,
                    fseq_tsr_sum<dtype> const   ftsr,
                    fseq_elm_sum<dtype> const   felm,
                    int const                   run_diag = 0);

    ctr<dtype> * construct_contraction( CTF_ctr_type_t const *    type,
                                        fseq_tsr_ctr<dtype> const ftsr,
                                        fseq_elm_ctr<dtype> const felm,
                                        dtype const               alpha,
                                        dtype const               beta,
                                        int const                 is_inner=0,
                                        iparam const *            inner_params=NULL,
                                        int *                     nvirt_C = NULL);

/*    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B);

    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B,
                                int ndim_C, int* idx_C, int* sym_C);

    dtype overcounting_factor(int ndim_A, int* idx_A, int* sym_A,
                            int ndim_B, int* idx_B, int* sym_B,
                            int ndim_C, int* idx_C, int* sym_C);
*/
    int home_contract(CTF_ctr_type_t const *    type,
                      fseq_tsr_ctr<dtype> const ftsr,
                      fseq_elm_ctr<dtype> const felm,
                      dtype const               alpha,
                      dtype const               beta,
                      int const                 map_inner);

    int sym_contract( CTF_ctr_type_t const *    type,
                      fseq_tsr_ctr<dtype> const ftsr,
                      fseq_elm_ctr<dtype> const felm,
                      dtype const               alpha,
                      dtype const               beta,
                      int const                 map_inner);

    int contract( CTF_ctr_type_t const *        type,
                  fseq_tsr_ctr<dtype> const     ftsr,
                  fseq_elm_ctr<dtype> const     felm,
                  dtype const                   alpha,
                  dtype const                   beta,
                  int const                     map_inner);

    int map_tensors(CTF_ctr_type_t const *      type,
                    fseq_tsr_ctr<dtype> const   ftsr,
                    fseq_elm_ctr<dtype> const   felm,
                    dtype const                 alpha,
                    dtype const                 beta,
                    ctr<dtype> **               ctrf,
                    int const                   do_remap=1);

    int map_sum_indices(int const *             idx_arr,
                        int const *             idx_sum,
                        int const               num_tot,
                        int const               num_sum,
                        int const               tid_A,
                        int const               tid_B,
                        topology const *        topo,
                        int const               idx_num);

    int map_weigh_indices(int const *             idx_arr,
                          int const *             idx_weigh,
                          int const               num_tot,
                          int const               num_weigh,
                          int const               tid_A,
                          int const               tid_B,
                          int const               tid_C,
                          topology const *        topo);

    int map_ctr_indices(int const *             idx_arr,
                        int const *             idx_ctr,
                        int const               num_tot,
                        int const               num_ctr,
                        int const               tid_A,
                        int const               tid_B,
                        topology const *        topo);

    int map_no_ctr_indices(int const *          idx_arr,
                           int const *          idx_ctr,
                           int const            num_tot,
                           int const            num_ctr,
                           int const            tid_A,
                           int const            tid_B,
                           int const            tid_C,
                           topology const *     topo);

    int map_extra_indices(int const *   idx_arr,
                          int const *   idx_extra,
                          int const     num_extra,
                          int const     tid_A,
                          int const     tid_B,
                          int const     tid_C);

    int map_self_indices(int const      tid,
                                           int const*   idx_map);

    int check_contraction_mapping(CTF_ctr_type_t const * type,
                                  int const is_inner = 0);

    int check_sum_mapping(int const     tid_A,
                          int const *   idx_A,
                          int const     tid_B,
                          int const *   idx_B);

    int check_self_mapping(int const    tid,
                           int const *  idx_map);

    int check_pair_mapping(const int tid_A, const int tid_B);

    int map_tensor_pair(const int tid_A, const int tid_B);

    int map_tensor_pair(const int       tid_A,
                        const int *     idx_map_A,
                        const int       tid_B,
                        const int *     idx_map_B);

    int daxpy_local_tensor_pair(dtype alpha, const int tid_A, const int tid_B);


    int cpy_tsr(int const tid_A, int const tid_B);

    int clone_tensor(int const tid, int const copy_data,
                     int * new_tid, int const alloc_data=1);

    int scale_tsr(dtype const alpha, int const tid);

    int scale_tsr(dtype const                   alpha,
                  int const                     tid,
                  int const *                   idx_map,
                  fseq_tsr_scl<dtype> const     ftsr,
                  fseq_elm_scl<dtype> const     felm);

    int dot_loc_tsr(int const tid_A, int const tid_B, dtype *product);

    int red_tsr(int const tid, CTF_OP op, dtype * result);

    int del_tsr(int const tid);

    int map_tsr(int const tid,
                dtype (*map_func)(int const ndim, int const * indices,
                       dtype const elem));

    int get_max_abs( int const  tid,
                     int const  n,
                     dtype *    data);

    int set_zero_tsr(int const tensor_id);

    int print_tsr(FILE * stream, int const tid, double cutoff = -1.0);

    int compare_tsr(FILE * stream, int const tid_A, int const tid_B, double cutoff = -1.0);

    int print_map(FILE * stream, int const tid,
                  int const all=1, int const is_inner=0) const;

    int print_ctr(CTF_ctr_type_t const * ctype,
                  dtype const            alpha,
                  dtype const            beta) const;

    int print_sum(CTF_sum_type_t const * stype,
                  dtype const            alpha,
                  dtype const            beta) const;

    int zero_out_padding(int const tensor_id);

    int try_topo_morph( int const       tid_A,
                        int const       tid_B,
                        int const       tid_C);

    int map_to_topology(int const       tid_A,
                        int const       tid_B,
                        int const       tid_C,
                        int const *     idx_map_A,
                        int const *     idx_map_B,
                        int const *     idx_map_C,
                        int const       itopo,
                        int const       order,
                        int *           idx_arr,
                        int *           idx_ctr,
                        int *           idx_extra,
                        int *           idx_no_ctr,
                        int *           idx_weigh);

    int map_inner(CTF_ctr_type_t const * type,
                  iparam * inner_params);

    int map_to_inr_topo(int const       tid_A,
                        int const       tid_B,
                        int const       tid_C,
                        int const *     idx_map_A,
                        int const *     idx_map_B,
                        int const *     idx_map_C,
                        int const       itopo,
                        int const       order,
                        int *           idx_ctr,
                        int *           idx_extra,
                        int *           idx_no_ctr);

    void unmap_inner(tensor<dtype> * tsr);

    void get_new_ordering(CTF_ctr_type_t const * type,
                          int ** new_odering_A,
                          int ** new_odering_B,
                          int ** new_odering_C);

    int remap_inr_tsr( tensor<dtype> *otsr,
                       tensor<dtype> *itsr,
                       long_int const old_size,
                       int const *      old_phase,
                       int const *      old_rank,
                       int const *      old_virt_dim,
                       int const *      old_pe_lda,
                       int const        was_padded,
                       int const        was_cyclic,
                       int const *      old_padding,
                       int const *      old_edge_len,
                       int const *      ordering);

    void get_fold_indices(CTF_sum_type_t const *type,
                          int *                 num_fold,
                          int **                fold_idx);

    void get_fold_indices(CTF_ctr_type_t const *type,
                          int *                 num_fold,
                          int **                fold_idx);

    int can_fold(CTF_sum_type_t const * type);

    int can_fold(CTF_ctr_type_t const * type);

    void fold_tsr(tensor<dtype> *       tsr,
                  int const             nfold,
                  int const *           fold_idx,
                  int const *           idx_map,
                  int *                 all_fdim,
                  int **                all_flen);

    void unfold_tsr(tensor<dtype> * tsr);

    int map_fold(CTF_sum_type_t const * type,
                 int *                  inner_stride);

    int map_fold(CTF_ctr_type_t const * type,
                 iparam *               inner_prm);

    int unfold_broken_sym(CTF_ctr_type_t const *        type,
                          CTF_ctr_type_t *              new_type);

    int unfold_broken_sym(CTF_sum_type_t const *        type,
                          CTF_sum_type_t *              new_type);

    void dealias(int const sym_tid, int const nonsym_tid);

    void desymmetrize(int const sym_tid,
                      int const nonsym_tid,
                      int const is_C);

    void symmetrize(int const sym_tid, int const nonsym_tid);

    void copy_type(CTF_ctr_type_t const *       old_type,
                   CTF_ctr_type_t *             new_type);
    
    void free_type(CTF_ctr_type_t * old_type);
    
    int is_equal_type(CTF_ctr_type_t const *       old_type,
                      CTF_ctr_type_t const *       new_type);

    void order_perm(tensor<dtype> const * tsr_A,
                    tensor<dtype> const * tsr_B,
                    tensor<dtype> const * tsr_C,
                    int *                 idx_arr,
                    int const             off_A,
                    int const             off_B,
                    int const             off_C,
                    int *                 idx_map_A,
                    int *                 idx_map_B,
                    int *                 idx_map_C,
                    dtype &               add_sign,
                    int &                  mod);

    void get_sym_perms(CTF_ctr_type_t const *           type,
                       dtype const                      alpha,
                       std::vector<CTF_ctr_type_t>&     perms,
                       std::vector<dtype>&              signs);

    void add_sym_perm(std::vector<CTF_ctr_type_t>&    perms,
                      std::vector<dtype>&             signs, 
                      CTF_ctr_type_t const *          new_perm,
                      dtype const                     new_sign);
    
    void copy_type(CTF_sum_type_t const *       old_type,
                   CTF_sum_type_t *             new_type);
    
    void free_type(CTF_sum_type_t * old_type);
    
    int is_equal_type(CTF_sum_type_t const *       old_type,
                      CTF_sum_type_t const *       new_type);

    void order_perm(tensor<dtype> const * tsr_A,
                    tensor<dtype> const * tsr_B,
                    int *                 idx_arr,
                    int const             off_A,
                    int const             off_B,
                    int *                 idx_map_A,
                    int *                 idx_map_B,
                    dtype &               add_sign,
                    int &                  mod);

    void get_sym_perms(CTF_sum_type_t const *           type,
                       dtype const                      alpha,
                       std::vector<CTF_sum_type_t>&     perms,
                       std::vector<dtype>&              signs);

    void add_sym_perm(std::vector<CTF_sum_type_t>&    perms,
                      std::vector<dtype>&             signs, 
                      CTF_sum_type_t const *          new_perm,
                      dtype const                     new_sign);

    void get_len_ordering(CTF_sum_type_t const *        type,
                          int **                        new_ordering_A,
                          int **                        new_ordering_B);

    void get_len_ordering(CTF_ctr_type_t const *        type,
                          int **                        new_ordering_A,
                          int **                        new_ordering_B,
                          int **                        new_ordering_C);

    int extract_diag(int const    tid,
                     int const *  idx_map,
                     int const    rw,
                     int *        tid_new,
                     int **       idx_map_new);

    void contract_mst();

    int elementalize(int const          tid,
                     int const          x_rank,
                     int const          x_np,
                     int const          y_rank,
                     int const          y_np,
                     long_int const     blk_sz,
                     dtype *            data);

    /* ScaLAPACK back-end */
    int load_matrix(dtype *             DATA,
                    int const *         DESC,
                    int *               tid,
                    int *               need_free);

    int pgemm( char const       TRANSA,
               char const       TRANSB,
               int const        M,
               int const        N,
               int const        K,
               dtype const      ALPHA,
               dtype *          A,
               int const        IA,
               int const        JA,
               int const *      DESCA,
               dtype *          B,
               int const        IB,
               int const        JB,
               int const *      DESCB,
               dtype const      BETA,
               dtype *          C,
               int const        IC,
               int const        JC,
               int const *      DESCC,
               CTF_ctr_type *   pct,
               fseq_tsr_ctr<dtype> * pfs,
               int *            need_free);

};
inline double GET_REAL(double const d) {
  return d;
}
inline  double GET_REAL(std::complex<double> const d) {
  return d.real();
}

#include "dist_tensor_internal.cxx"
#include "scala_backend.cxx"
#endif
