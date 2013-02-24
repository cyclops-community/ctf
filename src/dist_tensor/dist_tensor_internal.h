/* Copyright (c) 2011, Edgar Solomonik>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following
 * conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL EDGAR SOLOMONIK BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE. */

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
  int need_remap;
  int has_zero_edge_len;
  union {
    dtype * data;
    tkv_pair <dtype> * pairs;
  };
};



int get_buffer_space(int const len, void ** const ptr);
int free_buffer_space(void * ptr);
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
    int const * sym_A;
    int ndim_B;
    int * edge_len_B;
    int const * idx_map_B;
    int const * sym_B;
    int ndim_C;
    int * edge_len_C;
    int const * idx_map_C;
    int const * sym_C;
    fseq_tsr_ctr<dtype> func_ptr;

    int is_inner;
    iparam inner_params;

    void run();
    long_int mem_fp();
    ctr<dtype> * clone();

    seq_tsr_ctr(ctr<dtype> * other);
    ~seq_tsr_ctr(){ free(edge_len_A), free(edge_len_B), free(edge_len_C); };
    seq_tsr_ctr(){}
};

template<typename dtype>
class seq_tsr_sum : public tsum<dtype> {
  public:
    int ndim_A;
    int * edge_len_A;
    int const * idx_map_A;
    int const * sym_A;
    int ndim_B;
    int * edge_len_B;
    int const * idx_map_B;
    int const * sym_B;
    fseq_tsr_sum<dtype> func_ptr;

    int is_inner;
    int inr_stride;

    void run();
    long_int mem_fp();
    tsum<dtype> * clone();

    seq_tsr_sum(tsum<dtype> * other);
    ~seq_tsr_sum(){ free(edge_len_A), free(edge_len_B); };
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

    void run();
    long_int mem_fp();
    scl<dtype> * clone();

    seq_tsr_scl(scl<dtype> * other);
    ~seq_tsr_scl(){ free(edge_len); };
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
                      int const         alloc_data = 1);


    int set_tsr_data(int const  tensor_id,
                     int const  num_val,
                     dtype * tsr_data);

    topology * get_topo(int const itopo);
    int get_dim(int const tensor_id) const;
    int * get_edge_len(int const tensor_id) const;
    int * get_sym(int const tensor_id) const;
    dtype * get_raw_data(int const tensor_id, int64_t * size);

    int get_tsr_info(int const tensor_id,
                     int * ndim,
                     int ** edge_len,
                     int ** sym) const;

    int write_pairs(int const                 tensor_id,
                    long_int const            num_pair,
                    double const              alpha,
                    double const              beta,
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
                                fseq_tsr_sum<dtype> const       func_ptr,
                                int const               inr_str=-1);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(dtype const                 alpha,
                    dtype const                 beta,
                    int const                   tid_A,
                    int const                   tid_B,
                    int const *                 idx_map_A,
                    int const *                 idx_map_B,
                    fseq_tsr_sum<dtype> const   func_ptr);

    ctr<dtype> * construct_contraction( CTF_ctr_type_t const *  type,
                                        dtype *                 buffer,
                                        int const               buffer_len,
                                        fseq_tsr_ctr<dtype>     func_ptr,
                                        dtype const             alpha,
                                        dtype const             beta,
                                        int const               is_inner=0,
                                        iparam const *          inner_params=NULL,
                                        int *                   nvirt_C = NULL);

/*    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B);

    dtype align_symmetric_indices(int ndim_A, int* idx_A, int* sym_A,
                                int ndim_B, int* idx_B, int* sym_B,
                                int ndim_C, int* idx_C, int* sym_C);

    dtype overcounting_factor(int ndim_A, int* idx_A, int* sym_A,
                            int ndim_B, int* idx_B, int* sym_B,
                            int ndim_C, int* idx_C, int* sym_C);
*/
    int sym_contract( CTF_ctr_type_t const *    type,
                      dtype *                   buffer,
                      int const                 buffer_len,
                      fseq_tsr_ctr<dtype> const func_ptr,
                      dtype const               alpha,
                      dtype const               beta,
                      int const                 map_inner);

    int contract( CTF_ctr_type_t const *        type,
                  dtype *                       buffer,
                  int const                     buffer_len,
                  fseq_tsr_ctr<dtype> const     func_ptr,
                  dtype const                   alpha,
                  dtype const                   beta,
                  int const                     map_inner);

    int map_tensors(CTF_ctr_type_t const *      type,
                    dtype *                     buffer,
                    int const                   buffer_len,
                    fseq_tsr_ctr<dtype>         func_ptr,
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

    int sum_tsr(dtype const             alpha,
                dtype const             beta,
                int const               tid_A,
                int const               tid_B,
                int const *             idx_A,
                int const *             idx_B,
                fseq_tsr_sum<dtype>     func_ptr);

    int cpy_tsr(int const tid_A, int const tid_B);

    int clone_tensor(int const tid, int const copy_data,
                     int * new_tid, int const alloc_data=1);

    int scale_tsr(dtype const alpha, int const tid);

    int scale_tsr(dtype const                           alpha,
                  int const                               tid,
                  int const *                           idx_map,
                  fseq_tsr_scl<dtype> const     func_ptr);

    int dot_loc_tsr(int const tid_A, int const tid_B, dtype *product);

    int red_tsr(int const tid, CTF_OP op, dtype * result);

    int del_tsr(int const tid);

    int map_tsr(int const tid,
                dtype (*map_func)(int const ndim, int const * indices,
                       dtype const elem));

    int set_zero_tsr(int const tensor_id);

    int print_tsr(FILE * stream, int const tid);

    int print_map(FILE * stream, int const tid,
                  int const all=1, int const is_inner=0) const;

    int print_ctr(CTF_ctr_type_t const * ctype) const;

    int print_sum(CTF_sum_type_t const * stype) const;

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
                        int *           idx_no_ctr);

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

/*  void get_sym_perms(CTF_ctr_type_t const *   type,
                       dtype const              alpha,
                       int *                    nperm,
                       CTF_ctr_type_t **        perms,
                       dtype **                 signs);*/
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

    void get_len_ordering(CTF_sum_type_t const *        type,
                          int **                        new_ordering_A,
                          int **                        new_ordering_B);

    void get_len_ordering(CTF_ctr_type_t const *        type,
                          int **                        new_ordering_A,
                          int **                        new_ordering_B,
                          int **                        new_ordering_C);


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

    double GET_REAL(dtype const d) const;
};

#include "dist_tensor_internal.cxx"
#include "scala_backend.cxx"
#endif
