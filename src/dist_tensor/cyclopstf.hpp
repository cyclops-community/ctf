/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#ifndef __CYCLOPSTF_HPP__
#define __CYCLOPSTF_HPP__

#include "mpi.h"
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

/**
 * \brief reduction types for tensor data
 */
enum CTF_OP { CTF_OP_SUM, CTF_OP_SUMABS, CTF_OP_SQNRM2,
              CTF_OP_MAX, CTF_OP_MIN, CTF_OP_MAXABS, CTF_OP_MINABS };
/**
 * labels corresponding to symmetry of each tensor dimension
 * NS = 0 - nonsymmetric
 * SY = 1 - symmetric
 * AS = 2 - antisymmetric
 * SH = 3 - symmetric hollow
 */
#if (!defined NS && !defined SY && !defined SH)
#define NS 0
#define SY 1
#define AS 2
#define SH 3
#endif


typedef int64_t long_int;
typedef long_int key;

/* Force redistributions always by setting to 1 (use 2.5D algorithms) */
#define REDIST 0
#define VERIFY 0
#define VERIFY_REMAP 0
#define INNER_MAP 0
#define FOLD_TSR 1
#define PERFORM_DESYM 1
#define DIAG_RESCALE
//#define USE_SYM_SUM 
#define HOME_CONTRACT
#define USE_BLOCK_RESHUFFLE

template<typename dtype>
struct tkv_pair {
  key k;
  dtype d;
  tkv_pair() {}
  tkv_pair(key k, dtype d) : k(k), d(d) {}
  bool operator< (const tkv_pair<dtype>& other) const{
    return k < other.k;
  }
};

typedef tkv_pair<double> kv_pair;

template<typename dtype>
inline bool comp_tkv_pair(tkv_pair<dtype> i,tkv_pair<dtype> j) {
  return (i.k<j.k);
}

typedef struct CTF_ctr_type {
  int   tid_A;
  int   tid_B;
  int   tid_C;
  int * idx_map_A; /* map indices of tensor A to contraction */
  int * idx_map_B; /* map indices of tensor B to contraction */
  int * idx_map_C; /* map indices of tensor C to contraction */
} CTF_ctr_type_t;

typedef struct CTF_sum_type {
  int   tid_A;
  int   tid_B;
  int * idx_map_A; /* map indices of tensor A to sum */
  int * idx_map_B; /* map indices of tensor B to sum */
} CTF_sum_type_t;

enum { DIST_TENSOR_SUCCESS, DIST_TENSOR_ERROR, DIST_TENSOR_NEGATIVE };

enum CTF_MACHINE { MACHINE_GENERIC, MACHINE_BGP, MACHINE_BGQ,
                   MACHINE_8D, NO_TOPOLOGY };


/* These now have to live in a struct due to being templated, since one
   cannot typedef. */
template<typename dtype>
struct fseq_tsr_scl {
  /* Function signature for sub-tensor scale recrusive call */
  int  (*func_ptr) ( dtype const        alpha,
                     dtype *            A,
                     int const          ndim_A,
                     int const *        edge_len_A,
                     int const *        lda_A,
                     int const *        sym_A,
                     int const *        idx_map_A);
};

/* custom element-wise function for tensor scale */
template<typename dtype>
struct fseq_elm_scl {
  void  (*func_ptr)(dtype const alpha, 
                    dtype &     a);
};

/* Function signature for sub-tensor summation recrusive call */
template<typename dtype>
struct fseq_tsr_sum {
  int  (*func_ptr) ( dtype const          alpha,
                     dtype const *        A,
                     int const            ndim_A,
                     int const *          edge_len_A,
                     int const *          lda_A,
                     int const *          sym_A,
                     int const *          idx_map_A,
                     dtype const          beta,
                     dtype *              B,
                     int const            ndim_B,
                     int const *          edge_len_B,
                     int const *          lda_B,
                     int const *          sym_B,
                     int const *          idx_map_B);
};

/* custom element-wise function for tensor sum */
template<typename dtype>
struct fseq_elm_sum {
  void  (*func_ptr)(dtype const alpha, 
                    dtype const a,
                    dtype &     b);
};


template<typename dtype>
struct fseq_tsr_ctr {

    /* Function signature for sub-tensor contraction recrusive call */
    int  (*func_ptr) ( dtype const      alpha,
                       dtype const *    A,
                       int const        ndim_A,
                       int const *      edge_len_A,
                       int const *      lda_A,
                       int const *      sym_A,
                       int const *      idx_map_A,
                       dtype const *    B,
                       int const        ndim_B,
                       int const *      edge_len_B,
                       int const *      lda_B,
                       int const *      sym_B,
                       int const *      idx_map_B,
                       dtype const      beta,
                       dtype *          C,
                       int const        ndim_C,
                       int const *      edge_len_C,
                       int const *      lda_C,
                       int const *      sym_C,
                       int const *      idx_map_C);
};

/* custom element-wise function for tensor sum */
template<typename dtype>
struct fseq_elm_ctr {
  void  (*func_ptr)(dtype const alpha, 
                    dtype const a, 
                    dtype const b,
                    dtype &     c);
};

template<typename dtype>
class dist_tensor;

template<typename dtype>
class tCTF{
  private:
    dist_tensor<dtype> * dt;
    int initialized;

  public:


    tCTF();
    ~tCTF();

    /* initializes library. Sets topology to be a torus of edge lengths equal to the
       factorization of np. Main args can be sset for profiler output. */
    int init(MPI_Comm const global_context,
             int const      rank,
             int const      np,
             CTF_MACHINE    mach = MACHINE_GENERIC,
             int const      argc = 0,
             const char * const * argv = NULL);


    /* initializes library. Sets topology to be a mesh of dimension ndim with
       edge lengths dim_len. */
    int init(MPI_Comm const global_context,
             int const      rank,
             int const      np,
             int const      ndim,
             int const *    dim_len,
             int const      argc = 0,
             const char * const * argv = NULL);


    /* return MPI_Comm global_context */
    MPI_Comm get_MPI_Comm();
    
    /* return MPI processor rank */
    int get_rank();
    
    /* return number of MPI processes in the defined global context */
    int get_num_pes();

    /* define a tensor and retrive handle */
    int define_tensor(int const     ndim,
                      int const *   edge_len,
                      int const *   sym,
                      int *         tensor_id,
                      char const *  name = NULL,
                      int           profile = 0);

    /* Create identical tensor with identical data if copy_data=1 */
    int clone_tensor(int const  tensor_id,
                     int const  copy_data,
                     int *      new_tensor_id);

    /* get information about a tensor */
    int info_tensor(int const tensor_id,
                    int *     ndim,
                    int **    edge_len,
                    int **    sym) const;

    /* set the tensor name */
    int set_name(int const tensor_id, char const * name);
    
    /* get the tensor name */
    int get_name(int const tensor_id, char const ** name);

    /* turn on profiling */
    int profile_on(int const tensor_id);
    
    /* turn off profiling */
    int profile_off(int const tensor_id);

    /* get dimension of a tensor */
    int get_dimension(int const tensor_id, int *ndim) const;

    /* get lengths of a tensor */
    int get_lengths(int const tensor_id, int **edge_len) const;

    /* get symmetry of a tensor */
    int get_symmetry(int const tensor_id, int **sym) const;

    /* get raw data pointer WARNING: includes padding */
    int get_raw_data(int const tensor_id, dtype ** data, long_int * size);

    /* Input tensor data with <key, value> pairs where key is the
       global index for the value. */
    int write_tensor(int const                tensor_id,
                     long_int const           num_pair,
                     tkv_pair<dtype> const *  mapped_data);
    
    /* Add tensor data new=alpha*new+beta*old
       with <key, value> pairs where key is the 
       global index for the value. */
    int write_tensor(int const                tensor_id,
                     long_int const           num_pair,
                     dtype const              alpha,
                     dtype const              beta,
                     tkv_pair<dtype> const *  mapped_data);

    /* Add tensor data from A to a block of B, 
       B[offsets_B:ends_B] = beta*B[offsets_B:ends_B] + alpha*A[offsets_A:ends_A] */
    int slice_tensor(int const    tid_A,
                     int const *  offsets_A,
                     int const *  ends_A,
                     double const alpha,
                     int const    tid_B,
                     int const *  offsets_B,
                     int const *  ends_B,
                     double const beta);

    /* read a block from tensor_id, 
       new_tensor_id = tensor_id[offsets:ends] */
/*    int read_block_tensor(int const   tensor_id,
                          int const * offsets,
                          int const * ends,
                          int *       new_tensor_id);*/


    /* read tensor data with <key, value> pairs where key is the
       global index for the value, which gets filled in. */
    int read_tensor(int const               tensor_id,
                    long_int const          num_pair,
                    tkv_pair<dtype> * const mapped_data);

    /* read entire tensor with each processor (in packed layout).
       WARNING: will use a lot of memory. */
    int allread_tensor(int const  tensor_id,
                       long_int * num_pair,
                       dtype **   all_data);


    /* map input tensor local data to zero. */
    int set_zero_tensor(int tensor_id);

    /* read tensor data pairs local to processor. */
    int read_local_tensor(int const           tensor_id,
                          long_int *          num_pair,
                          tkv_pair<dtype> **  mapped_data);

    /* contracts tensors alpha*A*B + beta*C -> C,
       uses standard symmetric contraction sequential kernel */
    int contract(CTF_ctr_type_t const * type,
                 dtype const            alpha,
                 dtype const            beta);

    /* contracts tensors alpha*A*B + beta*C -> C,
       seq_func used to perform sequential op */
    int contract(CTF_ctr_type_t const *     type,
                 fseq_tsr_ctr<dtype> const  func_ptr,
                 dtype const                alpha,
                 dtype const                beta,
                 int const                  map_inner = 0);
    
    /* contracts tensors alpha*A*B + beta*C -> C,
       seq_func used to perform element-wise sequential op */
    int contract(CTF_ctr_type_t const *     type,
                 fseq_elm_ctr<dtype> const  felm,
                 dtype const                alpha,
                 dtype const                beta);

    /* DAXPY: a*A + B -> B. */
    int sum_tensors(dtype const  alpha,
                    int const    tid_A,
                    int const    tid_B);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(CTF_sum_type_t const *  type,
                    dtype const             alpha,
                    dtype const             beta);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(dtype const               alpha,
                    dtype const               beta,
                    int const                 tid_A,
                    int const                 tid_B,
                    int const *               idx_map_A,
                    int const *               idx_map_B,
                    fseq_tsr_sum<dtype> const func_ptr);

    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(CTF_sum_type_t const *    type,
                    dtype const               alpha,
                    dtype const               beta,
                    fseq_tsr_sum<dtype> const func_ptr);
    
    /* DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). */
    int sum_tensors(dtype const               alpha,
                    dtype const               beta,
                    int const                 tid_A,
                    int const                 tid_B,
                    int const *               idx_map_A,
                    int const *               idx_map_B,
                    fseq_elm_sum<dtype> const felm);

    /* copy A into B. Realloc if necessary */
    int copy_tensor(int const tid_A, int const tid_B);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(dtype const alpha, int const tid);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(dtype const                alpha,
                     int const                  tid,
                     int const *                idx_map_A);

    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(dtype const                alpha,
                     int const                  tid,
                     int const *                idx_map_A,
                     fseq_tsr_scl<dtype> const  func_ptr);
    
    /* scale tensor by alpha. A <- a*A */
    int scale_tensor(dtype const                alpha,
                     int const                  tid,
                     int const *                idx_map_A,
                     fseq_elm_scl<dtype> const  felm);
    
    /* aligns tensor mapping of tid_A to that of tid_B */
    int align(int const    tid_A,   
              int const    tid_B);

    /* product will contain the dot prodiuct if tsr_A and tsr_B */
    int dot_tensor(int const tid_A, int const tid_B, dtype *product);

    /* reduce data of tid_A with the given OP */
    int reduce_tensor(int const tid, CTF_OP op, dtype * result);

    /* map data of tid_A with the given function */
    int map_tensor(int const tid,
                   dtype (*map_func)(int const ndim, int const * indices,
                                     dtype const elem));

    /* obtains the largest n elements (in absolute value) of the tensor */
    int get_max_abs(int const tid, int const n, dtype * data);

    /* Prints a tensor on one processor. */
    int print_tensor(FILE * stream, int const tid, double cutoff = -1.0);

    /* Compares two tensors on one processor. */
    int compare_tensor(FILE * stream, int const tid_A, int const tid_B, double cutoff = -1.0);

    /* Prints contraction type. */
    int print_ctr(CTF_ctr_type_t const * ctype,
                  dtype const            alpha,
                  dtype const            beta) const;

    /* Prints sum type. */
    int print_sum(CTF_sum_type_t const * stype,
                  dtype const            alpha,
                  dtype const            beta) const;

    /* Deletes all tensor handles. Invalidates all tensor ids. */
    int clean_tensors();

    /* Deletes a tensor handle. Invalidates all tensor ids. */
    int clean_tensor(int const tid);

    /* Exits library and deletes all internal data */
    int exit();

    /* ScaLAPACK back-end */
    int pgemm( char const         TRANSA,
               char const         TRANSB,
               int const          M,
               int const          N,
               int const          K,
               dtype const        ALPHA,
               dtype *            A,
               int const          IA,
               int const          JA,
               int const *        DESCA,
               dtype *            B,
               int const          IB,
               int const          JB,
               int const *        DESCB,
               dtype const        BETA,
               dtype *            C,
               int const          IC,
               int const          JC,
               int const *        DESCC);

    /* define matrix from ScaLAPACK descriptor */
    int def_scala_mat(int const * DESCA, dtype const * data, int * tid);

    /* reads a ScaLAPACK matrix to the original data pointer */
    int read_scala_mat(int const tid, dtype * data);

    int pgemm( char const         TRANSA,
               char const         TRANSB,
               int const          M,
               int const          N,
               int const          K,
               dtype const        ALPHA,
               int const          tid_A,
               int const          tid_B,
               dtype const        BETA,
               int const          tid_C);

};

//template class tCTF<double>;
//template class tCTF< std::complex<double> >;

typedef tCTF<double> CTF;


//#include "cyclopstf.cxx"

#endif ////__CYCLOPSTF_HPP__

