/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/


#include "cyclopstf.hpp"
#include "dist_tensor.h"
#include "dist_tensor_internal.h"
#include "../shared/util.h"

CTF ctf_obj;
tCTF< std::complex<double> > zctf_obj;

/**
 * \brief  initializes library. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 */
int  CTF_init(MPI_Comm const  global_context,
              int const       rank, 
              int const       np){      
  return ctf_obj.init(global_context, rank, np);
}

/**
 * \brief  initializes complex library. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 */
int  CTF_init_complex(MPI_Comm const    global_context,
                      int const         rank, 
                      int const         np){        
  return zctf_obj.init(global_context, rank, np);
}


/**
 * \brief  initializes library. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] mach the type of machine we are running on
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 */
int  CTF_init(MPI_Comm const        global_context,
              CTF_MACHINE           mach,
              int const             rank, 
              int const             np,
              int const             inner_size){        
  return ctf_obj.init(global_context, mach, rank, np, inner_size);
}

/**
 * \brief  initializes library. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] mach the type of machine we are running on
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 */
int  CTF_init_complex(MPI_Comm const        global_context,
                      CTF_MACHINE           mach,
                      int const             rank, 
                      int const             np){        
  return zctf_obj.init(global_context, mach, rank, np);
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. FIXME: determine topology automatically 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] ndim is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 */

int CTF_init(MPI_Comm const       global_context,
             int const            rank, 
             int const            np, 
             int const            ndim, 
             int const *          dim_len){
  return ctf_obj.init(global_context, rank, np, ndim, dim_len);
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. FIXME: determine topology automatically 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] ndim is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 */

int CTF_init_complex(MPI_Comm const       global_context,
                     int const            rank, 
                     int const            np, 
                     int const            ndim, 
                     int const *          dim_len){
  return zctf_obj.init(global_context, rank, np, ndim, dim_len);
}


/**
 * \brief will be deprecated
 */
/*int CTF_init(CommData_t * cdt_global, int const ndim, int const * dim_len){
  return ctf_obj.init(cdt_global, ndim, dim_len);
}*/

/**
 * \brief  defines a tensor and retrieves handle
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] edge_len global edge lengths of tensor
 * \param[in] sym symmetry relations of tensor
 * \param[out] tensor_id the tensor index (handle)
 */
int CTF_define_tensor(int const         ndim,       
                      int const *       edge_len, 
                      int const *       sym,
                      int *                 tensor_id){
  return ctf_obj.define_tensor(ndim,edge_len,sym,tensor_id);
}

/* \brief get dimension of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 */
int CTF_get_dimension(int const tensor_id, int *ndim) {
  return ctf_obj.get_dimension(tensor_id, ndim);
}
    
/* \brief get lengths of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] edge_len edge lengths of tensor
 */
int CTF_get_lengths(int const tensor_id, int **edge_len) {
  return ctf_obj.get_lengths(tensor_id, edge_len);
}
    
/* \brief get symmetry of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] sym symmetries of tensor
 */
int CTF_get_symmetry(int const tensor_id, int **sym) {
  return ctf_obj.get_symmetry(tensor_id, sym);
}

/**
 * \brief get information about tensor
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 * \param[out] edge_len edge lengths of tensor
 * \param[out] sym symmetries of tensor
 */
int CTF_info_tensor(int const tensor_id,
                    int *     ndim,
                    int **    edge_len,
                    int **    sym){
  return ctf_obj.info_tensor(tensor_id, ndim, edge_len, sym);
}


/**
 * \brief  Input tensor data with <key, value> pairs where key is the
 *              global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to write
 * \param[in] mapped_data pairs to write
 */
int CTF_write_tensor(int const              tensor_id, 
                     int64_t const          num_pair,  
                     kv_pair * const        mapped_data){
  return ctf_obj.write_tensor(tensor_id, num_pair, mapped_data);
}

/**
 * \brief read tensor data with <key, value> pairs where key is the
 *              global index for the value, which gets filled in. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to read
 * \param[in,out] mapped_data pairs to read
 */
int CTF_read_tensor(int const               tensor_id, 
                    int64_t const           num_pair, 
                    kv_pair * const         mapped_data){
  return ctf_obj.read_tensor(tensor_id, num_pair, mapped_data);
}

/**
 * \brief read entire tensor with each processor (in packed layout).
 *         WARNING: will use a lot of memory. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[in,out] mapped_data values read
 */
int CTF_allread_tensor(int const        tensor_id, 
                       int64_t *        num_pair, 
                       double **        all_data){
  return ctf_obj.allread_tensor(tensor_id, num_pair, all_data);
}



/* input tensor local data or set buffer for contract answer. */
/*int CTF_set_local_tensor(int const    tensor_id, 
                         int const      num_val, 
                         double *       tsr_data){
  return set_tsr_data(tensor_id, num_val, tsr_data);  
}*/

/**
 * \brief  map input tensor local data to zero
 * \param[in] tensor_id tensor handle
 */
int CTF_set_zero_tensor(int const tensor_id){
  return ctf_obj.set_zero_tensor(tensor_id);
}

/**
 * \brief read tensor data pairs local to processor. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[out] mapped_data values read
 */
int CTF_read_local_tensor(int const       tensor_id, 
                          int64_t *       num_pair,  
                          kv_pair **      mapped_data){
  return ctf_obj.read_local_tensor(tensor_id, num_pair, mapped_data);
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
int CTF_contract(CTF_ctr_type_t const * type,
                 double const           alpha,
                 double const           beta){
  return ctf_obj.contract(type, alpha, beta);
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      accepts custom-sized buffer-space,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
int CTF_contract( CTF_ctr_type_t const *  type,
                  double *                buffer, 
                  int const               buffer_len, 
                  double const            alpha,
                  double const            beta){
  return ctf_obj.contract(type, buffer, buffer_len, alpha, beta);
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      accepts custom-sized buffer-space,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] buffer the buffer space to use, or NULL to allocate
 * \param[in] buffer_len length of buffer 
 * \param[in] func_ptr sequential ctr func pointer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
int CTF_contract(CTF_ctr_type_t const * type,
                 double *               buffer, 
                 int const              buffer_len, 
                 CTF_seq_tsr_ctr const  func_ptr, 
                 double const           alpha,
                 double const           beta){
  fseq_tsr_ctr<double> fs;
  fs.func_ptr=func_ptr;
  return ctf_obj.contract(type, buffer, buffer_len, fs, alpha, beta);
}

/**
 * \brief copy tensor from one handle to another
 * \param[in] tid_A tensor handle to copy from
 * \param[in] tid_B tensor handle to copy to
 */
int CTF_copy_tensor(int const tid_A, int const tid_B){
  return ctf_obj.copy_tensor(tid_A, tid_B);
}

/**
 * \brief scales a tensor by alpha
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 */
int CTF_scale_tensor(double const alpha, int const tid){
  return ctf_obj.scale_tensor(alpha, tid);
}

/**
 * \brief computes a dot product of two tensors A dot B
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[out] product the result of the dot-product
 */
int CTF_dot_tensor(int const tid_A, int const tid_B, double *product){
  return ctf_obj.dot_tensor(tid_A, tid_B, product);
}

/**
 * \brief Performs an elementwise reduction on a tensor 
 * \param[in] tid tensor handle
 * \param[in] CTF_OP reduction operation to apply
 * \param[out] result result of reduction operation
 */
int CTF_reduce_tensor(int const tid, CTF_OP op, double * result){
  return ctf_obj.reduce_tensor(tid, op, result);
}

/**
 * \brief Calls a mapping function on each element of the tensor 
 * \param[in] tid tensor handle
 * \param[in] map_func function pointer to apply to each element
 */
int CTF_map_tensor(int const tid, 
                   double (*map_func)(int const ndim, int const * indices, 
                                      double const elem)){
  return ctf_obj.map_tensor(tid, map_func);
}

/**
 * \brief daxpy tensors A and B, B = B+alpha*A
 * \param[in] alpha scaling factor
 * \param[in] tid_A tensor handle of A
 * \param[in] tid_B tensor handle of B
 */
int CTF_sum_tensors(double const  alpha,
                    int const     tid_A,
                    int const     tid_B){
  return ctf_obj.sum_tensors(alpha, tid_A, tid_B);
}

/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 *               uses standard summation pointer
 * \param[in] type idx_maps and tids of contraction
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
int CTF_sum_tensors( CTF_sum_type_t const * type,
                     double const           alpha,
                     double const           beta){
  
  return ctf_obj.sum_tensors(type, alpha, beta);

}

/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[in] idx_map_A index map of A
 * \param[in] idx_map_B index map of B
 * \param[in] func_ptr sequential ctr func pointer 
 */
int CTF_sum_tensors(double const          alpha,
                    double const          beta,
                    int const             tid_A,
                    int const             tid_B,
                    int const *           idx_map_A,
                    int const *           idx_map_B,
                    CTF_seq_tsr_sum const func_ptr){
  fseq_tsr_sum<double> fs;
  fs.func_ptr=func_ptr;
  return ctf_obj.sum_tensors(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, fs);
}

/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 * \param[in] type index mapping of tensors
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] func_ptr sequential ctr func pointer 
 */
int CTF_sum_tensors(CTF_sum_type_t const *  type,
                    double const            alpha,
                    double const            beta,
                    CTF_seq_tsr_sum const   func_ptr){
  fseq_tsr_sum<double> fs;
  fs.func_ptr=func_ptr;
  return ctf_obj.sum_tensors(type, alpha, beta, fs);
}

/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 * \param[in] func_ptr pointer to sequential scale function
 */
int CTF_scale_tensor(double const           alpha, 
                     int const              tid, 
                     int const *            idx_map,
                     CTF_seq_tsr_scl const  func_ptr){
  fseq_tsr_scl<double> fs;
  fs.func_ptr=func_ptr;
  return ctf_obj.scale_tensor(alpha, tid, idx_map, fs);
}

int CTF_print_tensor(FILE * stream, int const tid) {
  return ctf_obj.print_tensor(stream, tid);
}

/* Prints contraction type. */
int CTF_print_ctr(CTF_ctr_type_t const * ctype,
                  double const           alpha,
                  double const           beta) {
  return ctf_obj.print_ctr(ctype,alpha,beta);
}

/* Prints sum type. */
int CTF_print_sum(CTF_sum_type_t const * stype,
                  double const           alpha,
                  double const           beta) {
  return ctf_obj.print_sum(stype,alpha,beta);
}


/**
 * \brief removes all tensors, invalidates all handles
 */
int CTF_clean_tensors(){
  return ctf_obj.clean_tensors();
}

/**
 * \brief removes a tensor, invalidates its handle
 * \param tid tensor handle
 */
int CTF_clean_tensor(int const tid){
  return ctf_obj.clean_tensor(tid);
}

/**
 * \brief removes all tensors, invalidates all handles, and exits library.
 *              Do not use library instance after executing this.
 */
int CTF_exit(){
  zctf_obj.exit();
  return ctf_obj.exit();
}



/* ScaLAPACK PDGEMM back-end */
void CTF_pdgemm(char const    TRANSA, 
                char const    TRANSB, 
                int const           M, 
                int const           N, 
                int const           K, 
                double const    ALPHA,
                double *            A, 
                int const           IA, 
                int const           JA, 
                int const *       DESCA, 
                double *      B, 
                int const           IB, 
                int const           JB, 
                int const *       DESCB, 
                double const  BETA,
                double *            C, 
                int const           IC, 
                int const           JC, 
                int const *       DESCC){
  ctf_obj.pgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA,
                B, IB, JB, DESCB, BETA, C, IC, JC, DESCC);
}

/* ScaLAPACK back-end */
void CTF_pzgemm(char const                        TRANSA, 
                char const                        TRANSB, 
                int const                         M, 
                int const                         N, 
                int const                         K, 
                std::complex<double> const        ALPHA,
                std::complex<double> *            A, 
                int const                         IA, 
                int const                         JA, 
                int const *                       DESCA, 
                std::complex<double> *            B, 
                int const                         IB, 
                int const                         JB, 
                int const *                       DESCB, 
                std::complex<double> const        BETA,
                std::complex<double> *            C, 
                int const                         IC, 
                int const                         JC, 
                int const *                       DESCC){
  zctf_obj.pgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA,
                 B, IB, JB, DESCB, BETA, C, IC, JC, DESCC);
}

/**
 * \brief define matrix from ScaLAPACK descriptor
 *
 * \param[in] DESCA ScaLAPACK descriptor for a matrix
 * \param[in] data pointer to actual data
 * \param[out] tid tensor handle
 */
int CTF_def_scala_mat(int const * DESCA, double const * data, int * tid){
  return ctf_obj.def_scala_mat(DESCA, data, tid);
}

/**
 * \brief define matrix from ScaLAPACK descriptor
 *
 * \param[in] DESCA ScaLAPACK descriptor for a matrix
 * \param[in] data pointer to actual data
 * \param[out] tid tensor handle
 */
int CTF_def_scala_mat(int const *                   DESCA, 
                      std::complex<double> const *  data, 
                      int *                         tid){
  return zctf_obj.def_scala_mat(DESCA, data, tid);
}
    
/**
 * \brief reads a ScaLAPACK matrix to the original data pointer
 *
 * \param[in] tid tensor handle
 * \param[in,out] data pointer to buffer data
 */
int CTF_read_scala_mat(int const tid, double * data){
  return ctf_obj.read_scala_mat(tid, data);
}

/**
 * \brief reads a ScaLAPACK matrix to the original data pointer
 *
 * \param[in] tid tensor handle
 * \param[in,out] data pointer to buffer data
 */
int CTF_read_scala_mat(int const tid, std::complex<double> * data){
  return zctf_obj.read_scala_mat(tid, data);
}







