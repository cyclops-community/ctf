/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor_internal.h"
#include "cyclopstf.hpp"
#include "mach.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include <stdint.h>
#include <limits.h>
#if VERIFY
#include "../unit_test/unit_test.h"
#include "../unit_test/unit_test_ctr.h"
#endif

#ifdef HPM
extern "C" void HPM_Start(char *);  
extern "C" void HPM_Stop(char *);
#endif

#define DEF_INNER_SIZE 256

/** 
 * \brief destructor
 */
template<typename dtype>
tCTF<dtype>::~tCTF(){
  exit();
}

/** 
 * \brief constructor
 */
template<typename dtype>
tCTF<dtype>::tCTF(){
  initialized = 0;
}

template<typename dtype>
MPI_Comm tCTF<dtype>::get_MPI_Comm(){
  return (dt->get_global_comm())->cm;
}
    
/* return MPI processor rank */
template<typename dtype>
int tCTF<dtype>::get_rank(){
  return (dt->get_global_comm())->rank;
}
/* return number of MPI processes in the defined global context */
template<typename dtype>
int tCTF<dtype>::get_num_pes(){
  return (dt->get_global_comm())->np;
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] mach the type of machine we are running on
 * \param[in] argc number of arguments passed to main
 * \param[in] argv arguments passed to main
 */
template<typename dtype>
int tCTF<dtype>::init(MPI_Comm const  global_context,
                      int const       rank, 
                      int const       np,
                      CTF_MACHINE     mach,
                      int const       argc,
                      const char * const *  argv){
  int ndim, ret;
  int * dim_len;
  get_topo(np, mach, &ndim, &dim_len);
  ret = tCTF<dtype>::init(global_context, rank, np, ndim, dim_len, argc, argv);
  if (np > 1)
    CTF_free(dim_len);
  return ret;
}

/**
 * \brief  initializes library. 
 *      Sets topology to be a mesh of dimension ndim with
 *      edge lengths dim_len. 
 *
 * \param[in] global_context communicator decated to this library instance
 * \param[in] rank this pe rank within the global context
 * \param[in] np number of processors
 * \param[in] ndim is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 * \param[in] argc number of arguments passed to main
 * \param[in] argv arguments passed to main
 */
template<typename dtype>
int tCTF<dtype>::init(MPI_Comm const  global_context,
                      int const       rank, 
                      int const       np, 
                      int const       ndim, 
                      int const *     dim_len,
                      int const       argc,
                      const char * const *  argv){
  char * mst_size, * stack_size, * mem_size, * ppn;
  
  TAU_FSTART(CTF);
#ifdef HPM
  HPM_Start("CTF");
#endif
  CTF_set_context(global_context);
  CTF_set_main_args(argc, argv);

  
  mst_size = getenv("CTF_MST_SIZE");
  stack_size = getenv("CTF_STACK_SIZE");
  if (mst_size == NULL && stack_size == NULL){
#ifdef USE_MST
    if (rank == 0)
      DPRINTF(1,"Creating CTF stack of size %lld\n",1000*(long_int)1E6);
    CTF_mst_create(1000*(long_int)1E6);
#else
    if (rank == 0){
      DPRINTF(1,"Running CTF without stack, define CTF_STACK_SIZE ");
      DPRINTF(1,"environment variable to activate stack\n");
    }
#endif
  } else {
    uint64_t imst_size = 0 ;
    if (mst_size != NULL) 
      imst_size = strtoull(mst_size,NULL,0);
    if (stack_size != NULL)
      imst_size = MAX(imst_size,strtoull(stack_size,NULL,0));
    if (rank == 0)
      DPRINTF(1,"Creating CTF stack of size %llu due to CTF_STACK_SIZE enviroment variable\n",
                imst_size);
    CTF_mst_create(imst_size);
  }
  mem_size = getenv("CTF_MEMORY_SIZE");
  if (mem_size != NULL){
    uint64_t imem_size = strtoull(mem_size,NULL,0);
    if (rank == 0)
      DPRINTF(1,"CTF memory size set to %llu by CTF_MEMORY_SIZE environment variable\n",
                imem_size);
    CTF_set_mem_size(imem_size);
  }
  ppn = getenv("CTF_PPN");
  if (ppn != NULL){
    if (rank == 0)
      DPRINTF(1,"CTF assuming %lld processes per node due to CTF_PPN environment variable\n",
                atoi(ppn));
    LIBT_ASSERT(atoi(ppn)>=1);
    CTF_set_memcap(.75/atof(ppn));
  }
  initialized = 1;
  CommData_t * glb_comm = (CommData_t*)CTF_alloc(sizeof(CommData_t));
  SET_COMM(global_context, rank, np, glb_comm);
  dt = new dist_tensor<dtype>();
  return dt->initialize(glb_comm, ndim, dim_len, DEF_INNER_SIZE);
}


/**
 * \brief  defines a tensor and retrieves handle
 *
 * \param[in] ndim number of tensor dimensions
 * \param[in] edge_len global edge lengths of tensor
 * \param[in] sym symmetry relations of tensor
 * \param[out] tensor_id the tensor index (handle)
 * \param[in] name string name for tensor (optionary)
 * \param[in] profile wether to make profile calls for the tensor
 */
template<typename dtype>
int tCTF<dtype>::define_tensor(int const        ndim,             
                               int const *      edge_len, 
                               int const *      sym,
                               int *            tensor_id,
                               char const *     name,
                               int              profile){
  return dt->define_tensor(ndim, edge_len, sym, 
                           tensor_id, 1, name, profile);
}
    
/* \brief clone a tensor object
 * \param[in] tensor_id id of old tensor
 * \param[in] copy_data if 0 then leave tensor blank, if 1 copy data from old
 * \param[out] new_tensor_id id of new tensor
 */
template<typename dtype>
int tCTF<dtype>::clone_tensor(int const tensor_id,
                              int const copy_data,
                              int *     new_tensor_id){
  dt->clone_tensor(tensor_id, copy_data, new_tensor_id);
  return DIST_TENSOR_SUCCESS;
}
   

template<typename dtype>
int tCTF<dtype>::get_name(int const tensor_id, char const ** name){
  return dt->get_name(tensor_id, name);
}
 
template<typename dtype>
int tCTF<dtype>::set_name(int const tensor_id, char const * name){
  return dt->set_name(tensor_id, name);
}

template<typename dtype>
int tCTF<dtype>::profile_on(int const tensor_id){
  return dt->profile_on(tensor_id);
}

template<typename dtype>
int tCTF<dtype>::profile_off(int const tensor_id){
  return dt->profile_off(tensor_id);
}
    
/* \brief get dimension of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_dimension(int const tensor_id, int *ndim) const{
  *ndim = dt->get_dim(tensor_id);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get lengths of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] edge_len edge lengths of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_lengths(int const tensor_id, int **edge_len) const{
  int ndim, * sym;
  dt->get_tsr_info(tensor_id, &ndim, edge_len, &sym);
  CTF_untag_mem(edge_len);
  CTF_free(sym);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get symmetry of a tensor 
 * \param[in] tensor_id id of tensor
 * \param[out] sym symmetries of tensor
 */
template<typename dtype>
int tCTF<dtype>::get_symmetry(int const tensor_id, int **sym) const{
  *sym = dt->get_sym(tensor_id);
  CTF_untag_mem(*sym);
  return DIST_TENSOR_SUCCESS;
}
    
/* \brief get raw data pointer WARNING: includes padding 
 * \param[in] tensor_id id of tensor
 * \param[out] data raw local data
 */
template<typename dtype>
int tCTF<dtype>::get_raw_data(int const tensor_id, dtype ** data, long_int * size) {
  *data = dt->get_raw_data(tensor_id, size);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief get information about tensor
 * \param[in] tensor_id id of tensor
 * \param[out] ndim dimension of tensor
 * \param[out] edge_len edge lengths of tensor
 * \param[out] sym symmetries of tensor
 */
template<typename dtype>
int tCTF<dtype>::info_tensor(int const  tensor_id,
                             int *      ndim,
                             int **     edge_len,
                             int **     sym) const{
  dt->get_tsr_info(tensor_id, ndim, edge_len, sym);
  CTF_untag_mem(*sym);
  CTF_untag_mem(*edge_len);
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief  Input tensor data with <key, value> pairs where key is the
 *              global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to write
 * \param[in] mapped_data pairs to write
 */
template<typename dtype>
int tCTF<dtype>::write_tensor(int const               tensor_id, 
                              long_int const           num_pair,  
                              tkv_pair<dtype> const * mapped_data){
  return dt->write_pairs(tensor_id, num_pair, 1.0, 0.0, const_cast<tkv_pair<dtype>*>(mapped_data), 'w');
}

/** 
 * \brief  Add tensor data new=alpha*new+beta*old
 *         with <key, value> pairs where key is the 
 *         global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] 
 * \param[in] num_pair number of pairs to write
 * \param[in] mapped_data pairs to write
 */
template<typename dtype>
int tCTF<dtype>::write_tensor(int const               tensor_id, 
                              long_int const           num_pair,  
                              dtype const             alpha,
                              dtype const             beta,
                              tkv_pair<dtype> const * mapped_data){
  return dt->write_pairs(tensor_id, num_pair, alpha, beta, const_cast<tkv_pair<dtype>*>(mapped_data), 'w');
}

/**
 * \brief read tensor data with <key, value> pairs where key is the
 *              global index for the value, which gets filled in. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to read
 * \param[in,out] mapped_data pairs to read
 */
template<typename dtype>
int tCTF<dtype>::read_tensor(int const                tensor_id, 
                             long_int const            num_pair, 
                             tkv_pair<dtype> * const  mapped_data){
  return dt->write_pairs(tensor_id, num_pair, 1.0, 0.0, mapped_data, 'r');
}

template<typename dtype>
int tCTF<dtype>::slice_tensor( int const    tid_A,
                               int const *  offsets_A,
                               int const *  ends_A,
                               double const alpha,
                               int const    tid_B,
                               int const *  offsets_B,
                               int const *  ends_B,
                               double const beta){
  return dt->slice_tensor(tid_A, offsets_A, ends_A, alpha,
                          tid_B, offsets_B, ends_B, beta);
}

/**
 * \brief read entire tensor with each processor (in packed layout).
 *         WARNING: will use a lot of memory. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[in,out] mapped_data values read
 */
template<typename dtype>
int tCTF<dtype>::allread_tensor(int const   tensor_id, 
                                long_int *   num_pair, 
                                dtype **    all_data){
  int ret;
  long_int np;
  ret = dt->allread_tsr(tensor_id, &np, all_data);
  CTF_untag_mem(*all_data);
  *num_pair = np;
  return ret;
}

/* input tensor local data or set buffer for contract answer. */
/*int tCTF<dtype>::set_local_tensor(int const   tensor_id, 
                         int const      num_val, 
                         dtype *        tsr_data){
  return set_tsr_data(tensor_id, num_val, tsr_data);  
}*/

/**
 * \brief  map input tensor local data to zero
 * \param[in] tensor_id tensor handle
 */
template<typename dtype>
int tCTF<dtype>::set_zero_tensor(int const tensor_id){
  return dt->set_zero_tsr(tensor_id);
}

/**
 * \brief read tensor data pairs local to processor. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[out] mapped_data values read
 */
template<typename dtype>
int tCTF<dtype>::read_local_tensor(int const          tensor_id, 
                                   long_int *          num_pair,  
                                   tkv_pair<dtype> ** mapped_data){
  int ret;
  long_int np;
  ret = dt->read_local_pairs(tensor_id, &np, mapped_data);
  if (np > 0)
    CTF_untag_mem(*mapped_data);
  *num_pair = np;
  return ret;
}

/**
 * \brief contracts tensors alpha*A*B+beta*C -> C,
 *      uses standard symmetric contraction sequential kernel 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *  type,
                          dtype const             alpha,
                          dtype const             beta){
  fseq_tsr_ctr<dtype> fs;
  fs.func_ptr=sym_seq_ctr_ref<dtype>;
  return contract(type, fs, alpha, beta, 1);
}


/**
 * \brief contracts tensors alpha*A*B+beta*C -> C. 
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform sequential block op 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] func_ptr sequential block ctr func pointer 
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *    type,
                          fseq_tsr_ctr<dtype> const func_ptr, 
                          dtype const               alpha,
                          dtype const               beta,
                          int const                 map_inner){
  int i, ret;
#if DEBUG >= 1
  if (dt->get_global_comm()->rank == 0)
    printf("Head contraction :\n");
  dt->print_ctr(type,alpha,beta);
#endif
  fseq_elm_ctr<dtype> felm;
  felm.func_ptr = NULL;

  if ((*dt->get_tensors())[type->tid_A]->profile &&
      (*dt->get_tensors())[type->tid_B]->profile &&
      (*dt->get_tensors())[type->tid_C]->profile){
    char cname[200];
    cname[0] = '\0';
    if ((*dt->get_tensors())[type->tid_C]->name != NULL)
      sprintf(cname, (*dt->get_tensors())[type->tid_C]->name);
    else
      sprintf(cname, "%d", type->tid_C);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_C]->ndim; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_C[i]);
      else 
        sprintf(cname+strlen(cname),"%d",type->idx_map_C[i]);
    }
    sprintf(cname+strlen(cname),"]=");
    if ((*dt->get_tensors())[type->tid_A]->name != NULL)
      sprintf(cname+strlen(cname), (*dt->get_tensors())[type->tid_A]->name);
    else
      sprintf(cname+strlen(cname), "%d", type->tid_A);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_A]->ndim; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_A[i]);
      else
        sprintf(cname+strlen(cname),"%d",type->idx_map_A[i]);
    }
    sprintf(cname+strlen(cname),"]*");
    if ((*dt->get_tensors())[type->tid_B]->name != NULL)
      sprintf(cname+strlen(cname), (*dt->get_tensors())[type->tid_B]->name);
    else
      sprintf(cname+strlen(cname), "%d", type->tid_B);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_B]->ndim; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_B[i]);
      else 
        sprintf(cname+strlen(cname),"%d",type->idx_map_B[i]);
    }
    sprintf(cname+strlen(cname),"]");
    
   
    CTF_Timer tctr(cname);
    tctr.start(); 
    ret = dt->home_contract(type, func_ptr, felm, alpha, beta, map_inner);
    tctr.stop();
  } else 
    ret = dt->home_contract(type, func_ptr, felm, alpha, beta, map_inner);

  return ret;

}
    
/**
 * \brief contracts tensors alpha*A*B+beta*C -> C. 
        Accepts custom-sized buffer-space (set to NULL for dynamic allocs).
 *      seq_func used to perform elementwise sequential op 
 * \param[in] type the contraction type (defines contraction actors)
 * \param[in] felm sequential elementwise ctr func ptr
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::contract(CTF_ctr_type_t const *     type,
                          fseq_elm_ctr<dtype> const  felm,
                          dtype const                alpha,
                          dtype const                beta){
#if DEBUG >= 1
  if (dt->get_global_comm()->rank == 0)
    printf("Head custom contraction :\n");
  dt->print_ctr(type,alpha,beta);
#endif
  fseq_tsr_ctr<dtype> fs;
  fs.func_ptr=sym_seq_ctr_ref<dtype>;
  return dt->home_contract(type, fs, felm, alpha, beta, 0);

}

/**
 * \brief copy tensor from one handle to another
 * \param[in] tid_A tensor handle to copy from
 * \param[in] tid_B tensor handle to copy to
 */
template<typename dtype>
int tCTF<dtype>::copy_tensor(int const tid_A, int const tid_B){
  return dt->cpy_tsr(tid_A, tid_B);
}

/**
 * \brief scales a tensor by alpha
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const alpha, int const tid){
  return dt->scale_tsr(alpha, tid);
}
/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const               alpha, 
                              int const                 tid, 
                              int const *               idx_map){
  fseq_tsr_scl<dtype> fs;
  fs.func_ptr=sym_seq_scl_ref<dtype>;
  fseq_elm_scl<dtype> felm;
  felm.func_ptr = NULL;
  return dt->scale_tsr(alpha, tid, idx_map, fs, felm);
}

/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 * \param[in] func_ptr pointer to sequential scale function
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const               alpha, 
                              int const                 tid, 
                              int const *               idx_map,
                              fseq_tsr_scl<dtype> const func_ptr){
  fseq_elm_scl<dtype> felm;
  felm.func_ptr = NULL;
  return dt->scale_tsr(alpha, tid, idx_map, func_ptr, felm);
}

/**
 * \brief scales a tensor by alpha iterating on idx_map
 * \param[in] alpha scaling factor
 * \param[in] tid tensor handle
 * \param[in] idx_map indexer to the tensor
 * \param[in] felm pointer to sequential elemtwise scale function
 */
template<typename dtype>
int tCTF<dtype>::scale_tensor(dtype const               alpha, 
                              int const                 tid, 
                              int const *               idx_map,
                              fseq_elm_scl<dtype> const felm){
  fseq_tsr_scl<dtype> fs;
  fs.func_ptr=sym_seq_scl_ref<dtype>;
  return dt->scale_tsr(alpha, tid, idx_map, fs, felm);
}

  /**
   * \brief computes a dot product of two tensors A dot B
   * \param[in] tid_A tensor handle to A
 * \param[in] tid_B tensor handle to B
 * \param[out] product the result of the dot-product
 */
template<typename dtype>
int tCTF<dtype>::dot_tensor(int const tid_A, int const tid_B, dtype *product){
  int stat;
  /* check if the mappings of A and B are the same */
  stat = dt->check_pair_mapping(tid_A, tid_B);
  if (stat == 0){
    /* Align the mappings of A and B */
    stat = dt->map_tensor_pair(tid_A, tid_B);
    if (stat != DIST_TENSOR_SUCCESS)
      return stat;
  }
  /* Compute the dot product of A and B */
  return dt->dot_loc_tsr(tid_A, tid_B, product);
}

/**
 * \brief Performs an elementwise reduction on a tensor 
 * \param[in] tid tensor handle
 * \param[in] CTF::OP reduction operation to apply
 * \param[out] result result of reduction operation
 */
template<typename dtype>
int tCTF<dtype>::reduce_tensor(int const tid, CTF_OP op, dtype * result){
  return dt->red_tsr(tid, op, result);
}

/**
 * \brief Calls a mapping function on each element of the tensor 
 * \param[in] tid tensor handle
 * \param[in] map_func function pointer to apply to each element
 */
template<typename dtype>
int tCTF<dtype>::map_tensor(int const tid, 
                            dtype (*map_func)(int const   ndim, 
                                              int const * indices, 
                                              dtype const elem)){
  return dt->map_tsr(tid, map_func);
}

/**
 * \brief obtains a small number of the biggest elements of the 
 *        tensor in sorted order (e.g. eigenvalues)
 * \param[in] tid index of tensor
 * \param[in] n number of elements to collect
 * \param[in] data output data (should be preallocated to size at least n)
 */
template<typename dtype>
int tCTF<dtype>::get_max_abs(int const  tid,
                             int const  n,
                             dtype *    data){
  return dt->get_max_abs(tid, n, data);
}
    
/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 *               uses standard summation pointer
 * \param[in] type idx_maps and tids of contraction
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(CTF_sum_type_t const * type,
                             dtype const            alpha,
                             dtype const            beta){
  
  fseq_tsr_sum<dtype> fs;
  fs.func_ptr=sym_seq_sum_ref<dtype>;
  return sum_tensors(alpha, beta, type->tid_A, type->tid_B, 
                     type->idx_map_A, type->idx_map_B, fs);

}
    
/**
 * \brief DAXPY: a*idx_map_A(A) + b*idx_map_B(B) -> idx_map_B(B). 
 * \param[in] type idx_maps and tids of contraction
 * \param[in] alpha scaling factor for A*B
 * \param[in] beta scaling factor for C
 * \param[in] func_ptr sequential ctr func pointer 
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(CTF_sum_type_t const *     type,
                             dtype const                alpha,
                             dtype const                beta,
                             fseq_tsr_sum<dtype> const  func_ptr){
  return sum_tensors(alpha, beta, type->tid_A, type->tid_B, 
                     type->idx_map_A, type->idx_map_B, func_ptr);

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
template<typename dtype>
int tCTF<dtype>::sum_tensors(dtype const                alpha,
                             dtype const                beta,
                             int const                  tid_A,
                             int const                  tid_B,
                             int const *                idx_map_A,
                             int const *                idx_map_B,
                             fseq_tsr_sum<dtype> const  func_ptr){
  fseq_elm_sum<dtype> felm;
  felm.func_ptr = NULL;
  return dt->home_sum_tsr(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, func_ptr, felm);
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
template<typename dtype>
int tCTF<dtype>::sum_tensors(dtype const                alpha,
                             dtype const                beta,
                             int const                  tid_A,
                             int const                  tid_B,
                             int const *                idx_map_A,
                             int const *                idx_map_B,
                             fseq_elm_sum<dtype> const  felm){
  fseq_tsr_sum<dtype> fs;
  fs.func_ptr=sym_seq_sum_ref<dtype>;
  return dt->home_sum_tsr(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, fs, felm);
}

/**
 * \brief daxpy tensors A and B, B = B+alpha*A
 * \param[in] alpha scaling factor
 * \param[in] tid_A tensor handle of A
 * \param[in] tid_B tensor handle of B
 */
template<typename dtype>
int tCTF<dtype>::sum_tensors(dtype const  alpha,
                             int const    tid_A,
                             int const    tid_B){
  int stat;
  
  /* check if the mappings of A and B are the same */
  stat = dt->check_pair_mapping(tid_A, tid_B);
  if (stat == 0){
    /* Align the mappings of A and B */
    stat = dt->map_tensor_pair(tid_A, tid_B);
    if (stat != DIST_TENSOR_SUCCESS)
      return stat;
  }
  /* Sum tensors */
  return dt->daxpy_local_tensor_pair(alpha, tid_A, tid_B);
}

/**
 * \brief align mapping of tensor A to that of B
 * \param[in] tid_A tensor handle of A
 * \param[in] tid_B tensor handle of B
 */
template<typename dtype>
int tCTF<dtype>::align(int const    tid_A,
                       int const    tid_B){
  int stat;
  
  /* check if the mappings of A and B are the same */
  stat = dt->check_pair_mapping(tid_A, tid_B);
  if (stat == 0){
    /* Align the mappings of A and B */
    stat = dt->map_tensor_pair(tid_B, tid_A);
    if (stat != DIST_TENSOR_SUCCESS)
      return stat;
  }
  return DIST_TENSOR_SUCCESS;
}

template<typename dtype>
int tCTF<dtype>::print_tensor(FILE * stream, int const tid, double cutoff) {
  return dt->print_tsr(stream, tid, cutoff);
}

template<typename dtype>
int tCTF<dtype>::compare_tensor(FILE * stream, int const tid_A, int const tid_B, double cutoff) {
  int stat = align(tid_A, tid_B);
  if (stat != DIST_TENSOR_SUCCESS) return stat;
  return dt->compare_tsr(stream, tid_A, tid_B, cutoff);
}

/* Prints contraction type. */
template<typename dtype>
int tCTF<dtype>::print_ctr(CTF_ctr_type_t const * ctype,
                           dtype const            alpha,
                           dtype const            beta) const {
  return dt->print_ctr(ctype,alpha,beta);
}

/* Prints sum type. */
template<typename dtype>
int tCTF<dtype>::print_sum(CTF_sum_type_t const * stype,
                           dtype const            alpha,
                           dtype const            beta) const {
  return dt->print_sum(stype,alpha,beta);
}


/**
 * \brief removes all tensors, invalidates all handles
 */
template<typename dtype>
int tCTF<dtype>::clean_tensors(){
  unsigned int i;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  for (i=0; i<tensors->size(); i++){
    dt->del_tsr(i);
//    CTF_free((*tensors)[i]);
  }
  tensors->clear();
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief removes a tensor, invalidates its handle
 * \param tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::clean_tensor(int const tid){
  return dt->del_tsr(tid);
}

/**
 * \brief removes all tensors, invalidates all handles, and exits library.
 *              Do not use library instance after executing this.
 */
template<typename dtype>
int tCTF<dtype>::exit(){
  int ret;
  if (initialized){
    TAU_FSTOP(CTF);
#ifdef HPM
    HPM_Stop("CTF");
#endif
    ret = tCTF<dtype>::clean_tensors();
    LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
    delete dt;
    initialized = 0;
    return ret;
  } else
    return DIST_TENSOR_SUCCESS;
}

/* \brief ScaLAPACK back-end, see their DOC */
template<typename dtype>
int tCTF<dtype>::pgemm(char const   TRANSA, 
                       char const   TRANSB, 
                       int const    M, 
                       int const    N, 
                       int const    K, 
                       dtype const  ALPHA,
                       dtype *      A, 
                       int const    IA, 
                       int const    JA, 
                       int const *  DESCA, 
                       dtype *      B, 
                       int const    IB, 
                       int const    JB, 
                       int const *  DESCB, 
                       dtype const  BETA,
                       dtype *      C, 
                       int const    IC, 
                       int const    JC, 
                       int const *  DESCC){
  int ret, need_remap, i, j;
#if (!REDIST)
  int redist;
#endif
  int stid_A, stid_B, stid_C;
  int otid_A, otid_B, otid_C;
  long_int old_size_C;
  int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
  int * old_padding_C, * old_edge_len_C;
  int * need_free;
  int was_padded_C, was_cyclic_C;
  tensor<dtype> * tsr_nC, * tsr_oC;
  CTF_ctr_type ct;
  fseq_tsr_ctr<dtype> fs;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  CTF_alloc_ptr(3*sizeof(int), (void**)&need_free);
  ret = dt->pgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA,
                  B, IB, JB, DESCB,
                  BETA, C, IC, JC, DESCC, &ct, &fs, need_free);
  if (ret != DIST_TENSOR_SUCCESS)
    return ret;

  otid_A = ct.tid_A;
  otid_B = ct.tid_B;
  otid_C = ct.tid_C;
#if (!REDIST)
  ret = dt->try_topo_morph(otid_A, otid_B, otid_C);
  LIBT_ASSERT(ret == DIST_TENSOR_SUCCESS);
/*  dt->print_map(stdout, otid_A);
  dt->print_map(stdout, otid_B);
  dt->print_map(stdout, otid_C);*/
  redist = dt->check_contraction_mapping(&ct);
  if (redist == 0) {
    printf("REDISTRIBUTING\n");
#endif
    clone_tensor(ct.tid_A, 1, &stid_A);
    clone_tensor(ct.tid_B, 1, &stid_B);
    clone_tensor(ct.tid_C, 1, &stid_C);
    ct.tid_A = stid_A;
    ct.tid_B = stid_B;
    ct.tid_C = stid_C;
#if (!REDIST)
  }
#endif

  ret = this->contract(&ct, fs, ALPHA, BETA);
  if (ret != DIST_TENSOR_SUCCESS)
    return ret;
#if (!REDIST)
  if (redist == 0){
#endif
    tsr_oC = (*tensors)[otid_C];
    tsr_nC = (*tensors)[stid_C];
    need_remap = 0;
    if (tsr_oC->itopo == tsr_nC->itopo){
      if (!comp_dim_map(&tsr_oC->edge_map[0],&tsr_nC->edge_map[0]))
        need_remap = 1;
      if (!comp_dim_map(&tsr_oC->edge_map[1],&tsr_nC->edge_map[1]))
        need_remap = 1;
    } else
      need_remap = 1;
    if (need_remap){
      save_mapping(tsr_nC, &old_phase_C, &old_rank_C, &old_virt_dim_C, 
                   &old_pe_lda_C, &old_size_C, &was_padded_C, &was_cyclic_C, 
                   &old_padding_C, &old_edge_len_C, 
                   dt->get_topo(tsr_nC->itopo));
      if (need_free[2])
        CTF_free(tsr_oC->data);
      tsr_oC->data = tsr_nC->data;
      remap_tensor(otid_C, tsr_oC, dt->get_topo(tsr_oC->itopo), old_size_C, 
                   old_phase_C, old_rank_C, old_virt_dim_C, 
                   old_pe_lda_C, was_padded_C, was_cyclic_C, 
                   old_padding_C, old_edge_len_C, dt->get_global_comm());
    } else{
      if (need_free[2])
              CTF_free(tsr_oC->data);
      tsr_oC->data = tsr_nC->data;
    }
    /* If this process owns any data */
    if (!need_free[2]){
      memcpy(C,tsr_oC->data,tsr_oC->size*sizeof(dtype));
    } else
      CTF_free(tsr_oC->data);
    if (need_free[0])
      dt->del_tsr(otid_A);
    if (need_free[1])
      dt->del_tsr(otid_B);
    dt->del_tsr(stid_A);
    dt->del_tsr(stid_B);
    (*tensors)[stid_A]->is_alloced = 0;
    (*tensors)[stid_B]->is_alloced = 0;
    (*tensors)[stid_C]->is_alloced = 0;
#if (!REDIST)
  }
#endif
  if ((*tensors)[otid_A]->scp_padding[0] != 0 ||
      (*tensors)[otid_A]->scp_padding[1] != 0){
    CTF_free((*tensors)[otid_A]->data);
  }
  if ((*tensors)[otid_B]->scp_padding[0] != 0 ||
      (*tensors)[otid_B]->scp_padding[1] != 0){
    CTF_free((*tensors)[otid_B]->data);
  }
  if ((*tensors)[otid_C]->scp_padding[0] != 0 ||
      (*tensors)[otid_C]->scp_padding[1] != 0){
    int brow, bcol;
    brow = DESCC[4];
    bcol = DESCC[5];
    for (i=0; i<bcol-(*tensors)[otid_C]->scp_padding[1]; i++){
      for (j=0; j<brow-(*tensors)[otid_C]->scp_padding[0]; j++){
        C[i*(brow-(*tensors)[otid_C]->scp_padding[0])+j] 
          = (*tensors)[otid_C]->data[i*brow+j];
      }
    }
    CTF_free((*tensors)[otid_C]->data);
  }
  (*tensors)[otid_A]->is_data_aliased = 1;
  (*tensors)[otid_B]->is_data_aliased = 1;
  (*tensors)[otid_C]->is_data_aliased = 1;
  dt->del_tsr(otid_A);
  dt->del_tsr(otid_B);
  dt->del_tsr(otid_C);
  CTF_free(ct.idx_map_A);
  CTF_free(ct.idx_map_B);
  CTF_free(ct.idx_map_C);
  CTF_free(need_free);
  return DIST_TENSOR_SUCCESS;
}


/**
 * \brief define matrix from ScaLAPACK descriptor
 *
 * \param[in] DESCA ScaLAPACK descriptor for a matrix
 * \param[in] data pointer to actual data
 * \param[out] tid tensor handle
 */
template<typename dtype>
int tCTF<dtype>::def_scala_mat(int const * DESCA,
                               dtype const * data,
                               int * tid){
  int ret, stid;
  ret = dt->load_matrix((dtype*)data, DESCA, &stid, NULL);
  if (ret != DIST_TENSOR_SUCCESS) return ret;
  clone_tensor(stid, 1, tid);
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  tensor<dtype> * stsr = (*tensors)[stid];
  tensor<dtype> * tsr = (*tensors)[*tid];
  CTF_free(stsr->data);
  stsr->is_alloced = 0;
  tsr->is_matrix = 1;
  tsr->slay = stid;
  return DIST_TENSOR_SUCCESS;
}

/**
 * \brief reads a ScaLAPACK matrix to the original data pointer
 *
 * \param[in] tid tensor handle
 * \param[in,out] data pointer to buffer data
 */
template<typename dtype>
int tCTF<dtype>::read_scala_mat(int const tid,
                                dtype * data){
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda;
  int * old_padding, * old_edge_len;
  int was_padded, was_cyclic;
  long_int old_size;
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  tensor<dtype> * tsr = (*tensors)[tid];
  tensor<dtype> * stsr = (*tensors)[tsr->slay];
  dt->unmap_inner(tsr);
  save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, 
               &old_pe_lda, &old_size, &was_padded, &was_cyclic, 
               &old_padding, &old_edge_len, 
               dt->get_topo(tsr->itopo));
  LIBT_ASSERT(tsr->is_matrix);
  CTF_alloc_ptr(sizeof(dtype)*tsr->size, (void**)&stsr->data);
  memcpy(stsr->data, tsr->data, sizeof(dtype)*tsr->size);
  remap_tensor(tsr->slay, stsr, dt->get_topo(stsr->itopo), old_size, 
               old_phase, old_rank, old_virt_dim, 
               old_pe_lda, was_padded, was_cyclic, 
               old_padding, old_edge_len, dt->get_global_comm());
  if (data!=NULL)
    memcpy(data, stsr->data, stsr->size*sizeof(dtype));  
  CTF_free(stsr->data);
  return DIST_TENSOR_SUCCESS;
}
/**
 * \brief CTF interface for pgemm
 */
template<typename dtype>
int tCTF<dtype>::pgemm(char const   TRANSA, 
                       char const   TRANSB, 
                       int const    M, 
                       int const    N, 
                       int const    K, 
                       dtype const  ALPHA,
                       int const    tid_A,
                       int const    tid_B,
                       dtype const  BETA,
                       int const    tid_C){
  int herm_A, herm_B, ret;
  CTF_ctr_type ct;
  fseq_tsr_ctr<dtype> fs;
  ct.tid_A = tid_A;
  ct.tid_B = tid_B;
  ct.tid_C = tid_C;

  ct.idx_map_A = (int*)CTF_alloc(sizeof(int)*2);
  ct.idx_map_B = (int*)CTF_alloc(sizeof(int)*2);
  ct.idx_map_C = (int*)CTF_alloc(sizeof(int)*2);
  ct.idx_map_C[0] = 1;
  ct.idx_map_C[1] = 2;
  herm_A = 0;
  herm_B = 0;
  if (TRANSA == 'N' || TRANSA == 'n'){
    ct.idx_map_A[0] = 1;
    ct.idx_map_A[1] = 0;
  } else {
    LIBT_ASSERT(TRANSA == 'T' || TRANSA == 't' || TRANSA == 'c' || TRANSA == 'C');
    if (TRANSA == 'c' || TRANSA == 'C')
      herm_A = 1;
    ct.idx_map_A[0] = 0;
    ct.idx_map_A[1] = 1;
  }
  if (TRANSB == 'N' || TRANSB == 'n'){
    ct.idx_map_B[0] = 0;
    ct.idx_map_B[1] = 2;
  } else {
    LIBT_ASSERT(TRANSB == 'T' || TRANSB == 't' || TRANSB == 'c' || TRANSB == 'C');
    if (TRANSB == 'c' || TRANSB == 'C')
      herm_B = 1;
    ct.idx_map_B[0] = 2;
    ct.idx_map_B[1] = 0;
  }
  if (herm_A && herm_B)
    fs.func_ptr = &gemm_ctr<dtype,1,1>;
  else if (herm_A)
    fs.func_ptr = &gemm_ctr<dtype,1,0>;
  else if (herm_B)
    fs.func_ptr = &gemm_ctr<dtype,0,1>;
  else
    fs.func_ptr = &gemm_ctr<dtype,0,0>;
  ret = this->contract(&ct, fs, ALPHA, BETA);
  std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
  CTF_free(ct.idx_map_A);
  CTF_free(ct.idx_map_B);
  CTF_free(ct.idx_map_C);
  return ret;
};
  
/* Instantiate the ugly templates */
template class tCTF<double>;
#if (VERIFY==0)
template class tCTF< std::complex<double> >;
#endif


