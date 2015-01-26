/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "dist_tensor_internal.h"
#include "dt_aux_permute.hxx"
#include "dt_aux_sort.hxx"
#include "dt_aux_rw.hxx"
#include "dt_aux_map.hxx"
#include "dt_aux_topo.hxx"
#include "cyclopstf.hpp"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "../ctr_comm/ctr_comm.h"
#include "../ctr_comm/ctr_tsr.h"
#include "../ctr_comm/sum_tsr.h"
#include "../ctr_comm/strp_tsr.h"
#include <limits.h>
#include <float.h>
#include <stdint.h>
#include <vector>
#include <set>
#include <list>
#include <algorithm>
#include <errno.h>

#define MAX_NVIRT 256
#ifndef MIN_NVIRT
#define MIN_NVIRT 1
#endif
//#define USE_VIRT_25D


/* accessors */
template<typename dtype>
CommData dist_tensor<dtype>::get_global_comm(){ return global_comm; }
template<typename dtype>
void dist_tensor<dtype>::set_global_comm(CommData cdt){ global_comm = cdt; }



/**
 * \brief deallocates all internal state
 */
template<typename dtype>
int dist_tensor<dtype>::dist_cleanup(){
  int j;
  std::vector<topology>::iterator iter;
  
  std::set<MPI_Comm> commset;
  std::set<MPI_Comm>::iterator csiter;
  for (iter=topovec.begin(); iter<topovec.end(); iter++){
    for (j=0; j<iter->order; j++){
      if (iter->dim_comm[j].alive)
        commset.insert(iter->dim_comm[j].cm);
    }
    CTF_free(iter->dim_comm);
    CTF_free(iter->lda);
  }
  topovec.clear();
  for (csiter=commset.begin(); csiter!=commset.end(); csiter++){
    if (*csiter != MPI_COMM_WORLD && *csiter != MPI_COMM_SELF){
      MPI_Comm ctmp = *csiter;
      MPI_Comm_free(&ctmp);
    }
  } 
  commset.clear();
  int is_mpi_dead;
  MPI_Finalized(&is_mpi_dead);
  if (!is_mpi_dead)
    FREE_CDT(global_comm);
  return CTF_SUCCESS;
}

/**
 * \brief destructor
 */
template<typename dtype>
dist_tensor<dtype>::~dist_tensor(){
  int ret;
  ret = dist_cleanup();
  ASSERT(ret == CTF_SUCCESS);
}


/**
 * \brief  initializes library.
 * \param[in] order is the number of dimensions in the topology
 * \param[in] dim_len is the number of processors along each dimension
 * \param[in] inner_size is the total block size of dgemm calls
 */
template<typename dtype>
int dist_tensor<dtype>::initialize(CommData   cdt_global,
                                   int const    order,
                                   int const *  dim_len){
  int i, rank, stride, cut;
  int * srt_dim_len;


  CTF_alloc_ptr(order*sizeof(int), (void**)&srt_dim_len);
  memcpy(srt_dim_len, dim_len, order*sizeof(int));

  rank = cdt_global.rank;

  /* setup global communicator */
  set_global_comm(cdt_global);

  /* setup dimensional communicators */
  CommData *   phys_comm = (CommData*)CTF_alloc(order*sizeof(CommData));

/* FIXME: Sorting will fuck up dimensional ordering */
//  std::sort(srt_dim_len, srt_dim_len + order);

  if (cdt_global.rank == 0)
    VPRINTF(1,"Setting up initial torus physical topology P:\n");
  stride = 1, cut = 0;
  for (i=0; i<order; i++){
    ASSERT(dim_len[i] != 1);
    if (cdt_global.rank == 0)
      VPRINTF(1,"P[%d] = %d\n",i,srt_dim_len[i]);

    SETUP_SUB_COMM_SHELL(cdt_global, phys_comm[i],
                   ((rank/stride)%srt_dim_len[order-i-1]),
                   (((rank/(stride*srt_dim_len[order-i-1]))*stride)+cut),
                   srt_dim_len[order-i-1]);
    stride*=srt_dim_len[order-i-1];
    cut = (rank - (rank/stride)*stride);
  }
  set_phys_comm(phys_comm, order);
/*  for (i=0; i<(int)topovec.size(); i++){
    for (int j=0; j<topovec[i].order; j++){
      if (topovec[i].dim_comm[j].alive == 0){
        SHELL_SPLIT(cdt_global, topovec[i].dim_comm[j]);
      }
    }
  }*/
  CTF_free(srt_dim_len);

  return CTF_SUCCESS;
}



/**
 * \brief sets the physical torus topology
 * \param[in] cdt grid communicator
 * \param[in] order number of dimensions
 */
template<typename dtype>
void dist_tensor<dtype>::set_phys_comm(CommData *   cdt, int const order, int fold){
  int i, lda;
  topology new_topo;


  new_topo.order = order;
  new_topo.dim_comm = cdt;

  /* do not duplicate topologies */
  if (find_topology(&new_topo, topovec) != -1){
    CTF_free(cdt);
    return;
  }

  new_topo.lda = (int*)CTF_alloc(sizeof(int)*order);
  lda = 1;
  /* Figure out the lda of each dimension communicator */
  for (i=0; i<order; i++){
#if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("Added topo %d dim[%d] = %d:\n",(int)topovec.size(),i,cdt[i].np);
#endif
    ASSERT(cdt[i].np != 1);
    new_topo.lda[i] = lda;
    lda = lda*cdt[i].np;
    ASSERT(cdt[i].np > 0);
  }
  topovec.push_back(new_topo);

  if (order > 1 && fold)
    fold_torus(&new_topo, global_comm, this);
}



/**
 * \brief  defines a tensor and retrieves handle
 *
 * \param[in] order number of tensor dimensions
 * \param[in] edge_len global edge lengths of tensor
 * \param[in] sym symmetry relations of tensor
 * \param[out] tensor_id the tensor index (handle)
 * \param[in] alloc_data whether this tensor's data should be alloced
 * \param[in] name string name for tensor (optionary)
 * \param[in] profile wether to make profile calls for the tensor
 */
template<typename dtype>
int dist_tensor<dtype>::define_tensor( int const          order,
                                       int const *        edge_len, 
                                       int const *        sym,
                                       int *              tensor_id,
                                       int const          alloc_data,
                                       char const *       name,
                                       int                profile){
}

template<typename dtype>
std::vector< tensor<dtype>* > * dist_tensor<dtype>::get_tensors(){ return &tensors; }

/**
 * \brief sets the data in the tensor
 * \param[in] tensor_id tensor handle,
 * \param[in] num_val new number of data values
 * \param[in] tsr_data new tensor data
 */
template<typename dtype>
int dist_tensor<dtype>::set_tsr_data( int const tensor_id,
                                      int const num_val,
                                      dtype * tsr_data){
  tensors[tensor_id]->data = tsr_data;
  tensors[tensor_id]->size = num_val;
  return CTF_SUCCESS;
}

template<typename dtype>
topology * dist_tensor<dtype>::get_topo(int const itopo) {
  return &topovec[itopo];
}

template<typename dtype>
int dist_tensor<dtype>::get_dim(int const tensor_id) const { return tensors[tensor_id]->order; }

template<typename dtype>
int * dist_tensor<dtype>::get_edge_len(int const tensor_id) const {
  int i;
  int * edge_len;
  CTF_alloc_ptr(tensors[tensor_id]->order*sizeof(int), (void**)&edge_len);

  for (i=0; i<tensors[tensor_id]->order; i++){
    edge_len[i] = tensors[tensor_id]->edge_len[i]
                 -tensors[tensor_id]->padding[i];
  }

  return edge_len;
}

template<typename dtype>
int dist_tensor<dtype>::get_name(int const tensor_id, char const ** name){
  *name = tensors[tensor_id]->name;
  return CTF_SUCCESS;
}
 
template<typename dtype>
int dist_tensor<dtype>::set_name(int const tensor_id, char const * name){
  tensors[tensor_id]->name = name;
  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::profile_on(int const tensor_id){
  tensors[tensor_id]->profile = 1;
  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::profile_off(int const tensor_id){
  tensors[tensor_id]->profile = 0;
  return CTF_SUCCESS;
}

template<typename dtype>
int * dist_tensor<dtype>::get_sym(int const tensor_id) const {
  int * sym;
  CTF_alloc_ptr(tensors[tensor_id]->order*sizeof(int), (void**)&sym);
  memcpy(sym, tensors[tensor_id]->sym, tensors[tensor_id]->order*sizeof(int));

  return sym;
}

/* \brief get raw data pointer WARNING: includes padding
 * \param[in] tensor_id id of tensor
 * \return raw local data
 */
template<typename dtype>
dtype * dist_tensor<dtype>::get_raw_data(int const tensor_id, int64_t * size) {
  if (tensors[tensor_id]->has_zero_edge_len){
    *size = 0;
    return NULL;
  }
  *size = tensors[tensor_id]->size;
  return tensors[tensor_id]->data;
}

/**
 * \brief pulls out information about a tensor
 * \param[in] tensor_id tensor handle
 * \param[out] order the dimensionality of the tensor
 * \param[out] edge_len the number of processors along each dimension
 *             of the tensor
 * \param[out] sym the symmetries of the tensor
 */
template<typename dtype>
int dist_tensor<dtype>::get_tsr_info( int const         tensor_id,
                                      int *             order,
                                      int **            edge_len,
                                      int **            sym) const{
  int i;
  int nd;
  int * el, * s;

  const tensor<dtype> * tsr = tensors[tensor_id];

  nd = tsr->order;
  CTF_alloc_ptr(nd*sizeof(int), (void**)&el);
  CTF_alloc_ptr(nd*sizeof(int), (void**)&s);
  for (i=0; i<nd; i++){
    el[i] = tsr->edge_len[i] - tsr->padding[i];
  }
  memcpy(s, tsr->sym, nd*sizeof(int));

  *order = nd;
  *edge_len = el;
  *sym = s;

  return CTF_SUCCESS;
}


/* \brief clone a tensor object
 * \param[in] tensor_id id of old tensor
 * \param[in] copy_data if 0 then leave tensor blank, if 1 copy data from old
 * \param[out] new_tensor_id id of new tensor
 */
template<typename dtype>
int dist_tensor<dtype>::clone_tensor( int const tensor_id,
                                      int const copy_data,
                                      int *     new_tensor_id,
                                      int const alloc_data){
  int order, * edge_len, * sym;
  get_tsr_info(tensor_id, &order, &edge_len, &sym);
  define_tensor(order, edge_len, sym, 
                new_tensor_id, alloc_data, tensors[tensor_id]->name, tensors[tensor_id]->profile);
  if (global_comm.rank == 0)
    DPRINTF(2,"Cloned tensor %d from tensor %d\n", *new_tensor_id,tensor_id);
  CTF_free(edge_len), CTF_free(sym);
  if (copy_data){
    return cpy_tsr(tensor_id, *new_tensor_id);
  }
  return CTF_SUCCESS;
}

/**
 * \brief copies tensor A to tensor B
 * \param[in] tid_A handle to A
 * \param[in] tid_B handle to B
 */
template<typename dtype>
int dist_tensor<dtype>::cpy_tsr(int const tid_A, int const tid_B){
  int i;
  tensor<dtype> * tsr_A, * tsr_B;

  if (global_comm.rank == 0)
    DPRINTF(2,"Copying tensor %d to tensor %d\n", tid_A, tid_B);

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
  
  tsr_B->has_zero_edge_len = tsr_A->has_zero_edge_len;

  if (tsr_A->is_folded) unfold_tsr(tsr_A);

  if (tsr_A->is_mapped){
    if (tsr_B->is_mapped){
      if (tsr_B->size < tsr_A->size || tsr_B->size > 2*tsr_A->size){
        CTF_free(tsr_B->data);
        CTF_alloc_ptr(tsr_A->size*sizeof(dtype), (void**)&tsr_B->data);
      } 
    } else {
      if (tsr_B->pairs != NULL) 
        CTF_free(tsr_B->pairs);
      CTF_alloc_ptr(tsr_A->size*sizeof(dtype), (void**)&tsr_B->data);
    }
#ifdef HOME_CONTRACT
    if (tsr_A->has_home){
      if (tsr_B->has_home && 
          (!tsr_B->is_home && tsr_B->home_size != tsr_A->home_size)){ 
        CTF_free(tsr_B->home_buffer);
      }
      if (tsr_A->is_home){
        tsr_B->home_buffer = tsr_B->data;
        tsr_B->is_home = 1;
      } else {
        if (tsr_B->is_home || tsr_B->home_size != tsr_A->home_size){ 
          tsr_B->home_buffer = (dtype*)CTF_alloc(tsr_A->home_size);
        }
        tsr_B->is_home = 0;
        memcpy(tsr_B->home_buffer, tsr_A->home_buffer, tsr_A->home_size);
      }
      tsr_B->has_home = 1;
    } else {
      if (tsr_B->has_home && !tsr_B->is_home){
        CTF_free(tsr_B->home_buffer);
      }
      tsr_B->has_home = 0;
      tsr_B->is_home = 0;
    }
    tsr_B->home_size = tsr_A->home_size;
#endif
    memcpy(tsr_B->data, tsr_A->data, sizeof(dtype)*tsr_A->size);
  } else {
    if (tsr_B->is_mapped){
      CTF_free(tsr_B->data);
      CTF_alloc_ptr(tsr_A->size*sizeof(tkv_pair<dtype>), 
                       (void**)&tsr_B->pairs);
    } else {
      if (tsr_B->size < tsr_A->size || tsr_B->size > 2*tsr_A->size){
        CTF_free(tsr_B->pairs);
        CTF_alloc_ptr(tsr_A->size*sizeof(tkv_pair<dtype>), 
                         (void**)&tsr_B->pairs);
      }
    }
    memcpy(tsr_B->pairs, tsr_A->pairs, 
            sizeof(tkv_pair<dtype>)*tsr_A->size);
  } 
  if (tsr_B->is_folded){
    del_tsr(tsr_B->rec_tid);
  }
  tsr_B->is_folded = tsr_A->is_folded;
  if (tsr_A->is_folded){
    int new_tensor_id;
    tensor<dtype> * itsr = tensors[tsr_A->rec_tid];
    define_tensor(tsr_A->order, itsr->edge_len, tsr_A->sym, 
                              &new_tensor_id, 0);
    CTF_alloc_ptr(sizeof(int)*tsr_A->order, 
                     (void**)&tsr_B->inner_ordering);
    for (i=0; i<tsr_A->order; i++){
      tsr_B->inner_ordering[i] = tsr_A->inner_ordering[i];
    }
    tsr_B->rec_tid = new_tensor_id;
  }

  if (tsr_A->order != tsr_B->order){
    CTF_free(tsr_B->edge_len);
    CTF_free(tsr_B->padding);
    CTF_free(tsr_B->sym);
    CTF_free(tsr_B->sym_table);
    if (tsr_B->is_mapped)
      CTF_free(tsr_B->edge_map);

    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)&tsr_B->edge_len);
    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)tsr_B->padding);
    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)tsr_B->sym);
    CTF_alloc_ptr(tsr_A->order*tsr_A->order*sizeof(int), (void**)tsr_B->sym_table);
    CTF_alloc_ptr(tsr_A->order*sizeof(mapping), (void**)tsr_B->edge_map);
  }

  tsr_B->order = tsr_A->order;
  memcpy(tsr_B->edge_len, tsr_A->edge_len, sizeof(int)*tsr_A->order);
  memcpy(tsr_B->padding, tsr_A->padding, sizeof(int)*tsr_A->order);
  memcpy(tsr_B->sym, tsr_A->sym, sizeof(int)*tsr_A->order);
  memcpy(tsr_B->sym_table, tsr_A->sym_table, sizeof(int)*tsr_A->order*tsr_A->order);
  tsr_B->is_mapped      = tsr_A->is_mapped;
  tsr_B->is_cyclic      = tsr_A->is_cyclic;
  tsr_B->itopo          = tsr_A->itopo;
  if (tsr_A->is_mapped)
    copy_mapping(tsr_A->order, tsr_A->edge_map, tsr_B->edge_map);
  tsr_B->size = tsr_A->size;

  return CTF_SUCCESS;
}
    

/**
 * \brief permutes a tensor along each dimension skips if perm set to -1, generalizes slice.
 *        one of permutation_A or permutation_B has to be set to NULL, if multiworld read, then
 *        the parent world tensor should not be being permuted
 *
 * \param[in] tid_A id of first tensor
 * \param[in] permutation_A permutation for each edge lengths of the first tensor, can be NULL,
 *                          and subpointers can be NULL
 * \param[in] dt_A the dist_tensor object (world) on which A lives
 * \param[in] alpha scaling factor for the A tensor
 * \param[in] tid_B analogous to tid_A
 * \param[in] permutation_B analogous to permutation_A
 * \param[in] beta analogous to alpha
 * \param[in] dt_B analogous to dt_A
 */
template<typename dtype>
int dist_tensor<dtype>::permute_tensor(int const              tid_A,
                                       int * const *          permutation_A,
                                       dtype const            alpha,
                                       dist_tensor<dtype> *   dt_A,
                                       int const              tid_B,
                                       int * const *          permutation_B,
                                       dtype const            beta,
                                       dist_tensor<dtype> *   dt_B){
    
}

template<typename dtype>
void dist_tensor<dtype>::orient_subworld(int                 order,
                                        int                 tid_sub,
                                        dist_tensor<dtype> *dt_sub,
                                        int &               bw_mirror_rank,
                                        int &               fw_mirror_rank,
                                        distribution &      odst,
                                        dtype **            sub_buffer_){
}

//#define USE_SLICE_FOR_SUBWORLD
template<typename dtype>
int dist_tensor<dtype>::add_to_subworld(int                 tid,
                                        int                 tid_sub,
                                        dist_tensor<dtype> *dt_sub,
                                        dtype               alpha,
                                        dtype               beta){
  return CTF_SUCCESS; 
}

template<typename dtype>
int dist_tensor<dtype>::add_from_subworld(int                 tid,
                                          int                 tid_sub,
                                          dist_tensor<dtype> *dt_sub,
                                          dtype              alpha,
                                          dtype              beta){
  CTF_free(lens);
  CTF_free(sym);
  return CTF_SUCCESS; 
}
    

/**
 * Add tensor data from A to a block of B, 
 *      B[offsets_B:ends_B] = beta*B[offsets_B:ends_B] + alpha*A[offsets_A:ends_A] 
 * \param[in] tid_A id of tensor A
 * \param[in] offsets_A closest corner of tensor block in A
 * \param[in] ends_A furthest corner of tensor block in A
 * \param[in] permutation_A permutation of indices for each dimension (NULL if none)
 * \param[in] alpha scaling factor of A
 * \param[in] tid_B id of tensor B
 * \param[in] offsets_B closest corner of tensor block in B
 * \param[in] ends_B furthest corner of tensor block in B
 * \param[in] permutation_B permutation of indices for each dimension (NULL if none)
 * \param[in] alpha scaling factor of B
 */
template<typename dtype>
int dist_tensor<dtype>::slice_tensor(int const              tid_A,
                                     int const *            offsets_A,
                                     int const *            ends_A,
                                     dtype const            alpha,
                                     dist_tensor<dtype> *   dt_A,
                                     int const              tid_B,
                                     int const *            offsets_B,
                                     int const *            ends_B,
                                     dtype const            beta,
                                     dist_tensor<dtype> *   dt_B){
}


/**
 * \brief  Read or write tensor data by <key, value> pairs where key is the
 *              global index for the value. 
 * \param[in] tensor_id tensor handle
 * \param[in] num_pair number of pairs to write
 * \param[in] alpha multiplier for new value
 * \param[in] beta multiplier for old value
 * \param[in,out] mapped_data pairs to read/write
 * \param[in] rw weather to read (r) or write (w)
 */
template<typename dtype>
int dist_tensor<dtype>::write_pairs(int const           tensor_id, 
                                    int64_t const      num_pair,  
                                    dtype const         alpha,  
                                    dtype const         beta,  
                                    tkv_pair<dtype> *   mapped_data, 
                                    char const          rw){

}

/**
 * \brief Retrieve local data in the form of key value pairs
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of pairs read
 * \param[out] mapped_data pairs read
 */
template<typename dtype>
int dist_tensor<dtype>::read_local_pairs(int                tensor_id, 
                                         int64_t *          num_pair,  
                                         tkv_pair<dtype> ** mapped_data){
}


/**
 * \brief read entire tensor with each processor (in packed layout).
 *         WARNING: will use a lot of memory. 
 * \param[in] tensor_id tensor handle
 * \param[out] num_pair number of values read
 * \param[in,out] mapped_data values read
 * \param[in] is_prealloced set to 1 if all_data is already allocated, 0
 *            default
 */
template<typename dtype>
int dist_tensor<dtype>::allread_tsr(int const     tid, 
                                    int64_t *     num_val,  
                                    dtype **      all_data,
                                    int const     is_prealloced){
  int numPes;
  int * nXs;
  int nval, n, i;
  int * pXs;
  tkv_pair<dtype> * my_pairs, * all_pairs;

  numPes = global_comm.np;
  if (tensors[tid]->has_zero_edge_len){
    *num_val = 0;
    return CTF_SUCCESS;
  }

  CTF_alloc_ptr(numPes*sizeof(int), (void**)&nXs);
  CTF_alloc_ptr(numPes*sizeof(int), (void**)&pXs);
  pXs[0] = 0;

  int64_t ntt = 0;
  my_pairs = NULL;
  read_local_pairs(tid, &ntt, &my_pairs);
  n = (int)ntt;
  n*=sizeof(tkv_pair<dtype>);
  ALLGATHER(&n, 1, MPI_INT, nXs, 1, MPI_INT, global_comm);
  for (i=1; i<numPes; i++){
    pXs[i] = pXs[i-1]+nXs[i-1];
  }
  nval = pXs[numPes-1] + nXs[numPes-1];
  CTF_alloc_ptr(nval, (void**)&all_pairs);
  MPI_Allgatherv(my_pairs, n, MPI_CHAR,
                 all_pairs, nXs, pXs, MPI_CHAR, global_comm.cm);
  nval = nval/sizeof(tkv_pair<dtype>);

  std::sort(all_pairs,all_pairs+nval);
  if (n>0)
    CTF_free(my_pairs);
  if (!is_prealloced)
    CTF_alloc_ptr(nval*sizeof(dtype), (void**)all_data);
  for (i=0; i<nval; i++){
    (*all_data)[i] = all_pairs[i].d;
  }
  *num_val = (int64_t)nval;

  CTF_free(nXs);
  CTF_free(pXs);
  CTF_free(all_pairs);

  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::get_max_abs(int const  tid,
                                    int const  n,
                                    dtype *    data){
  printf("CTF: Currently unable to get largest values of non-double type array, exiting.\n");
  return CTF_ERROR;
}

/**
 * \brief obtains a small number of the biggest elements of the 
 *        tensor in sorted order (e.g. eigenvalues)
 * \param[in] tid index of tensor
 * \param[in] n number of elements to collect
 * \param[in] data output data (should be preallocated to size at least n)
 */
template<>
int dist_tensor<double>::get_max_abs(int const  tid,
                                     int const  n,
                                     double *    data){
  int i, j, con, np, rank;
  tensor<double> * tsr;
  double val, swp;
  double * recv_data, * merge_data;
  MPI_Status stat;

  CTF_alloc_ptr(n*sizeof(double), (void**)&recv_data);
  CTF_alloc_ptr(n*sizeof(double), (void**)&merge_data);

  tsr = tensors[tid];
  
  std::fill(data, data+n, get_zero<double>());
  for (i=0; i<tsr->size; i++){
    val = std::abs(tsr->data[i]);
    for (j=0; j<n; j++){
      if (val > data[j]){
        swp = val;
        val = data[j];
        data[j] = swp;
      }
    }
  }
  np = global_comm.np;
  rank = global_comm.rank;
  con = np/2;
  while (con>0){
    if (np%2 == 1) con++;
    if (rank+con < np){
      MPI_Recv(recv_data, n*sizeof(double), MPI_CHAR, rank+con, 0, global_comm.cm, &stat);
      i=0, j=0;
      while (i+j<n){
        if (data[i]<recv_data[j]){
          merge_data[i+j] = data[i];
          i++;
        } else {
          merge_data[i+j] = recv_data[j];
          j++;
        }
      }  
      memcpy(data, merge_data, sizeof(double)*n);
    } else if (rank-con >= 0 && rank < np){
      MPI_Send(data, n*sizeof(double), MPI_CHAR, rank-con, 0, global_comm.cm);
    }
    np = np/2 + (np%2);
    con = np/2;
  }
  MPI_Bcast(data, n*sizeof(double), MPI_CHAR, 0, global_comm.cm);
  CTF_free(merge_data);
  CTF_free(recv_data);
  return CTF_SUCCESS;
}


/* \brief deletes a tensor and deallocs the data
 */
template<typename dtype>
int dist_tensor<dtype>::del_tsr(int const tid){
  tensor<dtype> * tsr;

  tsr = tensors[tid];
  if (tsr != NULL){
    if (global_comm.rank == 0){
      DPRINTF(3,"Deleting tensor %d\n",tid);
    }
    //unfold_tsr(tsr);
    if (tsr->is_folded){ 
      del_tsr(tsr->rec_tid);
      CTF_free(tsr->inner_ordering);
    }
    CTF_free(tsr->edge_len);
    CTF_free(tsr->padding);
    if (tsr->is_scp_padded)
      CTF_free(tsr->scp_padding);
    CTF_free(tsr->sym);
    CTF_free(tsr->sym_table);
    if (tsr->is_mapped){
      if (!tsr->is_data_aliased){
        CTF_free(tsr->data);
        if (tsr->has_home && !tsr->is_home) 
          CTF_free(tsr->home_buffer);
      }
      clear_mapping(tsr);
    }
    CTF_free(tsr->edge_map);
    tsr->is_alloced = 0;
    CTF_free(tsr);
    tensors[tid] = NULL;
  }

  return CTF_SUCCESS;
}

/* WARNING: not a standard spec function, know what you are doing */
template<typename dtype>
int dist_tensor<dtype>::elementalize(int const      tid,
                                     int const      x_rank,
                                     int const      x_np,
                                     int const      y_rank,
                                     int const      y_np,
                                     int64_t const blk_sz,
                                     dtype *          data){
  tensor<dtype> * tsr;
  int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda, * old_padding;
  int * new_phase, * new_rank, * new_virt_dim, * new_pe_lda, * new_edge_len;
  int * new_padding, * old_edge_len;
  dtype * shuffled_data;
  int i, j, pad, my_x_dim, my_y_dim, was_cyclic;
  int64_t old_size;



  tsr = tensors[tid];
  unmap_inner(tsr);
  set_padding(tsr);

  assert(tsr->is_mapped);
  assert(tsr->order == 2);
  assert(tsr->sym[0] == NS);
  assert(tsr->sym[1] == NS);

  save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, &old_pe_lda, 
                     &old_size, &was_cyclic, &old_padding, &old_edge_len, &topovec[tsr->itopo]);

  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_phase);
  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_rank);
  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_pe_lda);
  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_virt_dim);
  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_padding);
  CTF_alloc_ptr(sizeof(int)*tsr->order,       (void**)&new_edge_len);

  new_phase[0]          = x_np;
  new_rank[0]           = x_rank;
  new_virt_dim[0]       = 1;
  new_pe_lda[0]         = 1;

  new_phase[1]          = y_np;
  new_rank[1]           = y_rank;
  new_virt_dim[1]       = 1;
  new_pe_lda[1]         = x_np;

  for (j=0; j<tsr->order; j++){
    new_edge_len[j] = tsr->edge_len[j];
    pad = (tsr->edge_len[j]-tsr->padding[j])%new_phase[j];
    if (pad != 0)
      pad = new_phase[j]-pad;
    new_padding[j] = pad;
  }

  if (false){
    padded_reshuffle(tid,
                     tsr->order,
                     old_size,
                     new_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     old_padding,
                     old_edge_len,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     new_padding,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     &shuffled_data,
                     get_global_comm());
  } else {
    ABORT;
/*    CTF_alloc_ptr(sizeof(dtype)*tsr->size, (void**)&shuffled_data);
    cyclic_reshuffle(tsr->order,
                     tsr->size,
                     new_edge_len,
                     tsr->sym,
                     old_phase,
                     old_rank,
                     old_pe_lda,
                     new_phase,
                     new_rank,
                     new_pe_lda,
                     old_virt_dim,
                     new_virt_dim,
                     tsr->data,
                     shuffled_data,
                     get_global_comm());
    new_blk_sz = tsr->size;
    new_padding = tsr->padding;*/
  }

  my_x_dim = (new_edge_len[0]-new_padding[0])/x_np;
  if (x_rank < (new_edge_len[0]-new_padding[0])%x_np)
    my_x_dim++;
  my_y_dim = (new_edge_len[1]-new_padding[1])/y_np;
  if (y_rank < (new_edge_len[1]-new_padding[1])%y_np)
    my_y_dim++;

  assert(my_x_dim*my_y_dim == blk_sz);

  for (i=0; i<my_y_dim; i++){
    for (j=0; j<my_x_dim; j++){
            data[j+i*my_x_dim] = shuffled_data[j+i*new_edge_len[0]];
    }
  }


  CTF_free((void*)new_phase);
  CTF_free((void*)new_rank);
  CTF_free((void*)new_virt_dim);
  CTF_free((void*)new_edge_len);
  CTF_free((void*)shuffled_data);

  return CTF_SUCCESS;
}

/* \brief set the tensor to zero, called internally for each defined tensor
 * \param tensor_id id of tensor to set to zero
 */
template<typename dtype>
int dist_tensor<dtype>::set_zero_tsr(int tensor_id){
  tensor<dtype> * tsr;
  int * restricted;
  int i, map_success, btopo;
  uint64_t nvirt, bnvirt;
  uint64_t memuse, bmemuse;
  tsr = tensors[tensor_id];

  if (tsr->is_mapped){
    std::fill(tsr->data, tsr->data + tsr->size, get_zero<dtype>());
  } else {
    if (tsr->pairs != NULL){
      for (i=0; i<tsr->size; i++) tsr->pairs[i].d = get_zero<dtype>();
    } else {
      CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&restricted);
//      memset(restricted, 0, tsr->order*sizeof(int));

      /* Map the tensor if necessary */
      bnvirt = UINT64_MAX, btopo = -1;
      bmemuse = UINT64_MAX;
      for (i=global_comm.rank; i<(int)topovec.size(); i+=global_comm.np){
        clear_mapping(tsr);
        set_padding(tsr);
        memset(restricted, 0, tsr->order*sizeof(int));
        map_success = map_tensor(topovec[i].order, tsr->order, tsr->edge_len,
                                 tsr->sym_table, restricted,
                                 topovec[i].dim_comm, NULL, 0,
                                 tsr->edge_map);
        if (map_success == CTF_ERROR) {
          ASSERT(0);
          return CTF_ERROR;
        } else if (map_success == CTF_SUCCESS){
          tsr->itopo = i;
          set_padding(tsr);
          memuse = (uint64_t)tsr->size;

          if ((uint64_t)memuse >= proc_bytes_available()){
            DPRINTF(1,"Not enough memory to map tensor on topo %d\n", i);
            continue;
          }

          nvirt = (uint64_t)calc_nvirt(tsr);
          ASSERT(nvirt != 0);
          if (btopo == -1 || nvirt < bnvirt){
            bnvirt = nvirt;
            btopo = i;
            bmemuse = memuse;
          } else if (nvirt == bnvirt && memuse < bmemuse){
            btopo = i;
            bmemuse = memuse;
          }
        }
      }
      if (btopo == -1)
        bnvirt = UINT64_MAX;
      /* pick lower dimensional mappings, if equivalent */
      ///btopo = get_best_topo(bnvirt, btopo, global_comm, 0, bmemuse);
      btopo = get_best_topo(bmemuse, btopo, global_comm);

      if (btopo == -1 || btopo == INT_MAX) {
        if (global_comm.rank==0)
          printf("ERROR: FAILED TO MAP TENSOR\n");
        return CTF_ERROR;
      }

      memset(restricted, 0, tsr->order*sizeof(int));
      clear_mapping(tsr);
      set_padding(tsr);
      map_success = map_tensor(topovec[btopo].order, tsr->order,
                               tsr->edge_len, tsr->sym_table, restricted,
                               topovec[btopo].dim_comm, NULL, 0,
                               tsr->edge_map);
      ASSERT(map_success == CTF_SUCCESS);

      tsr->itopo = btopo;

      CTF_free(restricted);

      tsr->is_mapped = 1;
      set_padding(tsr);

#if 0
      CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&phys_phase);
      CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&sub_edge_len);

      /* Pad the tensor */
      need_pad = 1;
      nvirt = 1;
      for (i=0; i<tsr->order; i++){
        if (tsr->edge_map[i].type == PHYSICAL_MAP){
          phys_phase[i] = tsr->edge_map[i].np;
          if (tsr->edge_map[i].has_child){
            phys_phase[i] = phys_phase[i]*tsr->edge_map[i].child->np;
            nvirt *= tsr->edge_map[i].child->np;
          } 
        } else {
          ASSERT(tsr->edge_map[i].type == VIRTUAL_MAP);
          phys_phase[i] = tsr->edge_map[i].np;
          nvirt *= tsr->edge_map[i].np;
        }
        if (tsr->edge_len[i] % phys_phase[i] != 0){
          need_pad = 1;
        }
      }

      old_size = packed_size(tsr->order, tsr->edge_len, 
                             tsr->sym, tsr->sym_type);

      if (need_pad){    
        tsr->is_padded = 1;
        CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&tsr->padding);
        for (i=0; i<tsr->order; i++){
          tsr->padding[i] = tsr->edge_len[i] % phys_phase[i];
          if (tsr->padding[i] != 0)
            tsr->padding[i] = phys_phase[i] - tsr->padding[i];
          tsr->edge_len[i] += tsr->padding[i];
        }
      }
      for (i=0; i<tsr->order; i++){
        sub_edge_len[i] = tsr->edge_len[i]/phys_phase[i];
      }
      /* Alloc and set the data to zero */
      DEBUG_PRINTF("tsr->size = nvirt = " PRIu64 " * packed_size = " PRIu64 "\n",
                    (unsigned int64_t int)nvirt, packed_size(tsr->order, sub_edge_len,
                                       tsr->sym, tsr->sym_type));
      if (global_comm.rank == 0){
        printf("Tensor %d initially mapped with virtualization factor of " PRIu64 "\n",tensor_id,nvirt);
      }
      tsr->size =nvirt*packed_size(tsr->order, sub_edge_len, 
                                   tsr->sym, tsr->sym_type);
      if (global_comm.rank == 0){
        printf("Tensor %d is of size " PRId64 ", has factor of %lf growth due to padding\n", 
              tensor_id, tsr->size,
              global_comm.np*(tsr->size/(double)old_size));

      }
#endif
     
#ifdef HOME_CONTRACT 
      if (tsr->order > 0){
        tsr->home_size = tsr->size; //MAX(1024+tsr->size, 1.20*tsr->size);
        tsr->is_home = 1;
        tsr->has_home = 1;
        //tsr->is_home = 0;
        //tsr->has_home = 0;
        if (global_comm.rank == 0)
          DPRINTF(3,"Initial size of tensor %d is " PRId64 ",",tensor_id,tsr->size);
        CTF_alloc_ptr(tsr->home_size*sizeof(dtype), (void**)&tsr->home_buffer);
        tsr->data = tsr->home_buffer;
      } else {
        CTF_alloc_ptr(tsr->size*sizeof(dtype), (void**)&tsr->data);
      }
#else
      CTF_mst_alloc_ptr(tsr->size*sizeof(dtype), (void**)&tsr->data);
#endif
#if DEBUG >= 2
      if (global_comm.rank == 0)
        printf("Tensor %d set to zero with mapping:\n", tensor_id);
      print_map(stdout, tensor_id);
#endif
      std::fill(tsr->data, tsr->data + tsr->size, get_zero<dtype>());
/*      CTF_free(phys_phase);
      CTF_free(sub_edge_len);*/
    }
  }
  return CTF_SUCCESS;
}
/*
 * \brief print tensor tid to stream
 * WARNING: serializes ALL data to ONE processor
 * \param stream output stream (stdout, stdin, FILE)
 * \param tid tensor handle
 */
template<typename dtype>
int dist_tensor<dtype>::print_tsr(FILE * stream, int const tid, double cutoff) {
  tensor<dtype> const * tsr;
  int i, j;
  int64_t my_sz, tot_sz =0;
  int * recvcnts, * displs, * adj_edge_len, * idx_arr;
  tkv_pair<dtype> * my_data;
  tkv_pair<dtype> * all_data;
  key k;

  print_map(stdout, tid, 1);

  tsr = tensors[tid];

  my_sz = 0;
  read_local_pairs(tid, &my_sz, &my_data);

  if (global_comm.rank == 0){
    CTF_alloc_ptr(global_comm.np*sizeof(int), (void**)&recvcnts);
    CTF_alloc_ptr(global_comm.np*sizeof(int), (void**)&displs);
    CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&adj_edge_len);
    CTF_alloc_ptr(tsr->order*sizeof(int), (void**)&idx_arr);

    for (i=0; i<tsr->order; i++){
       adj_edge_len[i] = tsr->edge_len[i] - tsr->padding[i];
    }
  }

  GATHER(&my_sz, 1, COMM_INT_T, recvcnts, 1, COMM_INT_T, 0, global_comm);

  if (global_comm.rank == 0){
    for (i=0; i<global_comm.np; i++){
      recvcnts[i] *= sizeof(tkv_pair<dtype>);
    }
    displs[0] = 0;
    for (i=1; i<global_comm.np; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
    }
    tot_sz = (displs[global_comm.np-1] 
                    + recvcnts[global_comm.np-1])/sizeof(tkv_pair<dtype>);
    CTF_alloc_ptr(tot_sz*sizeof(tkv_pair<dtype>), (void**)&all_data);
  }

  if (my_sz == 0) my_data = NULL;
  GATHERV(my_data, my_sz*sizeof(tkv_pair<dtype>), COMM_CHAR_T, 
          all_data, recvcnts, displs, COMM_CHAR_T, 0, global_comm);

  if (global_comm.rank == 0){
    std::sort(all_data, all_data + tot_sz);
    for (i=0; i<tot_sz; i++){
      if (std::abs(all_data[i].d) > cutoff)
      {
          k = all_data[i].k;
          for (j=0; j<tsr->order; j++){
              //idx_arr[tsr->order-j-1] = k%adj_edge_len[j];
              idx_arr[j] = k%adj_edge_len[j];
            k = k/adj_edge_len[j];
          }
          for (j=0; j<tsr->order; j++){
                  fprintf(stream,"[%d]",idx_arr[j]);
          }
          fprintf(stream," <%20.14E>\n",GET_REAL(all_data[i].d));
      }
    }
    CTF_free(recvcnts);
    CTF_free(displs);
    CTF_free(adj_edge_len);
    CTF_free(idx_arr);
    CTF_free(all_data);
  }
  //COMM_BARRIER(global_comm);
  return CTF_SUCCESS;
}
/*
 * \brief print tensors tid_A and tid_A side-by-side to stream
 * WARNING: serializes ALL data to ONE processor
 * \param stream output stream (stdout, stdin, FILE)
 * \param tid_A first tensor handle
 * \param tid_B second tensor handle
 */
template<typename dtype>
int dist_tensor<dtype>::compare_tsr(FILE * stream, int const tid_A, int const tid_B, double cutoff) {
  tensor<dtype> const * tsr_A;
  int i, j;
  int64_t my_sz, tot_sz =0, my_sz_B;
  int * recvcnts, * displs, * adj_edge_len, * idx_arr;
  tkv_pair<dtype> * my_data_A;
  tkv_pair<dtype> * my_data_B;
  tkv_pair<dtype> * all_data_A;
  tkv_pair<dtype> * all_data_B;
  key k;

  print_map(stdout, tid_A, 1);
  print_map(stdout, tid_B, 1);

  tsr_A = tensors[tid_A];

  my_sz = 0;
  read_local_pairs(tid_A, &my_sz, &my_data_A);
  my_sz_B = 0;
  read_local_pairs(tid_B, &my_sz_B, &my_data_B);
  assert(my_sz == my_sz_B);

  if (global_comm.rank == 0){
    CTF_alloc_ptr(global_comm.np*sizeof(int), (void**)&recvcnts);
    CTF_alloc_ptr(global_comm.np*sizeof(int), (void**)&displs);
    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)&adj_edge_len);
    CTF_alloc_ptr(tsr_A->order*sizeof(int), (void**)&idx_arr);

    for (i=0; i<tsr_A->order; i++){
            adj_edge_len[i] = tsr_A->edge_len[i] - tsr_A->padding[i];
    }
  }

  GATHER(&my_sz, 1, COMM_INT_T, recvcnts, 1, COMM_INT_T, 0, global_comm);

  if (global_comm.rank == 0){
    for (i=0; i<global_comm.np; i++){
      recvcnts[i] *= sizeof(tkv_pair<dtype>);
    }
    displs[0] = 0;
    for (i=1; i<global_comm.np; i++){
      displs[i] = displs[i-1] + recvcnts[i-1];
    }
    tot_sz = (displs[global_comm.np-1]
                    + recvcnts[global_comm.np-1])/sizeof(tkv_pair<dtype>);
    CTF_alloc_ptr(tot_sz*sizeof(tkv_pair<dtype>), (void**)&all_data_A);
    CTF_alloc_ptr(tot_sz*sizeof(tkv_pair<dtype>), (void**)&all_data_B);
  }

  if (my_sz == 0) my_data_A = my_data_B = NULL;
  GATHERV(my_data_A, my_sz*sizeof(tkv_pair<dtype>), COMM_CHAR_T,
          all_data_A, recvcnts, displs, COMM_CHAR_T, 0, global_comm);
  GATHERV(my_data_B, my_sz*sizeof(tkv_pair<dtype>), COMM_CHAR_T,
          all_data_B, recvcnts, displs, COMM_CHAR_T, 0, global_comm);

  if (global_comm.rank == 0){
      std::sort(all_data_A, all_data_A + tot_sz);
      std::sort(all_data_B, all_data_B + tot_sz);
    for (i=0; i<tot_sz; i++){
      if (std::abs(all_data_A[i].d) > cutoff ||
          std::abs(all_data_B[i].d) > cutoff)
      {
          k = all_data_A[i].k;
          for (j=0; j<tsr_A->order; j++){
              //idx_arr[tsr_A->order-j-1] = k%adj_edge_len[j];
              idx_arr[j] = k%adj_edge_len[j];
            k = k/adj_edge_len[j];
          }
          for (j=0; j<tsr_A->order; j++){
                  fprintf(stream,"[%d]",idx_arr[j]);
          }
          fprintf(stream," <%20.14E> <%20.14E>\n",GET_REAL(all_data_A[i].d),GET_REAL(all_data_B[i].d));
      }
    }
    CTF_free(recvcnts);
    CTF_free(displs);
    CTF_free(adj_edge_len);
    CTF_free(idx_arr);
    CTF_free(all_data_A);
    CTF_free(all_data_B);
  }
  //COMM_BARRIER(global_comm);
  return CTF_SUCCESS;
}

/*
 * \brief print mapping of tensor tid to stream
 * \param stream output stream (stdout, stdin, FILE)
 * \param tid tensor handle
 */
template<typename dtype>
int dist_tensor<dtype>::print_map(FILE *    stream,
                                  int const tid,
                                  int const all) const{
  tensor<dtype> const * tsr;
  tsr = tensors[tid];

#ifndef DEBUG
  if (!all || global_comm.rank == 0){
    tsr->print_map(stdout);
  }
#else
  mapping * map;
  int i;
  if (all)
    COMM_BARRIER(global_comm);
  fflush(stdout);
  if (/*tsr->is_mapped &&*/ (!all || global_comm.rank == 0)){
    printf("\nTensor %d of dimension %d is mapped to a ", tid, tsr->order);
    for (i=0; i<topovec[tsr->itopo].order-1; i++){
            printf("%d-by-", topovec[tsr->itopo].dim_comm[i].np);
    }
    if (topovec[tsr->itopo].order > 0)
            printf("%d topology (#%d).\n", topovec[tsr->itopo].dim_comm[i].np, tsr->itopo);
    for (i=0; i<tsr->order; i++){
      switch (tsr->edge_map[i].type){
        case NOT_MAPPED:
          printf("Dimension %d of length %d and symmetry %d is not mapped\n",i,tsr->edge_len[i],tsr->sym[i]);
          break;

        case PHYSICAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped to physical dimension %d with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].cdt,tsr->edge_map[i].np);
          map = &tsr->edge_map[i];
          while (map->has_child){
            map = map->child;
            if (map->type == VIRTUAL_MAP)
              printf("\tDimension %d also has a virtualized child of phase %d\n", i, map->np);
            else
              printf("\tDimension %d also has a physical child mapped to physical dimension %d with phase %d\n",
                      i, map->cdt, map->np);
          }
          break;

        case VIRTUAL_MAP:
          printf("Dimension %d of length %d and symmetry %d is mapped virtually with phase %d\n",
            i,tsr->edge_len[i],tsr->sym[i],tsr->edge_map[i].np);
          break;
      }
    }
  }
  fflush(stdout);
  if (all)
    COMM_BARRIER(global_comm);
#endif

  return CTF_SUCCESS;

}

/**
 * \brief sets the padded area of a tensor to zero, needed after contractions
 * \param[in] tensor_id identifies tensor
 */
template<typename dtype>
int dist_tensor<dtype>::zero_out_padding(int const tensor_id){
  int i, num_virt, idx_lyr;
  int64_t np;
  int * virt_phase, * virt_phys_rank, * phys_phase;
  tensor<dtype> * tsr;
  mapping * map;

  TAU_FSTART(zero_out_padding);

  tsr = tensors[tensor_id];
  if (tsr->has_zero_edge_len){
    return CTF_SUCCESS;
  }
  unmap_inner(tsr);
  set_padding(tsr);

  if (!tsr->is_mapped){
    return CTF_SUCCESS;
  } else {
    np = tsr->size;

    CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phase);
    CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&phys_phase);
    CTF_alloc_ptr(sizeof(int)*tsr->order, (void**)&virt_phys_rank);


    num_virt = 1;
    idx_lyr = global_comm.rank;
    for (i=0; i<tsr->order; i++){
      /* Calcute rank and phase arrays */
      map               = tsr->edge_map + i;
      phys_phase[i]     = calc_phase(map);
      virt_phase[i]     = phys_phase[i]/calc_phys_phase(map);
      virt_phys_rank[i] = calc_phys_rank(map, &topovec[tsr->itopo])
                                              *virt_phase[i];
      num_virt          = num_virt*virt_phase[i];

      if (map->type == PHYSICAL_MAP)
        idx_lyr -= topovec[tsr->itopo].lda[map->cdt]
                                *virt_phys_rank[i]/virt_phase[i];
    }
    if (idx_lyr == 0){
      zero_padding(tsr->order, np, num_virt,
                   tsr->edge_len, tsr->sym, tsr->padding,
                   phys_phase, virt_phase, virt_phys_rank, tsr->data); 
    } else {
      std::fill(tsr->data, tsr->data+np, 0.0);
    }
    CTF_free(virt_phase);
    CTF_free(phys_phase);
    CTF_free(virt_phys_rank);
  }
  TAU_FSTOP(zero_out_padding);

  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::print_ctr(CTF_ctr_type_t const * ctype,
                                  dtype const alpha,
                                  dtype const beta) const {
  int dim_A, dim_B, dim_C;
  int * sym_A, * sym_B, * sym_C;
  int i,j,max,ex_A, ex_B,ex_C;
  COMM_BARRIER(global_comm);
  if (global_comm.rank == 0){
    printf("Contracting Tensor %d with %d into %d\n", ctype->tid_A, ctype->tid_B,
                                                                                         ctype->tid_C);
    printf("alpha = %lf, beta = %lf\n", GET_REAL(alpha), GET_REAL(beta));
    dim_A = get_dim(ctype->tid_A);
    dim_B = get_dim(ctype->tid_B);
    dim_C = get_dim(ctype->tid_C);
    max = dim_A+dim_B+dim_C;
    sym_A = get_sym(ctype->tid_A);
    sym_B = get_sym(ctype->tid_B);
    sym_C = get_sym(ctype->tid_C);

    printf("Contraction index table:\n");
    printf("     A     B     C\n");
    for (i=0; i<max; i++){
      ex_A=0;
      ex_B=0;
      ex_C=0;
      printf("%d:   ",i);
      for (j=0; j<dim_A; j++){
        if (ctype->idx_map_A[j] == i){
          ex_A++;
          if (sym_A[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_A == 0)
        printf("      ");
      if (ex_A == 1)
        printf("   ");
      for (j=0; j<dim_B; j++){
        if (ctype->idx_map_B[j] == i){
          ex_B=1;
          if (sym_B[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_B == 0)
        printf("      ");
      if (ex_B == 1)
        printf("   ");
      for (j=0; j<dim_C; j++){
        if (ctype->idx_map_C[j] == i){
          ex_C=1;
          if (sym_C[j] != NS)
            printf("%d' ",j);
          else
            printf("%d ",j);
        }
      }
      printf("\n");
      if (ex_A + ex_B + ex_C == 0) break;
    }
    CTF_free(sym_A);
    CTF_free(sym_B);
    CTF_free(sym_C);
  }
  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::print_sum(CTF_sum_type_t const * stype, 
                                  dtype const            alpha, 
                                  dtype const            beta) const {
  int dim_A, dim_B;
  int i,j,max,ex_A,ex_B;
  int * sym_A, * sym_B;
  COMM_BARRIER(global_comm);
  if (global_comm.rank == 0){
    printf("Summing Tensor %lf*%d with %lf*%d into %d\n",
            GET_REAL(alpha), stype->tid_A, 
            GET_REAL(beta), stype->tid_B, stype->tid_B);
    dim_A = get_dim(stype->tid_A);
    dim_B = get_dim(stype->tid_B);
    max = dim_A+dim_B; //MAX(MAX((dim_A), (dim_B)), (dim_C));
    sym_A = get_sym(stype->tid_A);
    sym_B = get_sym(stype->tid_B);

    printf("Sum index table:\n");
    printf("     A     B \n");
    for (i=0; i<max; i++){
      ex_A=0;
      ex_B=0;
      printf("%d:   ",i);
      for (j=0; j<dim_A; j++){
        if (stype->idx_map_A[j] == i){
          ex_A++;
          if (sym_A[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      if (ex_A == 0)
        printf("      ");
      if (ex_A == 1)
        printf("   ");
      for (j=0; j<dim_B; j++){
        if (stype->idx_map_B[j] == i){
          ex_B=1;
          if (sym_B[j] != NS)
            printf("%d' ",j);
          else
            printf("%d  ",j);
        }
      }
      printf("\n");
      if (ex_A + ex_B == 0) break;
    }
    CTF_free(sym_A);
    CTF_free(sym_B);
  }
  COMM_BARRIER(global_comm);
  return CTF_SUCCESS;
}

template<typename dtype>
int dist_tensor<dtype>::check_contraction(CTF_ctr_type_t const * type){
}

/**
 * \brief checks the edge lengths specfied for this sum
 * \param type contains tensor ids and index maps
 */
template<typename dtype>
int dist_tensor<dtype>::check_sum(CTF_sum_type_t const *     type){
  return check_sum(type->tid_A, type->tid_B, type->idx_map_A, type->idx_map_B);
}

/**
 * \brief checks the edge lengths specfied for this sum
 * \param tid_A id of tensor A
 * \param tid_B id of tensor B
 * \param idx_map_A indices of tensor A
 * \param idx_map_B indices of tensor B
 */
template<typename dtype>
int dist_tensor<dtype>::check_sum(int const   tid_A, 
                                  int const   tid_B, 
                                  int const * idx_map_A, 
                                  int const * idx_map_B){
  int i, num_tot, len;
  int iA, iB;
  int order_A, order_B;
  int * len_A, * len_B;
  int * sym_A, * sym_B;
  int * idx_arr;
  
  tensor<dtype> * tsr_A, * tsr_B;

  tsr_A = tensors[tid_A];
  tsr_B = tensors[tid_B];
    

  get_tsr_info(tid_A, &order_A, &len_A, &sym_A);
  get_tsr_info(tid_B, &order_B, &len_B, &sym_B);
  
  inv_idx(tsr_A->order, idx_map_A, tsr_A->edge_map,
          tsr_B->order, idx_map_B, tsr_B->edge_map,
          &num_tot, &idx_arr);

  for (i=0; i<num_tot; i++){
    len = -1;
    iA = idx_arr[2*i+0];
    iB = idx_arr[2*i+1];
    if (iA != -1){
      len = len_A[iA];
    }
    if (len != -1 && iB != -1 && len != len_B[iB]){
      if (global_comm.rank == 0){
        printf("i = %d Error in sum call: The %dth edge length (%d) of tensor %d does not",
                i, iA, len, tid_A);
        printf("match the %dth edge length (%d) of tensor %d.\n",
                iB, len_B[iB], tid_B);
      }
      ABORT;
    }
  }
  CTF_free(sym_A);
  CTF_free(sym_B);
  CTF_free(len_A);
  CTF_free(len_B);
  CTF_free(idx_arr);
  return CTF_SUCCESS;
}

template<typename dtype>
void dist_tensor<dtype>::contract_mst(){

}



#include "tensor_object.cxx"
#include "dist_tensor_map.cxx"
#include "dist_tensor_op.cxx"
#include "dist_tensor_inner.cxx"
#include "dist_tensor_fold.cxx"
#include "dist_tensor_model.cxx"


/* Instantiate the ugly templates */
template class dist_tensor<double>;
#if (VERIFY==0)
template class dist_tensor< std::complex<double> >;
#endif

