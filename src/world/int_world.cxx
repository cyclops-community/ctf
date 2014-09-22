/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include <stdint.h>
#include <limits.h>
#include "int_topology.h"
#include "int_world.h"

#ifdef HPM
extern "C" void HPM_Start(char *);  
extern "C" void HPM_Stop(char *);
#endif

using namespace CTF;

namespace CTF_int {

  /** 
   * \brief destructor
   */
  world::~world(){
    exit();
  }

  /** 
   * \brief constructor
   */
  world::world(){
    initialized = 0;
  }

  MPI_Comm world::get_MPI_Comm(){
    return global_comm.cm;
  }
      
  /* return MPI processor rank */
  int world::get_rank(){
    return global_comm.rank;
  }
  /* return number of MPI processes in the defined global context */
  int world::get_num_pes(){
    return global_comm.np;
  }

  /**
   * \brief  initializes library. 
   *      Sets topology to be a mesh of dimension order with
   *      edge lengths dim_len. 
   *
   * \param[in] global_context communicator decated to this library instance
   * \param[in] rank this pe rank within the global context
   * \param[in] np number of processors
   * \param[in] mach the type of machine we are running on
   * \param[in] argc number of arguments passed to main
   * \param[in] argv arguments passed to main
   */
  int world::init(MPI_Comm const  global_context,
                  int             rank, 
                  int             np,
                  TOPOLOGY        mach,
                  int             argc,
                  const char * const *  argv){
    SET_COMM(global_context, rank, np, global_comm);
    phys_topology = get_phys_topo(global_comm, mach);
    topovec = peel_torus(phys_topology, global_comm);
    
    return initialize(argc, argv);
  }

  /**
   * \brief  initializes library. 
   *      Sets topology to be a mesh of dimension order with
   *      edge lengths dim_len. 
   *
   * \param[in] global_context communicator decated to this library instance
   * \param[in] rank this pe rank within the global context
   * \param[in] np number of processors
   * \param[in] order is the number of dimensions in the topology
   * \param[in] dim_len is the number of processors along each dimension
   * \param[in] argc number of arguments passed to main
   * \param[in] argv arguments passed to main
   */
  int world::init(MPI_Comm const  global_context,
                        int             rank, 
                        int             np, 
                        int             order, 
                        int const *     dim_len,
                        int             argc,
                        const char * const *  argv){
    SET_COMM(global_context, rank, np, global_comm);
    phys_topology = topology(order, dim_len, global_comm, 1);
    topovec = peel_torus(phys_topology, global_comm);

    return initialize(argc, argv);
  }

  /**
   * \brief initializes world stack and parameters
   * \param[in] argc number of arguments passed to main
   * \param[in] argv arguments passed to main
   */
  int world::initialize(int                   argc,
                        const char * const *  argv){
    char * mst_size, * stack_size, * mem_size, * ppn;
    int rank = global_comm.rank;  

    CTF_mem_create();
    if (CTF_get_num_instances() == 1){
      TAU_FSTART(CTF);
  #ifdef HPM
      HPM_Start("CTF");
  #endif
  #ifdef OFFLOAD
      offload_init();
  #endif
      CTF_set_context(global_comm.cm);
      CTF_set_main_args(argc, argv);

  #ifdef USE_OMP
      char * ntd = getenv("OMP_NUM_THREADS");
      if (ntd == NULL){
        omp_set_num_threads(1);
        if (rank == 0){
          VPRINTF(1,"Running with 1 thread using omp_set_num_threads(1), because OMP_NUM_THREADS is not defined\n");
        }
      } else {
        if (rank == 0 && ntd != NULL){
          VPRINTF(1,"Running with %d threads\n",omp_get_num_threads());
        }
      }
  #endif
    
      mst_size = getenv("CTF_MST_SIZE");
      stack_size = getenv("CTF_STACK_SIZE");
      if (mst_size == NULL && stack_size == NULL){
  #ifdef USE_MST
        if (rank == 0)
          VPRINTF(1,"Creating stack of size " PRId64 "\n",1000*(int64_t)1E6);
        CTF_mst_create(1000*(int64_t)1E6);
  #else
        if (rank == 0){
          VPRINTF(1,"Running without stack, define CTF_STACK_SIZE environment variable to activate stack\n");
        }
  #endif
      } else {
        uint64_t imst_size = 0 ;
        if (mst_size != NULL) 
          imst_size = strtoull(mst_size,NULL,0);
        if (stack_size != NULL)
          imst_size = MAX(imst_size,strtoull(stack_size,NULL,0));
        if (rank == 0)
          printf("Creating stack of size " PRIu64 " due to CTF_STACK_SIZE enviroment variable\n",
                    imst_size);
        CTF_mst_create(imst_size);
      }
      mem_size = getenv("CTF_MEMORY_SIZE");
      if (mem_size != NULL){
        uint64_t imem_size = strtoull(mem_size,NULL,0);
        if (rank == 0)
          VPRINTF(1,"Memory size set to " PRIu64 " by CTF_MEMORY_SIZE environment variable\n",
                    imem_size);
        CTF_set_mem_size(imem_size);
      }
      ppn = getenv("CTF_PPN");
      if (ppn != NULL){
        if (rank == 0)
          VPRINTF(1,"Assuming %d processes per node due to CTF_PPN environment variable\n",
                    atoi(ppn));
        ASSERT(atoi(ppn)>=1);
  #ifdef BGQ
        CTF_set_memcap(.75);
  #else
        CTF_set_memcap(.75/atof(ppn));
  #endif
      }
      if (rank == 0)
        VPRINTF(1,"Total amount of memory available to process 0 is " PRIu64 "\n", proc_bytes_available());
    } 
    initialized = 1;
    return SUCCESS;
  }


  /**
   * \brief  defines a tensor and retrieves handle
   *
   * \param[in] sr semiring defining type of tensor
   * \param[in] order number of tensor dimensions
   * \param[in] edge_len global edge lengths of tensor
   * \param[in] sym symmetry relations of tensor
   * \param[out] tensor_id the tensor index (handle)
   * \param[in] name string name for tensor (optionary)
   * \param[in] profile wether to make profile calls for the tensor
   */
  int world::define_tensor(semiring         sr,
                           int              order,             
                           int const *      edge_len, 
                           int const *      sym,
                           int *            tensor_id,
                           bool             alloc_data,
                           char const *     name,
                           bool             profile){
    int i;

    tensor * tsr = new tensor(sr, order, edge_len, sym, alloc_data, name, profile);
    (*tensor_id) = tensors.size();
    
    /* initialize map array and symmetry table */
  #if DEBUG >= 2
    if (global_comm.rank == 0)
      printf("Tensor %d of dimension %d defined with edge lengths", *tensor_id, order);
  #endif
    for (i=0; i<order; i++){
  #if DEBUG >= 1
      int maxlen;
      ALLREDUCE(tsr->sym+i,&maxlen,1,MPI_INT,MPI_MAX,global_comm);
      ASSERT(maxlen==sym[i]);
      ALLREDUCE(tsr->edge_len+i,&maxlen,1,MPI_INT,MPI_MAX,global_comm);
      ASSERT(maxlen==edge_len[i]);
  #endif
  #if DEBUG >= 2
      if (global_comm.rank == 0)
        printf(" %d", edge_len[i]);
  #endif
    }
  #if DEBUG >= 2
    if (global_comm.rank == 0)
      printf("\n");
  #endif

    tensors.push_back(tsr);

    return SUCCESS;
  }
      
  /* \brief clone a tensor object
   * \param[in] tensor_id id of old tensor
   * \param[in] copy_data if 0 then leave tensor blank, if 1 copy data from old
   * \param[out] new_tensor_id id of new tensor
   */
  int world::clone_tensor(int       tensor_id,
                          bool      copy_data,
                          int *     new_tensor_id,
                          bool      alloc_data){
    if (copy_data){
      tensor * tsr = new tensor(tensors[tensor_id]);
      (*new_tensor_id) = tensors.size();
      tensors.push_back(tsr);
    } else {
      tensor * other = tensors[tensor_id];
      tensor * tsr = new tensor(other->sr, other->order, other->edge_len, other->sym, alloc_data, other->name, other->profile);
      (*new_tensor_id) = tensors.size();
      tensors.push_back(tsr);
    }
    return SUCCESS;
  }
     

  int world::get_name(int       tensor_id, char const ** name){
    return dt->get_name(tensor_id, name);
  }
   
  int world::set_name(int       tensor_id, char const * name){
    return dt->set_name(tensor_id, name);
  }

  int world::profile_on(int       tensor_id){
    return dt->profile_on(tensor_id);
  }

  int world::profile_off(int       tensor_id){
    return dt->profile_off(tensor_id);
  }
      
  /* \brief get dimension of a tensor 
   * \param[in] tensor_id id of tensor
   * \param[out] order dimension of tensor
   */
  int world::get_dimension(int       tensor_id, int *order) const{
    *order = dt->get_dim(tensor_id);
    return SUCCESS;
  }
      
  /* \brief get lengths of a tensor 
   * \param[in] tensor_id id of tensor
   * \param[out] edge_len edge lengths of tensor
   */
  int world::get_lengths(int       tensor_id, int **edge_len) const{
    int order, * sym;
    dt->get_tsr_info(tensor_id, &order, edge_len, &sym);
    CTF_untag_mem(edge_len);
    CTF_free(sym);
    return SUCCESS;
  }
      
  /* \brief get symmetry of a tensor 
   * \param[in] tensor_id id of tensor
   * \param[out] sym symmetries of tensor
   */
  int world::get_symmetry(int       tensor_id, int **sym) const{
    *sym = dt->get_sym(tensor_id);
    CTF_untag_mem(*sym);
    return SUCCESS;
  }
      
  /* \brief get raw data pointer WARNING: includes padding 
   * \param[in] tensor_id id of tensor
   * \param[out] data raw local data
   */
  int world::get_raw_data(int       tensor_id, dtype ** data, int64_t * size) {
    *data = dt->get_raw_data(tensor_id, size);
    return SUCCESS;
  }

  /**
   * \brief get information about tensor
   * \param[in] tensor_id id of tensor
   * \param[out] order dimension of tensor
   * \param[out] edge_len edge lengths of tensor
   * \param[out] sym symmetries of tensor
   */
  int world::info_tensor(int        tensor_id,
                               int *      order,
                               int **     edge_len,
                               int **     sym) const{
    dt->get_tsr_info(tensor_id, order, edge_len, sym);
    CTF_untag_mem(*sym);
    CTF_untag_mem(*edge_len);
    return SUCCESS;
  }

  /**
   * \brief  Input tensor data with <key, value> pairs where key is the
   *              global index for the value. 
   * \param[in] tensor_id tensor handle
   * \param[in] num_pair number of pairs to write
   * \param[in] mapped_data pairs to write
   */
  int world::write_tensor(int                     tensor_id, 
                                int64_t const           num_pair,  
                                tkv_pair<dtype> const * mapped_data){
    return dt->write_pairs(tensor_id, num_pair, 1.0, 0.0, const_cast<tkv_pair<dtype>*>(mapped_data), 'w');
  }

  /** 
   * \brief  Add tensor data new=alpha*new+beta*old
   *         with <key, value> pairs where key is the 
   *         global index for the value. 
   * \param[in] tensor_id tensor handle
   * \param[in] num_pair number of pairs to write
   * \param[in] alpha scaling factor of written value
   * \param[in] beta scaling factor of old value
   * \param[in] mapped_data pairs to write
   */
  int world::write_tensor(int                     tensor_id, 
                                int64_t const          num_pair,  
                                dtype const              alpha,
                                dtype const              beta,
                                tkv_pair<dtype> const * mapped_data){
    return dt->write_pairs(tensor_id, num_pair, alpha, beta, const_cast<tkv_pair<dtype>*>(mapped_data), 'w');
  }

  /**
   * \brief read tensor data with <key, value> pairs where key is the
   *              global index for the value, which gets filled in. 
   * \param[in] tensor_id tensor handle
   * \param[in] num_pair number of pairs to read
   * \param[in] alpha scaling factor of read value
   * \param[in] beta scaling factor of old value
   * \param[in] mapped_data pairs to write
   */
  int world::read_tensor(int                      tensor_id, 
                               int64_t const           num_pair, 
                               dtype const               alpha, 
                               dtype const               beta, 
                               tkv_pair<dtype> * const  mapped_data){
    return dt->write_pairs(tensor_id, num_pair, alpha, beta, mapped_data, 'r');
  }


  /**
   * \brief read tensor data with <key, value> pairs where key is the
   *              global index for the value, which gets filled in. 
   * \param[in] tensor_id tensor handle
   * \param[in] num_pair number of pairs to read
   * \param[in,out] mapped_data pairs to read
   */
  int world::read_tensor(int                      tensor_id, 
                               int64_t const           num_pair, 
                               tkv_pair<dtype> * const  mapped_data){
    return read_tensor(tensor_id, num_pair, 1.0, 0.0, mapped_data);
  }

  int world::permute_tensor( int                    tid_A,
                                   int * const *          permutation_A,
                                   dtype const            alpha,
                                   world *          tC_A,
                                   int                    tid_B,
                                   int * const *          permutation_B,
                                   dtype const            beta,
                                   world *          tC_B){
    return dt->permute_tensor(tid_A, permutation_A, alpha, tC_A->dt, tid_B, permutation_B, beta, tC_B->dt);
  }
  int world::add_to_subworld(int          tid,
                                   int          tid_sub,
                                   world *tC_sub,
                                   dtype       alpha,
                                   dtype       beta){
    if (tC_sub == NULL)
      return dt->add_to_subworld(tid, tid_sub, NULL, alpha, beta);
    else
      return dt->add_to_subworld(tid, tid_sub, tC_sub->dt, alpha, beta);
  }
      
  int world::add_from_subworld(int          tid,
                                     int          tid_sub,
                                     world *tC_sub,
                                     dtype       alpha,
                                     dtype       beta){
    if (tC_sub == NULL)
      return dt->add_from_subworld(tid, tid_sub, NULL, alpha, beta);
    else
      return dt->add_from_subworld(tid, tid_sub, tC_sub->dt, alpha, beta);

  }

  int world::slice_tensor( int          tid_A,
                                 int const *  offsets_A,
                                 int const *  ends_A,
                                 dtype const  alpha,
                                 int          tid_B,
                                 int const *  offsets_B,
                                 int const *  ends_B,
                                 dtype const  beta){
    return dt->slice_tensor(tid_A, offsets_A, ends_A, alpha, dt,
                            tid_B, offsets_B, ends_B, beta, dt);
  }

  int world::slice_tensor( int            tid_A,
                                 int const *    offsets_A,
                                 int const *    ends_A,
                                 dtype const    alpha,
                                 world *  dt_other_A,
                                 int            tid_B,
                                 int const *    offsets_B,
                                 int const *    ends_B,
                                 dtype const    beta){
    return dt->slice_tensor(tid_A, offsets_A, ends_A, alpha, dt_other_A->dt,
                            tid_B, offsets_B, ends_B, beta, dt);
  }

  int world::slice_tensor( int            tid_A,
                                 int const *    offsets_A,
                                 int const *    ends_A,
                                 dtype const    alpha,
                                 int            tid_B,
                                 int const *    offsets_B,
                                 int const *    ends_B,
                                 dtype const    beta,
                                 world *  dt_other_B){
    return dt->slice_tensor(tid_A, offsets_A, ends_A, alpha, dt,
                            tid_B, offsets_B, ends_B, beta, dt_other_B->dt);
  }


  /**
   * \brief read entire tensor with each processor (in packed layout).
   *         WARNING: will use a lot of memory. 
   * \param[in] tensor_id tensor handle
   * \param[out] num_pair number of values read
   * \param[in,out] preallocated mapped_data values read
   */
  int world::allread_tensor(int         tensor_id, 
                                  int64_t *    num_pair, 
                                  dtype *     all_data){
    int ret;
    int64_t np;
    ret = dt->allread_tsr(tensor_id, &np, &all_data, 1);
    *num_pair = np;
    return ret;
  }

  /**
   * \brief read entire tensor with each processor (in packed layout).
   *         WARNING: will use a lot of memory. 
   * \param[in] tensor_id tensor handle
   * \param[out] num_pair number of values read
   * \param[in,out] mapped_data values read
   */
  int world::allread_tensor(int         tensor_id, 
                                  int64_t *    num_pair, 
                                  dtype **    all_data){
    int ret;
    int64_t np;
    ret = dt->allread_tsr(tensor_id, &np, all_data, 0);
    CTF_untag_mem(*all_data);
    *num_pair = np;
    return ret;
  }

  /* input tensor local data or set buffer for contract answer. */
  /*int world::set_local_tensor(int         tensor_id, 
                           int            num_val, 
                           dtype *        tsr_data){
    return set_tsr_data(tensor_id, num_val, tsr_data);  
  }*/

  /**
   * \brief  map input tensor local data to zero
   * \param[in] tensor_id tensor handle
   */
  int world::set_zero_tensor(int       tensor_id){
    return dt->set_zero_tsr(tensor_id);
  }



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
  int64_t world::estimate_cost(int          tid_A,
                        int const *  idx_A,
                        int          tid_B,
                        int const *  idx_B,
                        int          tid_C,
                        int const *  idx_C){
    return dt->estimate_cost(tid_A, idx_A, tid_B, idx_B, tid_C, idx_C);
    
  }

  /**
   * \brief estimate the cost of a sum B[idx_B] = A[idx_A]
   * \param[in] A first operand tensor
   * \param[in] idx_A indices of A in contraction, e.g. "ik" -> A_{ik}
   * \param[in] B second operand tensor
   * \param[in] idx_B indices of B in contraction, e.g. "kj" -> B_{kj}
   * \return cost as a int64_t type, currently a rought estimate of flops/processor
   */
  int64_t world::estimate_cost(int          tid_A,
                        int const *  idx_A,
                        int          tid_B,
                        int const *  idx_B){
    return dt->estimate_cost(tid_A, idx_A, tid_B, idx_B);
    
  }



  /**
   * \brief read tensor data pairs local to processor. 
   * \param[in] tensor_id tensor handle
   * \param[out] num_pair number of values read
   * \param[out] mapped_data values read
   */
  int world::read_local_tensor(int                tensor_id, 
                                     int64_t *           num_pair,  
                                     tkv_pair<dtype> ** mapped_data){
    int ret;
    int64_t np;
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
  int world::contract(CTF_ctr_type_t const *  type,
                            dtype const             alpha,
                            dtype const             beta){
    fseq_tsr_ctr<dtype> fs;
    fs.func_ptr=NULL;//sym_seq_ctr_ref<dtype>;
  #ifdef OFFLOAD
    fs.is_offloadable=0;
  #endif
    return contract(type, fs, alpha, beta);
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
  int world::contract(CTF_ctr_type_t const *    type,
                            fseq_tsr_ctr<dtype> const func_ptr, 
                            dtype const               alpha,
                            dtype const               beta){
    int i, ret;
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("Start head contraction :\n");
    dt->print_ctr(type,alpha,beta);
  #endif
    fseq_elm_ctr<dtype> felm;
    felm.func_ptr = NULL;

  /*  if ((*dt->get_tensors())[type->tid_A]->profile &&
        (*dt->get_tensors())[type->tid_B]->profile &&
        (*dt->get_tensors())[type->tid_C]->profile){*/
  #ifdef VERBOSE 
    char cname[200];
    cname[0] = '\0';
    if ((*dt->get_tensors())[type->tid_C]->name != NULL)
      sprintf(cname, "%.2lf*%s",beta, (*dt->get_tensors())[type->tid_C]->name);
    else
      sprintf(cname, "%d", type->tid_C);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_C]->order; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_C[i]);
      else 
        sprintf(cname+strlen(cname),"%d",type->idx_map_C[i]);
    }
    sprintf(cname+strlen(cname),"]+=%.2lf*",alpha);
    if ((*dt->get_tensors())[type->tid_A]->name != NULL)
      sprintf(cname+strlen(cname), "%s", (*dt->get_tensors())[type->tid_A]->name);
    else
      sprintf(cname+strlen(cname), "%d", type->tid_A);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_A]->order; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_A[i]);
      else
        sprintf(cname+strlen(cname),"%d",type->idx_map_A[i]);
    }
    sprintf(cname+strlen(cname),"]*");
    if ((*dt->get_tensors())[type->tid_B]->name != NULL)
      sprintf(cname+strlen(cname), "%s", (*dt->get_tensors())[type->tid_B]->name);
    else
      sprintf(cname+strlen(cname), "%d", type->tid_B);
    sprintf(cname+strlen(cname),"[");
    for (i=0; i<(*dt->get_tensors())[type->tid_B]->order; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",type->idx_map_B[i]);
      else 
        sprintf(cname+strlen(cname),"%d",type->idx_map_B[i]);
    }
    sprintf(cname+strlen(cname),"]");

    double dtt;
    CTF_Timer tctr(cname);
    if (global_comm.rank == 0){
      dtt = MPI_Wtime();
      VPRINTF(1,"Contracting: %s\n",cname);
    }
    if ((*dt->get_tensors())[type->tid_A]->profile &&
        (*dt->get_tensors())[type->tid_B]->profile &&
        (*dt->get_tensors())[type->tid_C]->profile){
      tctr.start(); 
    }
    ret = dt->home_contract(type, func_ptr, felm, alpha, beta);
    
  #if (VERBOSE>1)
    if (global_comm.rank == 0){
      VPRINTF(1,"Ended %s in %lf seconds\n",cname,MPI_Wtime()-dtt);   
    }
  #endif
    if ((*dt->get_tensors())[type->tid_A]->profile &&
        (*dt->get_tensors())[type->tid_B]->profile &&
        (*dt->get_tensors())[type->tid_C]->profile){
      tctr.stop();
    }
  #else
    ret = dt->home_contract(type, func_ptr, felm, alpha, beta);
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("End head contraction.\n");
  #endif
  #endif

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
  int world::contract(CTF_ctr_type_t const *     type,
                            fseq_elm_ctr<dtype> const  felm,
                            dtype const                alpha,
                            dtype const                beta){
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("Start head custom contraction:\n");
    dt->print_ctr(type,alpha,beta);
  #endif
    fseq_tsr_ctr<dtype> fs;
    fs.func_ptr=NULL;//sym_seq_ctr_ref<dtype>;
  #ifdef OFFLOAD
    fs.is_offloadable=0;
  #endif
    int ret = dt->home_contract(type, fs, felm, alpha, beta);
  #if DEBUG >= 1
    if (global_comm.rank == 0)
      printf("End head custom contraction.\n");
  #endif
    return ret;

  }

  /**
   * \brief copy tensor from one handle to another
   * \param[in] tid_A tensor handle to copy from
   * \param[in] tid_B tensor handle to copy to
   */
  int world::copy_tensor(int       tid_A, int       tid_B){
    return dt->cpy_tsr(tid_A, tid_B);
  }

  /**
   * \brief scales a tensor by alpha
   * \param[in] alpha scaling factor
   * \param[in] tid tensor handle
   */
  int world::scale_tensor(dtype const alpha, int       tid){
    return dt->scale_tsr(alpha, tid);
  }
  /**
   * \brief scales a tensor by alpha iterating on idx_map
   * \param[in] alpha scaling factor
   * \param[in] tid tensor handle
   * \param[in] idx_map indexer to the tensor
   */
  int world::scale_tensor(dtype const               alpha, 
                                int                       tid, 
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
  int world::scale_tensor(dtype const               alpha, 
                                int                       tid, 
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
  int world::scale_tensor(dtype const               alpha, 
                                int                       tid, 
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
  int world::dot_tensor(int       tid_A, int       tid_B, dtype *product){
    int stat;
    /* check if the mappings of A and B are the same */
    stat = dt->check_pair_mapping(tid_A, tid_B);
    if (stat == 0){
      /* Align the mappings of A and B */
      stat = dt->map_tensor_pair(tid_A, tid_B);
      if (stat != SUCCESS)
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
  int world::reduce_tensor(int       tid, CTF_OP op, dtype * result){
    return dt->red_tsr(tid, op, result);
  }

  /**
   * \brief Calls a mapping function on each element of the tensor 
   * \param[in] tid tensor handle
   * \param[in] map_func function pointer to apply to each element
   */
  int world::map_tensor(int       tid, 
                              dtype (*map_func)(int         order, 
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
  int world::get_max_abs(int        tid,
                               int        n,
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
  int world::sum_tensors(CTF_sum_type_t const * type,
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
  int world::sum_tensors(CTF_sum_type_t const *     type,
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
  int world::sum_tensors(dtype const                alpha,
                               dtype const                beta,
                               int                        tid_A,
                               int                        tid_B,
                               int const *                idx_map_A,
                               int const *                idx_map_B,
                               fseq_tsr_sum<dtype> const  func_ptr){
    fseq_elm_sum<dtype> felm;
    felm.func_ptr = NULL;

  #ifdef VERBOSE
    char cname[200];
    cname[0] = '\0';
    if ((*dt->get_tensors())[tid_B]->name != NULL)
      sprintf(cname, "%.2lf*%s", beta, (*dt->get_tensors())[tid_B]->name);
    else
      sprintf(cname, "%d", tid_B);
    sprintf(cname+strlen(cname),"[");
    for (int i=0; i<(*dt->get_tensors())[tid_B]->order; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",idx_map_B[i]);
      else 
        sprintf(cname+strlen(cname),"%d",idx_map_B[i]);
    }
    sprintf(cname+strlen(cname),"]+=%.2lf*",alpha);
    if ((*dt->get_tensors())[tid_A]->name != NULL)
      sprintf(cname+strlen(cname), "%s", (*dt->get_tensors())[tid_A]->name);
    else
      sprintf(cname+strlen(cname), "%d", tid_A);
    sprintf(cname+strlen(cname),"[");
    for (int i=0; i<(*dt->get_tensors())[tid_A]->order; i++){
      if (i>0)
        sprintf(cname+strlen(cname)," %d",idx_map_A[i]);
      else
        sprintf(cname+strlen(cname),"%d",idx_map_A[i]);
    }
    sprintf(cname+strlen(cname),"]");
    double dtt;
    if (dt->get_global_comm().rank == 0){
      dtt = MPI_Wtime();
      VPRINTF(1,"Summing: %s\n",cname);
    }
   
    CTF_Timer tctr(cname);
    if ((*dt->get_tensors())[tid_A]->profile &&
        (*dt->get_tensors())[tid_B]->profile)
      tctr.start(); 
    return dt->home_sum_tsr(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, func_ptr, felm);
    if ((*dt->get_tensors())[tid_A]->profile &&
        (*dt->get_tensors())[tid_B]->profile){
      tctr.stop();
      VPRINTF(1,"Ended %s in %lf seconds\n",cname,MPI_Wtime()-dtt);   
    }
  #else
    return dt->home_sum_tsr(alpha, beta, tid_A, tid_B, idx_map_A, idx_map_B, func_ptr, felm);
  #endif


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
  int world::sum_tensors(dtype const                alpha,
                               dtype const                beta,
                               int                        tid_A,
                               int                        tid_B,
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
  int world::sum_tensors(dtype const  alpha,
                               int          tid_A,
                               int          tid_B){
    int stat;
    
    /* check if the mappings of A and B are the same */
    stat = dt->check_pair_mapping(tid_A, tid_B);
    if (stat == 0){
      /* Align the mappings of A and B */
      stat = dt->map_tensor_pair(tid_A, tid_B);
      if (stat != SUCCESS)
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
  int world::align(int          tid_A,
                         int          tid_B){
    int stat;
    
    /* check if the mappings of A and B are the same */
    stat = dt->check_pair_mapping(tid_A, tid_B);
    if (stat == 0){
      /* Align the mappings of A and B */
      stat = dt->map_tensor_pair(tid_B, tid_A);
      if (stat != SUCCESS)
        return stat;
    }
    return SUCCESS;
  }

  int world::print_tensor(FILE * stream, int       tid, double cutoff) {
    return dt->print_tsr(stream, tid, cutoff);
  }

  int world::compare_tensor(FILE * stream, int       tid_A, int       tid_B, double cutoff) {
    int stat = align(tid_A, tid_B);
    if (stat != SUCCESS) return stat;
    return dt->compare_tsr(stream, tid_A, tid_B, cutoff);
  }

  /* Prints contraction type. */
  int world::print_ctr(CTF_ctr_type_t const * ctype,
                             dtype const            alpha,
                             dtype const            beta) const {
    return dt->print_ctr(ctype,alpha,beta);
  }

  /* Prints sum type. */
  int world::print_sum(CTF_sum_type_t const * stype,
                             dtype const            alpha,
                             dtype const            beta) const {
    return dt->print_sum(stype,alpha,beta);
  }


  /**
   * \brief removes all tensors, invalidates all handles
   */
  int world::clean_tensors(){
    unsigned int i;
    std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
    for (i=0; i<tensors->size(); i++){
      dt->del_tsr(i);
  //    CTF_free((*tensors)[i]);
    }
    tensors->clear();
    return SUCCESS;
  }

  /**
   * \brief removes a tensor, invalidates its handle
   * \param tid tensor handle
   */
  int world::clean_tensor(int       tid){
    return dt->del_tsr(tid);
  }

  /**
   * \brief removes all tensors, invalidates all handles, and exits library.
   *              Do not use library instance after executing this.
   */
  int world::exit(){
    int ret;
    if (initialized){
      int rank = global_comm.rank;
      ret = world::clean_tensors();
      ASSERT(ret == SUCCESS);
      delete dt;
      initialized = 0;
      CTF_mem_exit(rank);
      if (CTF_get_num_instances() == 0){
  #ifdef OFFLOAD
        offload_exit();
  #endif
  #ifdef HPM
        HPM_Stop("CTF");
  #endif
        TAU_FSTOP(CTF);
      }
      return ret;
    } else
      return SUCCESS;
  }

  /* \brief ScaLAPACK back-end, see their DOC */
  int world::pgemm(char         TRANSA, 
                         char         TRANSB, 
                         int const    M, 
                         int const    N, 
                         int const    K, 
                         dtype        ALPHA,
                         dtype *      A, 
                         int const    IA, 
                         int const    JA, 
                         int const *  DESCA, 
                         dtype *      B, 
                         int const    IB, 
                         int const    JB, 
                         int const *  DESCB, 
                         dtype        BETA,
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
    int64_t old_size_C;
    int * old_phase_C, * old_rank_C, * old_virt_dim_C, * old_pe_lda_C;
    int * old_padding_C, * old_edge_len_C;
    int * need_free;
    int was_cyclic_C;
    tensor<dtype> * tsr_nC, * tsr_oC;
    CTF_ctr_type ct;
    fseq_tsr_ctr<dtype> fs;
  #ifdef OFFLOAD
    fs.is_offloadable=1;
  #endif
    std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
    CTF_alloc_ptr(3*sizeof(int), (void**)&need_free);
    ret = dt->pgemm(TRANSA, TRANSB, M, N, K, ALPHA, A, IA, JA, DESCA,
                    B, IB, JB, DESCB,
                    BETA, C, IC, JC, DESCC, &ct, &fs, need_free);
    if (ret != SUCCESS)
      return ret;

    otid_A = ct.tid_A;
    otid_B = ct.tid_B;
    otid_C = ct.tid_C;
  #if (!REDIST)
    ret = dt->try_topo_morph(otid_A, otid_B, otid_C);
    ASSERT(ret == SUCCESS);
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
    if (ret != SUCCESS)
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
        save_mapping<dtype>(tsr_nC, &old_phase_C, &old_rank_C, &old_virt_dim_C, 
                     &old_pe_lda_C, &old_size_C, &was_cyclic_C, 
                     &old_padding_C, &old_edge_len_C, 
                     dt->get_topo(tsr_nC->itopo));
        if (need_free[2])
          CTF_free(tsr_oC->data);
        tsr_oC->data = tsr_nC->data;
        remap_tensor(otid_C, tsr_oC, dt->get_topo(tsr_oC->itopo), old_size_C,
                     old_phase_C, old_rank_C, old_virt_dim_C,
                     old_pe_lda_C, was_cyclic_C,
                     old_padding_C, old_edge_len_C, global_comm);
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
    return SUCCESS;
  }


  /**
   * \brief define matrix from ScaLAPACK descriptor
   *
   * \param[in] DESCA ScaLAPACK descriptor for a matrix
   * \param[in] data pointer to actual data
   * \param[out] tid tensor handle
   */
  int world::def_scala_mat(int const * DESCA,
                                 dtype const * data,
                                 int * tid){
    int ret, stid;
    ret = dt->load_matrix((dtype*)data, DESCA, &stid, NULL);
    if (ret != SUCCESS) return ret;
    clone_tensor(stid, 1, tid);
    std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
    tensor<dtype> * stsr = (*tensors)[stid];
    tensor<dtype> * tsr = (*tensors)[*tid];
    CTF_free(stsr->data);
    stsr->is_data_aliased = 1;
    tsr->is_matrix = 1;
    tsr->slay = stid;
    return SUCCESS;
  }

  /**
   * \brief reads a ScaLAPACK matrix to the original data pointer
   *
   * \param[in] tid tensor handle
   * \param[in,out] data pointer to buffer data
   */
  int world::read_scala_mat(int       tid,
                                  dtype * data){
    int * old_phase, * old_rank, * old_virt_dim, * old_pe_lda;
    int * old_padding, * old_edge_len;
    int was_cyclic;
    int64_t old_size;
    std::vector< tensor<dtype>* > * tensors = dt->get_tensors();
    tensor<dtype> * tsr = (*tensors)[tid];
    tensor<dtype> * stsr = (*tensors)[tsr->slay];
    dt->unmap_inner(tsr);

    save_mapping(tsr, &old_phase, &old_rank, &old_virt_dim, 
                 &old_pe_lda, &old_size, &was_cyclic, 
                 &old_padding, &old_edge_len, (dt->get_topo(tsr->itopo)));
    
  //  ASSERT(tsr->is_matrix);

    CTF_alloc_ptr(sizeof(dtype)*tsr->size, (void**)&stsr->data);
    memcpy(stsr->data, tsr->data, sizeof(dtype)*tsr->size);
    remap_tensor(tsr->slay, stsr, dt->get_topo(stsr->itopo), old_size,
                 old_phase, old_rank, old_virt_dim,
                 old_pe_lda, was_cyclic,
                 old_padding, old_edge_len, global_comm);
    if (data!=NULL)
      memcpy(data, stsr->data, stsr->size*sizeof(dtype));  
    CTF_free(stsr->data);
    return SUCCESS;
  }
  /**
   * \brief CTF interface for pgemm
   */
  int world::pgemm(char         TRANSA, 
                         char         TRANSB, 
                         dtype        ALPHA,
                         int          tid_A,
                         int          tid_B,
                         dtype        BETA,
                         int          tid_C){
    int herm_A, herm_B, ret;
    CTF_ctr_type ct;
    fseq_tsr_ctr<dtype> fs;
  #ifdef OFFLOAD
    fs.is_offloadable=1;
  #endif
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
      ASSERT(TRANSA == 'T' || TRANSA == 't' || TRANSA == 'c' || TRANSA == 'C');
      if (TRANSA == 'c' || TRANSA == 'C')
        herm_A = 1;
      ct.idx_map_A[0] = 0;
      ct.idx_map_A[1] = 1;
    }
    if (TRANSB == 'N' || TRANSB == 'n'){
      ct.idx_map_B[0] = 0;
      ct.idx_map_B[1] = 2;
    } else {
      ASSERT(TRANSB == 'T' || TRANSB == 't' || TRANSB == 'c' || TRANSB == 'C');
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
    CTF_free(ct.idx_map_A);
    CTF_free(ct.idx_map_B);
    CTF_free(ct.idx_map_C);
    return ret;
  };

}

