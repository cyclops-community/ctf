/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"

using namespace CTF_int;

namespace CTF {

  World::World(int            argc,
               char * const * argv) : cdt(MPI_COMM_WORLD) {
    comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
#ifdef BGQ
    this->init(comm, rank, np, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
    this->init(comm, rank, np, TOPOLOGY_BGP, argc, argv);
#else
    this->init(comm, rank, np, TOPOLOGY_GENERIC, argc, argv);
#endif
#endif
  }


  World::World(MPI_Comm       comm_,
               int            argc,
               char * const * argv)  : cdt(comm_) {
    comm = comm_;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
#ifdef BGQ
    this->init(comm, rank, np, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
    this->init(comm, rank, np, TOPOLOGY_BGP, argc, argv);
#else
    this->init(comm, rank, np, TOPOLOGY_GENERIC, argc, argv);
#endif
#endif
  }


  World::World(int             order, 
               int const *     lens, 
               MPI_Comm        comm_,
               int             argc,
               char * const *  argv) : cdt(comm_) {
    comm = comm_;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    this->init(comm, rank, np, order, lens, argc, argv);
  }

  World::~World(){
    for (int i=0; i<(int)topovec.size(); i++){
      delete topovec[i];
    }
    topovec.clear();
    delete phys_topology;

    initialized = 0;
    mem_exit(rank);
    if (get_num_instances() == 0){
#ifdef OFFLOAD
      offload_exit();
#endif
#ifdef HPM
      HPM_Stop("CTF");
#endif
      TAU_FSTOP(CTF);
    }

  }


  int World::init(MPI_Comm const  global_context,
                  int             rank, 
                  int             np,
                  TOPOLOGY        mach,
                  int             argc,
                  const char * const *  argv){
    phys_topology = get_phys_topo(cdt, mach);
    topovec = peel_perm_torus(phys_topology, cdt);
    
    return initialize(argc, argv);
  }

  int World::init(MPI_Comm const       global_context,
                  int                  rank,
                  int                  np,
                  int                  order,
                  int const *          dim_len,
                  int                  argc,
                  const char * const * argv){
    phys_topology = new topology(order, dim_len, cdt, 1);
    topovec = peel_perm_torus(phys_topology, cdt);


    return initialize(argc, argv);
  }

  int World::initialize(int                   argc,
                        const char * const *  argv){
    char * mst_size, * stack_size, * mem_size, * ppn;
    int rank = cdt.rank;  

    CTF_int::mem_create();
    if (CTF_int::get_num_instances() == 1){
      TAU_FSTART(CTF);
  #ifdef HPM
      HPM_Start("CTF");
  #endif
  #ifdef OFFLOAD
      offload_init();
  #endif
      CTF::set_context(cdt.cm);
      CTF::set_main_args(argc, argv);

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
  #if 0 //def USE_MST
        if (rank == 0)
          VPRINTF(1,"Creating stack of size %ld\n",1000*(int64_t)1E6);
        CTF_int::mst_create(1000*(int64_t)1E6);
  #else
        if (rank == 0){
//          VPRINTF(1,"Running without stack, define CTF_STACK_SIZE environment variable to activate stack\n");
        }
  #endif
      } else {
#if 0
        int64_t imst_size = 0 ;
        if (mst_size != NULL) 
          imst_size = strtoull(mst_size,NULL,0);
        if (stack_size != NULL)
          imst_size = MAX(imst_size,strtoull(stack_size,NULL,0));
        if (rank == 0)
          printf("Creating stack of size %ld due to CTF_STACK_SIZE enviroment variable\n",
                    imst_size);
        CTF_int::mst_create(imst_size);
#endif
      }
      mem_size = getenv("CTF_MEMORY_SIZE");
      if (mem_size != NULL){
        int64_t imem_size = strtoull(mem_size,NULL,0);
        if (rank == 0)
          VPRINTF(1,"Memory size set to %ld by CTF_MEMORY_SIZE environment variable\n",
                    imem_size);
        CTF_int::set_mem_size(imem_size);
      }
      ppn = getenv("CTF_PPN");
      if (ppn != NULL){
        if (rank == 0)
          VPRINTF(1,"Assuming %d processes per node due to CTF_PPN environment variable\n",
                    atoi(ppn));
        ASSERT(atoi(ppn)>=1);
  #ifdef BGQ
        CTF_int::set_memcap(.75);
  #else
        CTF_int::set_memcap(.75/atof(ppn));
  #endif
      }
      if (rank == 0)
        VPRINTF(1,"Total amount of memory available to process 0 is %ld\n", proc_bytes_available());
    } 
    initialized = 1;
    return CTF_int::SUCCESS;
  }

/*
  void World::contract_mst(){
    std::list<mem_transfer> tfs = CTF_int::contract_mst();
    if (tfs.size() > 0 && get_global_comm().rank == 0){
      DPRINTF(1,"CTF Warning: contracting memory stack\n");
    }
    std::list<mem_transfer>::iterator it;
    int i;
    int j = 0;
    for (it=tfs.begin(); it!=tfs.end(); it++){
      j++;
      for (i=0; i<(int)tensors.size(); i++){
        if (tensors[i]->data == (dtype*)it->old_ptr){
          tensors[i]->data = (dtype*)it->new_ptr;
          break;
        }
      }
      if (i == (int)tensors.size()){
        printf("CTF ERROR: pointer %d on mst is not tensor data, aborting\n",j);
        ASSERT(0);
      }
      for (i=0; i<(int)tensors.size(); i++){
        if (tensors[i]->data == (dtype*)it->old_ptr){
          tensors[i]->data = (dtype*)it->new_ptr;
        }
      }

  }*/


}
