/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"

using namespace CTF_int;

namespace CTF {

  World::World(int            argc,
               char * const * argv)  {
    MPI_Comm comm = MPI_COMM_WORLD;
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
                char * const * argv)  {
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
                                char * const *  argv) {
    comm = comm_;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &np);
    this->init(comm, rank, np, order, lens, argc, argv);
  }

  World::~World(){
  }


  int World::init(MPI_Comm const  global_context,
                  int             rank, 
                  int             np,
                  TOPOLOGY        mach,
                  int             argc,
                  const char * const *  argv){
    cdt = CommData(global_context);
    phys_topology = get_phys_topo(cdt, mach);
    topovec = peel_torus(phys_topology, cdt);
    
    return initialize(argc, argv);
  }

  int World::init(MPI_Comm const  global_context,
                        int             rank, 
                        int             np, 
                        int             order, 
                        int const *     dim_len,
                        int             argc,
                        const char * const *  argv){
    cdt = CommData(global_context);
    phys_topology = new topology(order, dim_len, cdt, 1);
    topovec = peel_torus(phys_topology, cdt);

    return initialize(argc, argv);
  }

  int World::initialize(int                   argc,
                        const char * const *  argv){
    char * mst_size, * stack_size, * mem_size, * ppn;
    int rank = cdt.rank;  

    CTF_mem_create();
    if (CTF_get_num_instances() == 1){
      TAU_FSTART(CTF);
  #ifdef HPM
      HPM_Start("CTF");
  #endif
  #ifdef OFFLOAD
      offload_init();
  #endif
      CTF_set_context(cdt.cm);
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




}
