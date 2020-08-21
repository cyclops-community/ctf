/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "common.h"
#include "world.h"
#include "../shared/lapack_symbs.h"
#include "../tensor/algstrct.h"
#include "../shared/util.h"
#include "../shared/memcontrol.h"
#include "../shared/offload.h"

extern "C"
{
	void CTF_linked() {}
}

using namespace CTF_int;

namespace CTF_int {
  bool grid_wrapper::operator<(grid_wrapper const & other) const {
    if (this->layout == other.layout){
      if (this->pr == other.pr)
        return this->pc < other.pc;
      else
        return this->pr < other.pr;
    } else
      return this->layout == 'C';
  }

  bool grid_map_wrapper::operator<(grid_map_wrapper const & other) const {
    if (this->layout == other.layout){
      if (this->pr == other.pr)
        if (this->pc != other.pc)
          return this->pc < other.pc;
        else {
          for (int i=0; i<pr*pc; i++){
            if (this->allranks[i] != other.allranks[i]){
              return this->allranks[i] < other.allranks[i];
            }
          }
          return false;
        }
      else
        return this->pr < other.pr;
    } else
      return this->layout == 'C';
  }


  /** \brief index for ScaLAPACK processor grids */
  std::set<grid_wrapper> scalapack_grids;
  std::set<grid_map_wrapper> scalapack_grid_maps;
}

namespace CTF {
  bool universe_exists = false;
  World universe("");

  World::World(int            argc,
               char * const * argv){
    comm = MPI_COMM_WORLD;
#ifdef BGQ
    this->init(comm, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
    this->init(comm, TOPOLOGY_BGP, argc, argv);
#else
    this->init(comm, TOPOLOGY_GENERIC, argc, argv);
#endif
#endif
  }


  World::World(MPI_Comm       comm_,
               int            argc,
               char * const * argv){
    comm = comm_;
#ifdef BGQ
    this->init(comm, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
    this->init(comm, TOPOLOGY_BGP, argc, argv);
#else
    this->init(comm, TOPOLOGY_GENERIC, argc, argv);
#endif
#endif
  }


  World::World(int             order, 
               int const *     lens, 
               MPI_Comm        comm_,
               int             argc,
               char * const *  argv){
    comm = comm_;
    this->init(comm, order, lens, argc, argv);
  }

  World::World(World const & other){
    comm        = other.comm;
#if DEBUG >= 1
    if (other.rank == 0){
      printf("CTF WARNING: Creating copy of World, which is not free or useful, pass original World by reference instead if possible.\n");
    }
#endif
    //ASSERT(0);
    this->init(comm, other.phys_topology->order, other.phys_topology->lens, 0, NULL);
/*    cdt         = other.cdt;
    rank        = other.rank;
    np          = other.np;
    initialized = other.initialized;
  
    ASSERT(0); 
    for (int i=0; i<(int)other.topovec.size(); i++){
      topovec.push_back(other.topovec[i]);
    }*/
  }

  World::World(char const * emptystring){}

  World::~World(){
    if (!is_copy && this != &universe){
      for (int i=0; i<(int)topovec.size(); i++){
        delete topovec[i];
      }
      delete phys_topology;
      if (this->cdt.cm == MPI_COMM_WORLD){
        ASSERT(universe_exists);
        universe_exists = false;
      }
      topovec.clear();
    }

    initialized = 0;
    mem_exit(rank);
    //if (rank == 0) printf("deleting subworld of size %d, #instances is now %d\n",np,CTF_int::get_num_instances());
    if (get_num_instances() == 0){
      for (std::set<grid_wrapper>::iterator it=scalapack_grids.begin(); it!=scalapack_grids.end(); it++){
        //printf("HERE %d %d %d\n",it->pr,it->pc,it->ctxt);
        if (it->ctxt != -1)
          CTF_SCALAPACK::cblacs_gridexit(it->ctxt);
      }
      scalapack_grids.clear();
      for (std::set<grid_map_wrapper>::iterator it=scalapack_grid_maps.begin(); it!=scalapack_grid_maps.end(); it++){
        //printf("HERE %d %d %d\n",it->pr,it->pc,it->ctxt);
        delete it->allranks;
        if (it->ctxt != -1)
          CTF_SCALAPACK::cblacs_gridexit(it->ctxt);
      }
      scalapack_grid_maps.clear();
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
                  TOPOLOGY        mach,
                  int             argc,
                  const char * const *  argv){
    cdt = CommData(comm);
    if (mach == TOPOLOGY_GENERIC)
      phys_topology = NULL;
    else
      phys_topology = get_phys_topo(cdt, mach);
    
    return initialize(argc, argv);
  }

  int World::init(MPI_Comm const       global_context,
                  int                  order,
                  int const *          dim_len,
                  int                  argc,
                  const char * const * argv){

    cdt = CommData(global_context);
    phys_topology = new topology(order, dim_len, cdt, 1);

    return initialize(argc, argv);
  }


  int World::initialize(int                   argc,
                        const char * const *  argv){
    char * mem_size, * ppn;
    if (comm == MPI_COMM_WORLD && universe_exists){
      delete phys_topology;
      *this = universe;
      is_copy = true;
    } else {
      is_copy = false;
      glob_wrld_rng.seed(CTF_int::get_num_instances());
      MPI_Comm_rank(comm, &rank);
      MPI_Comm_size(comm, &np);
      if (phys_topology == NULL){
        phys_topology = get_phys_topo(cdt, TOPOLOGY_GENERIC);
        topovec = get_generic_topovec(cdt);
/*        std::vector<topology*> topovec2;
        topovec2 = peel_perm_torus(get_phys_topo(cdt, TOPOLOGY_GENERIC), cdt);
        printf("topovec size is %ld, via old method was %ld\n",topovec.size(), topovec2.size());*/
      } else
        topovec = peel_perm_torus(phys_topology, cdt);
    }
    CTF_int::mem_create();
    //if (rank == 0) printf("creating subworld of size %d, #instances is now %d\n",np,CTF_int::get_num_instances());
    if (CTF_int::get_num_instances() == 1){
      TAU_FSTART(CTF);
  #ifdef HPM
      HPM_Start("CTF");
  #endif
  #ifdef OFFLOAD
      offload_init();
  #endif
      int all_np;
      MPI_Comm_size(MPI_COMM_WORLD, &all_np);
      if (all_np != np){
        if (rank == 0){
          printf("CTF ERROR: the first CTF instance created has to be on MPI_COMM_WORLD\n");
          fflush(stdout);
        }
        MPI_Barrier(comm);
        IASSERT(0);
      } 
      init_rng(rank);
  
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
          VPRINTF(1,"Running with %d threads\n",omp_get_max_threads());
        }
      }
  #endif
      // Get the environment variable FILE_PATH
      char * file_path = getenv("CTF_MODEL_FILE");
      if (file_path != NULL && strcmp(file_path,"")!=0){
        if (rank == 0)
          VPRINTF(1,"Reading model coefficients from file %s (CTF_MODEL_FILE)\n", file_path);
        std::string coeff_file;
        coeff_file = std::string(file_path);
        CTF_int::load_all_models(coeff_file);
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
          printf("Assuming %d processes per node due to CTF_PPN environment variable\n",
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
    if (comm == MPI_COMM_WORLD){
      if (!universe_exists){
        universe_exists = true;
        universe = *this;
      } 
    }
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

  World & get_universe(){
    if (!universe_exists){
      World * pscp_universe = new World();
      pscp_universe->is_copy=true;
      delete pscp_universe;
    } 
    return universe;
  }

}
