#ifndef __WORLD_H__
#define __WORLD_H__

#include "common.h"
#include <set>
#include "../mapping/topology.h"

namespace CTF {
  /**
   * \defgroup World CTF World interface
   * \addtogroup World 
   * @{
   */


  /**
   * \brief an instance of the CTF library (world) on a MPI communicator
   */
  class World {
    public:
      /** \brief set of processors making up this world */
      MPI_Comm comm;
      /** \brief rank of local processor */
      int rank;
      /** \brief number of processors */
      int np;
      /** \brief derived topologies */
      std::vector< CTF_int::topology* > topovec;
      /** \brief whether the world has been initialized */
      bool initialized;
      /** \brief communicator data for MPI comm defining this world */
      CTF_int::CommData cdt;
      /** \brief main torus topology corresponding to the world */
      CTF_int::topology * phys_topology;
      /** \brief random number generator for this world object (same seed for each rank) */
      std::mersenne_twister_engine<std::uint_fast64_t, 64, 312, 156, 31,
                               0xb5026f5aa96619e9, 29,
                               0x5555555555555555, 17,
                               0x71d67fffeda60000, 37,
                               0xfff7eee000000000, 43, 6364136223846793005> glob_wrld_rng;



      /**
       * \brief creates CTF library on comm that can output profile data 
       *        into a file with a name based on the main args
       * \param[in] argc number of main arguments 
       * \param[in] argv main arguments 
       */
      World(int argc, char * const * argv);
    
      /**
       * \brief copy constructor, reallocates copies of all topologies
       * \param[in] other world to copy
       */
      World(World const & other);

      /**
       * \brief creates CTF library on comm that can output profile data 
       *        into a file with a name based on the main args
       * \param[in] comm MPI communicator associated with this CTF instance
       * \param[in] argc number of main arguments 
       * \param[in] argv main arguments 
       */
      World(MPI_Comm       comm = MPI_COMM_WORLD,
            int            argc = 0,
            char * const * argv = NULL);

      /**
       * \brief creates CTF library on comm
       * \param[in] order number of torus network dimensions
       * \param[in] lens lengths of torus network dimensions
       * \param[in] comm MPI global context for this CTF World
       * \param[in] argc number of main arguments 
       * \param[in] argv main arguments 
       */
      World(int            order, 
            int const *    lens, 
            MPI_Comm       comm = MPI_COMM_WORLD,
            int            argc = 0,
            char * const * argv = NULL);

      /**
       * \brief substitute for default constructor that does not initialize the world
       * \param[in] emptystring should be set to ""
       */
      World(char const * emptystring);


      /**
       * \brief frees CTF library
       */
      ~World();


      bool operator==(World const & other){ return comm==other.comm; }
      bool is_copy;
    private:
      /* whether this world is a copy of the universe object */

      /**
       * \brief initializes world stack and parameters, args only needed for profiler output
       * \param[in] argc number of arguments to application
       * \param[in] argv arguments to application
       */
      int initialize(int                   argc,
                     const char * const *  argv);

      /**
       * \brief  initializes library by determining topology based on mach specifier. 
       *
       * \param[in] global_context communicator decated to this library instance
       * \param[in] rank this pe rank within the global context
       * \param[in] np number of processors
       * \param[in] mach the type of machine we are running on
       * \param[in] argc number of arguments passed to main
       * \param[in] argv arguments passed to main
       */
      int init(MPI_Comm             global_context,
               CTF_int::TOPOLOGY    mach = CTF_int::TOPOLOGY_GENERIC,
               int                  argc = 0,
               const char * const * argv = NULL);


      /**
       * \brief  initializes library by determining topology based on 
       *      mesh of dimension order with edge lengths dim_len. 
       *
       * \param[in] global_context communicator decated to this library instance
       * \param[in] rank this pe rank within the global context
       * \param[in] np number of processors
       * \param[in] order the number of dimensions in the torus
       * \param[in] dim_len the size of the span of each dimension
       * \param[in] argc number of arguments passed to main
       * \param[in] argv arguments passed to main
       */
      int init(MPI_Comm       global_context,
               int            order,
               int const *    dim_len,
               int            argc = 0,
               const char * const * argv = NULL);
  };

  World & get_universe();
  /**
   * @}
   */

}

namespace CTF_int {
  class grid_wrapper {
    public:
      int pr;
      int pc;
      char layout;
      int ctxt;

      bool operator<(grid_wrapper const & other) const;
  };
  class grid_map_wrapper : public grid_wrapper{
    public:
      int * allranks;
      bool operator<(grid_map_wrapper const & other) const;
      //~grid_map_wrapper(){ free(allranks); }
  }; 

  extern std::set<grid_wrapper> scalapack_grids;
  extern std::set<grid_map_wrapper> scalapack_grid_maps;
}
#endif
