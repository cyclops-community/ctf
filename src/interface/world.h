#ifndef __WORLD_H__
#define __WORLD_H__

#include "common.h"
#include "../mapping/topology.h"

/**
 * \defgroup CTF CTF: C++ World interface
 * @{
 */

namespace CTF {

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
      std::vector<CTF_int::topology> topovec;
      /** \brief whether the world has been initialized */
      bool initialized;
      /** \brief communicator data for MPI comm defining this world */
      CTF_int::CommData cdt;
      /** \brief main torus topology corresponding to the world */
      CTF_int::topology phys_topology;


      /**
       * \brief creates CTF library on comm that can output profile data 
       *        into a file with a name based on the main args
       * \param[in] comm MPI communicator associated with this CTF instance
       * \param[in] argc number of main arguments 
       * \param[in] argv main arguments 
       */
      World(int argc, char * const * argv);

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
       * \brief frees CTF library
       */
      ~World();
    private:
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
               int                  rank,
               int                  np,
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
               int            rank,
               int            np,
               int            order,
               int const *    dim_len,
               int            argc = 0,
               const char * const * argv = NULL);




  };

}

/**
 * @}
 */
#endif
