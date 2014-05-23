#ifndef __WORLD_H__
#define __WORLD_H__

#include <mpi.h>
#include "../world/int_world.h"

/**
 * \defgroup CTF CTF: C++ World interface
 * @{
 */

/**
 * \brief an instance of the CTF library (world) on a MPI communicator
 */
class World {
  public:
    MPI_Comm comm;
    Int_World * ctf;

  public:
    /**
     * \brief creates CTF library on comm_ that can output profile data 
     *        into a file with a name based on the main args
     * \param[in] comm_ MPI communicator associated with this CTF instance
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    World(int argc, char * const * argv);

    /**
     * \brief creates CTF library on comm_ that can output profile data 
     *        into a file with a name based on the main args
     * \param[in] comm_ MPI communicator associated with this CTF instance
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    World(MPI_Comm       comm_ = MPI_COMM_WORLD,
          int            argc = 0,
          char * const * argv = NULL);

    /**
     * \brief creates CTF library on comm_
     * \param[in] ndim number of torus network dimensions
     * \param[in] lens lengths of torus network dimensions
     * \param[in] comm MPI global context for this CTF World
     * \param[in] argc number of main arguments 
     * \param[in] argv main arguments 
     */
    World(int            ndim, 
          int const *    lens, 
          MPI_Comm       comm_ = MPI_COMM_WORLD,
          int            argc = 0,
          char * const * argv = NULL);

    /**
     * \brief frees CTF library
     */
    ~World();
};

/**
 * @}
 */
#endif
