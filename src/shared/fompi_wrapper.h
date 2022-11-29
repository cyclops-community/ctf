#ifndef __FOMPI_WRAPPER__
#define __FOMPI_WRAPPER__

#ifdef USE_FOMPI
#include <fompi.h>

typedef foMPI_Win CTF_Win;
#define MPI_Init(...) foMPI_Init(__VA_ARGS__)
#define MPI_Win_create(...) foMPI_Win_create(__VA_ARGS__)
#define MPI_Win_fence(...) foMPI_Win_fence(__VA_ARGS__)
#define MPI_Win_free(...) foMPI_Win_free(__VA_ARGS__)
#define MPI_Put(...) foMPI_Put(__VA_ARGS__)
#else
#ifdef CRITTER
#include "critter.h"
#else
#include <mpi.h>
#endif
typedef MPI_Win CTF_Win;
#endif

#endif
