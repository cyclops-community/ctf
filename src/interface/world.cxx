/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include <algorithm>
#include <iomanip>
#include <ostream>
#include <iostream>
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <limits.h>
#include <string.h>
#include "../../include/ctf.hpp"
#include "../shared/util.h"

namespace CTF {

World::World(int const      argc,
                              char * const * argv){
  int rank, np;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, rank, np, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
  ctf->init(comm, rank, np, TOPOLOGY_BGP, argc, argv);
#else
  ctf->init(comm, rank, np, TOPOLOGY_8D, argc, argv);
#endif
#endif
}


World::World(MPI_Comm       comm_,
                              int const      argc,
                              char * const * argv){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF();
#ifdef BGQ
  ctf->init(comm, rank, np, TOPOLOGY_BGQ, argc, argv);
#else
#ifdef BGP
  ctf->init(comm, rank, np, TOPOLOGY_BGP, argc, argv);
#else
  ctf->init(comm, rank, np, TOPOLOGY_8D, argc, argv);
#endif
#endif
}


World::World(int const       order, 
                              int const *     lens, 
                              MPI_Comm        comm_,
                              int const       argc,
                              char * const *  argv){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
  ctf->init(comm, rank, np, order, lens, argc, argv);
}

World::~World(){
  delete ctf;
}

}
