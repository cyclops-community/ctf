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

template<typename dtype>
tWorld<dtype>::tWorld(int const      argc,
                              char * const * argv){
  int rank, np;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, rank, np, MACHINE_BGQ, argc, argv);
#else
#ifdef BGP
  ctf->init(comm, rank, np, MACHINE_BGP, argc, argv);
#else
  ctf->init(comm, rank, np, MACHINE_8D, argc, argv);
#endif
#endif
}


template<typename dtype>
tWorld<dtype>::tWorld(MPI_Comm       comm_,
                              int const      argc,
                              char * const * argv){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, rank, np, MACHINE_BGQ, argc, argv);
#else
#ifdef BGP
  ctf->init(comm, rank, np, MACHINE_BGP, argc, argv);
#else
  ctf->init(comm, rank, np, MACHINE_8D, argc, argv);
#endif
#endif
}


template<typename dtype>
tWorld<dtype>::tWorld(int const       ndim, 
                              int const *     lens, 
                              MPI_Comm        comm_,
                              int const       argc,
                              char * const *  argv){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
  ctf->init(comm, rank, np, ndim, lens, argc, argv);
}

template<typename dtype>
tWorld<dtype>::~tWorld(){
  delete ctf;
}


template class tWorld<double>;
#ifdef COMPLEX
template class tWorld< std::complex<double> >;
#endif
