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
tCTF_World<dtype>::tCTF_World(MPI_Comm comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, MACHINE_BGQ, rank, np);
#else
#ifdef BGP
  ctf->init(comm, MACHINE_BGP, rank, np);
#else
  ctf->init(comm, MACHINE_8D, rank, np);
#endif
#endif
}

template<typename dtype>
tCTF_World<dtype>::tCTF_World() {
  int rank, np;
  comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
#ifdef BGQ
  ctf->init(comm, MACHINE_BGQ, rank, np);
#else
#ifdef BGP
  ctf->init(comm, MACHINE_BGP, rank, np);
#else
  ctf->init(comm, MACHINE_8D, rank, np);
#endif
#endif
}

template<typename dtype>
tCTF_World<dtype>::tCTF_World(int const   ndim, 
                              int const * lens, 
                              MPI_Comm    comm_){
  int rank, np;
  comm = comm_;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &np);
  ctf = new tCTF< dtype >();
  ctf->init(comm, rank, np, ndim, lens);
}

template<typename dtype>
tCTF_World<dtype>::~tCTF_World(){
  delete ctf;
}


template class tCTF_World<double>;
template class tCTF_World< std::complex<double> >;
