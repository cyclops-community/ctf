/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../../include/ctf.hpp"
#include "../shared/util.h"

CTF_Flop_Counter::CTF_Flop_Counter(){
  start_count = CTF_get_flops();
}

CTF_Flop_Counter::~CTF_Flop_Counter(){
}

void CTF_Flop_Counter::zero(){
  start_count = CTF_get_flops();
}

long_int CTF_Flop_Counter::count(MPI_Comm comm){
  long_int allf;
  long_int myf = (CTF_get_flops() - start_count);
  MPI_Allreduce(&myf,&allf,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
  return allf;
}
