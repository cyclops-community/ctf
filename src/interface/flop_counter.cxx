/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "../../include/ctf.hpp"
#include "../shared/util.h"

Flop_Counter::Flop_Counter(){
  start_count = get_flops();
}

Flop_Counter::~Flop_Counter(){
}

void Flop_Counter::zero(){
  start_count = get_flops();
}

int64_t Flop_Counter::count(MPI_Comm comm){
  int64_t allf;
  int64_t myf = (get_flops() - start_count);
  MPI_Allreduce(&myf,&allf,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
  return allf;
}
