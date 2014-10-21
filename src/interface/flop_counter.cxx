/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "timer.h"
#include "common.h"

namespace CTF {
  Flop_Counter::Flop_Counter(){
    start_count = CTF_int::get_flops();
  }

  Flop_Counter::~Flop_Counter(){
  }

  void Flop_Counter::zero(){
    start_count = CTF_int::get_flops();
  }

  int64_t Flop_Counter::count(MPI_Comm comm){
    int64_t allf;
    int64_t myf = (CTF_int::get_flops() - start_count);
    MPI_Allreduce(&myf,&allf,1,MPI_LONG_LONG_INT,MPI_SUM,comm);
    return allf;
  }
}
