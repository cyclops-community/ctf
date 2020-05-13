/*Copyright (c) 2011, Edgar Solomonik, all rights reserved.*/

#include "timer.h"
#include "common.h"

namespace CTF {
  Flop_counter::Flop_counter(){
    start_count = CTF_int::get_computed_flops();
  }

  Flop_counter::~Flop_counter(){
  }

  void Flop_counter::zero(){
    start_count = CTF_int::get_computed_flops();
  }

  int64_t Flop_counter::count(MPI_Comm comm){
    int64_t allf;
    int64_t myf = (CTF_int::get_computed_flops() - start_count);
    MPI_Allreduce(&myf,&allf,1,MPI_INT64_T,MPI_SUM,comm);
    return allf;
  }
}
