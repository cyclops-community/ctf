#include "partition.h"
#include "../shared/util.h"

namespace CTF {
  Partition::Partition(int order_, int const * lens_){
    order = order_;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, lens_, order*sizeof(int));
  }

  Partition::Partition(){
    order = 0;
    lens = NULL;
  }

  Partition::~Partition(){
    CTF_int::cdealloc(lens);
  }

  Partition::Partition(Partition const & other){
    order = other.order;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));
  }
  
  void Partition::operator=(Partition const & other){
    order = other.order;
    lens = (int*)CTF_int::alloc(order*sizeof(int));
    memcpy(lens, other.lens, order*sizeof(int));
  }


  Idx_Partition Partition::operator[](char const * idx){
    return Idx_Partition(*this, idx);
  }

  Idx_Partition::Idx_Partition(){
    part = Partition(0, NULL);
    idx = NULL;
  }

  Idx_Partition::Idx_Partition(Partition const & part_, char const * idx_){
    part = part_;
    idx = idx_;
  }
}
